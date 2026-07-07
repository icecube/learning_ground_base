import pytorch_lightning

import jammy_flows
import jammy_flows.helper_fns

import torch
#from ray.tune.integration.pytorch_lightning import TuneReportCallback

from .. import config_parser

from .. import lr_schedulers

from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from lightning_fabric.utilities.cloud_io import _load as pl_load
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector as CheckpointConnector
import ray.tune as tune

from pytorch_lightning import seed_everything
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
import pylab

import numpy
import collections

from dataclasses import dataclass

from torch.nn.utils import convert_parameters

import os
import glob
import copy
import argparse

try:
    import see3po
except:
    print("See3po could not be imported!")
    see3po=None

try:
    import shampoo
except:
    print("shampoo optimizer could not be imported")
    shampoo=None

class system_base(pytorch_lightning.LightningModule):

    def __init__(self, config=dict()):
        super().__init__()

      
        self.cfg_parser=config_parser.config_parser()

        """
        self.extra_flow_detail=dict()
        if("global_flow_extra_detail" in config.keys()):
            self.extra_flow_detail=config["global_flow_extra_detail"]
        """

        ## check for deterministic behavior here
        self.cfg_parser.add_default_kwarg("det", "switch_off_pytorch_deterministic_flag", 0, int)
        _, deterministic_kwargs=self.cfg_parser.parse_cfg(config, "det")
        self.switch_off_deterministic_pytorch=deterministic_kwargs["switch_off_pytorch_deterministic_flag"]

        ########################
        ## optimizer config
        #######################

        ## general
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.lr", 1e-4, float)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.name", "adam", str)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.automatic", 1, int)

        ## see3po specific

        ## samples
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.fixed_len_avg_numsteps", 1, int)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.num_average_groups", 1, int)

        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.eps", 1e-8, float)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.epsilon", 1.0, float)
        #self.cfg_parser.add_default_kwarg("optimizer", "optimizer.diagonal_approximation", 1, int)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.hvp_averages", 0, int)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.keep_running_avg_of_differences", 1, int)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.use_adam_in_unused_dims", 1, int)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.use_geometric_averaging", 1, int)
        #self.cfg_parser.add_default_kwarg("optimizer", "optimizer.fine_grained_sample_averaging", 0, int)

        ## fisher fn
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.fisher_mode", "fisher_function", str)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.running_max_rank", -1, int)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.rank_averaging_mode", "largest", str)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.marginalize_fisher_over_batch", 0, int)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.beta1", 0.9, float)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.beta2", 0.999, float)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.beta3", 0.0, float)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.see3po_eps", 1e-8, float)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.empirical_fisher", 0, int)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.svd_tolerance", 1e-12, float)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.relative_grad_norm_factor", -1.0, float)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.relative_chi_trust_region", 0.0, float)
        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.absolute_grad_norm_allowed", -1.0, float)

        self.cfg_parser.add_default_kwarg("optimizer", "optimizer.save_fisher_eigenvecs_interval", 50, int)

        _, opti_kwargs=self.cfg_parser.parse_cfg(config, "optimizer")

        self.optimizer_config=opti_kwargs

        if(self.optimizer_config["optimizer.relative_chi_trust_region"]>0.0):
            assert(self.optimizer_config["optimizer.relative_grad_norm_factor"]<=0.0)

        if(self.optimizer_config["optimizer.relative_grad_norm_factor"]>0.0):
            assert(self.optimizer_config["optimizer.relative_chi_trust_region"]==0.0)

        config_parser.pretty_print("Optimizer Config", self.optimizer_config)

        if(opti_kwargs["optimizer.automatic"]==0):
            self.automatic_optimization=False

        ## switch off always with see3po
        if(opti_kwargs["optimizer.name"]=="see3po"):
            self.automatic_optimization=False

        ########### scheduler options
        ########################

        scheduler_choices=["bayes_reduce", "cosine_annealing_reduce"]
        self.cfg_parser.add_default_kwarg("scheduler", "scheduler.name", "bayes_reduce", str, choices=scheduler_choices)
        self.cfg_parser.add_default_kwarg("scheduler", "scheduler.starting_lr_scaling_factor", 10.0, float)
        self.cfg_parser.add_default_kwarg("scheduler", "scheduler.target_lr_fudge_factor", 1.0, float)

        ## averaging steps = -1 means averaging steps are defined by N/B (total_dataset_size/batch_size)
        self.cfg_parser.add_default_kwarg("scheduler", "scheduler.num_validation_reducing_steps", -1, int) # -1 means it is automatically determined from dataset size
        self.cfg_parser.add_default_kwarg("scheduler", "scheduler.cumulative_averaging_value", -1.0, float) # < 0: standard avg, >0, cumulative meaning
        self.cfg_parser.add_default_kwarg("scheduler", "scheduler.eps", 1e-8, float)
        self.cfg_parser.add_default_kwarg("scheduler", "scheduler.threshold", 1e-4, float)
        self.cfg_parser.add_default_kwarg("scheduler", "scheduler.tracking_metric", "", str)
        self.cfg_parser.add_default_kwarg("scheduler", "scheduler.force_reduction_in_fixed_intervals", 0, int, choices=[0,1])
        self.cfg_parser.add_default_kwarg("scheduler", "scheduler.warmup_period_steps", -1, int)

        _, scheduler_kwargs = self.cfg_parser.parse_cfg(config, "scheduler")
        self.scheduler_config=scheduler_kwargs
        self.scheduler=None

        config_parser.pretty_print("Scheduler Config", self.scheduler_config)

        #########

        ## used to spawn clone of itself
        self.config_at_init=config

        self._setup(config)

        self.define_encoding_scheme_collates()
        #####

    def _setup(self, config):

        raise NotImplementedError("Main *_setup* routing has to implemented by each subclass!")

    ## subclasses can implement specific inits here
    def _set_training_settings(self, config, datamodule=None, current_epoch=0, current_total_iter=0):
        print("-- SUBCLASS DOES NOT IMPLEMENT CUSTOM INITIALIZATION OF TRAINING SETTINGS CURRENTLY --")
        print("-- implement *_set_training_settings()* in system subclass to init training-specific settings --")
        sys.exit(-1)
        return

    def set_training_settings(self, dataset_size, batch_size, val_interval, val_names, config, datamodule, current_epoch=0, current_total_iter=0):
        """
        Called by the optimization engine before starting the loop .. used to gauge scheduler settings.
        """
        print("-> Setting training settings on the module <-")
        self.val_interval=val_interval

        ## those items *have* to be defined!!!

        self.total_dataset_size=dataset_size
        self.train_batch_size=batch_size
       
        ## indepedent steps in an epoch
        self.num_independent_steps=int(self.total_dataset_size/self.train_batch_size)
    
        ## number of actual validation steps per epoch
        num_validation_steps_per_sweep=int(self.num_independent_steps/self.val_interval)
        #assert(num_independent_steps%self.val_interval==0), ("Number of independent steps must be divisible by validation interval.", num_independent_steps, self.val_interval)

        self.num_validation_steps_per_sweep=num_validation_steps_per_sweep

        ## also update validation 
        self.validation_names=[]
        for val_name in val_names:
            self.validation_names.append(val_name)

        ## call config/model specific training inits, useful for example for custom dtype settings
        self._set_training_settings(config, datamodule=datamodule, current_epoch=current_epoch, current_total_iter=current_total_iter)

        print(" -> updated training settings for model ...")


    def _init_scheduler(self, scheduler_name, optimizer, target_lr, dataset_num_averaging_steps, dataset_num_reducing_steps):

        reducing_steps=dataset_num_reducing_steps
        if(self.scheduler_config["scheduler.num_validation_reducing_steps"]>0):
            reducing_steps=self.scheduler_config["scheduler.num_validation_reducing_steps"]
        
     
        scheduler_eps=self.scheduler_config["scheduler.eps"]
        scheduler_threshold=self.scheduler_config["scheduler.threshold"]

        if(scheduler_name=="bayes_reduce"):
            
            

            scheduler=lr_schedulers.ModifiedPlateauReducer(optimizer, mode='min', factor=0.5, patience=10,
             threshold=scheduler_threshold, threshold_mode='abs', cooldown=0,
             min_lr=target_lr, eps=scheduler_eps, verbose=True, averaging_steps=dataset_num_averaging_steps, reducing_steps=reducing_steps, cumulative_averaging_value=self.scheduler_config["scheduler.cumulative_averaging_value"])

          
            ## this function is called manually in validation step
            #self.scheduler=scheduler_dict
            print("initialized PlateauReduce scheduler")
        elif(scheduler_name=="cosine_annealing_reduce"):
            print("initialized COS annealing scheduler")
            print("Num reducing steps handed", dataset_num_reducing_steps)
            print("num reducing steps scheduler ", self.scheduler_config["scheduler.num_validation_reducing_steps"])
            print("taken ... ", reducing_steps)

            scheduler=lr_schedulers.make_reducing_and_averaging_scheduler(CosineAnnealingLR)(optimizer, 
                                                                                             T_max=10, 
                                                                                             eta_min=target_lr, 
                                                                                             verbose=True,
                                                                                             #red_eps=scheduler_eps,
                                                                                             red_min_lr=target_lr,
                                                                                             red_threshold=scheduler_threshold,
                                                                                             red_factor=0.5,
                                                                                             red_patience=10,
                                                                                             red_mode="min",
                                                                                             red_threshold_mode="abs",
                                                                                             red_averaging_steps=dataset_num_averaging_steps,
                                                                                             red_reducing_steps=reducing_steps,
                                                                                             red_cumulative_averaging_value=self.scheduler_config["scheduler.cumulative_averaging_value"],
                                                                                             tracking_metric=self.scheduler_config["scheduler.tracking_metric"],
                                                                                             force_reduction_in_fixed_intervals=self.scheduler_config["scheduler.force_reduction_in_fixed_intervals"],
                                                                                             warmup_period_steps=self.scheduler_config["scheduler.warmup_period_steps"]
                                                                                              )

            #scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=target_lr, verbose=True)

        elif(scheduler_name!=""):
            raise Exception("Unsupported scheduler ", scheduler_name)

        return scheduler

    def configure_optimizers(self):
       
        self.scheduler=None

        starting_lr=self.optimizer_config["optimizer.lr"]

        #train_batch_size=self.data_module.train_batch_size
        #total_dataset_size=self.data_module.train_dataset_size

        ## target LR corresponds to a tempreature T=1
        ## this is the target LR when using a diagonal preconditioner ala ADAM (corollary 3 in Mandt et al 2017)
        target_lr=self.scheduler_config["scheduler.target_lr_fudge_factor"]*2*self.train_batch_size/self.total_dataset_size

        ## we start with a higher LR than the target LR, where the difference is given by a Temperature scaling (T=1 -> target LR)
        optimal_starting_lr=self.scheduler_config["scheduler.starting_lr_scaling_factor"]*target_lr

        ## TODO: include some check if target LR is too large
        ## Mandt et al (2017), p 20 -> LR < 2/lambda_max where lambda_max is max EV of Noise target Hessian
        ## in order to fullfill condition, LR might have to be decrased .. at the same time Batch size B has to be 
        ## increased accordingly 

        ## we use the theoretically determined optimal LR if the LR from the config is <= 0
        if(starting_lr<=0):
            starting_lr=optimal_starting_lr

            ## if dynamically determined starting lr is smaller than 1e-3 (typically smaller is too small), increase to 1e-3
            if(starting_lr<1e-3):
                starting_lr=1e-3

        ## make sure target LR is smaller than starting LR
        if(starting_lr<target_lr):
            target_lr=starting_lr

        print("dataset size: ", self.total_dataset_size)
        print("train batch size: ", self.train_batch_size)
        print("-------------")
        print("Starting Learning rate is ..", starting_lr)
        print("---------------")
        print("-------------")
        print("Target Learning rate is ..", target_lr)
        print("---------------")
        print(self.optimizer_config.keys())
        beta1=self.optimizer_config["optimizer.beta1"]
        beta2=self.optimizer_config["optimizer.beta2"]
        beta3=self.optimizer_config["optimizer.beta3"]

        if(self.optimizer_config["optimizer.name"]=="adam"):
            print("setting eps to ... ",self.optimizer_config["optimizer.eps"])
            opti=torch.optim.Adam(self.parameters(), starting_lr, betas=(beta1, beta2), eps=self.optimizer_config["optimizer.eps"])
        elif(self.optimizer_config["optimizer.name"]=="shampoo"):

            hyperparams=dict()
            """Shampoo hyper parameters."""
            hyperparams["beta2"]= beta2 # default=1.0
            hyperparams["diagonal_eps"] = self.optimizer_config["optimizer.eps"] # 1e-6 default
            hyperparams["matrix_eps"]= 1e-12
            hyperparams["weight_decay"] = 0.0
            hyperparams["inverse_exponent_override"]= 0  # fixed exponent for preconditioner, if >0
            hyperparams["start_preconditioning_step"]= 1
            # Performance tuning params for controlling memory and compute requirements.
            # How often to compute preconditioner.
            hyperparams["preconditioning_compute_steps"]= 1
            # How often to compute statistics.
            hyperparams["statistics_compute_steps"]= 1
            # Block size for large layers (if > 0).
            # Block size = 1 ==> Adagrad (Don't do this, extremely inefficient!)
            # Block size should be as large as feasible under memory/time constraints.
            hyperparams["block_size"] = 128
            # Automatic shape interpretation (for eg: [4, 3, 1024, 512] would result in
            # 12 x [1024, 512] L and R statistics. Disabled by default which results in
            # Shampoo constructing statistics [4, 4], [3, 3], [1024, 1024], [512, 512].
            hyperparams["best_effort_shape_interpretation"]= True
            # Type of grafting (SGD or AdaGrad).
            # https://arxiv.org/pdf/2002.11803.pdf
            hyperparams["graft_type"]= shampoo.LayerwiseGrafting.ADAGRAD ## grafting?
            # Nesterov momentum
            hyperparams["nesterov"]= False

            hyperparams=argparse.Namespace(**hyperparams)

            print("setting eps to ... ",self.optimizer_config["optimizer.eps"])
            opti=shampoo.Shampoo(self.parameters(), lr=starting_lr, momentum=beta1, hyperparams=hyperparams)

        elif(self.optimizer_config["optimizer.name"]=="sgd"):
            opti=torch.optim.SGD(self.parameters(), starting_lr)
     
        elif(self.optimizer_config["optimizer.name"]=="see3po"):
            
            epsilon=self.optimizer_config["optimizer.epsilon"]
            avg_hvp=self.optimizer_config["optimizer.hvp_averages"]
            keep_running_avg_of_differences=self.optimizer_config["optimizer.keep_running_avg_of_differences"]
            num_average_groups=self.optimizer_config["optimizer.num_average_groups"]
            averaging_len=self.optimizer_config["optimizer.fixed_len_avg_numsteps"]
            use_geometric_averaging=self.optimizer_config["optimizer.use_geometric_averaging"]
            #fine_grained_sample_averaging=self.optimizer_config["optimizer.fine_grained_sample_averaging"]

            use_adam_in_unused_dims=self.optimizer_config["optimizer.use_adam_in_unused_dims"]

            fisher_mode=self.optimizer_config["optimizer.fisher_mode"]
            rank_averaging_mode=self.optimizer_config["optimizer.rank_averaging_mode"]
            running_max_rank=self.optimizer_config["optimizer.running_max_rank"]
            marginalize_fisher_over_batch=self.optimizer_config["optimizer.marginalize_fisher_over_batch"]

            empirical_fisher=self.optimizer_config["optimizer.empirical_fisher"]
            svd_tolerance=self.optimizer_config["optimizer.svd_tolerance"]

            eps=self.optimizer_config["optimizer.see3po_eps"]

            save_fisher_eigenvecs_interval=self.optimizer_config["optimizer.save_fisher_eigenvecs_interval"]

            opti=see3po.see3po.see3po(self.parameters(), 
                        lr=starting_lr,
                        use_adam_in_unused_dims=use_adam_in_unused_dims, 
                        epsilon=epsilon, 
                        betas=(beta1, beta2, beta3),
                        hvp_averages=avg_hvp, 
                        num_average_groups=num_average_groups, 
                        keep_running_avg_of_differences=keep_running_avg_of_differences, 
                        fixed_len_avg_numsteps=averaging_len, 
                        use_geometric_averaging=use_geometric_averaging, 
                        empirical_fisher=empirical_fisher,
                        fisher_mode=fisher_mode,
                        rank_averaging_mode=rank_averaging_mode,
                        running_max_rank=running_max_rank,
                        svd_tolerance=svd_tolerance,
                        eps=eps,
                        marginalize_fisher_over_batch=marginalize_fisher_over_batch,
                        save_fisher_eigenvecs_interval=save_fisher_eigenvecs_interval
                        )

        else:
      
            raise Exception("UNKNOWN OPTIMIZER ",self.optimizer_config["optimizer.name"])

        ## init scheduler and save internally
        
        

        #num_independent_steps=int(total_dataset_size/train_batch_size)
        assert(self.total_dataset_size%self.train_batch_size==0), "Dataset size must be divisible by train batch size."

        #num_validation_steps_per_sweep=int(self.num_independent_steps/self.val_interval)
        assert(self.num_independent_steps%self.val_interval==0), "Number of independent steps must be divisible by validation interval."

        scheduler_obj=self._init_scheduler(self.scheduler_config["scheduler.name"], opti, target_lr, self.num_independent_steps, self.num_validation_steps_per_sweep)
        print("loaded sched obj ", scheduler_obj)

        used_monitor_name=self.scheduler_config["scheduler.tracking_metric"]#"s.val.%s.total" % self.validation_names[0]
       
        ### TODO: Overwrite monitor loss on demand!!!

        ## return optimizer  
        return {"optimizer": opti, "lr_scheduler": scheduler_obj, "monitor": used_monitor_name}

    def clone_system(self, new_parameters=None):

        # this should work without deepcopy..
        self_ref_class=self._get_self_ref_class()
        
        new_obj=self_ref_class(config=self.config_at_init)

        ## also put on same device as mother model
        new_obj.to(next(self.parameters()).device)

        if(new_parameters is not None):
            
           
            convert_parameters.vector_to_parameters(new_parameters, new_obj.parameters())
            
           

        return new_obj
           
            

       

    def get_labels(self, batch):
        
        with torch.no_grad():

            return {"labels": batch["labels"].clone()}

    def init_pdfs(self, labels):

        self._init_pdfs(labels)

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)   

    def obtain_new_model(self, new_param_vector):

        averaged_model=self.clone_system(new_parameters=new_param_vector)

        # set averaged model to desired device
        for p in self.parameters():
            averaged_model.to(p.device)
            break

        ## A LITTLE HACKY TO CALL virtual method here
        ## IN REALITY WE ONLY NEED TO SET PRECISION OF MODEL HERE
        ## None only given for backwrads compatibility (datmodule is now kwarg)
        averaged_model._set_training_settings(self.config_at_init, None)

        return averaged_model
        
    def training_step(self, batch, batch_idx):
       
        #print("-------------------> begin training step <----------------")
        loss=self._training_step(batch, batch_idx)

        sched=self.lr_schedulers()

        if(sched is not None):
                
            ## only update a param vector if LR scheduler supports it!
            if(hasattr(sched, "averaged_param_vector")):
                par_vec=convert_parameters.parameters_to_vector(self.parameters()).detach()

                sched.update_running_parameters(par_vec)


        #print("--------------------> end training step <-------------------")

        return loss

    def on_validation_epoch_start(self):
        """
        Obtain averaged model at beginning of validation epoch
        """
        print("VALIDATON START -------------- >")
        sched=self.lr_schedulers()

        self._averaged_model=None
        if(sched is not None):

            if(sched.averaged_param_vector is not None):

                self._averaged_model=self.obtain_new_model(sched.averaged_param_vector)

        self.validation_loss_dict=dict()

        self._num_cumulative_validation_steps=0

    def on_validation_epoch_end(self, *args, **kwargs):

        print("EPOCH END")
        
        if(self._num_cumulative_validation_steps>1):

            for val_key in self.validation_loss_dict:
                
                if(len(self.validation_loss_dict[val_key])>0):

                    # per batch is a single already averaged value
                    if( len(self.validation_loss_dict[val_key][0].shape)==0):
                        self.validation_loss_dict[val_key]=torch.stack(self.validation_loss_dict[val_key])
                    else:
                        ## per batch is an array
                        self.validation_loss_dict[val_key]=torch.cat([i.flatten() for i in self.validation_loss_dict[val_key]])
                    print(val_key)

                    print("1st mean ", self.validation_loss_dict[val_key])
                    self.validation_loss_dict[val_key]=self.validation_loss_dict[val_key][numpy.isfinite(self.validation_loss_dict[val_key])]
                    ###########
                    print("prev mean ", self.validation_loss_dict[val_key])
                    self.validation_loss_dict[val_key]=self.validation_loss_dict[val_key].mean()
                    
                    print("final mean ", self.validation_loss_dict[val_key])
                    self.log(val_key, self.validation_loss_dict[val_key])

                else:
                    print(self.validation_loss_dict[val_key])

                    raise Exception("Empty list should not happen here!")

               
            
            
            ## scheduler

            sched=self.lr_schedulers()


            if(sched is not None):

                ## only care about scheduling in first validation loader
                
                if("reduce" in self.scheduler_config["scheduler.name"]):

                    sched.step(metrics=self.validation_loss_dict)

                    ## log averaged result if possible
                    
                    if(hasattr(sched, "eta_min")):
                        self.log("target_lr", sched.eta_min, add_dataloader_idx=False)
                    elif(hasattr(sched, "min_lrs")):
                        self.log("target_lr", sched.min_lrs[0], add_dataloader_idx=False)

                else:
                    raise Exeption("Currently not supported.. must use reducer!")
                    sched.step()

                    # eta_min is specific to CosAnnealing
                    if(hasattr(sched, "eta_min")):
                        self.log("target_lr", sched.eta_min, add_dataloader_idx=False)
                print("last LR...", sched._last_lr[0])
                self.log("running_lr", sched._last_lr[0], add_dataloader_idx=False)
                
        self._averaged_model=None
        self.validation_loss_dict=dict()

        print("------------------------->")

    def validation_step(self, batch, batch_idx, *args):

        dataloader_id_to_use=0
        if(len(args)>0):
            print("extra args (validation idx?) ", args)
            dataloader_id_to_use=args[0]

        validation_name=self.validation_names[dataloader_id_to_use]
   
        if(self.switch_off_deterministic_pytorch):
            torch.use_deterministic_algorithms(False)

        print("################### validation %s ... batch %d ######################" % (validation_name, batch_idx))
        print(batch["labels"].shape)
        print("val energies..", min(batch["event_properties"]["log10_deposited_energy"]), max(batch["event_properties"]["log10_deposited_energy"]))

        this_batch_loss_dict=self._validation_step(batch, batch_idx, validation_name, log=False)
        ## add normal items to global dict
        for k in this_batch_loss_dict.keys():
            if(k in self.validation_loss_dict):
                self.validation_loss_dict[k].append(this_batch_loss_dict[k].cpu())
            else:
                self.validation_loss_dict[k]=[this_batch_loss_dict[k].cpu()]


        if(self._averaged_model is not None):
            this_batch_loss_dict_averaged=self._averaged_model._validation_step(batch, batch_idx, validation_name, log=False)

            ## add averaged items to global dict
            for k in this_batch_loss_dict_averaged.keys():
                avg_k=k+"_averaged"
                if(avg_k in self.validation_loss_dict):
                    self.validation_loss_dict[avg_k].append(this_batch_loss_dict_averaged[k].cpu())
                else:
                    self.validation_loss_dict[avg_k]=[this_batch_loss_dict_averaged[k].cpu()]

        else:
            # in the first step we dont average yet... copy over
            for k in this_batch_loss_dict.keys():
                if( (k+"_averaged") in self.validation_loss_dict):
                    self.validation_loss_dict[k+"_averaged"].append(this_batch_loss_dict[k].cpu())
                else:
                    self.validation_loss_dict[k+"_averaged"]=[this_batch_loss_dict[k].cpu()]

        """    
        sched=self.lr_schedulers()

        if(sched is not None):

            if(dataloader_id_to_use==0):
            
                ## only care about scheduling in first validation loader
                
                if("reduce" in self.scheduler_config["scheduler.name"]):

                    sched.step(metrics=loss_dict)

                    ## log averaged result if possible
                    
                    if(hasattr(sched, "eta_min")):
                        self.log("target_lr", sched.eta_min, add_dataloader_idx=False)
                    elif(hasattr(sched, "min_lrs")):
                        self.log("target_lr", sched.min_lrs[0], add_dataloader_idx=False)

                else:
                    raise Exeption("Currently not supported.. must use reducer!")
                    sched.step()

                    # eta_min is specific to CosAnnealing
                    if(hasattr(sched, "eta_min")):
                        self.log("target_lr", sched.eta_min, add_dataloader_idx=False)

                self.log("running_lr", sched._last_lr[0], add_dataloader_idx=False)
                

            ## average everything
            if(sched.averaged_param_vector is not None):

                averaged_model=self.obtain_new_model(sched.averaged_param_vector)

                loss_dict=averaged_model._validation_step(batch, batch_idx, validation_name, log=False)

                ## save averaged metrics
                for ld_key in loss_dict:
                    self.log("%s_averaged" % ld_key, loss_dict[ld_key].detach(), add_dataloader_idx=False)
        """

        self._num_cumulative_validation_steps+=1

    #####

    def helper_get_relative_indices_by_names(self, names):
        """
        This helper function returns a joint union of all parameter and group indices, respectively, of all the sub-modules named by *names*.
        """

        def check_list(total_list, partial_list):

            this_tot_count=0
            this_param_base_index=0

            global_indices=[]

            all_param_indices=0

            for ind, total_param in enumerate(total_list):
                
                if(total_param.data_ptr()==partial_list[0].data_ptr()):
               
                    this_param_base_index=all_param_indices

                for part in partial_list:
                    if(part.data_ptr()==total_param.data_ptr()):
                
                        global_indices.append(ind)

                        this_tot_count+=total_param.numel()

                all_param_indices+=total_param.numel()

            global_indices=numpy.array(global_indices)

            ## make sure the param indices are following each other and are in consecutive groups
            assert(len(global_indices)==len(partial_list))
            assert( ((global_indices[1:]-global_indices[:-1])!=1).sum()==0  )

            ## obtain group indices and overall param indices 
            return slice(global_indices[0], global_indices[0]+len(partial_list)), slice(this_param_base_index, this_param_base_index+this_tot_count)
        
        def merge(plist):
            """
            Merge a single ordered slice list when sometimes the slices can touch.
            """
            merged_list=[]

            current=plist[0]

            for item in plist[1:]:
                if(item.start==current.stop):
                    ## extend
                    current=slice(current.start, item.stop)
                else:
                    merged_list.append(current)
                    current=item

            merged_list.append(current)

            return merged_list

        
        tot_list=list(self.parameters())

     
      
        group_list=[]
        param_list=[]
        #print(self.named_modules().keys())
        for key,obj in self.named_modules():
            if(key in names):
                
                partial_list=list(obj.parameters())

                totel=0
               
                for p in obj.named_parameters():
                   
                    totel+=p[1].numel()

                
                local_groups, local_params=check_list(tot_list, partial_list)

                group_list.append(local_groups)
                param_list.append(local_params)
                

        group_list.sort()
        param_list.sort()

     
        merged_groups=merge(group_list)
        merged_params=merge(param_list)

        ## hisft params relative to the union
        relative_to_union_shifted_params=[]
        baseline_union=merged_params[0].start
        for p in merged_params:
            relative_to_union_shifted_params.append(slice(p.start-baseline_union, p.stop-baseline_union))

        

        return merged_params, merged_groups#, relative_to_union_shifted_params


    def obtain_parameter_specs_for_optimization(self, parameter_kwargs):
        """
        Should return a dictionary of "parameter specs" for different parameter groups to be passed on to the optimizer.
        Parameters:
        parameter_kwargs: dict containing parameter info to communicate which parameter groups should be part of fisher calculation.
        """

        def check_overlap(indices1, indices2):
            for ind1 in indices1:
                for ind2 in indices2:
                    ## exclusive higher borders -> equal/greater means no overlap
                    if(ind1.start>=ind2.stop):
                        continue

                    if(ind2.start >= ind1.stop):
                        continue

                    ## there is some overlap
                    return True

            return False


        def merge_indices(plist1, plist2):
            """
            Merge two individually sorted slice lists.
            """
           
            def update_merged(m):
                updated=[]

                cur_running=m[0]

                for ind in range(len(m)):
                    #if(ind<len(m)-1):
                    if(m[ind].start > cur_running.stop):
                        updated.append(cur_running)
                        cur_running=m[ind]
                    else:
                        assert(m[ind].start<=cur_running.stop and m[ind].stop >= cur_running.start)
                        ## merge

                        cur_running=slice(min(m[ind].start, cur_running.start), max(m[ind].stop, cur_running.stop))
                    
                    cur_running=slice(min(m[ind].start, cur_running.start), max(m[ind].stop, cur_running.stop))

                updated.append(cur_running)

                return updated

            merged=[]

            for p1 in plist1:
                current=p1
                for p2 in plist2:

                    ## do slices touch?
                    if(p2.start > current.stop):
                        continue
                    if(current.start > p2.stop):
                        continue

                    ## ok they touch .. merge them

                    current=slice(min(current.start,p2.start), max(current.stop, p2.stop))

                merged.append(current)
                
                merged=update_merged(merged)
          
            return merged



        tot_parameter_count=0

        param_list=list(self.parameters())
       
        for param in param_list:
            
            tot_parameter_count+=param.numel()
            
        individual_param_specs=self._obtain_parameter_specs_for_optimization(parameter_kwargs)

        ## go through all parameters and check which ones are missed by standard fisher

        ## look for overlap and generate overlapping param groups
        joint_param_groups=dict()
        joint_param_groups["joint"]=collections.OrderedDict()
        joint_param_groups["single"]=collections.OrderedDict()

        overall_used_names=[]

        joint_group_counter=0

        for parspec1 in individual_param_specs.keys():
            #joint_names=[parspec1]
            #joint_param_indices=(individual_param_specs[parspec1]["param_indices"][0][0], individual_param_specs[parspec1]["param_indices"][-1][1])
            #joint_group_indices=(individual_param_specs[parspec1]["group_indices"][0][0], individual_param_specs[parspec1]["group_indices"][-1][1])
          
            #overall_used_names.append(parspec1)
            if(len(joint_param_groups["joint"].keys())==0):
                ## create new joint group, no existing yet
                group_id="group_%d" % joint_group_counter
                joint_param_groups["joint"][group_id]=dict()

                joint_param_groups["joint"][group_id]["names"]=[parspec1]
                joint_param_groups["joint"][group_id]["boundary_param_indices"]=slice(individual_param_specs[parspec1]["param_indices"][0].start, individual_param_specs[parspec1]["param_indices"][-1].stop)
                joint_param_groups["joint"][group_id]["merged_param_indices"]=individual_param_specs[parspec1]["param_indices"].copy()
                joint_param_groups["joint"][group_id]["boundary_group_indices"]=slice(individual_param_specs[parspec1]["group_indices"][0].start, individual_param_specs[parspec1]["group_indices"][-1].stop)
                

                joint_group_counter+=1
                continue

            found=False
            for joint_group in joint_param_groups["joint"].keys():
               
                if(found):
                    break
                for name in joint_param_groups["joint"][joint_group]["names"]:
                    
                  
                    if(name!=parspec1):
                        if(check_overlap( individual_param_specs[parspec1]["param_indices"], individual_param_specs[name]["param_indices"])):
                            ## add parspec to group
                          
                            if(parspec1 not in joint_param_groups["joint"][joint_group]["names"]):
                                joint_param_groups["joint"][joint_group]["names"].append(parspec1)
                            else:
                                raise Exception("This should not happen! Check logic!")

                            joint_param_indices=slice(min(joint_param_groups["joint"][joint_group]["boundary_param_indices"].start, individual_param_specs[parspec1]["param_indices"][0].start), joint_param_groups["joint"][joint_group]["boundary_param_indices"].stop)
                            joint_param_indices=slice(joint_param_groups["joint"][joint_group]["boundary_param_indices"].start, max(joint_param_groups["joint"][joint_group]["boundary_param_indices"].stop, individual_param_specs[parspec1]["param_indices"][-1].stop))

                            joint_group_indices=slice(min(joint_param_groups["joint"][joint_group]["boundary_group_indices"].start, individual_param_specs[parspec1]["group_indices"][0].start), joint_param_groups["joint"][joint_group]["boundary_group_indices"].stop)
                            joint_group_indices=slice(joint_param_groups["joint"][joint_group]["boundary_group_indices"].start, max(joint_param_groups["joint"][joint_group]["boundary_group_indices"].stop, individual_param_specs[parspec1]["group_indices"][-1].stop))

                            joint_param_groups["joint"][group_id]["boundary_param_indices"]=joint_param_indices
                            joint_param_groups["joint"][group_id]["boundary_group_indices"]=joint_group_indices

                            ### merge the param indices

                            joint_param_groups["joint"][group_id]["merged_param_indices"]=merge_indices(joint_param_groups["joint"][group_id]["merged_param_indices"], individual_param_specs[parspec1]["param_indices"])

                          
                            found=True
                            break
                        

            if(found==False):

                ## no overlap found .. generate a new joint group
                group_id="group_%d" % joint_group_counter
                joint_param_groups["joint"][group_id]=dict()

                joint_param_groups["joint"][group_id]["names"]=[parspec1]
                joint_param_groups["joint"][group_id]["boundary_param_indices"]=slice(individual_param_specs[parspec1]["param_indices"][0].start, individual_param_specs[parspec1]["param_indices"][-1].stop)
                joint_param_groups["joint"][group_id]["boundary_group_indices"]=slice(individual_param_specs[parspec1]["group_indices"][0].start, individual_param_specs[parspec1]["group_indices"][-1].stop)

                joint_param_groups["joint"][group_id]["merged_param_indices"]=individual_param_specs[parspec1]["param_indices"].copy()
              
               
                joint_group_counter+=1

            ## generate a joint param group
        ## determine unused dimensions

        

        def update_unused_slices(used_sl, unused_sl):

            current_slices=unused_sl.copy()
            #print("startin with unused ARRA", current_slices)

            print("USED ", used_sl)
            print("UNUSED ", unused_sl)

            for used in used_sl:
                #print("checking ", used)
                new_slices=[]
                for unused in current_slices:
                    #print(".. comparing with unused", unused)
                    if(used.start <= unused.start and used.stop > unused.start and  used.stop<unused.stop):
                        ## make the slice smaller at the bottom
                        new_slices.append(slice(used.stop, unused.stop))
                        print(".......smaller bottom")

                    elif(used.start < unused.stop and used.stop >= unused.stop and used.start > unused.start):
                        ## make slice smaller at top
                        new_slices.append(slice(unused.start, used.start))
                        print(".........smaller top")

                    elif(used.start > unused.start and used.stop < unused.stop):
                        ## used slice is contained in unused one .. split in two

                        new_slices.append(slice(unused.start, used.start))
                        new_slices.append(slice(used.stop, unused.stop))
                        print(".........split")
                    elif(used.start<=unused.start and used.stop>=unused.stop):
                        ## do nothing .. unused part disappears
                        print("do nothing")
                        continue
                    else:
                        print("append unused")
                        new_slices.append(unused)

                current_slices=new_slices
                current_slices.sort()

                #print("new unused ARRAY.. ", current_slices)
            #print("FINAL UNUSED ARRAY: ", current_slices)
            return current_slices

        unused_slices=[slice(0, tot_parameter_count)]
        
        ## transfer individual information
        for k in individual_param_specs.keys():
            joint_param_groups["single"][k]=individual_param_specs[k]
           
            for joint_key in joint_param_groups["joint"].keys():
              
                if(k in joint_param_groups["joint"][joint_key]["names"]):
                    ## go through merged indices and construct new indices relative to the merged ones
                    global_offset=joint_param_groups["joint"][joint_key]["boundary_param_indices"].start

                    relative_indices=[]
                    internal_counter=0

                    merged_indices=joint_param_groups["joint"][joint_key]["merged_param_indices"]
                    
                    last_merge_index=-1
                    last_merge_remaining_indices=0
                    last_slice_stop=None

                    for this_par_slice in joint_param_groups["single"][k]["param_indices"]:
                        
                        for merge_index, merged in enumerate(merged_indices):
                            if(this_par_slice.start >= merged.start and this_par_slice.stop <= merged.stop):
                                ## ok we found the merged slice this one is a part of

                                ## first add any extra relative offset that is required
                                if(last_merge_index==-1):
                                    internal_counter+=this_par_slice.start-merged.start
                                else:

                                    if(last_merge_index==merge_index):
                                        ## we are a second time in the same merged slice
                                        assert(this_par_slice.start-last_slice_stop>0)
                                        internal_counter+=this_par_slice.start-last_slice_stop

                                    else:
                                        ## we are at least one merged slice further than last one
                                        assert(merge_index>last_merge_index)

                                        ## fill up remainig indices from "last_merge_index"
                                        internal_counter+=last_merge_remaining_indices

                                        # skip further merged slices in between
                                        for ii in [i for i in range(last_merge_index+1, merge_index-1)]:
                                            internal_counter+=merged_indices[ii]

                                        ## add the offset from this merge slice finally
                                        internal_counter+=this_par_slice.start-merged.start

                                relative_indices.append(slice(internal_counter, internal_counter+this_par_slice.stop-this_par_slice.start))
                                
                                internal_counter+=this_par_slice.stop-this_par_slice.start

                                ####

                                last_merge_index=merge_index
                                last_merge_remaining_indices=merged.stop-this_par_slice.stop
                                last_slice_stop=this_par_slice.stop
                                break

                            else:
                                assert(this_par_slice.start > merged.stop or this_par_slice.stop < merged.start)

                   

                    joint_param_groups["single"][k]["param_indices_relative_to_parent_joint"]=relative_indices
                    break
            print("update unused slices ", k)
            print("bef ", unused_slices)
            print("used slices ..... ",joint_param_groups["single"][k]["param_indices"])
            unused_slices=update_unused_slices(joint_param_groups["single"][k]["param_indices"], unused_slices)
            print("---", unused_slices)
        joint_param_groups["unused_slices"]=unused_slices
        
        
        ####################################
        ## check if any parameter ranges are unused


        return joint_param_groups

    def _obtain_parameter_specs_for_optimization(self):
        """
        Subclass and return parameter specs.
        """
        raise NotImplementedError()

    def _init_pdfs(self, labels):
        raise NotImplementedError()

    def _forward(self, x, extra_inputs=None):
        raise NotImplementedError()

    def _training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError()
    
    ## this function has to return the validation loss
    def _validation_step(self, batch, batch_idx, log=True):
        raise NotImplementedError()

    ## return structure of flows for debugging
    def obtain_flow_params(self, batch, batch_idx):
        raise NotImplementedError()

    def get_dataset_specific_variables(self, batch):
        """
        Overwrite this function by subclasses to add extra variables per event (e.g. NCH for neutrino events) for inference purposes
        """
        return dict()

    def define_encoding_scheme_collates(self, **kwargs):

        self._define_encoding_scheme_collates(**kwargs)

        assert(hasattr(self, "collate_fn")), "The function defining encoding schemes must define collate fn and attach *collate_fn* to model."
        assert(hasattr(self, "plot_uncollate_fn")), "The function defining encoding schemes must define collate fn and attach *plot_uncollate_fn* to model."

    def _define_encoding_scheme_collates(self, **kwargs):
        """
        Basic collation functions that does no real collation as default.
        """

        def collate_fn(batch):  
            return batch
           
        def uncollate_fn(batch):

            return batch

        self.collate_fn=collate_fn 
        self.plot_uncollate_fn=uncollate_fn
        

    def get_encoding_scheme_collates(self):
        """
        To be defined by subclasses. Returns collate/uncollate fn depending on the used encoding scheme
        """

        return self.collate_fn, self.plot_uncollate_fn

    def _get_self_ref_class(self):
        """
        Returns subclassed class reference.
        """

        raise NotImplementedError()


    
