#import matplotlib
#matplotlib.use("Agg")

from pytorch_lightning.callbacks import Callback
import pylab
import os
import io
#import tensorflow as tf
import torch
import jammy_flows
import re
import collections

import ray.tune as tune
import glob

from matplotlib.colors import Normalize

import numpy 
import time

try:
    from tensorboard.backend.event_processing import event_accumulator, event_multiplexer
except:
    print("tensorboard not installed -> cannot access multiplexer/accumulator for loss visualization")

from lightning_fabric.utilities.cloud_io import _load as pl_load

class val_plot_callback_base(Callback):

    def __init__(self, 
                 val_dataset_indices=None, 
                 plotting_dataset_indices=None, 
                 indices=None, 
                 data_on_the_left=1, 
                 plot_interval=100, 
                 constraints=None):
        super().__init__()
        ## indices to plot is either None or a str of indices separated by commas

        self.indices=[]
        if(indices is not None):
            self.indices=[int(i) for i in indices.split(",")]

        ## optionally plot various validation datasets ?
        self.val_dataset_indices=[0]
        if(val_dataset_indices is not None):
            self.val_dataset_indices=[int(i) for i in val_dataset_indices.split(",")]

        ## do the same for plotting dataset indices .. 
        self.plotting_dataset_indices=None
        if(plotting_dataset_indices is not None):
            self.plotting_dataset_indices=[int(i) for i in plotting_dataset_indices.split(",")]

        self.other_options=[]
        self.plot_name=None
        self.num_horizontal_gridspecs=1

        self.data_on_the_left=data_on_the_left

        self.plot_interval=plot_interval

        ## can be used to select specific regions in label space
        self.constraints=constraints

    
    """
    def on_epoch_start(self, trainer, system):
        print("EPOCH IS STARTING!!!!")


    def on_init_end(self, trainer):
        print("ON INIT END!!!!")
    """
    def on_validation_batch_end(self, *args, **kwargs):
        
        # we only plot based on the first validation batch
        batch_idx=args[4]
        if(batch_idx>0):
            print("BATCH IDX > 0 ... EXITING ( NO PLOTTING HERE) ", batch_idx)
            return
     
        # check that we actually want to plot .. args[0]==trainer
        ## args[1] = dataloader
        if( (args[0].global_step==0) or ((args[0].global_step)%self.plot_interval==0)):
            
            #assert(len(args)==6), ("Args should be trainer,system,output,batch,batch_idx,dataset_idx.. but number of args does not match", len(args), args, kwargs)
            print("plotting args ... len", len(args))
            print("KWARGS", kwargs)
            
            tbef=time.time()
            dataset_index=0
            if(len(args)==6):
                dataset_index=args[5]

            print("dataset index ..", dataset_index)
            
            if(dataset_index in self.val_dataset_indices):
                print("--> plotting ... ", self.plot_name)
                print(" ... for val dataset %d (%s)" % (dataset_index, args[1].validation_names[dataset_index]))

                ### normal validation batch plotting if no speciic plotting datasets are requested
                if(self.plotting_dataset_indices is None):
                    if(len(self.indices)>0):
                        for bi in self.indices:
                            print(" ..... batch index ..")
                            print(bi)
                            self.do_the_plotting(*args, batch_index=bi, dataset_identifier="val_%d" % dataset_index)
                    else:
                        self.do_the_plotting(*args, dataset_identifier="val_%d" % dataset_index)

                ### we are in the first validation batch ... lets piggy back here and als plot the "plotting datasets" 

                if(dataset_index==0):

                    trainer=args[0]
                    system=args[1]

                    batch_loss_return=args[2]

                    _=args[3] 

                    used_data_module=trainer.data_module

                    
                    if(self.plotting_dataset_indices is not None):
                        ## obtain a batch

                        for pindex in self.plotting_dataset_indices:
                        
                            used_dataset=trainer.data_module.plotting_dict_list[dataset_index]["dataset"]
                            used_batch_size=trainer.data_module.plotting_dict_list[dataset_index]["batch_size"]
                            used_name=trainer.data_module.plotting_dict_list[dataset_index]["name"]
                            #############

                            assert(len(self.indices)>0), "We need plotting indices for a plotting dataset, since we form the batch manually!"
                            ## need global event properties for this to work...
                            if(self.constraints is not None):

                                assert(hasattr(used_dataset, "global_event_properties"))
                                derived_indices=numpy.where(self._check_constraints_and_get_mask(used_dataset.global_event_properties, len(used_dataset)))[0]

                                assert(max(self.indices)<len(derived_indices))

                                t=[used_dataset.__getitem__(derived_indices[i]) for i in self.indices]


                            else:


                                assert(max(self.indices)<len(used_dataset))
                                t=[used_dataset.__getitem__(i) for i in self.indices]
            
                                print("-----> dataset %s does not have attribute *global_event_properties*... this means it as a whole will be indexed. Please add this parameter to change this." % used_name)


                            new_batch=used_data_module.collate_fn(t)

                            for k in new_batch:
                                if(type(new_batch[k])==torch.Tensor):
                                   
                                    ## hack - copy GPU status from official validation batch, which must have same stucture
                                    new_batch[k]=new_batch[k].to(args[3][k])
                           
                            assert(len(self.indices)>0), "Plotting datasets must invoke *indices* to specify specific events to be plotted!"

                            
                            for real_index, bi in enumerate(self.indices):
                                self.do_the_plotting(trainer,system,batch_loss_return, new_batch, batch_index=real_index, dataset_identifier="plotting_%d" % pindex)
                            

            print("--> took ", time.time()-tbef, " secs")


        else:
            print("skipping plotting since cur_step=%d and plot_interval %d" % (args[0].global_step+1, self.plot_interval))

    ## generic fig and layout (either 1 or 2 grid specs) -- can be overwritten to be dependent on system
    def make_fig_and_layout(self, system, batch_info):

        fig=pylab.figure(figsize=(self.num_horizontal_gridspecs*6,6))

        ## add 

        total_gridspec=fig.add_gridspec(1, self.num_horizontal_gridspecs)
       
        return fig, total_gridspec

    def _check_constraints_and_get_mask(self, array_dict, array_size):
        ## var1>=&var1<=x&var2<xxx&var2> 

        ## only allow & masking atm

        constraint_mask=numpy.array([True]*array_size)

        if(self.constraints is None):
            return constraint_mask

        for c in self.constraints.split("&"):
            if(">" in c):
                s=c.split(">")
                ar=array_dict[s[0]]
                if(type(ar)==list):
                    ar=numpy.array(ar)

                constraint_mask=constraint_mask & (ar>float(s[1]))
            elif("<" in c):
                s=c.split("<")
                ar=array_dict[s[0]]
                if(type(ar)==list):
                    ar=numpy.array(ar)
               
                constraint_mask=constraint_mask & (ar<float(s[1]))

            elif("--" in c): ## double dash instead of == for equal
                s=c.split("--")
                ar=array_dict[s[0]]
                if(type(ar)==list):
                    ar=numpy.array(ar)
               
                constraint_mask=constraint_mask & (ar==float(s[1]))
                print("constriant mask after ", c, constraint_mask.sum())
            else:
                raise Exception("Unknown constraint operation .. ", c)

        return constraint_mask

    ## generic visualization .. visualizes data given by dataloader in first gridspec if required, and custom plot in second gridspec
    def visualize(self, 
                  fig, 
                  total_gridspec, 
                  data_module, 
                  dataset, 
                  system, 
                  batch_info, 
                  uncoll_batch_info, 
                  batch_index=None, 
                  global_index=None, 
                  dataset_identifier="val_0"):

        ## if we have constraints based on labels, first check which events fulfill constraints and calculate new batch_index based on constraints
        ## *batch_index* will then serve only as a relative index
      
        bounds=None

        data_vis_returns=None

        if(batch_index != None):

            #used_global_index=batch_index
            #if(global_index is not None):
            #    used_global_index=global_index

            if(self.data_on_the_left):
                ## data is on the left most subplotspec
                data_vis_returns=data_module.visualize_data(dataset, fig, total_gridspec[0,0], batch_info, batch_index, dataset_identifier=dataset_identifier)
            else:
                ## data is on the top and gets its own subplotspec
                data_vis_returns=data_module.visualize_data(dataset, fig, total_gridspec[0,:], batch_info, batch_index, dataset_identifier=dataset_identifier)
        
       
        bounds=dataset.get_bounds()
        print("bounds ", bounds)

        ## this function has to be overridden by child classes
        
        self._visualize_model(fig, total_gridspec, system, batch_info, uncoll_batch_info, bounds=bounds, batch_index=batch_index, data_vis_returns=data_vis_returns)
        

    def do_the_plotting(self, 
                        *args, 
                        batch_index=None, 
                        total_batch_idx_overwrite=None, 
                        dataset_identifier="val_0" # val_0, val_1 ... plotting_0, plotting_1...
                        ):

        if(self.plot_name==None):
            raise NotImplementedError("Implement a plot name in the __init__ of this callback!")

        trainer=args[0]
        system=args[1]
        averaged_system=None
        sched=None
        ## update system with averaged model if possible
        if(hasattr(system, "_averaged_model")):
            if(system._averaged_model is not None):
                averaged_system=system._averaged_model
        """
        else:
        try:
            sched=system.lr_schedulers()
        except:
            print("Scheduler could not be loaded from model (likely because in *check_plots* in beginning, and trainer is not connected yet!")
          
        if(sched is not None):
            print("found sched", sched)
            if(hasattr(sched, "averaged_param_vector")):
                if(sched.averaged_param_vector is not None):
                    print("updating plotting system with averaged model!")
                    averaged_system=system.obtain_new_model(sched.averaged_param_vector)    

        """
        
        batch_loss_return=args[2]

        batch_info=args[3] ## label/data in supervised case

        ## also get uncollated stuff
        uncollated_batch_info=trainer.plot_uncollate_fn(batch_info)

        ## we are in standard training mode .. data module is referenced to by the trainer, the dataset is the validation dataset
        used_data_module=trainer.data_module

        used_dataset_dict_list=trainer.data_module.validation_dict_list
        if("plotting" in dataset_identifier):
            used_dataset_dict_list=trainer.data_module.plotting_dict_list

        dataset_index=int(dataset_identifier.split("_")[-1])

        used_dataset=used_dataset_dict_list[dataset_index]["dataset"]

        ## first check that we actually want to create a plot here...

        used_batch_index=None
        if(batch_index is not None):

            ## plotting dataset already has constraints in it
            if("plotting" in dataset_identifier):
                used_batch_index=batch_index
            else:
                ## validation dataset can use constraints here
                ev_props=None
                assert("batch_size" in batch_info)
                if("event_properties" in batch_info):
                    ev_props=batch_info["event_properties"]
               
                constraint_mask=self._check_constraints_and_get_mask(ev_props, batch_info["batch_size"])
              
                #assert(constraint_mask.sum()>0), "Constraints %s for plotting are not leaving ANY possible batch items .. loosen constriants!" % self.constraints
                num_available_events=constraint_mask.sum()
                
                if(batch_index>=num_available_events):
                    ### the index is too large to index the available events .. just exit gracefully here
                    print("index %d is too large for available events .. do not plot" % batch_index)
                    return

                constraint_indices=numpy.where(constraint_mask)[0]
                
                ## define the actually used index here
                used_batch_index=constraint_indices[batch_index]

        for tuple_info in [(system, ""), (averaged_system, "_average")]:
            if(tuple_info[0] is not None):
                with torch.no_grad():
                  
                    fig, total_gridspec=self.make_fig_and_layout(system, batch_info)
                    
                    #try:
                    ## we do not want a try block here, as it might hide underlying other issues
                    self.visualize(fig, total_gridspec, used_data_module, used_dataset, tuple_info[0], batch_info, uncollated_batch_info, batch_index=used_batch_index, dataset_identifier=dataset_identifier)
                    #except:
                    #    print("Visualization had an error ... some plots might be corrupted.")

                    total_gridspec.tight_layout(fig)
                    #############

                    log_path=trainer.logger.save_dir

                    image_basename="images/%s__ds_%s" % (self.plot_name, dataset_identifier)
                    img_writer_basename="%s__ds_%s" % (self.plot_name, dataset_identifier)
                    if(batch_index!=None):
                        image_basename="images/%s__ds_%s__batchidx_%d" % (self.plot_name, dataset_identifier,batch_index)
                        img_writer_basename="%s__ds_%s__batchidx_%d" % (self.plot_name, dataset_identifier, batch_index)
                    elif(len(self.other_options)>0):
                        image_basename="images/%s" % (self.plot_name+"__ds_%s__"%(dataset_identifier)+"+".join(self.other_options))
                        img_writer_basename="%s" % (self.plot_name+"__ds_%s__"%(dataset_identifier)+"+".join(self.other_options))
                    image_path=os.path.join(log_path, image_basename)

                    if(not os.path.exists(image_path)):
                        os.makedirs(image_path)
                    ## save to png

                    average_suffix=tuple_info[1]
                    img_name="%s_%.12d_%s.png" % (average_suffix, trainer.global_step, self.plot_name)
                    img_integer_id=trainer.global_step

                    if(total_batch_idx_overwrite is not None):
                       img_name="%s_%s_%s.png" % (average_suffix,total_batch_idx_overwrite, self.plot_name) 
                       img_integer_id=int(total_batch_idx_overwrite)

                    #try:
                    fig.savefig(os.path.join(image_path, img_name))
                    print("saved fig on disk ...", os.path.join(image_path, img_name))
                    """
                    except:
                        pylab.close(fig)
                        fig, total_gridspec=self.make_fig_and_layout(system, batch_info)
                        fig.savefig(os.path.join(image_path, img_name))
                        print("problem with saving image... probably because inf/-infs in plots ...", os.path.join(image_path, img_name))
                    """

                    ## generate img
                    tb_summary_writer = trainer.logger.experiment
                    tb_summary_writer.add_figure(img_writer_basename, fig, img_integer_id)
                        
                    

                    pylab.close(fig)
        
    
    def _visualize_model(self, fig, total_gridspec, system, batch_info, uncoll_batch_info, bounds=None, batch_index=None, data_vis_returns=None):
        raise NotImplementedError()

  
    
    def check_plots(self, trainer, load_model_fn, val_batches, config=None, batch_loss_fn=None):
        print(" >>>>>>>>>> CHECK PLOTS.. <<<<<<<<<<<<")
        plot_data=1
        ## an index of -1 indicates that the whole batch should be used for plotting
    
        all_plotting_folders=[]

        if(self.plotting_dataset_indices is not None):

            for plotting_ds_index in self.plotting_dataset_indices:
                if(len(self.indices)>0):

                    for index in self.indices:

                        plotting_folder=os.path.join(trainer.logger.save_dir, "images/%s__ds_plotting_%d__batchidx_%d" % (self.plot_name,plotting_ds_index, index) )

                        all_plotting_folders.append(plotting_folder)
                elif(len(self.other_options)>0):

                    opt_string="+".join(self.other_options)

                    plotting_folder=os.path.join(trainer.logger.save_dir, "images/%s__ds_plotting_%d__%s" % (self.plot_name, plotting_ds_index, opt_string) ) 

                    all_plotting_folders.append(plotting_folder)
                else:

                    plotting_folder=os.path.join(trainer.logger.save_dir, "images/%s__ds_plotting_%d" % (self.plot_name, plotting_ds_index) ) 

                    all_plotting_folders.append(plotting_folder)

        else:

            for val_ds_index in self.val_dataset_indices:
                if(len(self.indices)>0):

                    for index in self.indices:

                        plotting_folder=os.path.join(trainer.logger.save_dir, "images/%s__ds_val_%d__batchidx_%d" % (self.plot_name,val_ds_index, index) )

                        all_plotting_folders.append(plotting_folder)
                elif(len(self.other_options)>0):

                    opt_string="+".join(self.other_options)

                    plotting_folder=os.path.join(trainer.logger.save_dir, "images/%s__ds_val_%d__%s" % (self.plot_name, val_ds_index, opt_string) ) 

                    all_plotting_folders.append(plotting_folder)
                else:

                    plotting_folder=os.path.join(trainer.logger.save_dir, "images/%s__ds_val_%d" % (self.plot_name, val_ds_index) ) 

                    all_plotting_folders.append(plotting_folder)

        ## get list of figures to check from checkpoints

       
        checkpoint_dirs=glob.glob(os.path.join(trainer.logger.save_dir, "checkpoint_*/cp_validation_end"))
        checkpoint_dirs.sort()

        for cp_path in checkpoint_dirs:

            if("checkpoint_tmp" in cp_path):
                continue

            ## newer ray version has =
            if("=" in cp_path.split("/")[-2]):
                cp_index_string=cp_path.split("/")[-2].split("=")[-1]
            else:
                cp_index_string=cp_path.split("/")[-2].split("_")[-1]

            # make sure we have a 12-digit int
            cp_index_string="%.12d" % int(cp_index_string)

            #print("all plotting folders ", all_plotting_folders)
            for f in all_plotting_folders:
                
                img_path=glob.glob(os.path.join(f, "*"+cp_index_string+"*"))
                img_path2=glob.glob(os.path.join(f, cp_index_string+"*"))
                
                ## also check for plot interval
                #print("cp index str ", cp_index_string)

                if(len(img_path)==0 and ((int(cp_index_string)) % self.plot_interval==0)):
                    print("-------> replotting %s .... " % (os.path.join(f, "*"+cp_index_string+"*")))
                    print("img path len ", len(img_path), img_path, f)
                    print("img path 2, ", img_path2)
                    print("overall ", glob.glob(os.path.join(f, "*")))
                    ## 
                    cp = torch.load(cp_path)
                    #cp = pl_load(cp_path, map_location=lambda storage, loc: storage)

                    model=load_model_fn(cp, config=config)

                    #if(not hasattr(model, "num_validation_steps_per_sweep")):
                    #    ##
                    print(".. resetting sweep attributes ..")
                    train_dataset_size=model.config_at_init["data.train.dataset_size"]
                    train_batch_size=model.config_at_init["data.train.batch_size"]
                    val_interval=model.config_at_init["val_interval"]

                    ## make sure we get a copy here
                    val_names=[]
                    for ditem in trainer.data_module.validation_dict_list:
                        newname=""
                        for c in ditem["name"]:
                            newname=newname+c
                        val_names.append(newname)

                    model.set_training_settings(train_dataset_size, 
                                                     train_batch_size, 
                                                     val_interval,
                                                     val_names,
                                                     model.config_at_init,
                                                     trainer.data_module)

                    ## ensoure double precision
                    #model.double()
                    model.eval()

                    this_batch_index=None

                    if(len(self.indices)>0):
                        
                        this_batch_index=int(f.split("/")[-1].split("_")[-1])
                        #print("in check plots.. batch index ", this_batch_index)
                    
                    ## find out dataset index
                    used_ds_index=0

                    last_split=f.split("/")[-1]
                    for further_split in last_split.split("__"):
                        if("ds_" in further_split):
                            used_ds_index=int(further_split[-1])


                    ## is ds_ in the folder name of the plot? If yes, extract dataset_index from str.
                    #print("USED DS INDEX ID ", used_ds_index)
                    ds_index_in_str=f.split("/")[-1].split("__")[-2][:3]=="ds_"
                    
                    if(used_ds_index>=len(val_batches)):
                        continue

                    used_ds_identifier="val_%d" % used_ds_index
                    if(self.plotting_dataset_indices is not None):
                        used_ds_identifier="plotting_%d" % used_ds_index

                    this_val_batch=val_batches[used_ds_index]

                    #print("-----------_> USED DS IDENTIFIER!!!!! ", used_ds_identifier)

                    self.do_the_plotting(trainer, model, None, this_val_batch, batch_index=this_batch_index, total_batch_idx_overwrite=cp_index_string, dataset_identifier=used_ds_identifier)

class visualize_fisher_grad_projections(val_plot_callback_base):

    def __init__(self, validation_base_string="s.val_loss", plot_interval=100):
        super().__init__(indices=None, plot_interval=plot_interval)

        self.plot_name="visualize_fisher_grad_projections"
        #self.validation_batch_size=validation_batch_size
        #self.minibatch_size=minibatch_size
        self.base_string=validation_base_string


    def make_fig_and_layout(self,system):

        self.num_horizontal_gridspecs=1
    
        fig=pylab.figure(figsize=(15,6))

        total_gridspec=fig.add_gridspec(1, self.num_horizontal_gridspecs)
       
        return fig, total_gridspec

    def do_the_plotting(self, *args, batch_index=None, total_batch_idx_overwrite=None, dataset_index=0):

        if(self.plot_name==None):
            raise NotImplementedError("Implement a plot name in the __init__ of this callback!")
        trainer=args[0]
        system=args[1]

        batch_loss_return=args[2]

        batch_info=args[3] ## label/data in supervised case

        ## also get uncollated stuff
        uncollated_batch_info=system.plot_uncollate_fn(batch_info)

        with torch.no_grad():
          
            fig, total_gridspec=self.make_fig_and_layout(system)

            self.visualize(fig, total_gridspec, trainer, system, batch_info, uncollated_batch_info, batch_index=batch_index, total_batch_idx_overwrite=total_batch_idx_overwrite)

            total_gridspec.tight_layout(fig)
            #############

            log_path=trainer.logger.save_dir

            image_basename="images/%s" % (self.plot_name)
            img_writer_basename="%s" % (self.plot_name)
            if(batch_index!=None):
                image_basename="images/%s_batchidx_%d" % (self.plot_name, batch_index)
                img_writer_basename="%s_%d" % (self.plot_name, batch_index)
            image_path=os.path.join(log_path, image_basename)

            if(not os.path.exists(image_path)):
                os.makedirs(image_path)
            ## save to png

            img_name="%.12d_%s.png" % (trainer.global_step, self.plot_name)
            img_integer_id=trainer.global_step

            if(total_batch_idx_overwrite is not None):
               img_name="%s_%s.png" % (total_batch_idx_overwrite, self.plot_name) 
               img_integer_id=int(total_batch_idx_overwrite)

            fig.savefig(os.path.join(image_path, img_name))

            print("saved fig on disk ...", os.path.join(image_path, img_name))
            ## save to img buffer
            #img_buffer=self._fig_to_image_buffer(fig)

  
            ## generate img
            tb_summary_writer = trainer.logger.experiment
            tb_summary_writer.add_figure(img_writer_basename, fig, img_integer_id)

            pylab.close(fig)

  
    def visualize(self, fig, total_gridspec, trainer, system, batch_info, uncoll_batch_info, batch_index=None, total_batch_idx_overwrite=None):

        def adjust_min_max(vec, prev_min, prev_max):

            new_min=prev_min
            new_max=prev_max
            if(min(vec)<prev_min):
                new_min=min(vec)

            if(max(vec)>prev_max):
                new_max=max(vec)

            return new_min, new_max
        
        gs_counter=0
        bounds=None
      
        lookback_steps=2*system.num_validation_steps_per_sweep

        data_vis_returns=None

        opti=system.optimizers()

        if(type(opti)==list or type(opti)==tuple):
            raise NotImplementedError("Dont support multiple optimizres for plotting ... implement!")

        if(opti is None):
            print("optimizer is None .. cannot produce grad projections plot")
            return
       
        if("fish_max_eigenvecs" not in opti.state["global"]):
            return

        cur_step_idx=trainer.global_step
        if(total_batch_idx_overwrite is not None):
            cur_step_idx=int(total_batch_idx_overwrite)

        statedict=opti.state["global"]

        num_saved_eigenvecs=statedict["fish_max_eigenvecs"].shape[0]

        new_xmin=numpy.inf
        new_xmax=-numpy.inf

        new_ymin=numpy.inf
        new_ymax=-numpy.inf

        for ind in range(num_saved_eigenvecs):
            finite_mask=numpy.isfinite(statedict["fish_min_eigenvec_grad_projection"][ind,:])

            if(ind==(num_saved_eigenvecs-1) ):
                new_xmin,new_xmax=adjust_min_max(statedict["fish_min_eigenvec_grad_projection"][ind,:][finite_mask], new_xmin, new_xmax)
                
                new_xmin, new_xmax=adjust_min_max(statedict["fish_min_eigenvec_scaled_grad_projection"][ind,:][finite_mask], new_xmin, new_xmax)
                new_xmin, new_xmax=adjust_min_max(statedict["fish_min_eigenvec_mean_grad_projection"][ind,:][finite_mask], new_xmin, new_xmax)
                new_xmin, new_xmax=adjust_min_max(statedict["fish_min_eigenvec_scaled_mean_grad_projection"][ind,:][finite_mask], new_xmin, new_xmax)

                new_ymin,new_ymax=adjust_min_max(statedict["fish_max_eigenvec_grad_projection"][ind,:][finite_mask], new_ymin, new_ymax)
                new_ymin, new_ymax=adjust_min_max(statedict["fish_max_eigenvec_scaled_grad_projection"][ind,:][finite_mask], new_ymin, new_ymax)
                new_ymin, new_ymax=adjust_min_max(statedict["fish_max_eigenvec_mean_grad_projection"][ind,:][finite_mask], new_ymin, new_ymax)
                new_ymin, new_ymax=adjust_min_max(statedict["fish_max_eigenvec_scaled_mean_grad_projection"][ind,:][finite_mask], new_ymin, new_ymax)

        ## make axes symmetric
        if(new_xmin < 0 and new_xmax>0):
            abs_greater=max([numpy.fabs(new_xmin), numpy.fabs(new_xmax)])
            new_xmin=-abs_greater
            new_xmax=abs_greater

        if(new_ymin < 0 and new_ymax>0):
            abs_greater=max([numpy.fabs(new_ymin), numpy.fabs(new_ymax)])
            new_ymin=-abs_greater
            new_ymax=abs_greater



        for ind in range(num_saved_eigenvecs):
            ax=fig.add_subplot(num_saved_eigenvecs,1,ind+1)

            """
            statedict["fish_min_eigenvec_grad_projection"][ind,:]=statedict["fish_min_eigenvec_grad_projection"][cur_ind_to_copy_backwards,:]
            statedict["fish_min_eigenvec_scaled_grad_projection"][ind,:]=statedict["fish_min_eigenvec_scaled_grad_projection"][cur_ind_to_copy_backwards,:]


            statedict["fish_max_eigenvec_grad_projection"][ind,:]=statedict["fish_max_eigenvec_grad_projection"][cur_ind_to_copy_backwards,:]
            statedict["fish_max_eigenvec_scaled_grad_projection"][ind,:]=statedict["fish_max_eigenvec_scaled_grad_projection"][cur_ind_to_copy_backwards,:]


            statedict["fish_min_eigenvec_mean_grad_projection"][ind,:]=statedict["fish_min_eigenvec_mean_grad_projection"][cur_ind_to_copy_backwards,:]
            statedict["fish_min_eigenvec_scaled_mean_grad_projection"][ind,:]=statedict["fish_min_eigenvec_scaled_mean_grad_projection"][cur_ind_to_copy_backwards,:]

            statedict["fish_max_eigenvec_mean_grad_projection"][ind,:]=statedict["fish_max_eigenvec_mean_grad_projection"][cur_ind_to_copy_backwards,:]
            statedict["fish_max_eigenvec_scaled_mean_grad_projection"]

            """

            ## finite mask can be shared among all entries
            finite_mask=numpy.isfinite(statedict["fish_min_eigenvec_grad_projection"][ind,:])

            ax.plot(statedict["fish_min_eigenvec_grad_projection"][ind,:][finite_mask], statedict["fish_max_eigenvec_grad_projection"][ind,:][finite_mask], marker="o", lw=0.0, label="grad")
            ax.plot(statedict["fish_min_eigenvec_scaled_grad_projection"][ind,:][finite_mask], statedict["fish_max_eigenvec_scaled_grad_projection"][ind,:][finite_mask], marker="x", lw=0.0,label="scaled grad")
            
            ax.plot(statedict["fish_min_eigenvec_mean_grad_projection"][ind,:][finite_mask], statedict["fish_max_eigenvec_mean_grad_projection"][ind,:][finite_mask], lw=0.0, marker="o", label="mean grad")
            ax.plot(statedict["fish_min_eigenvec_scaled_mean_grad_projection"][ind,:][finite_mask], statedict["fish_max_eigenvec_scaled_mean_grad_projection"][ind,:][finite_mask], lw=0.0, marker="x", label="scaled mean grad")
            
            ax.legend(loc="upper right")
            ax.set_xlim(new_xmin, new_xmax)
            ax.set_ylim(new_ymin, new_ymax)


        fig.tight_layout()


class val_loss_vis(val_plot_callback_base):

    def __init__(self, validation_base_string="s.val_loss", plot_interval=100):
        super().__init__(indices=None, plot_interval=plot_interval)

        self.plot_name="val_loss_vis"
        #self.validation_batch_size=validation_batch_size
        #self.minibatch_size=minibatch_size
        self.base_string=validation_base_string

        if( len(self.indices)>0):
            raise Exception("Loss stochasticity does not allow indices for plotting!")

    def make_fig_and_layout(self,system):

        self.num_horizontal_gridspecs=1
    
        fig=pylab.figure(figsize=(15,6))

        total_gridspec=fig.add_gridspec(1, self.num_horizontal_gridspecs)
       
        return fig, total_gridspec

    def do_the_plotting(self, *args, batch_index=None, total_batch_idx_overwrite=None, dataset_identifier="val_0"):

        if(self.plot_name==None):
            raise NotImplementedError("Implement a plot name in the __init__ of this callback!")
        trainer=args[0]
        system=args[1]

        batch_loss_return=args[2]

        batch_info=args[3] ## label/data in supervised case

        ## also get uncollated stuff
        uncollated_batch_info=system.plot_uncollate_fn(batch_info)

        with torch.no_grad():
          
            fig, total_gridspec=self.make_fig_and_layout(system)

            self.visualize(fig, total_gridspec, trainer, system, batch_info, uncollated_batch_info, batch_index=batch_index, total_batch_idx_overwrite=total_batch_idx_overwrite)

            total_gridspec.tight_layout(fig)
            #############

            log_path=trainer.logger.save_dir

            ### image_basename="images/%s__ds_%d" % (self.plot_name, dataset_i
            image_basename="images/%s__ds_%s" % (self.plot_name, dataset_identifier)
            img_writer_basename="%s__ds_%s" % (self.plot_name, dataset_identifier)
            if(batch_index!=None):
                image_basename="images/%s__ds_%s_batchidx_%d" % (self.plot_name, dataset_identifier, batch_index)
                img_writer_basename="%s__ds_%s_batchidx_%d" % (self.plot_name, dataset_identifier, batch_index)
            image_path=os.path.join(log_path, image_basename)

            if(not os.path.exists(image_path)):
                os.makedirs(image_path)
            ## save to png

            img_name="%.12d_%s.png" % (trainer.global_step, self.plot_name)
            img_integer_id=trainer.global_step

            if(total_batch_idx_overwrite is not None):
               img_name="%s_%s.png" % (total_batch_idx_overwrite, self.plot_name) 
               img_integer_id=int(total_batch_idx_overwrite)

            fig.savefig(os.path.join(image_path, img_name))

            print("saved fig on disk ...", os.path.join(image_path, img_name))
            ## save to img buffer
            #img_buffer=self._fig_to_image_buffer(fig)

  
            ## generate img
            tb_summary_writer = trainer.logger.experiment
            tb_summary_writer.add_figure(img_writer_basename, fig, img_integer_id)

            pylab.close(fig)

    def extract_all_scalars_logical(self, event_files, val_interval, disallowed_strings=["ray/tune/", "hp_metric", "train", "epoch"]):

        em= event_multiplexer.EventMultiplexer().AddRunsFromDirectory(os.path.dirname(event_files[-1]), name="run")
        em.Reload()
        
        tags=em.Runs()["run/."]["scalars"]

        first_extract_dict=dict()

        masks=dict()

        smallest_len=999999999999999999999

        """ 
        The loop extracts all "standard" tags that do not start later (like s.val_loss_averaged)
        """

        tags_beginning_late=[]
        for tag in tags:

            found=False
            for a in disallowed_strings:
                if(a in tag):
                    found=True
                    break
            if(found==True):
                continue



            """
            found=False
            for a in allowed_strings:
                if(a in tag):
                    found=True
                    break

            if(found==False):
                continue
            """

            ### get a coherent list of all scalars
            scals=em.Scalars("run/.", tag)
          
            steps=numpy.array([i.step for i in scals])
            values=numpy.array([i.value for i in scals])

            ## the first entry should be the 0-indexed validation interval .. othrewise logging begins later than at the beginning
            if(steps[0]>val_interval-1):
               
                tags_beginning_late.append(tag)

            ## continue if the tag is starting late in the training
            if(tag in tags_beginning_late):
                first_extract_dict[tag]=(steps, values)
                continue

            step_mask=None
            
            if(len(steps) not in masks.keys()):

                step_diff=steps[1:]-steps[:-1]

                #print("step_diffs >= 0 before cleaning" ,(step_diff<=0).sum())

                step_mask=[True]
                largest_step=max(steps)

                ## largest step is at end
                if(largest_step==steps[-1]):
                    #print("start with largest")
                    last_step=largest_step

                    for step in steps[::-1][1:]:
                        if(step<last_step):
                            step_mask.append(True)
                            last_step=step
                        else:
                            step_mask.append(False)

                        

                

                else:
                    #print("largest is somewhere in the middle")
                    largest_index=numpy.where(max(steps)==steps)[-1][-1]

                    max_index=len(steps)-1

                    index_diff=(max_index-largest_index)

                    step_mask=[False]*index_diff+[True]

                    last_step=largest_step

                    #print("last step: ", last_step)
                    for step in steps[:-index_diff][::-1][1:]:
                        if(step<last_step):
                            step_mask.append(True)
                            last_step=step
                        else:
                            step_mask.append(False)
                    
                step_mask=numpy.array(step_mask)[::-1]

                masks[len(step_mask)]=step_mask
            else:

                step_mask=masks[len(steps)]


            new_steps=steps[step_mask]
            new_values=values[step_mask]

            for ind,step in enumerate(new_steps):
                if(ind>0):
                    if(new_steps[ind]-new_steps[ind-1]<=0):
                        print(ind)
                        print(new_steps[ind])
                        print(new_steps[ind-1])
                        
           
            if(len(new_steps)<smallest_len):

                smallest_len=len(new_steps)

            
            step_diff=new_steps[1:]-new_steps[:-1]

            if( (step_diff<=0).sum() > 0):
                raise Exception("Stepdiff > 0 even after cleaning!?")

            first_extract_dict[tag]=(new_steps, new_values)


        ## a second dict that will hold the actual tags for plotting

        second_extract_dict=dict()

        #print("smallest len: ", smallest_len)
        for tag in first_extract_dict.keys():
            
            if(tag in tags_beginning_late):
               
                second_extract_dict[tag]=first_extract_dict[tag]
                
            else:
                if("steps" not in second_extract_dict.keys()):
                    second_extract_dict["steps"]=first_extract_dict[tag][0][:smallest_len]

                second_extract_dict[tag]=first_extract_dict[tag][1][:smallest_len]

                
                ## assert that all steps are now the same
                assert( (second_extract_dict["steps"]==first_extract_dict[tag][0][:smallest_len]).sum()==len(second_extract_dict["steps"]) ), (tag, second_extract_dict["steps"], first_extract_dict[tag][0][:smallest_len])

        #print("FINAL STEP LEN")
        #print(len(second_extract_dict["steps"]))

        return second_extract_dict

    def get_val_losses(self, event_files, val_interval):


        return_dict=self.extract_all_scalars_logical(event_files, val_interval)
        
        """
        assert(self.validation_batch_size%self.minibatch_size==0)

        num_minibatches=int(self.validation_batch_size/self.minibatch_size)

        return_dict["minibatch_size"]=self.minibatch_size
        return_dict["num_minibatches"]=num_minibatches
        """

        return return_dict

    def calculate_expected_sigma(self, sigma, old_n, new_n):
        return sigma*numpy.sqrt(old_n)/numpy.sqrt(new_n)

    def get_expected_sigmas(self, observed_sigmas, num_average, normalize_to_index):

        new_sigmas=[]
        obs_sigma=observed_sigmas[normalize_to_index]
        base_counter=num_average[normalize_to_index]

        for ind in range(len(observed_sigmas)):
            if(ind==normalize_to_index):
                new_sigmas.append(observed_sigmas[ind])
            else:
                new_sig=self.calculate_expected_sigma(observed_sigmas[ind], base_counter, num_average[ind])
                new_sigmas.append(new_sig)

        return new_sigmas

    def visualize(self, fig, total_gridspec, trainer, system, batch_info, uncoll_batch_info, batch_index=None, total_batch_idx_overwrite=None):

        
        gs_counter=0
        bounds=None
      
        lookback_steps=2*system.num_validation_steps_per_sweep

        data_vis_returns=None

        trainer.logger.experiment.flush()
        logdir=trainer.logger.log_dir
        
        event_files=glob.glob(os.path.join(logdir, "event*"))
        event_files.sort()

        loss_dict=self.get_val_losses(event_files, system.val_interval)

        if("s.val_loss" not in loss_dict.keys()):
            return

        cur_step_idx=trainer.global_step
        if(total_batch_idx_overwrite is not None):
            cur_step_idx=int(total_batch_idx_overwrite)



        valid_mask=loss_dict["steps"] <= cur_step_idx

        val_loss_history_step=loss_dict["steps"][valid_mask]

        if(len(val_loss_history_step)==0):
            print("VALID MASK CUR IDX ", cur_step_idx)
            print("VAL LOSS HISTORY HAS LEN 0!?, check if there is issue with log files")
            return

        val_loss_history_values=loss_dict["s.val_loss"][valid_mask]
        running_lr=loss_dict["running_lr"][valid_mask]

        val_loss_history_averaged=None
        steps_loss_history_averaged=None
        if("s.val_loss_averaged" in loss_dict.keys()):
            steps_loss_history_averaged=loss_dict["s.val_loss_averaged"][0]
            val_loss_history_averaged=loss_dict["s.val_loss_averaged"][1]
            
        index_mask=(running_lr[1:]-running_lr[:-1]) < 0
        
        lr_shifts=val_loss_history_step[:-1][index_mask]

        loss_ax=fig.add_subplot(2,1,1)

        loss_ax.fill_betweenx( [1.0,20.0], [ val_loss_history_step[-1] , val_loss_history_step[-1]  ], [ val_loss_history_step[-lookback_steps:][0] , val_loss_history_step[-lookback_steps:][0]  ] ,color="yellow", alpha=0.5, label="Lookback window")
        
        loss_ax.plot(val_loss_history_step, val_loss_history_values,color="black")

        #for lr_shift in lr_shifts:
        #    loss_ax.axvline(lr_shift,color="black")

        loss_ax.set_ylim(val_loss_history_values.min()-1e-4, val_loss_history_values.max()+1e-4)

        loss_ax.legend(loc="upper right")

        #loss_ax.fill_between( [ loss_dict["step"][-self.lookback_steps:][0], loss_dict["step"][-1]  ], [0,0],[10,10] ,color="red", alpha=0.5)

        ####

        #avg_mean=numpy.mean(val_loss_history_values[-self.lookback_steps:])

      
        average_ax=fig.add_subplot(2,1,2)


        num_half_window_steps=int(len(val_loss_history_step[-lookback_steps:])/2.0)

        average_ax.plot( val_loss_history_step[-lookback_steps:], val_loss_history_values[-lookback_steps:],color="black", label="val loss")

        average_ax.axhline( numpy.mean(val_loss_history_values[-lookback_steps:]),ls="--",color="gray", label="average val loss")
        average_ax.axvline(val_loss_history_step[-lookback_steps:][num_half_window_steps],color="yellow", ls="--")

        #for lr_shift in lr_shifts:
        #    if((lr_shift > val_loss_history_step[-lookback_steps:].min()) and (lr_shift < val_loss_history_step[-lookback_steps:].max())):
        #        average_ax.axvline(lr_shift, color="black")

        if(val_loss_history_averaged is not None):
            average_ax.plot(steps_loss_history_averaged[-lookback_steps:], val_loss_history_averaged[-lookback_steps:],color="red", label="averaged weights loss %.5f" % float(val_loss_history_averaged[-lookback_steps:][-1]) )

        average_ax.legend(loc="upper right")
        average_ax.set_title("Lookback window length: Two full dataset sweeps")
       
        ## smallest batch size
        #gauss_ax.axvline(val_loss_history_values[-1], color="red", label="latest val loss")
        

   
class visualize_params(val_plot_callback_base):

    def __init__(self, indices=None, batch_par=None, plot_interval=100):
        super().__init__(indices=indices, plot_interval=plot_interval)

        self.plot_name="visualize_params"
        self.batch_par=batch_par

        if(self.batch_par is not None):
            self.other_options.append(batch_par)

        # we can plot either neither specific flow params or from one event
        #assert(len(self.indices)==0 or len(self.indices)==1)
    
    def make_fig_and_layout(self,system, batch_info):

        self.num_horizontal_gridspecs=1

        if(len(self.indices)==1):
            
            for mod in system._modules:

                if(type(system._modules[mod])==jammy_flows.pdf or type(system._modules[mod]==jammy_flows.fully_amortized_pdf)):
                    print("COUNTER!")

                    self.num_horizontal_gridspecs+=1
               
    
        fig=pylab.figure(figsize=(15,6))

        total_gridspec=fig.add_gridspec(1, self.num_horizontal_gridspecs)
       
        return fig, total_gridspec

    def visualize_params(self, ax, params, params_name):
        
        if(len(params.shape)==0):
            ax.plot([0], [params.cpu().type(torch.float64).numpy()])
            #ax.set_ylim(-1,1)
        elif(len(params.shape)==1):
            # 1-d

            np_array=params.cpu().type(torch.float64).numpy()

            ax.plot(numpy.arange(len(np_array)), np_array)
            ax.set_ylim(np_array.min(), np_array.max())
            #ax.set_ylim(-1,1)
        elif(len(params.shape)==2):
            # 2-d
            if(1 in params.shape):
                np_array=params.view(-1).cpu().type(torch.float64).numpy()

                ax.plot(numpy.arange(len(np_array)), np_array)

                ax.set_ylim(np_array.min(), np_array.max())
            else:

                img=ax.matshow(params.cpu().type(torch.float64).numpy(), aspect="auto")#, norm=Normalize(vmin=-1, vmax=1))
                pylab.colorbar(img, ax=ax)

        else:
            ## try to śqueeze first dimension if it is 1-d (can happen for flow params)
            ## -> intrinsically 2-d param vector
            one_counter=0
            for sh in params.shape:
                if(sh==1):
                    one_counter+=1

            if(one_counter==1):
                for ind, p in enumerate(params.shape):

                    img=ax.matshow(params.cpu().type(torch.float64).squeeze(ind).numpy(), aspect="auto")#, norm=Normalize(vmin=-1, vmax=1))
                    pylab.colorbar(img, ax=ax)

                    break
            elif(one_counter==2 or one_counter==3):
                np_array=params.view(-1).type(torch.float64).cpu().numpy()
                print("3 way array ", params_name, np_array)
                ax.plot(numpy.arange(len(np_array)), np_array)
                ax.set_ylim(np_array.min(), np_array.max())
            else:
                raise Exception("Shape not visualizable: ", params.shape, params_name)

      
    def visualize(self, fig, total_gridspec, data_module, dataset, system, batch_info, uncoll_batch_info, batch_index=None, total_batch_idx_overwrite=None, dataset_identifier="val_0"):


        model_state_dict=system.state_dict()

        ## no batch index
        num_par_groups=0


        total_groups=0
        ### with batch index
        num_flow_groups=0

        beginning_strings=[]
        all_strings=[]

        ## get flow sub modules
        
        flow_params=None
        if(batch_index is not None):

            flow_params=system.obtain_flow_params(batch_info, batch_index)

            all_strings_flow=[]

            flattened_dict=dict()

            if(flow_params is not None):
                for top_key in flow_params.keys():
                    for layer_key in flow_params[top_key].keys():
                        for id_key in flow_params[top_key][layer_key].keys():
                            tot_str=top_key+"."+layer_key+"."+id_key
                            all_strings_flow.append(tot_str)
                            num_flow_groups+=1

                            flattened_dict[tot_str]=flow_params[top_key][layer_key][id_key]

            total_groups=num_flow_groups
        else:

            if(self.batch_par is not None):
                ### visualize a particular param for the whole batch
                flow_params=system.obtain_flow_params(batch_info, None)

                flattened_dict=collections.OrderedDict()

                for top_key in flow_params.keys():
                    for layer_key in flow_params[top_key].keys():
                        for id_key in flow_params[top_key][layer_key].keys():
                            tot_str=top_key+"."+layer_key+"."+id_key
                            print(tot_str, self.batch_par)
                            res=re.match(self.batch_par, tot_str)

                            if(res):
                                assert(flow_params[top_key][layer_key][id_key].dim()>=2), "Plotting param over whole batch requires currently > 2-dim param"

                                for i in range(flow_params[top_key][layer_key][id_key].shape[1]):
                                    print("angle pars shape", flow_params[top_key][layer_key][id_key].shape)
                                    if(len(flow_params[top_key][layer_key][id_key].shape)==3):
                                        for j in range(flow_params[top_key][layer_key][id_key] .shape[2]):
                                            flattened_dict[tot_str+"_%.2d_%.2d" % (i,j)]=flow_params[top_key][layer_key][id_key][:,i,j]
                                    else:
                                        flattened_dict[tot_str+"_%.2d" % (i)]=flow_params[top_key][layer_key][id_key][:,i]

                                total_groups+=numpy.prod(flow_params[top_key][layer_key][id_key].shape[1:])
                                
                                
                                
                                #all_strings.append(tot_str)

               

            else:
        
                for par in model_state_dict:

                    num_par_groups+=1

                    beginning_string=par.split(".")[0]

                    if(beginning_string not in beginning_strings):
                        beginning_strings.append(beginning_string)

                    all_strings.append(par)

                total_groups=num_par_groups

        #########################

        
        

        max_num_cols=6
        num_cols=max_num_cols

        num_rows=int(total_groups/num_cols)+1

        ## correct num_cols if it is less than max_num_cols

        if(total_groups < max_num_cols):
            num_cols=total_groups
            
        ### params

        if(batch_index is None):
            if(self.batch_par is not None):

                index=0
                ## we have not found the param... do something to create empty ax
                
                for k in flattened_dict.keys():

                    ax=fig.add_subplot(num_rows, num_cols, index+1)

                    assert(flattened_dict[k].dim()==1)

                    ax.hist(flattened_dict[k].cpu().numpy(), density=True, bins=50)
            
                    ax.set_title(k, fontdict={"fontsize": 11})

                    index+=1
                   
            else:
                for index in range(num_par_groups):
                    ax=fig.add_subplot(num_rows, num_cols, index+1)

                    self.visualize_params(ax, model_state_dict[all_strings[index]], all_strings[index])

                    ax.set_title(all_strings[index]+" / ".join([str(shape_val) for shape_val in model_state_dict[all_strings[index]].shape ]), fontdict={"fontsize": 11})
        else:
            ### final flow params
            for index in range(num_flow_groups):
                ax=fig.add_subplot(num_rows, num_cols, index+1+num_par_groups)

                self.visualize_params(ax, flattened_dict[all_strings_flow[index]], all_strings_flow[index])

                ax.set_title(all_strings_flow[index]+" / ".join([str(shape_val) for shape_val in flattened_dict[all_strings_flow[index]].shape ]), fontdict={"fontsize": 11})



        width_inches=3.5*num_cols
        height_inches=2*num_rows
        if(width_inches>0 and height_inches>0):
            fig.set_size_inches(width_inches, height_inches)

        fig.tight_layout()





