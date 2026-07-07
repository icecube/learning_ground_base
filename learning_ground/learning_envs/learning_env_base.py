from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from pytorch_lightning.callbacks import Callback
import pylab
import os
import io
import copy
import gc
import collections
import sys

from typing import Union, List

from .. import config_parser
from .. import lr_schedulers

import pytorch_lightning
#from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback, TuneCallback
from pytorch_lightning.loggers import TensorBoardLogger
import ray.tune as tune
import glob
import numpy

from memory_profiler import profile

try:
    import torch_geometric
except:
    print("torch geometric could not be imported!")

from torch.utils.data import DataLoader

import pickle
import json

from os import kill
from signal import SIGKILL
import time

import psutil

from tqdm import tqdm


from . import plotting_callback_base


def merge_dicts(dict1, dict2):
    merged_dict = copy.deepcopy(dict1)

    for key, value in dict2.items():
        if key in merged_dict and isinstance(merged_dict[key], dict) and isinstance(value, dict):
            merged_dict[key] = merge_dicts(merged_dict[key], value)
        else:
            merged_dict[key] = value

    return merged_dict

def get_last_epoch_cp(dir, config, trial_dir, resume_only_end_of_epoch=0, maximum_global_iter=None):

    cleaned_dir=None
    #cleaned_dir=trial_dir

    if(dir is None):
        if("resume" in config.keys()):
            if(config["resume"]=="True" or config["resume"]=="LOCAL"):
                print("Custom Restoration! We should switch to official restoration working!")
                raise Exception()
                cleaned_dir=trial_dir
        else:
            print("RESUME IS NOT In keys!?")
            return None

    else:

        cleaned_dir=dir[:dir.rfind("checkpoint_tmp")]

    print("cleaned dir", cleaned_dir)

    if(cleaned_dir is None):
        return None

    final_cp_string="cp_epoch_end"

    if(resume_only_end_of_epoch==0):
        final_cp_string="cp_validation_end"

    print("FINAL CP STRING RESUME :" , final_cp_string)
    cp_epoch_end=glob.glob(os.path.join(cleaned_dir, "checkpoint_*/%s" % final_cp_string))
            
    possible_epoch_files=[]
    for f in cp_epoch_end:
        if("tmp" not in f):
            possible_epoch_files.append(f)
    
    possible_epoch_files.sort()

    if(len(possible_epoch_files)>0):
        if("=" in possible_epoch_files[0]):
            # we deal with the new logging convenction ... perform different manual sorting

            int_list=numpy.array([int(pef.split("=")[-1].split("/")[0]) for pef in possible_epoch_files])

            sorta=numpy.argsort(int_list)
            
            possible_epoch_files=numpy.array(possible_epoch_files)[sorta]

            possible_epoch_files=[f for f in possible_epoch_files]
    
    

    if(len(possible_epoch_files)>0):

        if(maximum_global_iter is not None):

            if("=" not in possible_epoch_files[0]):
                raise Exception("Require newest pytorch/lihgtning combi with = in naming convention to extract step")

            this_index=len(possible_epoch_files)-1

            while this_index > 0:

                this_step=int(possible_epoch_files[this_index].split("=")[-1].split("/")[0])

                if(this_step<=maximum_global_iter):
                    print("MAX GLOBAL ITER FOUND ", this_step, " ... starting from here..")
                    return possible_epoch_files[this_index]

                this_index-=1

            raise Exception("Maximum iter %d not found in logged checkpoints!" % maximum_global_iters)
        else:
            return possible_epoch_files[-1]

    return None

###
## CALLBACKS
###

class GPUStatsCallback(Callback):

    def __init__(self):
        super().__init__()
  
    def print_gpu_stats(self, name="", kill_previous_processes=False):

        if(torch.cuda.is_available()):
            dev_count=torch.cuda.device_count()
            dev_count=0
            if(dev_count>0):
                print("///////////////// GPU STATS %s /////////////////////" % name)
                
                # device count
                print("device count ", dev_count)

                # arch list
                print("arch list of compilation ", torch.cuda.get_arch_list())

                # current device
                print("current device ", torch.cuda.current_device())
                current_device=torch.cuda.current_device()

                # device name
                print("device name ", torch.cuda.get_device_name(current_device))

                # device capability
                print("device capability ", torch.cuda.get_device_capability(current_device))

                # device properties
                print("device properties ", torch.cuda.get_device_properties(current_device))

                # nvcc gencode flags
                print("nvcss gencode flags ", torch.cuda.get_gencode_flags())

                # percent memory usage
                print("perc memory usage ", torch.cuda.memory_usage())

                # percent utilization
                print("perc gpu utilziation as given by nvidia-smi ", torch.cuda.utilization())
                
                print("mem get info ", torch.cuda.mem_get_info())

                #print("memory stats ", torch.cuda.memory_stats())

                print("memory summary ", torch.cuda.memory_summary())

                #print("memory snapshot ", torch.cuda.memory_snapshot())

                print("memory allocated ", torch.cuda.memory_allocated())

                print("max memory allocated ", torch.cuda.max_memory_allocated())

                print("memory reserved ", torch.cuda.memory_reserved())

                print("/////////////////////////////////////////////////////")

        else:
            print("///// CUDA NOT AVAILABLE! //////////")

    def on_validation_batch_end(self, *args, **kwargs):
        
        self.print_gpu_stats(name="VAL BATCH END")

    def on_validation_batch_start(self, *args, **kwargs):

        self.print_gpu_stats(name="VAL BATCH START")

class TuneSaveMetricCallback(TuneCallback):
    """PyTorch Lightning checkpoint callback

    Saves checkpoints after each validation step.

    Checkpoint are currently not registered if no ``tune.report()`` call
    is made afterwards. Consider using ``TuneReportCheckpointCallback``
    instead.

    Args:
        filename: Filename of the checkpoint within the checkpoint
            directory. Defaults to "checkpoint".
        on: When to trigger checkpoint creations. Must be one of
            the PyTorch Lightning event hooks (less the ``on_``), e.g.
            "batch_start", or "train_end". Defaults to "validation_end".


    """

    def __init__(
        self, filename="metrics", on: Union[str, List[str]] = "validation_end", metrics=dict()
    ):
        super(TuneSaveMetricCallback, self).__init__(on)
        self._filename = filename
        self._metrics = metrics

    def _get_report_dict(self, trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule):
        # Don't report if just doing initial validation sanity checks.
        if trainer.sanity_checking:
            return
        if not self._metrics:
            report_dict = {k: v.item() for k, v in trainer.callback_metrics.items()}
        else:
            report_dict = {}
            for key in self._metrics:
                if isinstance(self._metrics, dict):
                    metric = self._metrics[key]
                else:
                    metric = key
                print("cur metric ", metric)
                print("cb metrics: ", trainer.callback_metrics)
                if metric in trainer.callback_metrics:
                    report_dict[key] = trainer.callback_metrics[metric].item()
                else:
                    print("-------- WARNING ---- cannot save metric ", metric)
                    

        return report_dict

    def _handle(self, trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule):
        if trainer.sanity_checking:
            return

        report_dict = self._get_report_dict(trainer, pl_module)
        step = f"epoch={trainer.current_epoch}-step={trainer.global_step}"
        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            print("TUNE CALLBCK")
            print("REPORT DICT ", report_dict)
            print(checkpoint_dir)
            print("---------------")

            path = os.path.join(checkpoint_dir, "metrics")
            with open(path, "w") as f:
                f.write(json.dumps(report_dict))


class OverwriteSchedulerCallback(Callback):
    """
    Overwrites scheduler settings (num_red_steps, metric, and target lr)
    later in training, if desired
    """
    def __init__(self, used_config=None):
        super().__init__()

        self.used_config=None
        if(used_config is not None):
            self.used_config=copy.deepcopy(used_config)

    def on_sanity_check_start(self, trainer, pl_module):

        sched=pl_module.lr_schedulers()
        print("ON SANITY START -- SCHEDULER OVERWRITE --")
        if(sched is not None and self.used_config is not None):
            print("overwrite scheduler settings..")
            if(hasattr(sched, "tracking_metric")):
                if("scheduler.tracking_metric" in self.used_config):
                    setattr(sched, "tracking_metric",self.used_config["scheduler.tracking_metric"])
                    print("overwriting tracking metric...")
            if(hasattr(sched, "red_reducing_steps")):
                if("scheduler.num_validation_reducing_steps" in self.used_config):
                    new_red_steps=self.used_config["scheduler.num_validation_reducing_steps"]
                    red_multiple=sched.red_backwards_multiple_lr_scheduling
                    setattr(sched, "red_reducing_steps",self.used_config["scheduler.num_validation_reducing_steps"])
                    print("overwriting reducing steps...")
                    if(sched.past_metrics.maxlen != (new_red_steps*red_multiple)):
                        ## the bakcwards list is *backwards_multiple* as long as the amount of averaging
                        print("Resetting the past_metric deque list because it is of different length")
                        setattr(sched, "past_metrics", collections.deque(maxlen=(new_red_steps*red_mutiple)))

            if(hasattr(sched, "red_min_lr")):
                if("optimizer.min_lr" in self.used_config):
                    setattr(sched, "red_min_lr", self.used_config["optimizer.min_lr"])
                    print("overwriting min_lr...")

                    ## if it is cos annealing, make sure the minimum lr is still larger than red_min_lr
            
            ## fix to ensure in cos annealing scheduler we cannot go negative
            if(hasattr(sched, "eta_min")):
                if(sched.eta_min<sched.red_min_lr):
                    sched.red_min_lr=sched.eta_min*1.00001

            print("RED MIN LR ON LOAD CHECKPOINT CHECK.... ", sched.red_min_lr if(hasattr(sched, "red_min_lr")) else "NO RED MIN LR", "CONFIG OPTI MIN LR : ", self.used_config["optimizer.min_lr"] if "optimizer.min_lr" in self.used_config.keys() else "NO MIN LR IN USEDCONFIG", "eta min : ", sched.eta_min if(hasattr(sched, "eta_min")) else "NO ETA MIN"   ,file=sys.stderr)
                    
                  
            
        else:
            print("sched is None or used config is none.. no scheduler overwrite!")

    def on_load_checkpoint(self, trainer, pl_module, cp):

        sched=pl_module.lr_schedulers()
        print("ON LOAD CHECKPOINT -- SCHEDULER OVERWRITE --")
        #print(cp)
        if("lr_schedulers" in cp.keys() and self.used_config is not None):
            

            print("overwrite scheduler settings..")
            if("tracking_metric" in cp["lr_schedulers"][0].keys()):
                if("scheduler.tracking_metric" in self.used_config):
                    cp["lr_schedulers"][0]["tracking_metric"]=self.used_config["scheduler.tracking_metric"]
                    print("overwriting tracking metric...")
            if("red_reducing_steps" in cp["lr_schedulers"][0].keys()):
                if("scheduler.num_validation_reducing_steps" in self.used_config):
                    cp["lr_schedulers"][0]["red_reducing_steps"]=self.used_config["scheduler.num_validation_reducing_steps"]
                    new_red_steps=self.used_config["scheduler.num_validation_reducing_steps"]
                    red_multiple=cp["lr_schedulers"][0]["red_backwards_multiple_lr_scheduling"]
                    print("overwriting reducing steps...")
                    if(cp["lr_schedulers"][0]["past_metrics"].maxlen != (new_red_steps*red_multiple)):
                        ## the bakcwards list is *backwards_multiple* as long as the amount of averaging
                        print("Resetting the past_metric deque list because it is of different length")
                        cp["lr_schedulers"][0]["past_metrics"]=collections.deque(maxlen=(new_red_steps*red_multiple))

            if("red_min_lr" in cp["lr_schedulers"][0].keys()):
                if("optimizer.min_lr" in self.used_config):
                    cp["lr_schedulers"][0]["red_min_lr"]=self.used_config["optimizer.min_lr"]
                    print("overwriting min_lr...")

                   
            ## fix to ensure in cos annealing scheduler we cannot go negative

            if("red_min_lr" in cp["lr_schedulers"][0].keys()):
                if("eta_min" in cp["lr_schedulers"][0].keys()):
                    
                    cp["lr_schedulers"][0]["red_min_lr"]=cp["lr_schedulers"][0]["eta_min"]*1.00001
                    
            print("RED MIN LR ON LOAD CHECKPOINT CHECK.... ", cp["lr_schedulers"][0]["red_min_lr"], "CONFIG OPTI MIN LR : ", self.used_config["optimizer.min_lr"] if "optimizer.min_lr" in self.used_config.keys() else "NO MIN LR IN USEDCONFIG", "eta min : ",cp["lr_schedulers"][0]["eta_min"] if("eta_min" in cp["lr_schedulers"][0].keys()) else "NO ETA MIN"   ,file=sys.stderr)
                    
                  
        else:
            print("sched is None or used config is none.. no scheduler overwrite!")

## base class
class learning_env:

    ##############################################
    ## Init and a few functions below are only used for inference
    ##############################################
    def __init__(self, inference_config=None, inference_model_path=None, averaging_mode="single", inference_batch_size=50, use_gpu=0, overwrite_config=dict()):
        """
        inference_config (str): Path to pkl or json file with parameters of the model.
        inference_model_path: (str or list of str): Path or list of paths to model files.
        averaging_mode (str): "single" - only a single path has to be given , "average" - average over all model paths, swa" - take model from
        lr_scheduler which calculates the swa model
        """
        self.data_loaders=dict()

        ## the data modules are not used with standard train/batch datasets, just as mock modules to allow plotting functionality
        self.mock_data_modules=dict()

        self.dataset_sizes=dict()

        self.averaging_mode=averaging_mode # can be None / "swa" / "average"
        self.inference_batch_size=inference_batch_size
        self.collate_fn=None
        self.plot_collate_fn=None

        ## check config first
        if(inference_config is not None):
            self.inference_config=self._load_inference_config(inference_config)
        else:
            self.inference_config=None
        if(inference_model_path is not None):
            if(type(inference_model_path)==list):
                self.inference_model_paths=inference_model_path
            elif(type(inference_model_path)==str):
                self.inference_model_paths=[inference_model_path]
            else:
                raise Exception("wrong type for model path.. model path must be str or list of str", inference_model_path)
        else:
            self.inference_model_paths=None 
        
        # inference model pat his not None but inference config is None -> try to infer config from model file
        if( (self.inference_model_paths is not None) and (self.inference_config is None)):
            
            test_path=self.inference_model_paths[0]
            assert(type(test_path)==str)
            
            dirname=os.path.dirname(test_path)
            
            top_dirname=dirname[:dirname.rfind("/")]

            ## go through model path and one higher folder than model path
            for tp in [dirname, top_dirname]:
         
                params_dict=dict()
                params_file=os.path.join(tp, "params.pkl")
               
                if(not os.path.exists(params_file)):
                    params_file=os.path.join(tp, "params.json")
                    if(not os.path.exists(params_file)):
                        raise Exception("Neither pkl nor json params file found for inference!")

                params_dict=self._load_inference_config(params_file)

                self.inference_config=params_dict

                break

        ## at this point, inference config is not None meanse we want to do inference

        if(self.inference_config is not None):

            assert(type(self.inference_config)==dict), "Inference config has to be a dict"
            assert(self.inference_model_paths is not None), "Path for inference models have to be given, at least 1 -- checkpoint files."
            assert(len(self.inference_model_paths)>0), "We need at least one model to start inference.. one model path must be defined at loading"
            ## ok we have some config .. load the model path
            
            #raise NotImplementedError("Fix collate ordering for inference")
            #collate_fn, plot_uncollate_fn, _=self.get_collate_n_loss_fn(self.inference_config)
            # object that holds sum of models for inference
            #self.collate_fn=collate_fn
            #self.plot_uncollate_fn=plot_uncollate_fn

            ##########

            
            """
            if(torch.cuda.is_available()==False or use_gpu==0):
                map_loc=torch.device("cpu")
                print("<<< WARNING: USING CPU FOR INFERENCE, BECAUSE NO GPU FOUND... IS IT INTENDED? >>>")
            else:
                vis_device=os.environ.get("CUDA_VISIBLE_DEVICES")
                ### TODO: do we want 0 always? NO
                map_loc=torch.device(0)
            """

    
            ## merge overwrite options into config
            self.inference_config=merge_dicts(self.inference_config, overwrite_config)

            ## first map to CPU
            ckpt = torch.load(self.inference_model_paths[0], map_location=torch.device("cpu"))
            self.num_model_averages=1

            if(averaging_mode=="single"):
                assert(len(self.inference_model_paths)==1), "Want to perform inference for a single model but multiple paths are given ..."
                self.inference_model=self.load_model(ckpt, config=self.inference_config)
            elif(averaging_mode=="average"):
                raise Exception("Outdated!")
                #assert(len(self.inference_model_paths)>1), "Averaging only makes sense for many paths .. please give more than 1 path!"
                #for mp in self.inference_model_paths[1:]:
                #    self.add_path_to_model(mp)

            elif(averaging_mode=="swa"):
                self.inference_model=self.load_model(ckpt, config=self.inference_config, load_averaged=True)
      
            else:
                raise Exception("Unknown averaging_type ", averaging_mode)  

            ## push to GPU at the end of model loading
            self.gpu_index=-1
            print("CUDA AVIALBALE: ", torch.cuda.is_available(), " USE GPU: ", use_gpu)
            if( (torch.cuda.is_available()==True) and (use_gpu>0)):
                self.gpu_index=use_gpu-1
                self.inference_model.to(torch.device("cuda:%d"%self.gpu_index))
                print("PUSH TO GPU !!!!!-------------------->")

            self.collate_fn, self.plot_uncollate_fn = self.inference_model.get_encoding_scheme_collates()
               

    def _load_inference_config(self, filename):

        if(type(filename)==dict):
            params_dict=filename
        else:
            if("pkl" in filename):
                with open(filename, "rb") as f:
                    params_dict=pickle.load(f)

            elif("json" in filename):
             ## json file
                with open(filename) as f:
                    params_dict = json.load(f)
            else:
                raise Exception("Unknown config file format", filename)

        ## if there are fixed choices, use them to overwrite actual config parameters

        if("fixed_choices" in params_dict.keys()):
            for k in params_dict["fixed_choices"].keys():
                params_dict[k]=params_dict["fixed_choices"][k]

        return params_dict

    def add_path_to_model(self, additional_model_path):
        """ 
        Adds a model/models from *path* to the current model (if it is a multi-model) and updates its parmaeters
        pars:
            inference_model_path: (str or list of str) path to a model or a list of models. Can either point to the model file
            or to the path containing the model.
        """ 

        new_num_averages=self.num_model_averages+1


        if(torch.cuda.is_available()==False):
            map_loc=torch.device("cpu")
        else:
            vis_device=os.environ.get("CUDA_VISIBLE_DEVICES")
            ### TODO: do we want 0 always? NO

            if(vis_device==""):
                map_loc=torch.device(0)
            else:
                map_loc=torch.device(vis_device)

        ckpt = torch.load(additional_model_path)
        

        tmp_model=self.load_model(ckpt, config=self.inference_config, map_location=map_loc)

        tmp_state_dict=tmp_model.state_dict()

        model_state_dict=self.inference_model.state_dict()

        prefac_1=float(self.num_model_averages)/float(new_num_averages)
        prefac_2=1.0/float(new_num_averages)
       
        for par in tmp_state_dict:
            
            model_state_dict[par].data=prefac_1*model_state_dict[par].data+prefac_2*tmp_state_dict[par].data
           
        
        ## load updated state dict back into system
        self.inference_model.load_state_dict(model_state_dict)


        ## update number of model averages
        self.num_model_averages=new_num_averages


    def add_dataset(self, datamodule_class, name, extra_dataset_params=dict()):

        assert(self.inference_config is not None)

        ## make new inference config where every *data.train* is replaced by *name*
        new_inference_config=dict()

        for k in self.inference_config:
            new_inference_config[k.replace("data.train", name)]=copy.deepcopy(self.inference_config[k])

        self.mock_data_modules[name]=datamodule_class(new_inference_config)

        init_params=copy.deepcopy(extra_dataset_params)
            
        print("overwriting some params for inference ... ")
        for extra_p in init_params:
            print(extra_p, " : ", init_params[extra_p])

        print("extra params for inference .... ", init_params)

        ## remove inference_dataset_size from keys for dataset
        if("inference_dataset_size" in init_params.keys()):
            del init_params["inference_dataset_size"]

        ds_name, dataset, bs, num_workers=self.mock_data_modules[name].get_dataset(new_inference_config, init_params, name)

        setattr(self.mock_data_modules[name], "dataset", dataset)

        inference_dataset_size=None
        if(hasattr(dataset, "__len__")):
            inference_dataset_size=len(dataset)
     

        self.data_loaders[name]=DataLoader(dataset, batch_size=self.inference_batch_size, collate_fn=self.collate_fn)

        ## Iterable datasets should also define a size if possible - given by extra params
        if("inference_dataset_size" in extra_dataset_params.keys()):
            inference_dataset_size=extra_dataset_params["inference_dataset_size"]

        if(inference_dataset_size is None):
            raise Exception("Dataset is an iterable dataset, must specify *inference_dataset_size* in *extra_dataset_params* within add_dataset!")
       
        self.dataset_sizes[name]=inference_dataset_size
        

    def inference_collated_batch(self, name, collated_batch, **kwargs):
        # batch is already in collated form .. we can directly evaluate it

        print("inference col batch name. .. ", name)
        assert(hasattr(self.inference_model, name)), (name, " is not an inference function of the model")
        
        inference_fn=getattr(self.inference_model, name)

        assert(hasattr(inference_fn, "__call__")), ("inference function ", name, " is not of type *function*")
        print("KWARGS 2", kwargs)
        returns=inference_fn(collated_batch, **kwargs)

        # add dataset specific variables (e.g. total NCH for neutrino events) that are not generic
        dataset_specific_variables=self.inference_model.get_dataset_specific_variables(collated_batch)

        for k in dataset_specific_variables.keys():
            returns[k]=dataset_specific_variables[k]

        return returns

    def inference_plot_events(self, plotting_type, dataset_name, absolute_indices, target_folder, basename="event"):

        assert(self.inference_model is not None), "Plotting event information in inference mode requires a loaded inference model!"

        if(not os.path.exists(target_folder)):
            os.makedirs(target_folder)

        ## give fake indices just to create plotting object
        plotting_object=self.get_plot_callback(plotting_type)(indices="0") 

        ###
        
        generator=iter(self.data_loaders[dataset_name])
        used_dataset_size=self.dataset_sizes[dataset_name]

        if(type(absolute_indices)==numpy.ndarray):
            absolute_indices_numpy=absolute_indices
        elif(type(absolute_indices)==list):
            absolute_indices_numpy=numpy.array(absolute_indices)
        else:
            raise Exception("Indices must either be numpy array or list!", type(absolute_indices))

        tot_events=0

        while tot_events<used_dataset_size:

            b=next(generator)

            cur_index_min=tot_events

            cur_index_max=tot_events+self.inference_batch_size-1

           
            indices_in_interval=absolute_indices_numpy[(absolute_indices_numpy>=cur_index_min) & (absolute_indices_numpy<=cur_index_max)]

            if(len(indices_in_interval)>0):
                relative_batch_indices=indices_in_interval-tot_events

                for ind, rel_batch_index in enumerate(relative_batch_indices):
                    
                    fig, total_gridspec=plotting_object.make_fig_and_layout(self.inference_model)

                    plotting_object.visualize(fig, total_gridspec, self.mock_data_modules[dataset_name], self.mock_data_modules[dataset_name].dataset, self.inference_model, b, b, batch_index=rel_batch_index, global_index=indices_in_interval[ind])

                    fig.savefig(os.path.join(target_folder, basename+"_%.5d.png" % indices_in_interval[ind]))
               
            tot_events+=self.inference_batch_size

    def inference(self, inference_type, batch_indices="all", dataset_name=None, **kwargs):
        """    
        Performs inference with the given function *name*. The given batch_index is either an index, a tuple, or "all", which will 
        loop over the whole dataset.
        Parameters:
            name: function name to call from parent class (e.g. GOF, sample, log_eval)
            batch_indices: int (no of , tuple(int, int) tuple(min_index [incl.], batchsize), str "all"
            dataset_name: Which connected dataset to use for inference. If None, will check that only one dataset is connected, and use that one.
            **kwargs: Any further keywords given to the inference function. If name is a list, this may only contain one entry with name *inference_function_options*.
        """
        assert(self.inference_model is not None), "Inference model has to be loaded."
        
        if(type(inference_type)==str):
            inference_types=[inference_type]
            inference_function_options=[kwargs]
        else:
            ## emtpy extra dict as default
            inference_types=inference_type
            inference_function_options=[dict() for i in range(len(inference_type))]

            if(len(kwargs)>0):
                assert("inference_function_options" in kwargs)
                inference_function_options=kwargs["inference_function_options"]
                assert(len(inference_function_options) == len(inference_types))

        
        assert(type(inference_types)==list), "We can have several inference functions"
        ### we feed in the collated batch

        used_dataset_name=dataset_name
        if(dataset_name==None):
            if( len(self.data_loaders) != 1):
                raise Exception("There can be only one dataset loader when no dataset name is given. Specific dataset_name!")
            used_dataset_name=next(iter(self.data_loaders))

        ## list to contain returns of inference - list of tensors, at the end concatonated
        inference_returns=dict()

        used_dataset_size=None
        max_index=None

        if(type(batch_indices)==str):

            if(batch_indices=="all"):

                #used_dataset_size=self.dataset_sizes[used_dataset_name]
                max_index=self.dataset_sizes[used_dataset_name]-1
                batch_indices=numpy.arange(used_dataset_size)

            else:
                raise Exception("batch index is string and must be all")

        elif(type(batch_indices)==int):
            # batch_indices is a single integer index
            max_index=batch_indices

            assert(batch_indices<self.dataset_sizes[used_dataset_name])

            #used_dataset_size=batch_indices+1+self.inference_batch_size
            #used_dataset_size=(used_dataset_size//self.inference_batch_size) * self.inference_batch_size

            batch_indices=numpy.array([batch_indices])

        else:
            assert( (type(batch_indices)==list) or (type(batch_indices)==numpy.ndarray))

            max_index=max(batch_indices)

            if(type(batch_indices)==list):
                batch_indices=numpy.array(batch_indices)

        assert(max_index<self.dataset_sizes[used_dataset_name])

        used_dataset_size=max_index+1#+self.inference_batch_size
        #used_dataset_size=(used_dataset_size//self.inference_batch_size) * self.inference_batch_size

        ####################

        #generator=iter(self.data_loaders[used_dataset_name])

        #assert(used_dataset_size%self.inference_batch_size==0), ("total number of events requested, ", used_dataset_size, " has to divide by batch sizes", self.inference_batch_size)
        assert(self.dataset_sizes[used_dataset_name]>=used_dataset_size), ("dataset size ,", self.dataset_sizes[used_dataset_name], " has to be larger or equal than number of events requested", used_dataset_size)

        tot_events=0
        current_slice=None
        
        num_total_batches=len(batch_indices)//self.inference_batch_size

        all_slices=[]

        if(num_total_batches>0):


            all_slices=batch_indices[:num_total_batches*self.inference_batch_size].reshape(-1, self.inference_batch_size).tolist()
            
            all_slices=[numpy.array(i) for i in all_slices]

        remainder=len(batch_indices)%self.inference_batch_size

        if(remainder>0):

            
            last_indices=batch_indices[num_total_batches*self.inference_batch_size: num_total_batches*self.inference_batch_size+remainder]

            num_total_batches+=1

            all_slices.append(last_indices)
        
        ## store keys that are added so no key is added twice
        keylist_crosscheck=[]

        for test_slice in tqdm(all_slices):

            ## obtain a collated batch
            t=[self.data_loaders[used_dataset_name].dataset.__getitem__(i) for i in test_slice]
            b=self.data_loaders[used_dataset_name].collate_fn(t)


            ## push all tensors in batch to GPU if GPU is used
            if(self.gpu_index>-1):
                for bkey in b.keys():
                    if( (type(b[bkey])==torch.Tensor)  or (type(b[bkey]) == torch.nn.utils.rnn.PackedSequence) ):# or (type(b[bkey])==torch_geometric.data.batch.DataBatch)):
                        
                        b[bkey]=b[bkey].to(device=torch.device("cuda:%d" % self.gpu_index))
                    elif((type(b[bkey])==list)):
                        if(type(b[bkey][0])==torch.Tensor):
                            ## put tensor list also on GPU
                            b[bkey]=[litem.to(device=torch.device("cuda:%d" % self.gpu_index)) for litem in b[bkey]]
                        else:
                            print(bkey, " (list) not put on GPU.....")
                            print("list item type: ", type(b[bkey][0]))
                            print(".......................")
                    else:
                        print(bkey, " not put on GPU.....")
                        print(type(b[bkey]))
                        print(".......................")
                        
                        



            ## requires a batch_size item in the batch
            #tot_events+=b["batch_size"]

            #pbar.update(b["batch_size"])
            """
            if(type(batch_indices)==int):
                if(batch_indices>=tot_events):
                    continue
                ## ok we are in index mode and the next batch contains the event .. calculate its relative index
                current_slice=slice(batch_indices-(tot_events-self.inference_batch_size))
            elif(type(batch_indices)==numpy.ndarray):
                
                if(min(batch_indices)>=tot_events):
                    continue

                new_indices=batch_indices-(tot_events-self.inference_batch_size)
                new_indices=new_indices[ (new_indices>=0) & (new_indices<self.inference_batch_size)]

                if(len(new_indices)==0):
                    continue

                current_slice=new_indices
            """
            these_kw_counter=dict()

            for inf_index, inf_type in enumerate(inference_types):

                ret=self.inference_collated_batch(inf_type, b, **inference_function_options[inf_index])

                
                assert("labels" not in ret), "*labels* keyword is not allowed to be returned by an inference function!"
                if("event_properties" in b):
                    for ev_prop in b["event_properties"]:
                        assert(ev_prop not in ret), ("Event property ", ev_prop, " is also returned by inference function ", inf_type, " .. this may not happen, please fix inference function or event_properties.")

                for k in ret.keys():
                    if(k not in inference_returns.keys()):
                        inference_returns[k]=[]

                    if(k not in these_kw_counter):
                        these_kw_counter[k]=1
                    else:
                        these_kw_counter[k]+=1

                    if(type(ret[k])==torch.Tensor):
                        ret[k]=ret[k].cpu().numpy()

                    inference_returns[k].append(ret[k])
           
            for k in these_kw_counter:
                assert(these_kw_counter[k]==1), ("Multiple inference functions return the same key ... this may not happen!", k, these_kw_counter)

            ## also get true labels of batch if available
            if("labels" in b):
                if(b["labels"] is not None):
                    if("labels" not in inference_returns.keys()):
                        inference_returns["labels"]=[]

                    inference_returns["labels"].append(b["labels"].cpu().numpy())

            if("event_properties" in b):
                for ev_prop in b["event_properties"]:
                    if(ev_prop not in inference_returns):
                        inference_returns[ev_prop]=[]

                    if(type(b["event_properties"][ev_prop])==list):
                        inference_returns[ev_prop].append(numpy.array(b["event_properties"][ev_prop]))
                    elif(type(b["event_properties"][ev_prop])==torch.Tensor):
                        inference_returns[ev_prop].append(b["event_properties"][ev_prop].clone().detach().numpy())
                    else:
                        assert(type(b["event_properties"][ev_prop])==numpy.ndarray)
                        inference_returns[ev_prop].append(b["event_properties"][ev_prop])


        ## another check
        #assert(tot_events<=self.dataset_sizes[used_dataset_name])

        for k in inference_returns.keys():
            
            inference_returns[k]=numpy.concatenate(inference_returns[k], axis=0)

        return inference_returns

    

    #############################################################
    ## The following functions are subclassed
    #############################################################
    ## every subclass has to define how a model is generated
    def make_model(self, config=None):

        raise NotImplementedError

    ## every subclass has to define how a model is loaded
    def load_model(self, checkpoint, config=None):

        raise NotImplementedError

    def update_model(self, model, ckpt, config):
        """
        Update model weights n biases based on ckpt.
        """
        raise NotImplementedError

    ## every subclass has to define some crosschecks for model and data_module
    """
    def model_datamodule_checks(self, model, data_module):

        raise NotImplementedError
    """
    
    def _model_datamodule_inits(self, model, data_module):

        print("--------------------------------------")
        print("No Model/Datamodule shared initialization! Implement function in subclass to overwrite!")
        print("--------------------------------------")

    ## every subclass has to define the data collate / uncollate (for plotting) / batch_loss 
    def get_collate_n_loss_fn(self, config):

        
        raise NotImplementedError

    ## function to return a plotting callback class
    def get_plot_callback(self, name):

        ### general plots that work for any model
        if(name == "visualize_params"):
            return plotting_callback_base.visualize_params
        elif(name == "val_loss_vis"):
            return plotting_callback_base.val_loss_vis
        elif(name == "visualize_fisher_grad_projections"):
            return plotting_callback_base.visualize_fisher_grad_projections
        else:
            ## model specific plots
            return self._get_plot_callback(name)

    def _get_plot_callback(self, name):

        raise NotImplementedError()
       
    #############################################################
    ## The following function is used for training only
    ###############################################################
   
    def get_trainable(self, data_module_func):
        """
        Returns a *ray-trainable* function for training.
        """
        
        def trainable(config, checkpoint_dir=None, overwrite_config=dict()):

           
            
            #######################

            ## callback stuff
            used_callbacks=[]

            ## progress bar
            bar=TQDMProgressBar(refresh_rate=1)
            #used_callbacks.append(bar)

            ###### OVERWRITE stuff if desired

            ### overwritable general keys

            #other_keys = ["train_batch_size", "plotting_args", "deterministic", "resume_only_end_of_epoch", "gpu"]
            ## train_batch_size can not be changed now 
            other_keys = ["plotting_args", 
                          "deterministic", 
                          "resume_only_end_of_epoch", 
                          "gpu", 
                          "data.train_num_workers", 
                          "data.train.batch_size",
                          "scheduler.num_validation_reducing_steps", 
                          "scheduler.tracking_metric",
                          "optimizer.min_lr"]
            
            for k in other_keys:
                if(k in overwrite_config.keys()):
                    config[k]=overwrite_config[k]

            ####################################

            change_optimizer=False
            change_scheduler=False

            ## overwrite scheduler or optimizer in resume
            if("overwrite_optimizer" in overwrite_config.keys()):
                if(overwrite_config["overwrite_optimizer"]==1):
                    #if(config["optimizer.name"]!=overwrite_config["optimizer.name"]):
                    change_optimizer=True

            if("overwrite_scheduler" in overwrite_config.keys()):
                if(overwrite_config["overwrite_scheduler"]==1):
                    change_scheduler=True

            ###############

            ## overwrite optimizer-specific keys
            opti_overwrite_keys=["optimizer.lr", "optimizer.name", "scheduler.name"]
            for ok in opti_overwrite_keys:
                if(ok in overwrite_config.keys()):
                    if(ok in config.keys()):

                        ## enforce overwriting optimizer or scheduler if a different name is given
                        if(ok=="optimizer.name"):
                            if(config["optimizer.name"]!=overwrite_config["optimizer.name"]):
                                change_optimizer=True

                        if(ok=="optimizer.scheduler"):
                            if(config["optimizer.scheduler"]!=overwrite_config["optimizer.scheduler"]):
                                change_scheduler=True



                        config[ok]=overwrite_config[ok]

            print("############# sorted (with overwrites) config:")
            for k in sorted(config):
                print(k, config[k])
            print("###############################################")
            #### get general config

            ## define the config parser
            cfg_parser=config_parser.config_parser()

            ## deterministic seeds  ?
            cfg_parser.add_default_kwarg("gen", "deterministic", 1, int)

            ## for gpu training in some pytorch versions deterministic behavior can not be guaranteed for *scatter* and related functions .. by switching off the flag we can ignore this
            ## and still train .. no idea of the exact effect 
            cfg_parser.add_default_kwarg("gen", "switch_off_pytorch_deterministic_flag", 0, int)
            
            ## number of epochs for training
            cfg_parser.add_default_kwarg("gen", "scheduler.add_safety_callback", 1, int)

            ## number of epochs for training
            cfg_parser.add_default_kwarg("gen", "num_epochs", 10000, int)

             ## resume only at the end of each epoch or also in between at validation checkpoints
            cfg_parser.add_default_kwarg("gen", "resume_only_end_of_epoch", 0, int)


            ## gpu_support
            cfg_parser.add_default_kwarg("gen", "gpu", 0, int)

            _, general_learning_kwargs=cfg_parser.parse_cfg(config, "gen")

            ## train_size and train_batch_size are always required to be defined by script! -> No default values!
            #general_learning_kwargs["data.train.dataset_size"]=config["data.train.dataset_size"]
            #general_learning_kwargs["data.train.batch_size"]=config["data.train.batch_size"]

            ### TODO: check do we want this check here? probably not, should be in dataloading classes
            #assert(general_learning_kwargs["train_size"] % general_learning_kwargs["train_batch_size"]==0), "Dataset size must be divisible by train batch size."

            deterministic=general_learning_kwargs["deterministic"]
            switch_off_deterministic_scatter_n_related=general_learning_kwargs["switch_off_pytorch_deterministic_flag"]

            if(deterministic):
                seed_everything(0)

            #if(switch_off_deterministic_scatter_n_related):
            #    print("--- SWITCHING OFF DETERMINISTIC PYTORCH TRAINING ---")
            #    torch.use_deterministic_algorithms(False)

            #####################################

            ## add GPU benchmarking
            #used_callbacks.append(GPUStatsCallback())


            ## plotting stuff
            indices_for_plotting=[]
            cfg_parser.add_default_kwarg("plotting", "plotting_args", "basic__indices=0,10,50", str)
            cfg_parser.add_default_kwarg("plotting", "val_interval", 50, int)
            cfg_parser.add_default_kwarg("plotting", "plot_interval", 50, int)
            _, plotting_kw=cfg_parser.parse_cfg(config, "plotting")

            assert(plotting_kw["plot_interval"]>=plotting_kw["val_interval"]), "Plotting interval must be >= validation interval!"
            assert(plotting_kw["plot_interval"]%plotting_kw["val_interval"]==0), "Plotting interval must be divisible by validation interval without rest."
            
            plotting_callbacks=[]

            for plot_option in plotting_kw["plotting_args"].split("+"):
                if(plot_option!=""):

                    splits=plot_option.split("__")
                    #these_indices="all"
                    this_plot_def=splits[0]

                    options=dict()

                    for opt in splits[1:]:
                        opt_splits=opt.split("=")

                        assert(len(opt_splits)==2), opt_splits

                        options[opt_splits[0]]=opt_splits[1]

                        if(opt_splits[0]=="indices"):
                            for index in opt_splits[1].split(","):
                                indices_for_plotting.append(int(index))

                    options["plot_interval"]=plotting_kw["plot_interval"]
                    ## allow for flexible options for plotting scripts
                    
                    new_callback=self.get_plot_callback(this_plot_def)(**options)
                    used_callbacks.append(new_callback)
                    plotting_callbacks.append(new_callback)

                    """
                    if("__" in plot_option):
                        these_indices=plot_option.split("__")[1]
                        this_plot_def=plot_option.split("__")[0]
                        
                    if(these_indices!="all"):
                        index_splits=[int(i) for i in these_indices.split(",")]

                        for i in index_splits:
                            if(i not in indices_for_plotting):
                                indices_for_plotting.append(i)
                        

                        new_callback=self.get_plot_callback(this_plot_def)(indices_to_plot=index_splits)
                        used_callbacks.append(new_callback)
                        plotting_callbacks.append(new_callback)
                    else:
                        new_callback=self.get_plot_callback(this_plot_def)(indices_to_plot=None)
                        used_callbacks.append(new_callback)
                        plotting_callbacks.append(new_callback)
                    """
            # scheduler callback first to overwrite model if it is bad
            #if(general_learning_kwargs["scheduler.add_safety_callback"]):
            #    used_callbacks.append(lr_schedulers.SchedulerCallback())

            # checkpoint at each valdiation checkpoint
            checkpoint_callback1=TuneReportCheckpointCallback(
                        #metrics={
                        #    "mean_accuracy": "s.val_loss"
                        #},
                        filename="cp_validation_end",
                        on="validation_end")

            
            used_callbacks.append(checkpoint_callback1)

            ## 

            metric_callback=TuneSaveMetricCallback(on="validation_end", filename="metrics.txt", metrics={
                            "s.val_loss": "s.val_loss"
                        })

            used_callbacks.append(metric_callback)

            # checkpoint each epoch end
            checkpoint_callback2=TuneReportCheckpointCallback(
                        filename="cp_epoch_end",
                        on="train_end")

          
            used_callbacks.append(checkpoint_callback2)
            
            """
            ### REPORT callback
            TuneReportCallback(
                {
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy"
                },
                on="validation_end"
            """
            
            #############################################
            
            ### DEPRECATED BLOCK ... move data init AFTER Model init!!! -> NEW ORDER:
            ### MODEL -> model.get_encoding_scheme_collates() -> DATAModule
            """
            ## define loss functions and stuff
            collate_fn, plot_uncollate_fn, _=self.get_collate_n_loss_fn(config)
            
            ## load the data module
            indices_for_plotting.sort()

            data_module=data_module_func(config, collate_fn=collate_fn, log_validation_indices_for_plotting=indices_for_plotting)
            data_module.setup()
           
            setattr(data_module, "train_dataset_size", general_learning_kwargs["train_size"])


            ## configure training batch size and num steps per epoch
            train_batch_size=None

            if(hasattr(data_module, "train_batch_size")):
                train_batch_size=data_module.train_batch_size
            else:
                raise Exception("Data module must define train_batch_size as an attribute! Fix your data module please!")
            """
            print("------------------ TORCH CUDA TEST ----------------")
            print(torch.cuda.is_available())
            if(torch.cuda.is_available()):

                dev_count=torch.cuda.device_count()
                print("-------- DEV COUNT ------ ", dev_count)
            print("now with kill process optionc")
            GPUStatsCallback().print_gpu_stats(name="ENV STARTUP", kill_previous_processes=True)

            gc.collect()
            torch.cuda.empty_cache()

            GPUStatsCallback().print_gpu_stats(name="ENV STARTUP 2 ")
            maximum_global_iter=None

            print("ENV VARS")
            print("-----------------------------")
            for k in os.environ:
                print(k, " : ", os.environ[k])
            print("-----------------------------")

            ## before loading or starting a model, lets check the GPU drivers etc
            if(torch.cuda.is_available()):
                os.system("nvidia-smi")
                os.system("ps aux | grep ray")

            if("maximum_global_iter" in overwrite_config.keys()):
                maximum_global_iter=overwrite_config["maximum_global_iter"]
            
            model=None
            ## Restoration of checkpoints
            if(checkpoint_dir is not None):
                print("Tune CP dir is not NONE:", checkpoint_dir)

            valid_checkpoint_dir=get_last_epoch_cp(checkpoint_dir, config, tune.get_trial_dir(), resume_only_end_of_epoch=general_learning_kwargs["resume_only_end_of_epoch"], maximum_global_iter=maximum_global_iter)

            print("valid cp dir ", valid_checkpoint_dir)
            cur_epoch=0
            cur_overall_iter=0

            if(valid_checkpoint_dir is not None):
                ### extract epoch and iter from last checkpoint

                cp_str=valid_checkpoint_dir.split("/")[-2]
                cur_epoch=int(cp_str.split("_")[-1].split("-")[0].split("=")[-1])
                cur_overall_iter=int(cp_str.split("_")[-1].split("-")[1].split("=")[-1])
                print("extract epoch and iter from checkpoint")
                print("cur epoch: ", cur_epoch)
                print("cur overall iter", cur_overall_iter)
                
            ############## 1) MODEL INITS ####################
            model = self.make_model(config=config)

            ########## 2) DATA LOADER AND ENCODING/COLLATING SCHEME DEFS ##########
            ####################################################################

            collate_fn, plot_uncollate_fn=model.get_encoding_scheme_collates()
            
            ## load the data module
            indices_for_plotting.sort()

            data_module=data_module_func(config, collate_fn=collate_fn)
            data_module.setup()


            ## initalize datamodule/model relations
            self._model_datamodule_inits(model, data_module)
           
            #setattr(data_module, "train_dataset_size", general_learning_kwargs["train_size"])

            ## configure training batch size and num steps per epoch
            train_batch_size=None
            train_dataset_size=None
            if(hasattr(data_module, "train_batch_size")):
                train_batch_size=data_module.train_batch_size
            else:
                raise Exception("Data module must define *train_batch_size* as an attribute! Fix your data module please!")

            if(hasattr(data_module, "train_dataset_size")):
                train_dataset_size=data_module.train_dataset_size
            else:
                raise Exception("Data module must define *train_dataset_size* as an attribute! Fix your data module please!")

            assert(train_dataset_size%train_batch_size==0), "Dataset size must be divisible by batch size!"

            steps_per_epoch=int(train_dataset_size/train_batch_size)

            assert(hasattr(data_module, "validation_dict_list")), "Please implement a dictionary in the datamodule with name *validation_dict_list* to associate a validation dataset id with its name." 
                
            ## make sure we get a copy here
            val_names=[]
            for ditem in data_module.validation_dict_list:
                newname=""
                for c in ditem["name"]:
                    newname=newname+c
                val_names.append(newname)

            ## set training settings in model .. required for proper training paramters .. can vary between training runs
            ## e.g. set up a dicitionary with *key* identifiers for each validation dataloader, uniquely identify different validaiton losses
            model.set_training_settings(train_dataset_size, 
                                         train_batch_size, 
                                         plotting_kw["val_interval"],
                                         val_names,
                                         config,
                                         data_module)

            ########## 3) UPDATE WEIGHTS N BIASES ##########
            ####################################################################

            if valid_checkpoint_dir is not None:
                print("VALID CP dir", valid_checkpoint_dir)
                
                ckpt = torch.load(valid_checkpoint_dir, map_location=torch.device('cpu'))

                """
                ckpt = pl_load(
                    valid_checkpoint_dir,
                    map_location=lambda storage, loc: storage)
                """
                self.update_model(model, ckpt, config)
                
                print("the model is loaded .....")
                print("change optimizer ", change_optimizer)

            else:
                
                
                make_new_model=True
                if("initialize_from_checkpoint_file" in config.keys()):
                    if(config["initialize_from_checkpoint_file"] is not None):
                        print("initializing weights form checkpoint ...")
                        assert(type(config["initialize_from_checkpoint_file"])==str)


                        ckpt = torch.load(config["initialize_from_checkpoint_file"])
                        """
                        ckpt = pl_load(
                        config["initialize_from_checkpoint_file"],
                        map_location=lambda storage, loc: storage)
                        """

                        model = self.load_model(ckpt, config=config)

                        make_new_model=False

                elif("pbt_config" in config.keys()):
                    # we use PBT
                    model_path=None
                    blueprint_path=None
                    print("CONIG START ", config)
                    assert("genetic_config" in config["pbt_config"].keys()), "Require genetic configuration in pbt config!"
                    assert("resample_options" in config["pbt_config"].keys()), "Require resample configuration in pbt config!"

                    resample_options=config["pbt_config"]["resample_options"]
                    assert("num_cont_params_constraint" in resample_options.keys()), resample_options

                    genetic_config=config["pbt_config"]["genetic_config"]

                    if("cp_file" in config.keys() and "blueprint_file" in config.keys()):
                        ## this is a derivative of a model
                        model_path=config["cp_file"]
                        blueprint_path=config["blueprint_file"]

                    genetic_model=self._make_genetic_model(blueprint_path, resample_options, genetic_config=genetic_config, **config)

                    par_constraint=resample_options["num_cont_params_constraint"]

                    if(model_path is not None and blueprint_path is not None):
                        ## incorporate model and perturb it
                        print("PERTURBING MODEL!->")
                        ckpt = torch.load(model_path)

                        temp_model = self.load_model(ckpt, config=config)

                        ## incorporate previous model into current genetic config
                        genetic_model.incorporate_continuous_parameters(temp_model)

                        ## perturb
                        genetic_model.perturb(param_constraints=par_constraint)

                    else:
                        ## make sure both are either None or defined
                        print("Starting new MODEL -> Sample uniformly")
                        assert(model_path is None)
                        assert(blueprint_path is None)

                        genetic_model.sample_quasi_uniform(param_constraints=par_constraint)

                    ## save new genetic config in current path
                    trialdir=tune.get_trial_dir()
                    blueprint_filename=os.path.join(trialdir, "dna_blueprint")
                    genetic_model.save(blueprint_filename)

                    ## return new model
                    model=genetic_model.return_model()

            ## add scheduler callback
            scheduler_callback=OverwriteSchedulerCallback(used_config=config)
            used_callbacks.append(scheduler_callback)

            ############ 4) TRAINER INIT 
            #########################

            trainer_kwargs=dict()
            trainer_kwargs["max_epochs"]=general_learning_kwargs["num_epochs"]
            trainer_kwargs["limit_train_batches"]=steps_per_epoch
            #trainer_kwargs["limit_val_batches"]=1
            trainer_kwargs["log_every_n_steps"]=100 # training log every n steps
            trainer_kwargs["num_sanity_val_steps"]=1 # switch off sanity validation in beginning
            trainer_kwargs["val_check_interval"]=plotting_kw["val_interval"]
            #trainer_kwargs["gpus"]=general_learning_kwargs["gpu"]
            
            if(general_learning_kwargs["gpu"]):
                trainer_kwargs["accelerator"]="gpu"
                trainer_kwargs["devices"]=1
            
            trainer_kwargs["logger"]=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".")
            #trainer_kwargs["progress_bar_refresh_rate"]=0
            trainer_kwargs["callbacks"]=used_callbacks
            #trainer_kwargs["automatic_optimization"]=do_automatic_optimization
            trainer_kwargs["deterministic"]=bool(deterministic)
           
            trainer_kwargs["gradient_clip_val"]=None
            if("max_grad_norm" in config.keys()):
                trainer_kwargs["gradient_clip_val"]=config["max_grad_norm"]

            if("max_grad_norm" in overwrite_config.keys()):
                trainer_kwargs["gradient_clip_val"]=overwrite_config["max_grad_norm"]


            if(valid_checkpoint_dir is not None):
                # we initialized from checkpoint
                assert( ( (change_optimizer==1) and (change_scheduler==1) ) or ( (change_optimizer==0) and (change_scheduler==0) ) ), ("Optiḿizer and schedulre either have both to be loaded or both be reset at the same time for compatibility reasons! Optimizer overwrite: %d - Scheduler overwrite: %d " % (change_optimizer, change_scheduler))
                #trainer=pytorch_lightning.Trainer(resume_from_checkpoint=valid_checkpoint_dir, resume_skip_opti=change_optimizer, resume_skip_scheduler=change_scheduler, **trainer_kwargs)
                trainer=pytorch_lightning.Trainer(**trainer_kwargs)

                ## differentiate between old and new ray version in terms of logging nomenclature
                
                #trainer.current_epoch = ckpt["epoch"]
                if("=" in valid_checkpoint_dir):
                    batch_id=int(valid_checkpoint_dir.split("/")[-2].split("=")[-1])
                else:
                    ## old version
                    batch_id=int(valid_checkpoint_dir.split("/")[-2].split("_")[-1])
                trainer.total_batch_idx=batch_id+1
                print("BATCH ID START ", trainer.total_batch_idx)
                print("RESUMING FROM CHECKPOINT .... ", valid_checkpoint_dir)
            else:

                ## initialize model with datamodule if necessary
                ### TODO: make datamodule initialization dependent on previous settings
                """
                if(make_new_model):
                    if( (not "cp_file" in config.keys()) and (not "blueprint_file" in config.keys())):

                        self._model_datamodule_inits(model, data_module)
                """     

                trainer=pytorch_lightning.Trainer(**trainer_kwargs)
                print("NEW TRAINER OBJ!!!")
            

            ################################################

            ## set the option to "uncollate" the data, to backtransform it into a format suitable for the dataloader to do plotting
            setattr(trainer, "plot_uncollate_fn", plot_uncollate_fn)
            setattr(trainer, "data_module", data_module)
                
            ## check if uncommenting helps pickling process of "model"
            #setattr(model ,"trainer", trainer)
            #setattr(model, "data_module", data_module)


            ##setattr(model, "plot_uncollate_fn", plot_uncollate_fn)
            #setattr(model, "collate_fn", collate_fn)

            ## before start to fit, check if any plots need to be renewed .. get one single time another val batch

            val_batches=[]
            validation_list=trainer.data_module.val_dataloader()

            print("VALIDATION LIST!")

            if(type(validation_list)==list):
                print("VALIDATION NAMES ", model.validation_names)
                for ds_index in range(len(model.validation_names)):
                    print("DS INDEX ", ds_index)
                    val_batch=next(iter(validation_list[ds_index]))  
                    val_batches.append(val_batch)
            else:
                val_batch=next(iter(validation_list))
                val_batches=[val_batch]

            GPUStatsCallback().print_gpu_stats(name="AFTER TRAINER CREATION")
            
            print("before plotting cb loop")
            for plot_callback in plotting_callbacks:
                plot_callback.check_plots(trainer, self.load_model, val_batches, config=config)
            
            ## reset seeds again to be sure (after plotting)
            if(deterministic):
                seed_everything(0)

            print("collecting GPU cache once before training....")
            gc.collect()
            torch.cuda.empty_cache()




            GPUStatsCallback().print_gpu_stats(name="BEFORE MODEL FIT")
            trainer.fit(model, data_module,ckpt_path=valid_checkpoint_dir)

        return trainable

    ### loss and collate functions must match!!! here we define some standard ones for orientation or direct usage
    def _make_genetic_model(self, **kwargs):
        """
        Subclasses must create the specific system.
        """
        raise NotImplementedError()
