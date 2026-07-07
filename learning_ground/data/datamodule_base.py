from .. import config_parser

from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler

import pytorch_lightning
import fnmatch

from memory_profiler import profile

import torch
import numpy

def collate_training_wrapper(orig_collate_fn, training=False):

    def new_collate_fn(batch):
        return orig_collate_fn(batch, training=training)

    return new_collate_fn

def obtain_further_configs(kwargs, config_descriptor="val"):

    ## first check basic train keys
    basic_train_keys=fnmatch.filter([k for k in kwargs.keys()], "data.train.*")
    
    ## first check basic val keys
    basic_validation_keys=fnmatch.filter([k for k in kwargs.keys()], "data.%s.*" % config_descriptor)

    num_extras=0

    val_dicts=[]
    for val_ind in range(10):
        further_keys=fnmatch.filter([k for k in kwargs.keys()], "data.%s%d.*" % (config_descriptor, val_ind))

        if(len(further_keys)>0):

            ## create new config
            new_config=dict()
            for k in basic_train_keys:
                new_config[k.replace("data.train", "data.%s" % config_descriptor)]=kwargs[k]

            for k in basic_validation_keys:
                new_config[k]=kwargs[k]

            for k in further_keys:
                new_config[k.replace("%s%d." % (config_descriptor, val_ind), "%s."%config_descriptor)]=kwargs[k]

            ## scan validation entries

            val_dicts.append(new_config)
            assert(num_extras==val_ind), "Require extra validation keys in ascending order of index..... no breaks allowed, and start at 0!"
            
            num_extras+=1


    ## if no extra validation args are defined ... define the basic validation args as the first index
    if(num_extras==0):
        new_config=dict()
        for k in basic_train_keys:
            new_config[k.replace("data.train", "data.%s" % config_descriptor)]=kwargs[k]

        for k in basic_validation_keys:
            new_config[k]=kwargs[k]

        val_dicts.append(new_config)
    
    return val_dicts


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled
    https://github.com/pytorch/pytorch/issues/2576
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.normed_np_weights=self.weights.numpy()/torch.sum(self.weights).numpy()

    def __iter__(self):
        rand_tensor = numpy.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.normed_np_weights,
                                       replace=self.replacement)

        rand_tensor = list(rand_tensor)

        return iter(rand_tensor)

class DataModuleBase(pytorch_lightning.LightningDataModule):

    def __init__(self, config, collate_fn=None, **kwargs):
        """
        Pytorch Lightning DataModule class which combines a dataset and dataloading.
        Can Unify streamed and file loading and also return data in desired format.

        Parameter:
        config (dict): Config for dataset loading, train, and validation set options. Accepted config parameters are defined in the code by setting them as default parameters.
        collate_fn (function or None) A function handed over to do the custom batch formation for variable length sequences.
        """
        super().__init__()
        
        self.dataset_kwargs=config
        self.collate_fn=collate_fn

    def parse_vital_pars(self, config, data_descriptor="data.train"):
        """
        All parameters that define the dataset in general. 
        Not included: dataset_size, batch_size, seed
        """
        cfg_parser=config_parser.config_parser()

        ## every dataset has a name which must be given
        cfg_parser.add_default_arg(data_descriptor, "%s.name" % data_descriptor, "dataset", str)
        cfg_parser.add_default_arg(data_descriptor, "%s.batch_size" % data_descriptor, 32, int)
        cfg_parser.add_default_arg(data_descriptor, "%s.num_workers" % data_descriptor, 0, int)
        #cfg_parser.add_default_arg(data_descriptor, "%s.sampling_strategy" % data_descriptor, "weighted_sampling", str, choices=["weighted_sampling", "sampling_with_shuffle", "sampling_no_shuffle"])

        ## parse custom parameters .. defined in subclass
        self._parse_vital_pars(cfg_parser, data_descriptor=data_descriptor)
        print("C>ONFIG", config)
        args, kwargs=cfg_parser.parse_cfg(config, "%s"%data_descriptor, drop_name_piece=True)
       
        return args, kwargs

    def _parse_vital_pars(self, parser, data_descriptor="data.train"):
        raise NotImplementedError()

    def get_dataset(self, config, specific_or_overwrite_kwargs, data_descriptor):
        """ 
        Returns name, dataset obj and keyword arguments. 

        config: The full config passed down
        specific_or_overwrite_kwargs: Extra config kwargs.
        data_descriptor (str): A string tag describing the dataset in the config. Typically either data.train or data.val, data.val0 etc.
        """ 
        print("data descriptor .. ", data_descriptor)
        print(config)
        args,kwargs=self.parse_vital_pars(config, data_descriptor=data_descriptor)
        print("####################||>>")
        print("... Dataset settings of ", args[0], " .. ", data_descriptor)
        print("args .. ")
        print(args[3:])
        
        name=args[0]
        batch_size=args[1]
        num_workers=args[2]

        for kw in specific_or_overwrite_kwargs.keys():
            kwargs[kw]=specific_or_overwrite_kwargs[kw]

        print("####")
        print("kwargs ..")
        config_parser.pretty_print(args[0],kwargs)
        
        ### print keys before creating the dataset
        #for key, item in sorted(kwargs.items()):
        #    print(key, " : ", item)
       
        # hand over *name* as mandatory arg and then all other args (after index 3)
        dataset_obj=self._obtain_dataset_obj(*(args[0:1]+args[3:]), **kwargs)
        print("<<||####################")
        return name, dataset_obj, batch_size, num_workers

    def _obtain_dataset_obj(self, *args, **kwargs):
        """
        Initialize the dataset class with *args, **kwargs here.
        """
        raise NotImplementedError()

    def _set_data_and_label_dim(self, kwargs):
        raise NotImplementedError()

    
    def setup(self, stage=None):
        print("begin setup ...")
        ########## set general settings.. all validation batch sizes must be similar in size

        cfg_parser=config_parser.config_parser()

        cfg_parser.add_default_kwarg("general", "data.val.batch_size", 500, int)
        cfg_parser.add_default_kwarg("general", "val_interval", 100, int)

        args, kwargs=cfg_parser.parse_cfg(self.dataset_kwargs, "general")

        #self.val_batch_size=kwargs["data.val.batch_size"]
        self.val_interval=kwargs["val_interval"]

        ########## setup training dataset

        self.train_dataset_name, self.train_dataset, self.train_batch_size, self.train_num_workers=self.get_dataset(self.dataset_kwargs, {}, "data.train")

        ########## find all validation tags... and obtain datasets and related configs for each of them

        val_configs=obtain_further_configs(self.dataset_kwargs, config_descriptor="val")

        self.validation_dict_list=[]

        for cfg in val_configs:
            
            val_name, val_dataset, val_batch_size, num_workers=self.get_dataset(cfg, {}, "data.val")

            ## TODO .. do we need these constraints? 
            #assert(val_batch_size==self.val_batch_size) # in principle we only need this for the first val dataset for consistent scheduling
            #assert(num_workers==0) # no we dont need this, but seems convenient

            this_dict=dict()
            this_dict["dataset"]=val_dataset
            this_dict["name"]=val_name
            this_dict["num_workers"]=num_workers ## no workers for validation currently
            this_dict["batch_size"]=val_batch_size ## all val datasets share batch size

            self.validation_dict_list.append(this_dict)

        ### plotting datasets

        ########## find all plotting tags... and obtain datasets and related configs for each of them

        plotting_configs=obtain_further_configs(self.dataset_kwargs, config_descriptor="plotting")

        self.plotting_dict_list=[]

        """
        for cfg in plotting_configs:
            
            plotting_name, plotting_dataset, plotting_batch_size, num_workers=self.get_dataset(cfg, {}, "data.plotting")

            ## TODO .. do we need these constraints? 
            #assert(val_batch_size==self.val_batch_size) # in principle we only need this for the first val dataset for consistent scheduling
            #assert(num_workers==0) # no we dont need this, but seems convenient

            this_dict=dict()
            this_dict["dataset"]=plotting_dataset
            this_dict["name"]=plotting_name
            this_dict["num_workers"]=0 ## no workers for validation currently
            this_dict["batch_size"]=plotting_batch_size 

            self.plotting_dict_list.append(this_dict)
        """
        
        ########## label and data information  

        self.data_dim, self.label_dim=self._set_data_and_label_dim(self.dataset_kwargs)

        ###

        ########## calculate the effective dataset size used .. must be based on val_interval etc for nice training vehi
        
        self.train_dataset_size=self.calculate_effective_training_size()

        print("------> configured datamodule ...")
        ###

    def calculate_effective_training_size(self, round_upwards=True):

        num_train_events=len(self.train_dataset)
        num_events_one_val_interval=self.val_interval*self.train_batch_size

        leftover=num_train_events%num_events_one_val_interval

        if(leftover==0):
            print("effective training size (for one epoch) is: ", num_train_events)
            return num_train_events
        else:
            if(round_upwards):
                ## roundup to add a few more events for one epoch to have a number divisible by val_interval*train_batch_size
                effective_training_size=num_train_events+(num_events_one_val_interval-leftover)

                print("effective training size (for one epoch) is (rounded upwards): ", effective_training_size)
                return effective_training_size
            else:
                effective_training_size=num_train_events-leftover
                print("effective training size (for one epoch) is (rounded downwards): ", effective_training_size)
                return effective_training_size

    def visualize_data(self, dataset, fig, gridspec, batch, batch_index, dataset_identifier="val_0"):

        return self._visualize_data(dataset, fig, gridspec, batch, batch_index, dataset_identifier=dataset_identifier)

    def _visualize_data(self, dataset, fig, gridspec, batch, batch_index, dataset_identifier="val_0"):
        """
        To subclass.
        """
        raise NotImplementedError()

    def get_data_dim(self):
        return self.data_dim

    def get_label_dim(self):
        return self.label_dim

    def _obtain_train_dataloader_obj(self, *args, **kwargs):
        return CustomWeightedRandomSampler(args[0].get_weights(), self.train_dataset_size)

    def _obtain_val_dataloader_obj(self, *args, **kwargs):
        return SequentialSampler(args[0]["dataset"])

    def train_dataloader(self):
        print("INITIALIZE TRAIN DATALOADER ... dataset size ... ", self.train_dataset_size)
        
        return DataLoader(self.train_dataset, 
                          sampler=self._obtain_train_dataloader_obj(self.train_dataset),
                          batch_size=self.train_batch_size,
                          num_workers=self.train_num_workers,
                          worker_init_fn=self.train_dataset.worker_init_fn,
                          collate_fn=collate_training_wrapper(self.collate_fn, training=True))

    def val_dataloader(self):
        """ 
        Loop through all validation datasets.
        """
        print("INITIALIZE VAL DATALOADER ... ")
        
        val_loaders=[]

        for ind in range(len(self.validation_dict_list)):

            cur_item=self.validation_dict_list[ind]

            val_loaders.append(DataLoader(cur_item["dataset"], 
                              sampler=self._obtain_val_dataloader_obj(cur_item),
                              batch_size=cur_item["batch_size"],
                              num_workers=cur_item["num_workers"], ## just have a single worker for val dataset?
                              worker_init_fn=cur_item["dataset"].worker_init_fn,
                              collate_fn=collate_training_wrapper(self.collate_fn, training=False)))
        return val_loaders
        
        
    def test_dataloader(self):
        return None