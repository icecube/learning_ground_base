import pytorch_lightning

import jammy_flows

import torch
#from ray.tune.integration.pytorch_lightning import TuneReportCallback

from .. import learning_env_base
from ... import config_parser
from ...encoders import variable_length_encoder

from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from lightning_fabric.utilities.cloud_io import _load as pl_load
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector as CheckpointConnector
import ray.tune as tune

from pytorch_lightning import seed_everything
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pylab

import numpy

import os
import glob

from .supervised_base_plotting import supervised_base_plotting, vis_compressed_correlations, vis_labels_umap, vis_coverage, vis_per_item_loss_and_entropy, vis_per_item_loss_and_entropy_fixed_yaxis
from .supervised_base_system import supervised_base_system

from ... import helper_fns


class supervised_base_env(learning_env_base.learning_env):

    def _get_plot_callback(self, name):

        if(name=="basic"):
            return supervised_base_plotting
        elif(name=="vis_compressed_correlations"):
            return vis_compressed_correlations
        elif(name=="vis_labels_umap"):
            return vis_labels_umap
        elif(name=="coverage"):
            return vis_coverage
        elif(name=="vis_per_item_loss_and_entropy"):
            return vis_per_item_loss_and_entropy
        elif(name=="vis_per_item_loss_and_entropy_fixed_yaxis"):
            return vis_per_item_loss_and_entropy_fixed_yaxis
        else:
            raise Exception("Undefined plotting name ", name)
    

    ## every subclass has to define some crosschecks for model and data_module

    """
    def model_datamodule_checks(self, model, data_module):

        #if(data_module.get_data_dim()!=model.encoder.input_dim):
        #    raise Exception("Error! Data from Dataloader has dimension %d, but encoder expects dimension %d" % (data_module.get_data_dim(), model.encoder.input_dim))

        
        next_iterator=data_module.val_dataloader()
        if(type(next_iterator)==list):
            next_iterator=next_iterator[0]
        elif(type(next_iterator)==dict):
            for k in next_iterator.keys():
                next_iterator=next_iterator[k]
                break

        val_batch=next(iter(next_iterator)) 
      
        model.init_pdfs(val_batch["labels"])

        return True
    """

    def update_model(self, model, checkpoint, config):

        model.load_state_dict( checkpoint["state_dict"])
        model.double()
        
        if(config["encoder.dtype"]=="float32"):
            model.encoder.float()
            #self.attn_dtype=torch.float32
        elif(config["encoder.dtype"]=="float16"):
            model.encoder.half()
            #self.attn_dtype=torch.float16
        elif(config["encoder.dtype"]=="bfloat16"):
            model.encoder.bfloat16()
            #self.attn_dtype=torch.bfloat16

    
    def _model_datamodule_inits(self, model, data_module):

        #if(data_module.get_data_dim()!=model.encoder.input_dim):
        #    raise Exception("Error! Data from Dataloader has dimension %d, but encoder expects dimension %d" % (data_module.get_data_dim(), model.encoder.input_dim))
        
        next_iterator=data_module.val_dataloader()
        if(type(next_iterator)==list):
            next_iterator=next_iterator[0]
        elif(type(next_iterator)==dict):
            for k in next_iterator.keys():
                next_iterator=next_iterator[k]
                break

        val_batch=next(iter(next_iterator)) 
      
        model.init_pdfs(val_batch["labels"])

        return True


    def get_collate_n_loss_fn(self, config):

        raise NotImplementedError("Collate loss option ", collate_loss_type, " is not implemented ...!")





       
        





