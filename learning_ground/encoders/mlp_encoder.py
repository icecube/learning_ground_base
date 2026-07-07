from .. import config_parser
from jammy_flows.amortizable_mlp import AmortizableMLP

import numpy
from torch import nn
import torch

def list_from_str(dims):

    return [int(i) for i in dims.split("-")]

class MLPEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(MLPEncoder, self).__init__()
        
        if("init_dict" in kwargs.keys()):
            raise NotImplementedError()

            assert("rnn_type" in kwargs.keys())

            self.mlp=kwargs["init_dict"]["mlp"]

        else:
            cfg_parser=config_parser.config_parser()

            cfg_parser.add_default_kwarg("settings", "input_dim", 3, int)
            cfg_parser.add_default_kwarg("settings", "output_dim", 20, int)

            cfg_parser.add_default_kwarg("settings", "hidden_dims", "128", str)
            cfg_parser.add_default_kwarg("settings", "use_custom_mlp", 0, int)
            cfg_parser.add_default_kwarg("settings", "mlp_highway_mode", 0, int)
            cfg_parser.add_default_kwarg("settings", "mlp_lowrank", 0, int)

            settings_args, settings_kwargs=cfg_parser.parse_cfg(kwargs, "settings")

            self.input_dim = settings_kwargs["input_dim"]
            self.output_dim = settings_kwargs["output_dim"]
           
            self.hidden_dims = settings_kwargs["hidden_dims"]
            self.use_custom_mlp = settings_kwargs["use_custom_mlp"]
            self.mlp_highway_mode = settings_kwargs["mlp_highway_mode"]
            self.mlp_lowrank = settings_kwargs["mlp_lowrank"]
            
            if(self.use_custom_mlp):
                self.mlp=amortizable_mlp.AmortizableMLP(self.input_dim, self.hidden_dims, self.output_dim, low_rank_approximations=self.mlp_lowrank, use_permanent_parameters=True, highway_mode=self.mlp_highway_mode, svd_mode="smart")
            else:
                mlp_in_dims = [self.input_dim] + list_from_str(self.hidden_dims)
                mlp_out_dims = list_from_str(self.hidden_dims) + [self.output_dim]
               
                nn_list = []
                for i in range(len(mlp_in_dims)):
                   
                    l = torch.nn.Linear(mlp_in_dims[i], mlp_out_dims[i])

                    nn_list.append(l)
                    
                    if i < (len(mlp_in_dims) - 1):
                        nn_list.append(nn.Tanh())
                
            self.mlp=torch.nn.Sequential(*nn_list)

    def forward(self, x):

        return self.mlp(x)
        