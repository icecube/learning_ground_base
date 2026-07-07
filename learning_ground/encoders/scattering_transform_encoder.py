from .. import config_parser
from jammy_flows.amortizable_mlp import AmortizableMLP

import numpy
from torch import nn
import torch


from kymatio.torch import Scattering1D

from .skip_mlp import SkipMLP

def list_from_str(dims):

    if(dims==""):
        return []
        
    return [int(i) for i in dims.split("-")]

class STEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(STEncoder, self).__init__()
        
        if("init_dict" in kwargs.keys()):
            raise NotImplementedError()

            assert("rnn_type" in kwargs.keys())

            self.mlp=kwargs["init_dict"]["mlp"]

        else:
            cfg_parser=config_parser.config_parser()

            cfg_parser.add_default_kwarg("settings", "input_dim", 2048, int)
            cfg_parser.add_default_kwarg("settings", "J", 7, int)
            cfg_parser.add_default_kwarg("settings", "Q", 6, int)

            cfg_parser.add_default_kwarg("settings", "output_dim", 10, int)

            cfg_parser.add_default_kwarg("settings", "hidden_dims", "128", str)
            cfg_parser.add_default_kwarg("settings", "use_custom_mlp", 0, int)
            cfg_parser.add_default_kwarg("settings", "mlp_highway_mode", 0, int)
            cfg_parser.add_default_kwarg("settings", "mlp_lowrank", 0, int)

            cfg_parser.add_default_kwarg("settings", "mlp_add_skip_connection", 0, int)

            settings_args, settings_kwargs=cfg_parser.parse_cfg(kwargs, "settings")

            self.input_dim = settings_kwargs["input_dim"]
            self.output_dim = settings_kwargs["output_dim"]
           
            self.hidden_dims = settings_kwargs["hidden_dims"]
            self.use_custom_mlp = settings_kwargs["use_custom_mlp"]
            self.mlp_highway_mode = settings_kwargs["mlp_highway_mode"]
            self.mlp_lowrank = settings_kwargs["mlp_lowrank"]

            self.J=settings_kwargs["J"]
            self.Q=settings_kwargs["Q"]


            self.st_object=Scattering1D(self.J,self.input_dim,self.Q)

            log2_base_total=int(numpy.log2(self.input_dim))

            ## number of *timesteps* after scattering transform
            self.num_independent_bases=2**(log2_base_total-self.J)

            ## information about order 0-2 coeffs
            meta = self.st_object.meta()
            self.order0 = numpy.where(meta['order'] == 0)
            self.order1 = numpy.where(meta['order'] == 1)
            self.order2 = numpy.where(meta['order'] == 2)

            num_coeffs_0=self.order0[0][-1]+1
            num_coeffs_1=self.order1[0][-1]+1-num_coeffs_0
            num_coeffs_2=self.order2[0][-1]+1-num_coeffs_0-num_coeffs_1

            ## save number of coeffs
            self.nc_0=num_coeffs_0
            self.nc_1=num_coeffs_1
            self.nc_2=num_coeffs_2
            self.nc_total=self.nc_0+self.nc_1+self.nc_2

            """
            ## no skip
            mlp_in_dims = [self.nc_total*self.num_independent_bases] + list_from_str(self.hidden_dims)
            mlp_out_dims = list_from_str(self.hidden_dims) + [self.output_dim]
           
            nn_list = []
            for i in range(len(mlp_in_dims)):
               
                l = torch.nn.Linear(mlp_in_dims[i], mlp_out_dims[i])

                nn_list.append(l)
                
                if i < (len(mlp_in_dims) - 1):
                    nn_list.append(nn.Tanh())
            
            self.mlp=torch.nn.Sequential(*nn_list)
            """
            
            if(self.use_custom_mlp):
                self.mlp=amortizable_mlp.AmortizableMLP(self.nc_total*self.num_independent_bases, self.hidden_dims, self.output_dim, low_rank_approximations=self.mlp_lowrank, use_permanent_parameters=True, highway_mode=self.mlp_highway_mode, svd_mode="smart")
            else:


                self.mlp=SkipMLP(self.nc_total*self.num_independent_bases, self.hidden_dims, self.output_dim, add_skip_connection=settings_kwargs["mlp_add_skip_connection"])
            

    def forward(self, x):

        intermediate=self.st_object(x)

        intermediate=intermediate.reshape(intermediate.shape[0], -1)

        return self.mlp(intermediate)
        