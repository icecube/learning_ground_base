from .. import config_parser
from jammy_flows.amortizable_mlp import AmortizableMLP

import numpy
from torch import nn
import torch


from kymatio.torch import Scattering1D

from . import scattering_transform_encoder, mh_attention_encoder, skip_mlp
def list_from_str(dims):

    return [int(i) for i in dims.split("-")]

class two_step_encoder(nn.Module):
    def __init__(self, **kwargs):
        super(two_step_encoder, self).__init__()
        

        if("init_dict" in kwargs.keys()):
            raise NotImplementedError()

        else:
            cfg_parser=config_parser.config_parser()

            cfg_parser.add_default_kwarg("settings", "input_dim", 2048, int)
            cfg_parser.add_default_kwarg("settings", "output_dim", 10, int)
            cfg_parser.add_default_kwarg("settings", "intermediate_dim", 128, int)
            cfg_parser.add_default_kwarg("settings", "first_step", "st", str, choices=["st"])
            cfg_parser.add_default_kwarg("settings", "second_step", "mh_attention", str, choices=["mh_attention", "mlp"])
            cfg_parser.add_default_kwarg("settings", "dtype", "float32", str)
            

            global_settings_args, global_settings_kwargs=cfg_parser.parse_cfg(kwargs, "settings")

            self.input_dim = global_settings_kwargs["input_dim"]
            self.output_dim = global_settings_kwargs["output_dim"]
            self.intermediate_dim = global_settings_kwargs["intermediate_dim"]

            self.first_encoder_str=global_settings_kwargs["first_step"]
            self.second_encoder_str=global_settings_kwargs["second_step"]

            #######################

            if(self.first_encoder_str=="st"):
                ## st options

                cfg_parser=config_parser.config_parser()

                cfg_parser.add_default_kwarg("st", "st.J", 7, int)
                cfg_parser.add_default_kwarg("st", "st.Q", 6, int)

                cfg_parser.add_default_kwarg("st", "st.hidden_dims", "128", str)
                cfg_parser.add_default_kwarg("st", "st.use_custom_mlp", 0, int)
                cfg_parser.add_default_kwarg("st", "st.mlp_highway_mode", 0, int)
                cfg_parser.add_default_kwarg("st", "st.mlp_lowrank", 0, int)
                cfg_parser.add_default_kwarg("st", "st.mlp_add_skip_connection", 0, int)

                settings_args, settings_kwargs=cfg_parser.parse_cfg(kwargs, "st", drop_name_piece=True)

                settings_kwargs["input_dim"]=self.input_dim
                settings_kwargs["output_dim"]=self.intermediate_dim
                settings_kwargs["dtype"]=global_settings_kwargs["dtype"]

                print("-----> ST Settings: ")
                for k in sorted(settings_kwargs):
                    print(k, " : ", settings_kwargs[k])
                print("------------------------")

                self.first_encoder=scattering_transform_encoder.STEncoder(*settings_args, **settings_kwargs)

            else:
                raise NotImplementedError()

            if(self.second_encoder_str=="mh_attention"):

                cfg_parser=config_parser.config_parser()

                cfg_parser.add_default_kwarg("mhattn", "mhattn.io_mlp_hidden_dims", "128", str)
                cfg_parser.add_default_kwarg("mhattn", "mhattn.io_add_skip_connection", 0, int)

                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_num_layers", 3, int)
                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_num_heads_per_layer", 3, int)
                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_use_layer_norm_1", 1, int)
                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_use_layer_norm_2", 1, int)
                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_use_extra_layer_norm", 0, int)
                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_layer_norm_first", 0, int)
                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_use_residual_addition", 1, int)

                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_computational_dim", -1, int)
                
                
                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_projection_type", "joint_qkv", str, choices=["single_self", "single_qkv", "joint_qkv"])
                cfg_parser.add_default_kwarg("mhattn", "mhattn.io_attn_projection_type", "single_self", str, choices=["single_self", "single_qkv", "joint_qkv"])


                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_dropout", 0.0, float)
                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_internal_mlp_dim", 512, int)
                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_use_weighted_mean", 0, int)

                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_package", "pytorch", str, choices=["pytorch", "xformer", "flash"])
                cfg_parser.add_default_kwarg("mhattn", "mhattn.attn_operator", "none", str, choices=["none", "flash", "small_k", "triton"])


                settings_args, settings_kwargs=cfg_parser.parse_cfg(kwargs, "mhattn", drop_name_piece=True)

                settings_kwargs["input_dim"]=self.intermediate_dim
                settings_kwargs["output_dim"]=self.output_dim
                settings_kwargs["dtype"]=global_settings_kwargs["dtype"]


                print("-----> MH Attention Settings: ")
                for k in sorted(settings_kwargs):
                    print(k, " : ", settings_kwargs[k])
                print("------------------------")

                self.second_encoder=mh_attention_encoder.almagate_multihead_attention(*settings_args, **settings_kwargs)

            elif(self.second_encoder_str=="mlp"):

                cfg_parser=config_parser.config_parser()

                cfg_parser.add_default_kwarg("mlp", "mlp.hidden_dims", "128", str)
                cfg_parser.add_default_kwarg("mlp", "mlp.add_skip_connection", 0, int)
                cfg_parser.add_default_kwarg("mlp", "mlp.num_items", 10, int)

                settings_args, settings_kwargs=cfg_parser.parse_cfg(kwargs, "mlp", drop_name_piece=True)

                settings_kwargs["input_dim"]=self.intermediate_dim*settings_kwargs["num_items"]
                settings_kwargs["output_dim"]=self.output_dim

                print("-----> SECOND MLP Settings: ")
                for k in sorted(settings_kwargs):
                    print(k, " : ", settings_kwargs[k])
                print("------------------------")
                
                modules=[]
                modules.append(torch.nn.Flatten(1,-1))
                modules.append(skip_mlp.SkipMLP(self.intermediate_dim*settings_kwargs["num_items"], settings_kwargs["hidden_dims"], settings_kwargs["output_dim"], add_skip_connection=settings_kwargs["add_skip_connection"]))

            
                self.second_encoder=torch.nn.Sequential(*modules)

            else:
                raise NotImplementedError()

    def forward(self, x, first_kwargs=dict(), second_kwargs=dict()):
        #### input must be of shape B X N X input_dim
        #### reformed to B*N X input_dim
        
        assert(len(x.shape)==3)
        batch_size=x.shape[0]
        items_per_batch=x.shape[1]
        input_dim=x.shape[2]
        assert(input_dim==self.input_dim)

        reshaped_x=x.reshape(batch_size*items_per_batch, -1)

        intermediate=self.first_encoder(reshaped_x, **first_kwargs)
       
        assert(intermediate.shape[1]==self.intermediate_dim)

        ## now it is B*N X intermediate_dim
        ## reshape to B X N X intermediate_dim

        reshaped_intermediate=intermediate.reshape(batch_size, items_per_batch, -1)
        
        assert(reshaped_intermediate.shape[2]==self.intermediate_dim)
        
        ## second encoder must aggregate information
        return self.second_encoder(reshaped_intermediate, **second_kwargs)
        
        
        