from .. import config_parser
from jammy_flows.amortizable_mlp import AmortizableMLP

import numpy
from torch import nn
import torch


class VariableLengthEncoder(nn.Module):
    def __init__(self, **kwargs):#encoder_layers=1, dropout=0.0, rnn_type="lstm", encoder_hidden_dim=10, num_encoder_mlp_layers=0, nonlinearity="tanh"):
        super(VariableLengthEncoder, self).__init__()
        
        if("init_dict" in kwargs.keys()):
            assert("rnn_type" in kwargs.keys())

            self.rnn_type=kwargs["rnn_type"]

            self.rnn=kwargs["init_dict"]["rnn"]
            self.mlp=kwargs["init_dict"]["mlp"]

        else:
            cfg_parser=config_parser.config_parser()

            cfg_parser.add_default_kwarg("settings", "input_dim", 3, int)
            cfg_parser.add_default_kwarg("settings", "output_dim", 20, int)

            cfg_parser.add_default_kwarg("settings", "rnn_hidden_dim", 50, int)
            cfg_parser.add_default_kwarg("settings", "rnn_num_layers", 1, int)
            cfg_parser.add_default_kwarg("settings", "rnn_type", "gru", str)
            cfg_parser.add_default_kwarg("settings", "rnn_bidirectional", 0, int, choices=[0,1])

            cfg_parser.add_default_kwarg("settings", "rnn_aggregate_mlp_hidden_structure", "20-20", str)
            cfg_parser.add_default_kwarg("settings", "rnn_mlp_highway_mode", 0, int)
            cfg_parser.add_default_kwarg("settings", "rnn_use_aggregate_mlp", 0, int)

            cfg_parser.add_default_kwarg("settings", "dtype", "float64", str)

            settings_args, settings_kwargs=cfg_parser.parse_cfg(kwargs, "settings")

            self.input_dim = settings_kwargs["input_dim"]
            self.output_dim = settings_kwargs["output_dim"]
            self.rnn_hidden_dim = settings_kwargs["rnn_hidden_dim"]
            self.rnn_num_layers = settings_kwargs["rnn_num_layers"]
            self.rnn_type = settings_kwargs["rnn_type"]
            self.bidirectional= settings_kwargs["rnn_bidirectional"]

            self.rnn_aggregate_mlp_hidden_structure = settings_kwargs["rnn_aggregate_mlp_hidden_structure"]
            self.rnn_mlp_highway_mode = settings_kwargs["rnn_mlp_highway_mode"]
            self.use_aggregate_mlp = settings_kwargs["rnn_use_aggregate_mlp"]

            self.dtype = settings_kwargs["dtype"]

            if(self.dtype=="float64"):
                self.dtype=torch.float64
            elif(self.dtype=="float32"):
                self.dtype=torch.float32
            elif(self.dtype=="float16"):
                self.dtype=torch.float16
            elif(self.dtype=="bfloat16"):
                self.dtype=torch.bfloat16

            ### 0 -- encode hits as x,y,t and start with a h_0 that is fitted anda lso includes x_y, so h_0 = (x,y,h_02, h_03...)
            ### 1 -- encode hits as t1, t2 .. and also start with a h_0 + concatenate result with x,y and apply 1-layer MLP 
            #self.individual_dom_encoding_type=basic_arg_check(args, "individual_dom_encoding_type")

            ## first per dom encoding
            if(self.rnn_type=="lstm"):
                self.rnn = nn.LSTM(self.input_dim, self.rnn_hidden_dim, self.rnn_num_layers, bidirectional=bool(self.bidirectional), batch_first=True)
            elif(self.rnn_type=="gru"):
                self.rnn = nn.GRU(self.input_dim, self.rnn_hidden_dim, self.rnn_num_layers, bidirectional=bool(self.bidirectional), batch_first=True)
            #elif(self.rnn_type=="hopfield"):
            #    self.rnn = hp(self.input_dim, hidden_size=self.encoder_hidden_dim, output_size=self.output_dim, num_heads=self.encoder_layers, dropout = self.dropout, batch_first=True)

            ## total hidden dim output of LSTM/GRU

            total_rnn_output_dim=self.rnn_num_layers*self.rnn_hidden_dim*(self.bidirectional+1)
            
            mlp_hidden_dims=[int(i) for i in self.rnn_aggregate_mlp_hidden_structure.split("-")]

            if(self.use_aggregate_mlp):
                self.mlp=AmortizableMLP(total_rnn_output_dim, mlp_hidden_dims, self.output_dim, highway_mode=self.rnn_mlp_highway_mode)
            else:
                # we must assure that the overall output dim aggrees with aggregate hidden output
                assert( (self.bidirectional+1)*self.rnn_hidden_dim*self.rnn_num_layers==self.output_dim ), ("Using no aggregate MLP..... dimensions must match!",(self.bidirectional+1)*self.rnn_hidden_dim*self.rnn_num_layers, self.output_dim)
                self.mlp=lambda x: x

    def forward(self, x):
        
        
        hidden=None
        
        if(self.rnn_type=="lstm"):
            outputs, (hidden, cell) = self.rnn(x)
        elif(self.rnn_type=="gru"):
            outputs,hidden=self.rnn(x)
        
        ## aggregate all hidden rnn output into an MLP flat input
       
        hidden=hidden.permute(1, 0, 2)

        hidden_aggregate=hidden.contiguous().view(hidden.shape[0], -1)
        
        res=self.mlp(hidden_aggregate)
        
        return res