from .. import config_parser
from jammy_flows.amortizable_mlp import AmortizableMLP

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class rnn_plus_joint_transformer(nn.Module):
    def __init__(self, **kwargs):#encoder_layers=1, dropout=0.0, rnn_type="lstm", encoder_hidden_dim=10, num_encoder_mlp_layers=0, nonlinearity="tanh"):
        super(rnn_plus_joint_transformer, self).__init__()
        
        print(kwargs)
        cfg_parser=config_parser.config_parser()

        cfg_parser.add_default_kwarg("settings", "input_dim", 1, int)
        cfg_parser.add_default_kwarg("settings", "output_dim", 20, int)

        cfg_parser.add_default_kwarg("settings", "rnn_hidden_dim", 20, int)
        cfg_parser.add_default_kwarg("settings", "rnn_num_layers", 1, int)
        cfg_parser.add_default_kwarg("settings", "rnn_type", "gru", str)

        cfg_parser.add_default_kwarg("settings", "rnn_aggregate_mlp_hidden_structure", "20-20", str)

        ### transformer settings
        cfg_parser.add_default_kwarg("settings", "position_dimensionality", 2, int)

        cfg_parser.add_default_kwarg("settings", "attention_input_dim", 50, int)
        cfg_parser.add_default_kwarg("settings", "attention_num_layers", 3, int)
        cfg_parser.add_default_kwarg("settings", "attention_num_heads_per_layer", 3, int)
        cfg_parser.add_default_kwarg("settings", "attention_use_layer_norm", 1, int)
        cfg_parser.add_default_kwarg("settings", "attention_dropout", 0.1, float)
        cfg_parser.add_default_kwarg("settings", "attention_internal_mlp_dim", 512, int)

        settings_args, settings_kwargs=cfg_parser.parse_cfg(kwargs, "settings")

        self.input_dim = settings_kwargs["input_dim"]

        assert(self.input_dim==1)
        
        self.output_dim = settings_kwargs["output_dim"]
        self.rnn_hidden_dim = settings_kwargs["rnn_hidden_dim"]
        self.rnn_num_layers = settings_kwargs["rnn_num_layers"]
        self.rnn_type = settings_kwargs["rnn_type"]

        self.rnn_aggregate_mlp_hidden_structure = settings_kwargs["rnn_aggregate_mlp_hidden_structure"]

        ### 0 -- encode hits as x,y,t and start with a h_0 that is fitted anda lso includes x_y, so h_0 = (x,y,h_02, h_03...)
        ### 1 -- encode hits as t1, t2 .. and also start with a h_0 + concatenate result with x,y and apply 1-layer MLP 
        #self.individual_dom_encoding_type=basic_arg_check(args, "individual_dom_encoding_type")

        ## first per dom encoding
        if(self.rnn_type=="lstm"):
            self.rnn = nn.LSTM(self.input_dim, self.rnn_hidden_dim, self.rnn_num_layers, batch_first=True)
        elif(self.rnn_type=="gru"):
            self.rnn = nn.GRU(self.input_dim, self.rnn_hidden_dim, self.rnn_num_layers, batch_first=True)
        #elif(self.rnn_type=="hopfield"):
        #    self.rnn = hp(self.input_dim, hidden_size=self.encoder_hidden_dim, output_size=self.output_dim, num_heads=self.encoder_layers, dropout = self.dropout, batch_first=True)

        ## size num_layers X batch_dim X hidden_dim
        self.h0=nn.Parameter(torch.randn((self.rnn_num_layers,1, self.rnn_hidden_dim)))

        ## total hidden dim output of LSTM/GRU


        ### num_hidden*num_layers+2 (2 for 2-dim dom positions)
        total_rnn_output_dim=self.rnn_num_layers*self.rnn_hidden_dim+settings_kwargs["position_dimensionality"]
        
        mlp_hidden_dims=[int(i) for i in self.rnn_aggregate_mlp_hidden_structure.split("-")]

        
        ## one MLP for query / key / value input ?
        

        self.attention_input_dim=settings_kwargs["attention_input_dim"]
        self.attention_num_layers=settings_kwargs["attention_num_layers"]
        self.attention_num_heads_per_layer=settings_kwargs["attention_num_heads_per_layer"]
        self.attention_use_layer_norm=settings_kwargs["attention_use_layer_norm"]
        self.attention_dropout=settings_kwargs["attention_dropout"]
        self.attention_internal_mlp_dim=settings_kwargs["attention_internal_mlp_dim"]

        self.rnn_to_attention_mlp1=AmortizableMLP(total_rnn_output_dim, mlp_hidden_dims, self.attention_input_dim)
        #self.rnn_to_attention_mlp2=AmortizableMLP(total_rnn_output_dim, mlp_hidden_dims, self.attention_input_dim)
        #self.rnn_to_attention_mlp3=AmortizableMLP(total_rnn_output_dim, mlp_hidden_dims, self.attention_input_dim)

        print("dropout ", self.attention_dropout)
        encoder_layer = nn.TransformerEncoderLayer(self.attention_input_dim, self.attention_num_heads_per_layer, self.attention_internal_mlp_dim, self.attention_dropout, "relu")

        #layer_norm = LayerNorm(d_model)
        self.layer_norm=None
        if(self.attention_use_layer_norm):
            self.layer_norm=nn.LayerNorm(self.attention_input_dim)
        
        print("LAYER NORM", self.layer_norm)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.attention_num_layers, norm=self.layer_norm)
        #self.transformer_encoder=nn.MultiheadAttention(self.attention_input_dim, self.attention_num_heads_per_layer, dropout=self.attention_dropout)


        self.attention_to_output_mlp=AmortizableMLP(self.attention_input_dim, mlp_hidden_dims, self.output_dim)


    def forward(self, data, geo):
        
        hidden=self.h0

        joint_aggregate_encoding=True

        batch_num=len(data)
        num_doms=len(geo.keys())

        total_hit_list=[]
        total_hit_list_lengths=[]

        total_dom_positions_hit=[]
        total_dom_positions_nohit=[]

        hit_numdoms_list=[]
        nohit_numdoms_list=[]

        if(joint_aggregate_encoding):

            ## find maxlen
            maxlen=0

            for ev_data in data:
                
                for dom_data in ev_data.values():
                
                    if(len(dom_data)>maxlen):
                        maxlen=len(dom_data)

            print("-------------> maxlen is ", maxlen)
            ## jointly encode all with maxlen

            

            

            for ev_data in data:

                #dom_positions=[]
                #dom_hit_mask=[]
                #dom_no_hit_mask=[]
                num_hit=0
                num_nohit=0

                for ind, geokey in enumerate(geo.keys()):
                
                    num_hits=0
                    if(geokey in ev_data.keys()):

                        this_dom_numhits=len(ev_data[geokey])
                        if(this_dom_numhits>0):
                          
                            #dom_hit_mask.append(ind)

                            diff=maxlen-this_dom_numhits
                            total_hit_list.append(torch.cat([torch.from_numpy(ev_data[geokey]), torch.zeros((diff))]).unsqueeze(0))
                            total_hit_list_lengths.append(this_dom_numhits)
                            total_dom_positions_hit.append(torch.Tensor([[geo[geokey][0], geo[geokey][1]]]))

                            num_hit+=1
                        else:
                            #dom_no_hit_mask.append(ind)
                            num_nohit+=1
                            total_dom_positions_nohit.append(torch.Tensor([[geo[geokey][0], geo[geokey][1]]]))

                    else:
                        num_nohit+=1
                        #dom_no_hit_mask.append(ind)
                        total_dom_positions_nohit.append(torch.Tensor([[geo[geokey][0], geo[geokey][1]]]))


            
                hit_numdoms_list.append(num_hit)
                nohit_numdoms_list.append(num_nohit)

                ####

                #position_list.append(dom_positions)

                #print(dom_positions)
                #print(dom_hit_mask)

                #sys.exit(-1)

        total_hit_list=torch.cat(total_hit_list, dim=0).unsqueeze(-1)

        packed_data=pack_padded_sequence(total_hit_list, total_hit_list_lengths, batch_first=True, enforce_sorted=False)

        total_num_hit_doms=len(total_dom_positions_hit)
        total_num_nohit_doms=len(total_dom_positions_nohit)
  
        ## get dom positions

        total_dom_positions_hit=torch.cat(total_dom_positions_hit)
        total_dom_positions_nohit=torch.cat(total_dom_positions_nohit)

        hh=hidden.repeat(1, total_num_hit_doms,1)
        nh=hidden.repeat(1, total_num_nohit_doms,1)

        #def testfn(hit_hid,nohit_hid):

        if(self.rnn_type=="lstm"):
            outputs, (hit_hidden, cell) = self.rnn(packed_data, hh)
        elif(self.rnn_type=="gru"):
            # repeat along batch-dim which is middle dimension
            outputs, hit_hidden=self.rnn(packed_data,hh)

        ## aggregate all hidden rnn output into an MLP flat input
        hit_hidden=hit_hidden.permute(1, 0, 2)
        nohit_hidden=nh.permute(1, 0, 2)
   
        hit_hidden_aggregate=hit_hidden.contiguous().view(hit_hidden.shape[0], -1)
        nohit_hidden_aggregate=nohit_hidden.contiguous().view(nohit_hidden.shape[0], -1)
       
        hit_hidden_aggregate=torch.cat([hit_hidden_aggregate, total_dom_positions_hit],dim=1)
        nohit_hidden_aggregate=torch.cat([nohit_hidden_aggregate, total_dom_positions_nohit],dim=1)

        prev_nohit=nohit_hidden_aggregate

        ##unify nohit and hit
        ####
       
        batch_aggregate=[]
        tot_index_hit=0
        tot_index_nohit=0

        for ind in range(len(hit_numdoms_list)):
            batch_aggregate.append(torch.cat([hit_hidden_aggregate[tot_index_hit:tot_index_hit+hit_numdoms_list[ind]], nohit_hidden_aggregate[tot_index_nohit:tot_index_nohit+nohit_numdoms_list[ind]]]))

            tot_index_hit+=hit_numdoms_list[ind]
            tot_index_nohit+=nohit_numdoms_list[ind]

        batch_aggregate=torch.cat(batch_aggregate)

        transformer_input1=self.rnn_to_attention_mlp1(batch_aggregate)
        #transformer_input2=self.rnn_to_attention_mlp2(batch_aggregate)
        #transformer_input3=self.rnn_to_attention_mlp3(batch_aggregate)
        #transformer_input=transformer_input.contiguous().view(num_doms, batch_num, transformer_input.shape[-1])


        transformer_input1=torch.reshape(transformer_input1, (batch_num, num_doms, self.attention_input_dim)).permute(1,0,2)
        #transformer_input2=torch.reshape(transformer_input2, (batch_num, num_doms, self.attention_input_dim)).permute(1,0,2)
        #transformer_input3=torch.reshape(transformer_input3, (batch_num, num_doms, self.attention_input_dim)).permute(1,0,2)

        transformer_result=self.transformer_encoder(transformer_input1)#.mean(axis=0)
        
      
        transformer_result=transformer_result.mean(axis=0)
        ret_val=self.attention_to_output_mlp(transformer_result)

        #return ret_val


        """
        ret_val=testfn(hh,nh)

        if(ret_val.requires_grad):

            test2=torch.autograd.functional.jacobian(testfn,(hh,nh))
            
            for ev_index in range(len(hit_numdoms_list)):
                print(ev_index)
                print("NUM HITS", hit_numdoms_list[ev_index])
                print("NO HITS", nohit_numdoms_list[ev_index])
                print(test2[0].shape)
                print(test2[0][0][0][0].shape)

                print("22 HITS")
                print(test2[0][ev_index][0][0][:22])
                
                print("22 NO-HITS")
                print(test2[1][ev_index][0][0][:22])
            sys.exit(-1)
        """
        return ret_val