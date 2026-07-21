from .. import config_parser

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy
import math

from argparse import Namespace


import copy
from typing import Optional, Any, Union, Callable

import gc
import time 

from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from torch_scatter import scatter_softmax, scatter_add, scatter_mean, scatter_sum, scatter_max, scatter_min

from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from torch.nested._internal.nested_tensor import jagged_from_list, buffer_from_jagged, ViewNestedFromBuffer
import subprocess

try:
    import xformers
except:
    print("cannot import xformers.. install package via pip!")
else:
    globals()["xformers"] = xformers
    from xformers.ops import fmha#, MemoryEfficientAttentionFlashAttentionOp,MemoryEfficientAttentionOp,TritonFlashAttentionOp

try:
    import flash_attn
except:
    print("cannot import flash_attn.. install package via pip!")
else:
    globals()["flash_attn"] = flash_attn
  

from .skip_mlp import SkipMLP

def jagged_from_buffer(buffer, offsets, maxval, minval):
    return ViewNestedFromBuffer.apply(buffer, offsets, maxval, minval)

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_tensor_of_diffs_flattened(list_of_tensors, element_list=None, reverse=False):
    ## B X L X DIM

    if(type(list_of_tensors)==torch.Tensor):
        # this is a padded tensor.. element list must be given
        assert(element_list is not None)
        input_dim=list_of_tensors.shape[-1]
      
        distances_flattened=torch.cat([ (t[:element_list[tind],None,:]-t[None,:element_list[tind],:]).reshape(-1,input_dim) for tind,t in enumerate(list_of_tensors)])

    else:
        assert(len(list_of_tensors[0].shape)==3)

        input_dim=list_of_tensors[0].shape[-1]
        element_list=[t.shape[1] for t in list_of_tensors]
        if(reverse):
            distances_flattened=torch.cat([ (t[:, None,:,:]-t[:, :,None,:]).reshape(-1,input_dim) for t in list_of_tensors])

        else:
            distances_flattened=torch.cat([ (t[:, :,None,:]-t[:, None,:,:]).reshape(-1,input_dim) for t in list_of_tensors])

    
    return distances_flattened, element_list

def _get_tensors_flattened(list_of_tensors, element_list=None):
    ## turn B X L(masked) X DIM -> B X L_individual*DIM

    if(type(list_of_tensors)==torch.Tensor):
        # this is a padded tensor.. element list must be given
        assert(element_list is not None)
        input_dim=list_of_tensors.shape[-1]
        vecs_flattened=torch.cat([ t[:element_list[tind],:].reshape(-1,input_dim) for tind,t in enumerate(list_of_tensors)])

    else:
        ## shape of each tensor is L X H X D ?
        assert(len(list_of_tensors[0].shape)==3)

        input_dim=list_of_tensors[0].shape[-1]
        element_list=[t.shape[1] for t in list_of_tensors]
        vecs_flattened=torch.cat([ t[:, :,:].reshape(-1,input_dim) for t in list_of_tensors])

    
    return vecs_flattened, element_list


def _recreate_list_of_matrices(flat_matrixarray, side_lengths):
    """
    Recreate matrices by putting them in a list, each having shape (1,M_b,M_b)
    """

    input_dim=flat_matrixarray.shape[-1]

    spl=torch.split(flat_matrixarray, split_size_or_sections=[s**2 for s in side_lengths])
    return [t.reshape(1,side_lengths[ind],side_lengths[ind],input_dim) for ind,t in enumerate(spl)]

def _recreate_list_of_matrices_padded(flat_matrixarray, side_lengths):
    """
    Recreating matrices by puttnig them in a padded tensor, where excess entries are padded to zero
    """
    input_dim=flat_matrixarray.shape[-1]
    base_tensor=torch.zeros(len(side_lengths),max(side_lengths), max(side_lengths), input_dim).to(flat_matrixarray)

    spl=torch.split(flat_matrixarray, split_size_or_sections=[s**2 for s in side_lengths])

    for ind, t in enumerate(spl):
        base_tensor[ind][:side_lengths[ind], :side_lengths[ind]]=t.reshape(side_lengths[ind], side_lengths[ind],input_dim)

    return base_tensor


## transform_input

def _transform_pytorch_to_xformer_rep(datavecs, mask):
    print("######## PYTORCH -> XFORMER TRANSFORM ##########")
    element_list=[datavecs.shape[1] for i in range(datavecs.shape[0])]
    if(mask is not None):
        element_list=[int(sum(mitem==0).cpu().detach()) for mitem in mask]

    data=[vec[:element_list[ind]].unsqueeze(0) for ind, vec in enumerate(datavecs)]
    attn_bias, x = fmha.BlockDiagonalMask.from_tensor_list(data)

    return x, attn_bias


def _transform_xformer_to_pytorch_rep(datavecs, mask):

    print("######## XFORMER -> PYTORCH TRANSFORM ##########")

    datalens=[i.shape[1] for i in range(src.shape[0])]

    maxlen=max(datalens)
    input_dim=datavecs[0].shape[-1]

    assert( (maxlen==datalens).sum()<len(datalens))

    for ind, dvec in enumerate(datavecs):

        assert(len(dvec.shape)==3)

        diff=maxlen-dvec.shape[1]

        if(diff>0):
            data.append(torch.cat([ dvec, torch.zeros((1,diff, input_dim)).to(dvec) ] , dim=1) )
        else:
            data.append(dvec)

    
    data=torch.cat(data)

    padding_mask=torch.zeros_like(data[:,:,0]).to(data[0])

    for b_ind, datalen in enumerate(datalens):
        padding_mask[b_ind,datalen:]=1

    padding_mask=padding_mask.type(torch.bool)

    print(data,padding_mask)
    sys.exit(-1)
    return data, padding_mask



class CustomTransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, 
                 encoder_layer_args, 
                 encoder_layer_kwargs,
                 num_layers, 
                 original_input_dim,
                 extra_layer_norm=None,
                 rel_position_layer_indices="0", # relative position encoding is expensive, and is done typically only at first layer
                 abs_position_layer_indices="0", # absolute positional encoding is also done only at first layer
                 abs_position_range="",
                 #abs_position_scale="logarithmic",
                 rel_position_as_parallel_to_normal_track=0,
                 rel_position_max_computational_dim=100,
                 use_global_query_input=0):
                 
                
        super().__init__()
    
        self.rel_position_feeding_overwrites=[]
        self.rel_indices_list=[]
        ## rel position config
        if(rel_position_layer_indices=="-1"):
            self.rel_indices_list=numpy.arange(num_layers)
            self.rel_position_feeding_overwrites=[None for i in range(num_layers)]
        else:
            self.rel_position_feeding_overwrites=[None for i in range(num_layers)]
            for split_rel_info in rel_position_layer_indices.split(","):
                ## using input
                if(split_rel_info[-1]=="i"):
                    self.rel_indices_list.append(int(split_rel_info[:-1]))
                    self.rel_position_feeding_overwrites[self.rel_indices_list[-1]]=2
                elif(split_rel_info[-1]=="v"):
                    self.rel_indices_list.append(int(split_rel_info[:-1]))
                    self.rel_position_feeding_overwrites[self.rel_indices_list[-1]]=0
                else:
                    self.rel_indices_list.append(int(split_rel_info))
                    

        print ("FEEDING OVERWRITES ... ", self.rel_position_feeding_overwrites)
        ## abs position config

        if(abs_position_layer_indices=="-1"):
            self.abs_indices_list=numpy.arange(num_layers)
        else:
            self.abs_indices_list=[int(i) for i in abs_position_layer_indices.split(",")]

        self.absolute_position_range=[] if abs_position_range=="" else [float(i) for i in abs_position_range.split("_")]
        

        ########################

        list_of_layers=[]
        #self.indices_for_original_input_feeding=[]

        self.input_formats=[]

        self.use_absolute_position_encoding=False
        self.absolute_position_type=""
        self.absolute_position_input_dim=None

        self.use_resi_dual=False

        prev_residual_setting=None

        for ind in range(num_layers):
            these_args=copy.deepcopy(encoder_layer_args)
            these_kwargs=copy.deepcopy(encoder_layer_kwargs)

            if(use_global_query_input):
                assert(these_kwargs["use_query_class_token"]==1)

                ## swtich off out projection in last layer if using query token
                if(ind==(num_layers-1)):
                    these_kwargs["do_perlayer_out_projection"]=0
                    these_kwargs["skip_ff_and_other_trafos"]=1

            if(these_kwargs["abs_position_mode"]=="sinusoidal"):
                if(ind not in self.abs_indices_list):
                    these_kwargs["abs_position_mode"]="none"
                else:
                    self.use_absolute_position_encoding=True
                    assert(self.absolute_position_type=="" or self.absolute_position_type=="sinusoidal"), "Mixed absolute position in different layers not allowed!"
                    self.absolute_position_type="sinusoidal"

                    assert(len(self.absolute_position_range)>0)
                    #self.absolute_position_range=[float(i) for i in abs_position_range.split("_")]

                    ## obtain absolute position dimension
                    assert(self.absolute_position_input_dim is None or self.absolute_position_input_dim==these_kwargs["original_input_position_feature_number"])
                    self.absolute_position_input_dim=these_kwargs["original_input_position_feature_number"]



            elif(these_kwargs["abs_position_mode"]=="roformer"):
                raise Exception("Roformer not implemented currently!")

                if(ind not in self.abs_indices_list):
                    these_kwargs["abs_position_mode"]="none"
                else:
                    self.use_absolute_position_encoding=True
                    assert(self.absolute_position_type=="" or self.absolute_position_type=="roformer"), "Mixed absolute position in different layers not allowed!"
                    self.absolute_position_type="roformer"

                    assert(len(self.absolute_position_range)>0)
                    
                    ## obtain absolute position dimension
                    assert(self.absolute_position_input_dim is None or self.absolute_position_input_dim==these_kwargs["original_input_position_feature_number"])
                    self.absolute_position_input_dim=these_kwargs["original_input_position_feature_number"]

            if("pytorch" in these_kwargs["attn_package"]):
                self.input_formats.append("p")
            else:
                self.input_formats.append("x")

            ## check residual setting for resi-dual setting
            if(these_kwargs["use_residual_addition"]==3):
                self.use_resi_dual=True
                if(prev_residual_setting is None):
                    prev_residual_setting==3
                else:
                    assert(prev_residual_setting==3)
            else:
                assert(self.use_resi_dual==False)
                prev_residual_setting=these_kwargs["use_residual_addition"]

            # get specific overwrite for relative position config
            if(these_kwargs["rel_position_mode_value"]!="none"):
              
                if(ind not in self.rel_indices_list):
                    # set to None if this layer index does not contain relative positional encoding
                    these_kwargs["rel_position_mode_value"]="none"
                    list_of_layers.append(CustomTransformerEncoderLayer(*these_args, **these_kwargs))

                else:

                    if(self.rel_position_feeding_overwrites[ind] is not None):
                        these_kwargs["rel_position_input_feeding_type"]=self.rel_position_feeding_overwrites[ind]
                    #print(these_kwargs)
                    if(rel_position_as_parallel_to_normal_track):
                        ## create a moduleList of two layers.. one normal and one with relative attnention
                        ## in the forward pass, the output of both wil lbe added
                        ## attn_rel_position_max_computational_dim must be smaller than normal computational_dim

                        used_input_dim_here=these_args[0]

                        new_args=copy.deepcopy(these_args)
                        new_kwargs=copy.deepcopy(these_kwargs)

                        new_kwargs["abs_position_mode"]="none"
                        new_kwargs["rel_position_max_computational_dim"]=-1
                        new_kwargs["rel_position_mode_value"]="only_rel"

                        assert(these_args[0]!=-1), "When using parallel relative positional track, must define computational dim"
                        new_args[0]=min(used_input_dim_here, rel_position_max_computational_dim) 
                        new_args[1]=1 # nhead = 1

                        ## relative layer requires flash attn of official_w_weights
                        if("pytorch" in new_kwargs["attn_package"]):
                            new_kwargs["attn_package"]="official_pytorch_w_weights"
                        else:
                            ## change relative attention package to geometric_scatter if "non-pytorch type" and also not flash_attn
                            #if(these_kwargs["attn_package"]!="flash_attn"):
                            new_kwargs["attn_package"]="flash_attn"

                        rel_projection_layer=None
                        rel_projection_layer2=None

                        if(used_input_dim_here!=new_args[0]):
                            rel_projection_layer=torch.nn.Linear(used_input_dim_here, new_args[0])

                            if(self.use_resi_dual):
                                rel_projection_layer2=torch.nn.Linear(used_input_dim_here, new_args[0])
                        
                        rel_layer=CustomTransformerEncoderLayer(*new_args, **new_kwargs)

                        ## absolute sub layer
                        new_args=copy.deepcopy(these_args)
                        new_kwargs=copy.deepcopy(these_kwargs)

                        new_kwargs["rel_position_mode_value"]="none"

                        abs_layer=CustomTransformerEncoderLayer(*new_args, **new_kwargs)

                        two_layers=[]
                        two_layers.append(abs_layer)
                        two_layers.append(rel_layer)

                        if(rel_projection_layer is not None):
                            two_layers.append(rel_projection_layer)
                        if(rel_projection_layer2 is not None):
                            two_layers.append(rel_projection_layer2)

                        list_of_layers.append(ModuleList(two_layers))

                        
                    else:

                        ## relative layer requires flash attn of official_w_weights
                        if("pytorch" in these_kwargs["attn_package"]):
                            these_kwargs["attn_package"]="official_pytorch_w_weights"
                        else:
                            ## change relative attention package to geometric_scatter if "non-pytorch type" and also not flash_attn
                            #if(these_kwargs["attn_package"]!="flash_attn"):
                            these_kwargs["attn_package"]="flash_attn"

                        ################################
                        these_args[1]=1 # nhead=1 for relative 
                        these_kwargs["rel_position_max_computational_dim"]=rel_position_max_computational_dim


                        list_of_layers.append(CustomTransformerEncoderLayer(*these_args, **these_kwargs))
            else:
                ## we dont have relative positional encoding at all.. just append layer
                list_of_layers.append(CustomTransformerEncoderLayer(*these_args, **these_kwargs))
            
            #print(ind, these_kwargs["attn_package"],these_kwargs["rel_position_mode_value"])
        
        self.layers=ModuleList(list_of_layers)
        #self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.final_layer_norm = extra_layer_norm
        # this attribute saves the value providedat object construction
        #self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        #self.use_nested_tensor = enable_nested_tensor
        #self.mask_check = mask_check
        #self.rel_position_mode_value=rel_position_mode_value
        #self.rel_position_input_feeding_type=rel_position_input_feeding_type
        self.original_input_dim=original_input_dim
        #self.absolute_position_scale=abs_position_scale

        self.embedding_dim=encoder_layer_args[0]
        self.num_heads=encoder_layer_args[1]
        
        ## define a global query input for class token if desired

        self.global_query_input=None
        if(use_global_query_input):
            self.global_query_input=torch.nn.Parameter(torch.randn(1,1,self.embedding_dim))
        

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None,
            original_input_feature_vecs=None,
            positional_features=None,
            datalens=None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """

        output = src
        used_mask=src_key_padding_mask

        if(self.use_absolute_position_encoding):
            ## do absolute position encoding here

            assert(positional_features.shape[-1]==self.absolute_position_input_dim), (positional_features.shape, self.absolute_position_input_dim)

            if(self.absolute_position_type=="sinusoidal"):

                ## 
                encoded_pos_features=torch.zeros_like(src)

                ## define everything in units of the expected range width of the position params
                ## default in word encodings: 10000
                range_width=self.absolute_position_range[1]-self.absolute_position_range[0]
                offset=self.absolute_position_range[0]

                # mult factor to be roughly equivalent to word embedding settings
                mult_factor=10000.0/range_width
             
                ## num

                total_dim_per_positional_dim_and_trigfn=self.embedding_dim // (self.absolute_position_input_dim*2)

                used_total_dim=total_dim_per_positional_dim_and_trigfn*(self.absolute_position_input_dim*2)
                
                assert(used_total_dim==self.embedding_dim), ("Please make sure embedding dim is divsible by ", self.absolute_position_input_dim*2, " (num_pos_dims*2) for posiitonal encoding... embedding dim is: ", self.embedding_dim)

                num_per_pos=self.embedding_dim//self.absolute_position_input_dim

                arange_vec=torch.arange(0, num_per_pos)

                div = torch.exp(-math.log(10000.0) * (2 * (arange_vec // 2) / num_per_pos))[None,None,:].to(device=src.device)
              
                # go through each positioinal feature and encoded it via sin/cos
                for pos_feature_ind in range(self.absolute_position_input_dim):
                    
                    scaled_features=(positional_features[:,:,pos_feature_ind:pos_feature_ind+1]-offset)*mult_factor
                 
                    encoded_pos_features[:, :, (pos_feature_ind*num_per_pos):(pos_feature_ind+1)*num_per_pos][:,:,0::2]=torch.sin(scaled_features*div[:,:,0::2])
                    encoded_pos_features[:, :, (pos_feature_ind*num_per_pos):(pos_feature_ind+1)*num_per_pos][:,:,1::2]=torch.cos(scaled_features*div[:,:,1::2])

                ## add position encodings to input vector
                
                output=output+encoded_pos_features

                
            else:
                print(self.absolute_position_type)
                raise NotImplementedError()

        res_connection=output

        last_query_output=None
        last_connection_query=None

        if(self.global_query_input is not None):
            if(self.input_formats[0]=="p"):
                last_query_output=self.global_query_input.repeat(len(datalens),1,1)
                last_connection_query=self.global_query_input.repeat(len(datalens),1,1)
            else:
                last_query_output=self.global_query_input.repeat(1,len(datalens),1)
                last_connection_query=self.global_query_input.repeat(1,len(datalens),1)

        for layer_ind, mod in enumerate(self.layers):
            

            ## check if we have absolute input as input for relative positional encoding
            """
            used_relative_input=None
            if(layer_ind in self.rel_indices_list):
                if(layer_ind in self.indices_for_original_input_feeding):
                    assert(positional_features is not None)
                    used_relative_input=positional_features
            """

            ## usually not used, as either pytorch or xformer (flash_attn) should be used
            if(layer_ind>0):
                if(self.input_formats[layer_ind]=="p"):
                    if(self.input_formats[layer_ind-1]=="x"):
                        assert(self.use_resi_dual==False), "No change of pytorch to xformer allowed with resi_dual setting (residual_setting=3)"
                        # change x->p
                        output, used_mask=_transform_xformer_to_pytorch_rep(output, used_mask)

                elif(self.input_formats[layer_ind]=="x"):
                    if(self.input_formats[layer_ind-1]=="p"):
                        assert(self.use_resi_dual==False), "No change of pytorch to xformer allowed with resi_dual setting (residual_setting=3)"
                        # change p->x
                        output, used_mask=_transform_pytorch_to_xformer_rep(output, used_mask)
            
            if(self.use_resi_dual):

                if(type(mod)==ModuleList):
                    
                    assert(last_query_output is None), "More complićated relative positional encoding not supported with global query vector!"

                    ## absolute part
                    o1, res_connection1,_,_=mod[0](output, 
                                 src_mask=None, 
                                 src_key_padding_mask=used_mask, 
                                 original_feature_input=original_input_feature_vecs, 
                                 positional_feature_input=positional_features,
                                 extra_res=res_connection,
                                 datalens=datalens)

                   
                    ## relative part
                    rel_input=output
                    rel_input_res=res_connection
                    if(len(mod)>2):
                        assert(len(mod)==4)
                        rel_input=mod[2](rel_input)
                        rel_input_res=mod[3](rel_input_res)


                    o2, res_connection2,_,_=mod[1](rel_input, 
                                 src_mask=None, 
                                 src_key_padding_mask=used_mask, 
                                 original_feature_input=original_input_feature_vecs, 
                                 positional_feature_input=positional_features,
                                 extra_res=rel_input_res,
                                 datalens=datalens)

                    
                        ##
                   
                    output=o1
                    res_connection=res_connection1

                    if(len(mod)>2):
                        assert(len(mod))
                        # add relative output into subset of absolute output
                        assert(o2.shape[-1]<output.shape[-1])
                        output[...,:o2.shape[-1]]=output[...,:o2.shape[-1]]+o2
                        res_connection[..., :res_connection2.shape[-1]]=res_connection[...,:res_connection2.shape[-1]]+res_connection2
                       
                    else:
                        output=output+o2
                        res_connection=res_connection+res_connection2

                else:
                    output, res_connection, last_query_output, last_connection_query = mod(output, 
                                 src_mask=None, 
                                 src_key_padding_mask=used_mask, 
                                 original_feature_input=original_input_feature_vecs, 
                                 positional_feature_input=positional_features,
                                 extra_res=res_connection,
                                 datalens=datalens,
                                 query_input=last_query_output,
                                 extra_res_query=last_connection_query)
            else:
                if(type(mod)==ModuleList):
                    
                    assert(last_query_output is None), "More complićated relative positional encoding not supported with global query vector!"

                    ## absolute part
                    o1,_=mod[0](output, 
                                 src_mask=None, 
                                 src_key_padding_mask=used_mask, 
                                 original_feature_input=original_input_feature_vecs, 
                                 positional_feature_input=positional_features,
                                 datalens=datalens)

                   
                    ## relative part
                    rel_input=output
                    if(len(mod)>2):
                        assert(len(mod)==3)
                        rel_input=mod[2](rel_input)


                    o2,_=mod[1](rel_input, 
                                 src_mask=None, 
                                 src_key_padding_mask=used_mask, 
                                 original_feature_input=original_input_feature_vecs, 
                                 positional_feature_input=positional_features,
                                 datalens=datalens)

                    
                        ##
                   
                    output=o1

                    if(len(mod)==3):
                        # add relative output into subset of absolute output
                        assert(o2.shape[-1]<output.shape[-1])
                        output[...,:o2.shape[-1]]=output[...,:o2.shape[-1]]+o2
                    else:
                        output=output+o2

                else:
                    output, last_query_output = mod(output, 
                                 src_mask=None, 
                                 src_key_padding_mask=used_mask, 
                                 original_feature_input=original_input_feature_vecs, 
                                 positional_feature_input=positional_features,
                                 datalens=datalens,
                                 query_input=last_query_output)

        if(last_query_output is not None):
            ## just return the single query token.. no aggregation needed later
            if(self.use_resi_dual):
                if(self.final_layer_norm is not None):
                    return last_query_output+self.final_layer_norm(last_connection_query)
                else:
                    return last_query_output+last_connection_query
            else:
                if(self.final_layer_norm is not None):
                    return self.final_layer_norm(last_query_output)
                else:
                    return last_query_output
        else:
            if(self.use_resi_dual):
                if(self.final_layer_norm is not None):
                    output=output+self.final_layer_norm(res_connection)
                else:
                    output=output+res_connection
            else:
                if self.final_layer_norm is not None:
                    output = self.final_layer_norm(output)

            return output

class CustomTransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - norm_first is ``False`` (this restriction may be loosened in the future)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, 
                 embed_dim: int, 
                 nhead: int, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.0,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, 
                 norm_first: bool = False,
                 device=None, 
                 dtype=None,
                 use_layer_norm_1=True, 
                 use_layer_norm_2=True, 
                 use_residual_addition=1,
                 attn_package="pytorch", # pytorch / xformers / flash
                 projection_hidden_dims="",
                 projection_add_skip_connection=0,
                 projection_add_mean_diff=0,
                 projection_type="single_self",
                 xformers_operator=None,
                 add_original_input_to_feature_input=False, # add the original input to the feature vecs
                 original_input_feature_dim=None, # dimension of 1st feature vec
                 original_input_position_feature_number=None, # how many dimensions encode position in original (1st) feature vec
                 rel_position_mode_value="none",
                 rel_position_input_feeding_type=0,
                 rel_pos_mlp_hidden_dims="512",
                 rel_pos_mlp_use_skip_connection=0,
                 rel_position_max_computational_dim=-1,
                 abs_position_mode="none",
                 do_perlayer_out_projection=1,
                 force_sdpa_precision=None,
                 use_query_class_token=0,
                 skip_ff_and_other_trafos=False) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(CustomTransformerEncoderLayer, self).__init__()
        #print("... out projection this layer?", do_perlayer_out_projection)
        #print("skip ff ? ", skip_ff_and_other_trafos)
        if(skip_ff_and_other_trafos):
            assert(use_query_class_token)
        ## outdated option: should always be joint_qkv
        assert(projection_type=="joint_qkv")

        ## position indices.. only used for 
        self.original_input_position_feature_number=original_input_position_feature_number
        self.original_input_feature_dim=original_input_feature_dim
        self.add_original_input_to_feature_input=add_original_input_to_feature_input

        proj_input_multiplier=1
        if(projection_add_mean_diff):
            proj_input_multiplier=2

        self.projection_add_mean_diff=projection_add_mean_diff

        if(self.add_original_input_to_feature_input):
            assert(self.original_input_feature_dim is not None)
        
            self.in_projector=qkv_projector(input_dim=proj_input_multiplier*(embed_dim+self.original_input_feature_dim), 
                                            mlp_hidden_dims=projection_hidden_dims, 
                                            output_dim=embed_dim,
                                            projection_type=projection_type,
                                            add_skip_connection=projection_add_skip_connection)
        else:
            self.in_projector=qkv_projector(input_dim=proj_input_multiplier*embed_dim, 
                                            mlp_hidden_dims=projection_hidden_dims, 
                                            output_dim=embed_dim,
                                            projection_type=projection_type,
                                            add_skip_connection=projection_add_skip_connection)

       
        self.nhead=nhead
        self.embed_dim=embed_dim
        self.dim_per_head=self.embed_dim//self.nhead
        assert(self.embed_dim % self.nhead ==0 ), "Embedding dim must be divisble by nhead!"

        ## required to work with pytorch 2.xx .. just a hack
        self_attn=dict()
        self_attn["batch_first"]=True
        self.self_attn=Namespace(**self_attn)

        ## TODO: make this optional?
        self.out_projector=None
        self.do_perlayer_out_projection=do_perlayer_out_projection
        if(self.do_perlayer_out_projection):
            self.out_projector=qkv_projector(input_dim=embed_dim, 
                                        mlp_hidden_dims="", # out projection is only linear
                                        output_dim=embed_dim,
                                        projection_type="single_self")
        else:
            assert(use_residual_addition!=1), "Residual addition of 1 (default transformer) requires out projection.. otherwise it is too restricting"

        # skip ff - only for query token and last layer
        if(skip_ff_and_other_trafos):
            self.linear1=lambda x: x
            self.linear2=lambda x: x
        else:
            # Implementation of Feedforward model
            self.linear1 = Linear(embed_dim, dim_feedforward, **factory_kwargs)
            self.linear2 = Linear(dim_feedforward, embed_dim, **factory_kwargs)

        self.dropout = Dropout(dropout)
        self.norm_first = norm_first

        if(skip_ff_and_other_trafos):
            self.norm1=lambda x: x
            self.norm2=lambda x: x
        else:
            if(use_layer_norm_1):
                self.norm1 = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
            else:
                self.norm1 = lambda x: x

            if(use_layer_norm_2):
                self.norm2 = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
            else:
                self.norm2 = lambda x: x

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0

        self.activation = activation

        self.use_query_class_token=use_query_class_token
        ## if we have a class passthrough query.. we have an additional MLP
        if(use_query_class_token):
            ## require RELU right now and share activation for other MLP
            assert(self.activation_relu_or_gelu==1)

            self.query_activation=activation#_get_activation_fn(activation)
            self.query_linear1 = Linear(embed_dim, dim_feedforward, **factory_kwargs)
            self.query_dropout = Dropout(dropout)
            self.query_linear2 = Linear(dim_feedforward, embed_dim, **factory_kwargs)

            if(use_layer_norm_2):
                self.norm2_query = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
            else:
                self.norm2_query = lambda x: x


            ## only if we do *not* norm first or have resi-dual...
            if(self.norm_first==0 or use_residual_addition==3):
                if(use_layer_norm_1):
                    self.norm1_query = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
                else:
                    self.norm1_query = lambda x: x
                

        self.use_residual_addition=use_residual_addition

        self.projection_type=projection_type
        self.attn_package=attn_package



        self.dropout_p=dropout

        self.xformers_operator=None
        if(xformers_operator is not None):
            if(xformers_operator=="flash"):
                self.xformers_operator=MemoryEfficientAttentionFlashAttentionOp
            elif(xformers_operator=="small_k"):
                self.xformers_operator=MemoryEfficientAttentionOp
            elif(xformers_operator=="triton"):
                self.xformers_operator=TritonFlashAttentionOp
            else:
                raise Exception("Unknown attn operator ", self.xformers_operator)

        
        ## relative positions
        self.rel_position_mode_value=rel_position_mode_value
        self.rel_position_input_feeding_type=rel_position_input_feeding_type
        self.rel_position_max_computational_dim=rel_position_max_computational_dim

        if( (self.attn_package!="official_pytorch_w_weights") and (self.attn_package != "geometric_scatter") and (self.attn_package != "flash_attn")):
            assert(self.rel_position_mode_value=="none"), "Relative position mode currently only supported in *official_pytorch_w_weights* mode! But used ... %s" % self.attn_package

        if(self.rel_position_mode_value!="none"):

            assert(nhead==1), "Only nhead=1 currently supported for rel position encoding!"

            target_rel_dim=embed_dim
            if(rel_position_max_computational_dim!=-1):
                target_rel_dim=min(embed_dim, rel_position_max_computational_dim)

            ## 
            ### value_mlp_1 "concat_value_to_diff", "split_in_heads_2"
            #if(self.rel_position_mode_value=="concat_abs_and_rel_and_add"):
            if(self.rel_position_input_feeding_type==0):## feed value embed dim + diffs of value
                self.rel_pos_value_mlp=SkipMLP(2*embed_dim, rel_pos_mlp_hidden_dims, target_rel_dim, add_skip_connection=rel_pos_mlp_use_skip_connection)
                #self.rel_pos_value_projector=torch.nn.Linear(2*embed_dim, embed_dim)
            elif(self.rel_position_input_feeding_type==1):

                used_additional_rel_input_dim=self.original_input_feature_dim

                ## if positional indices are subset of total vector, only pass those
                #if(original_input_position_feature_number is not None):
                #    used_additional_rel_input_dim=original_input_position_feature_number


                self.rel_pos_value_mlp=SkipMLP(embed_dim+used_additional_rel_input_dim, rel_pos_mlp_hidden_dims, target_rel_dim, add_skip_connection=rel_pos_mlp_use_skip_connection)
            
            elif(self.rel_position_input_feeding_type==2):
                used_additional_rel_input_dim=self.original_input_feature_dim

                ## if positional indices are subset of total vector, only pass those
                #if(original_input_position_feature_number is not None):
                #    used_additional_rel_input_dim=original_input_position_feature_number

                
                ## just push the official one through
                self.rel_pos_value_mlp=SkipMLP(2*used_additional_rel_input_dim, rel_pos_mlp_hidden_dims, target_rel_dim, add_skip_connection=rel_pos_mlp_use_skip_connection)
            
            else:
                raise NotImplementedError()


        self.force_sdpa_precision=force_sdpa_precision
        if(self.force_sdpa_precision is not None):
            # check that attn implementation supports switching precicion ad hoc
            supported=self.attn_package=="custom_pytorch" or self.attn_package=="flash_attn" or self.attn_package == "xformer"
            assert(supported), ("Changing precision for sdpa (scaled dot product attention) adhoc is only allowed for specific attention implementations: custom_pytorch/flash_attn/xformer .. but used ", self.attn_package)
            
            if(self.force_sdpa_precision=="float16"):
                self.force_sdpa_precision=torch.float16
            elif(self.force_sdpa_precision=="bfloat16"):
                self.force_sdpa_precision=torch.bfloat16
            elif(self.force_sdpa_precision=="float32"):
                self.force_sdpa_precision=torch.float32
            else:
                raise Exception("Unknown or unsupported precision for spda in *force_sdpa_precision*: ", self.force_sdpa_precision)
        
        self.abs_position_mode=abs_position_mode
        
    def __setstate__(self, state):
        super(CustomTransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, is_causal=False,
                src_key_padding_mask: Optional[Tensor] = None,
                original_feature_input=None,
                positional_feature_input = None,
                absolute_position_encoding = None,
                extra_res=None,
                datalens=None,
                query_input=None,
                extra_res_query=None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        ## replace default encoder layer with custom blocks
        ## right now we want no causal encoding
        assert(is_causal==False)
        assert(src_mask is None)

        x = src

        ## resi-dual block (https://arxiv.org/pdf/2304.14802)
        if(self.use_residual_addition==3):
            ## Resi-Dual requires an extra residual pass (extra_res) to be passed through
            ## make sure it is defined
            assert(extra_res is not None)

            ## sa block
            
            sa_result, query_ca_result=self._sa_block(x, 
                                                      None, 
                                                      src_key_padding_mask, 
                                                      original_feature_input=original_feature_input, 
                                                      positional_feature_input=positional_feature_input, 
                                                      absolute_position_encoding=absolute_position_encoding, 
                                                      datalens=datalens,
                                                      query_input=query_input)

            x=x+sa_result

            extra_res_result=extra_res+sa_result

            x=self.norm1(x)

            ## ff block

            ff_result=self._ff_block(x)

            x=x+ff_result

            extra_res_result=extra_res_result+ff_result

            x=self.norm2(x)

            if(query_input is None):

                return x,extra_res_result, None, None
            else:
                # copy resi-dual structure over to cross attention
                query_output=query_input+query_ca_result
                extra_res_result_query=extra_res_query+query_ca_result

                query_output=self.norm1_query(query_output)

                ff_result_query=self._ff_block_query(query_output)

                query_output=query_output+ff_result_query

                extra_res_result_query=extra_res_result_query+ff_result_query

                query_output=self.norm2_query(query_output)

                ## return 4 tensors, output, extra_ouptut, output_query, extra_output_query
                return x, extra_res_result, query_output, extra_res_result_query

        else:
            assert(extra_res==None)

            ## norm_first = pre LN (recommended)
            ## norm_last = post LN (older settings)
            query_output=None
            ## only use residual addition on SA block?
            if(self.use_residual_addition==2):
                if self.norm_first:
                    
                    sa_result, query_ca_result = self._sa_block(self.norm1(x), None, src_key_padding_mask, original_feature_input=original_feature_input, positional_feature_input=positional_feature_input, absolute_position_encoding=absolute_position_encoding, datalens=datalens, query_input=query_input)
                    x = x + self._ff_block(self.norm2(sa_result))

                    if(query_input is not None):
                        query_output=query_input+self._ff_block_query(self.norm2_query(quera_ca_result))

                else:
                    pre_res, query_ca_result=self._sa_block(x, None, src_key_padding_mask, original_feature_input=original_feature_input, positional_feature_input=positional_feature_input, absolute_position_encoding=absolute_position_encoding, datalens=datalens, query_input=query_input)
                    sa_result = self.norm1(x + pre_res)
                    x = self.norm2(x + self._ff_block(sa_result))

                    if(query_input is not None):
                        normed_query=self.norm1_query(query_input+query_ca_result)
                        query_output=self.norm2_query(query_input+self._ff_block_query(normed_query))

            ## DEFAULT TRANSFORMER -- both SA and FF blocks use residual addition
            elif(self.use_residual_addition==1):
                if self.norm_first:
                    
                    sa_result, query_ca_result = self._sa_block(self.norm1(x), None, src_key_padding_mask, original_feature_input=original_feature_input, positional_feature_input=positional_feature_input, absolute_position_encoding=absolute_position_encoding, datalens=datalens, query_input=query_input)
                    x = x + sa_result
                    x = x + self._ff_block(self.norm2(x))

                    if(query_input is not None):
                        query_output=query_input+query_ca_result
                        query_output=query_output+self._ff_block_query(self.norm2_query(query_output))
                else:
                    pre_res, query_ca_result=self._sa_block(x, None, src_key_padding_mask, original_feature_input=original_feature_input, positional_feature_input=positional_feature_input, absolute_position_encoding=absolute_position_encoding, datalens=datalens, query_input=query_input)
                    x = self.norm1(x + pre_res)
                    x = self.norm2(x + self._ff_block(x))

                    if(query_input is not None):
                        query_output=self.norm1_query(query_input+query_ca_result)
                        query_output=self.norm2_query(query_output+self._ff_block_query(query_input))


            elif(self.use_residual_addition==0):
                if self.norm_first:
                    sa_result, query_ca_result = self._sa_block(self.norm1(x), None, src_key_padding_mask, original_feature_input=original_feature_input, positional_feature_input=positional_feature_input, absolute_position_encoding=absolute_position_encoding, datalens=datalens, query_input=query_input)
                    x = sa_result
                    x = self._ff_block(self.norm2(x))

                    if(query_input is not None):

                        query_output=query_ca_result
                        query_output=self.ff_result_query(self.norm2_query(query_output))

                else:

                    pre_res, query_ca_result=self._sa_block(x, None, src_key_padding_mask, original_feature_input=original_feature_input, positional_feature_input=positional_feature_input, absolute_position_encoding=absolute_position_encoding, datalens=datalens, query_input=query_input)
                    x = self.norm1(pre_res)
                    x = self.norm2(self._ff_block(x))

                    if(query_input is not None):
                        query_output=self.norm1_query(query_ca_result)
                        query_output=self.norm2_query(self._ff_block_query(query_output))

            return x, query_output

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], 
                  key_padding_mask: Optional[Tensor],
                  original_feature_input=None,
                  positional_feature_input=None,
                  absolute_position_encoding=None,
                  datalens=None,
                  query_input=None) -> Tensor:

        assert(attn_mask is None), "No support for causal masks right now."
        ## in projection

        ## Q, k, V shape: B X num_length X Num_heads*dim_per_head
   
        
        total_embed_dim=self.nhead*self.dim_per_head
            
        ## only used when query input is not None
        query_output=None

        if(self.add_original_input_to_feature_input):
            assert(original_feature_input is not None)
            assert(self.attn_package!="official_pytorch_w_weights"), "*official_pytorch_w_weights* uses a pytorch internal projector and cannot handle ad-hoc addition of original input!"
            
            x=torch.cat( [x, original_feature_input], dim=-1)

        if(self.projection_add_mean_diff):
            ## add diff to input
            ## x mean calculation depends on masking type
            if("pytorch" in self.attn_package):

                if(key_padding_mask is not None):
                    tempvec=x*(~key_padding_mask).type(torch.int).unsqueeze(-1)
                else:
                    tempvec=x

               
                ## requires datalens here
                temp_mean=(tempvec.sum(axis=1)/datalens.unsqueeze(-1))[:,None,:]

                x=torch.cat([x, x-temp_mean], dim=-1)
               
                
            elif( ("xformer" in self.attn_package) or ("flash" in self.attn_package)):

                split_vec=key_padding_mask.split(x)
            
                new_split_vec=[ torch.cat([i,i-i.mean(dim=1,keepdims=True)], dim=-1) for i in split_vec]

                ## go back to x

                _, x = fmha.BlockDiagonalMask.from_tensor_list(new_split_vec)

            else:
                raise Exception("attn package ", self.attn_package, " does not support adding mean diff")
        
        if(self.attn_package=="custom_pytorch"):
            """ 
            First implementation .. should not be used
            """
            batch_size=x.size(0)

            q = self.in_projector(x)

            if(self.projection_type=="single_self"):
                q=q
                k=q
                v=q
            else:
                q,k,v = q # q is tuple of q,k,v
            
           
            ## but batch size at 1st pos
            q=q.transpose(1,0)
            k=k.transpose(1,0)
            v=v.transpose(1,0)

            if(query_input is not None):
                #print("query -input ", query_input.shape)
                query_input=query_input.transpose(1,0)
                assert(query_input.shape[0]==1)
            
            ## reshape to make qkv dimension work nicely with scaled dot product attention
            ## shape (S, batch_size * num_heads, per_head _dim) instead of (S, batch_size, num_heads*per_head_dim=embed_dim)
            q = q.contiguous().view(q.shape[0], q.shape[1] * self.nhead, self.dim_per_head).transpose(1, 0)
            k = k.contiguous().view(k.shape[0], k.shape[1] * self.nhead, self.dim_per_head).transpose(1, 0)
            v = v.contiguous().view(v.shape[0], v.shape[1] * self.nhead, self.dim_per_head).transpose(1, 0)

            if(query_input is not None):
                query_input=query_input.contiguous().view(query_input.shape[0], query_input.shape[1] * self.nhead, self.dim_per_head).transpose(1, 0)
                
            #print(q[0])
            new_attn_mask=key_padding_mask
            
            ## require mask with inf istead of bool for default dot product attn
            ## if given by the encoder ,this should be INF automatically

            max_num_samples=x.size(1)


            prev_type=q.dtype
            if(self.force_sdpa_precision is not None):
                q=q.type(self.force_sdpa_precision)
                k=k.type(self.force_sdpa_precision)
                v=v.type(self.force_sdpa_precision)

                query_input=query_input.type(self.force_sdpa_precision)

            if key_padding_mask is not None:
                
                #assert(key_padding_mask.dtype != torch.bool)
                assert(len(key_padding_mask.shape)==2)
                temp_mask = key_padding_mask.view(batch_size, 1, 1, max_num_samples).   \
                expand(-1, self.nhead, -1, -1).reshape(batch_size * self.nhead, 1, max_num_samples)

                new_attn_mask = torch.zeros_like(temp_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(temp_mask, float("-inf"))
            
        
            x  = F.scaled_dot_product_attention(q, k, v, new_attn_mask, self.dropout_p)

            if(query_input is not None):
                query_output=F.scaled_dot_product_attention(query_input, k,v, new_attn_mask, self.dropout_p)

            if(self.force_sdpa_precision is not None):
                x=x.type(prev_type)
                query_output=query_output.type(prev_type)
            

            x = x.transpose(1, 0).contiguous().view(-1, self.embed_dim)

                
            ## out projection not really needed without dropout
            if(self.do_perlayer_out_projection==1):
                x=self.out_projector(x)

            x = x.view(max_num_samples, batch_size, self.embed_dim).transpose(1, 0)

            if(query_input is not None):
                query_output = query_output.transpose(1, 0).contiguous().view(-1, self.embed_dim)
                query_output = query_output.view(1, batch_size, self.embed_dim).transpose(1, 0)
                
            return self.dropout1(x), query_output

        elif(self.attn_package=="flash_attn"):

            joint_qkv = self.in_projector(x, split_qkv=False)
            joint_qkv=joint_qkv[0].view(-1, 3,self.nhead, self.dim_per_head)

            if(query_input is not None):
               query_input=query_input.reshape(-1, self.nhead, self.dim_per_head)
            
            prev_type=joint_qkv.dtype
            if(self.force_sdpa_precision is not None):
                joint_qkv=joint_qkv.type(self.force_sdpa_precision)
                query_input=query_input.type(self.force_sdpa_precision)
            

            if(self.rel_position_mode_value=="none"):

                out=flash_attn.flash_attn_varlen_qkvpacked_func(joint_qkv,key_padding_mask.q_seqinfo.seqstart.to(device=joint_qkv.device), key_padding_mask.q_seqinfo.max_seqlen)
                
                if(query_input is not None):

                    q_seqlen=torch.arange(len(datalens)+1).to(device=joint_qkv.device, dtype=torch.int32)
                    query_output=flash_attn.flash_attn_varlen_kvpacked_func(query_input, joint_qkv[:,1:,:,:], q_seqlen, key_padding_mask.q_seqinfo.seqstart.to(device=joint_qkv.device), 1, key_padding_mask.q_seqinfo.max_seqlen)
                
                if(self.force_sdpa_precision is not None):
                    out=out.type(prev_type)
                    query_output=query_output.type(prev_type)

                if(query_input is not None):
                    
                    query_output=query_output.reshape(1, query_output.shape[0], -1)
                    
                out=out.reshape(1, out.shape[0], -1)

                ## out projection not really needed without dropout
                if(self.do_perlayer_out_projection==1):
                    out=self.out_projector(out)

                return self.dropout1(out), query_output
            else:
                #print("-------> flash attn .. relative")
                if(self.force_sdpa_precision is None):
                    # do we actually need bfloat16? HMM?
                    # force bfloat16 for relative pos encoding
                    prev_type=joint_qkv.dtype
                    joint_qkv=joint_qkv.type(torch.bfloat16)

                    if(query_input is not None):
                        query_input=query_input.type(torch.bfloat16)
                    

                out, _, weights=flash_attn.flash_attn_varlen_qkvpacked_func(joint_qkv,key_padding_mask.q_seqinfo.seqstart.to(device=joint_qkv.device), key_padding_mask.q_seqinfo.max_seqlen, return_attn_probs=True, dropout_p=0.0000001)
                out=out.view(1, out.shape[0], -1)

                if(query_input is not None):

                    q_seqlen=torch.arange(len(datalens)+1).to(device=joint_qkv.device, dtype=torch.int32)
                    query_output=flash_attn.flash_attn_varlen_kvpacked_func(query_input, joint_qkv[:,1:,:,:], q_seqlen, key_padding_mask.q_seqinfo.seqstart.to(device=joint_qkv.device), 1, key_padding_mask.q_seqinfo.max_seqlen)
                
                
                #if(self.force_sdpa_precision is not None):
                out=out.type(prev_type)
                weights=weights.type(prev_type)

                if(query_input is not None):
                    query_output=query_output.type(prev_type)
                    query_output=query_output.reshape(1, query_output.shape[0], -1)
                    

                element_list=key_padding_mask.q_seqinfo.seqstart.to(device=joint_qkv.device)
                element_list=element_list[1:]-element_list[:-1]

                repeated_lens=torch.repeat_interleave(element_list, element_list)

                if(self.rel_position_input_feeding_type==0):
                    ## create new diffs first
                    ## only_rel: only use relative value encoding .. weights*rel_distances
           
                    ## first index (first tesnor) - second index (second tensor)
                    ## 0 - 0
                    ## 0 - 1
                    ## 0 - 2
                    ## ...
                    ## 1 - 0
                    ## 1 - 1
                    ## 
                    ##
                   
                    ## get flattened distances based on current input
                    recovered=key_padding_mask.split(x)
                    flattened_distances,_=_get_tensor_of_diffs_flattened(recovered, element_list=element_list, reverse=False)
                    
                    # we take embed dim as input
                    flattened_inputs=x[0]

                    
                    repeated_inputs=torch.repeat_interleave(flattened_inputs, repeated_lens.to(device=x.device), dim=0)
                    
                    # stack absolute + relative on top
                    stacked_input=torch.cat([repeated_inputs, flattened_distances], dim=1)

                elif(self.rel_position_input_feeding_type==1):

                    
                    assert(original_feature_input is not None)

                    ## get flattened distances based on absolute overall input
               
                    recovered=key_padding_mask.split(original_feature_input)
                    flattened_distances,_=_get_tensor_of_diffs_flattened(recovered, element_list=element_list, reverse=False)
                    
                    # we take embed dim as input
                    flattened_inputs=x[0]

                    repeated_inputs=torch.repeat_interleave(flattened_inputs, repeated_lens.to(device=x.device), dim=0)

                    # stack absolute + relative on top
                    stacked_input=torch.cat([repeated_inputs, flattened_distances], dim=1)

                else:
                   
                    assert(original_feature_input is not None)
                    ## get flattened distances based on absolute overall input
                    #assert(positional_feature_input is not None)
                    recovered=key_padding_mask.split(original_feature_input)

                    flattened_distances,_=_get_tensor_of_diffs_flattened(recovered, element_list=element_list, reverse=False)
                    
                    # we take embed dim as input
                    flattened_inputs=original_feature_input[0]

                    repeated_inputs=torch.repeat_interleave(flattened_inputs, repeated_lens.to(device=x.device), dim=0)

                    stacked_input=torch.cat([repeated_inputs, flattened_distances], dim=1)

                ###################

                relative_values = self.rel_pos_value_mlp(stacked_input)

                ### strategy 2 seems to be most efficient
                approach=2
                if(approach==0):

                    ## WEIGHTS shape B X 1 X L X L
                    weights=weights[:,:,:key_padding_mask.q_seqinfo.max_seqlen,:key_padding_mask.q_seqinfo.max_seqlen]
                    weights=weights/weights.sum(dim=-1, keepdims=True)
                    relative_values=_recreate_list_of_matrices_padded(relative_values, element_list)
                    
                    final_rel_value_result=(weights[:, 0,:,:,None]*relative_values).sum(dim=2)

                    flattened_rel_result,_=_get_tensors_flattened(final_rel_value_result, element_list=element_list)
                    flattened_rel_result=flattened_rel_result.unsqueeze(0)
                elif(approach==1):
                    weights=weights[:,:,:key_padding_mask.q_seqinfo.max_seqlen,:key_padding_mask.q_seqinfo.max_seqlen]
                    weights=weights/weights.sum(dim=-1, keepdims=True)

                    index_scatter=torch.repeat_interleave(torch.arange(len(repeated_lens)).to(device=repeated_lens.device), repeated_lens)

                    rr=torch.cat([i[0][:element_list[ind],:element_list[ind]].reshape(-1) for ind,i in enumerate(weights)]).unsqueeze(-1)

                    flattened_rel_result=scatter_add(rr*relative_values, index_scatter, dim=0).unsqueeze(0)

                elif(approach==2):
                    index_scatter=torch.repeat_interleave(torch.arange(len(repeated_lens)).to(device=repeated_lens.device), repeated_lens)

                    rr=torch.cat([i[0][:element_list[ind],:element_list[ind]].reshape(-1) for ind,i in enumerate(weights)]).unsqueeze(-1)
                    
                    repeated_sums=scatter_sum(rr, index_scatter, dim=0).repeat_interleave(repeated_lens, dim=0)
                  
                    flattened_rel_result=scatter_add( (rr/repeated_sums)*relative_values, index_scatter, dim=0).unsqueeze(0)
                    

                if(self.do_perlayer_out_projection==1):
                    out=self.out_projector(out)

                ## same dim
                if(flattened_rel_result.shape[-1]==out.shape[-1]):

                    if(self.rel_position_mode_value=="only_rel"):
                        # only relative encoding
                        return flattened_rel_result,query_output
                    else:
                        return flattened_rel_result+out,query_output
                ## relative output is smaller in dimensionality.. embed or add to subspace
                else:
                    if(self.rel_position_mode_value=="only_rel"):
                        # only relative encoding
                        out_res=torch.zeros_like(out)
                        out_res[...,:flattened_rel_result.shape[-1]]=out_res[...,:flattened_rel_result.shape[-1]]+flattened_rel_result
                        return out_res,query_output
                    else:
                        
                        out[...,:flattened_rel_result.shape[-1]]=out[...,:flattened_rel_result.shape[-1]]+flattened_rel_result
                      
                        return out,query_output


        elif(self.attn_package=="geometric_scatter"):

            assert(query_input is None), "Query cross attention not implemented for geometric scatter!"
            
            q = self.in_projector(x)

            if(self.projection_type=="single_self"):
                q=q
                k=q
                v=q
            else:
                q,k,v = q # q is tuple of q,k,v

            q=q.reshape(q.shape[0], q.shape[1], self.nhead, -1)
            k=k.reshape(k.shape[0], k.shape[1], self.nhead, -1)
            v=v.reshape(v.shape[0], v.shape[1], self.nhead, -1)

            #q_scaled = q * numpy.sqrt(1.0 / float(q.shape[-1]))
           
            lens=key_padding_mask.q_seqinfo.seqstart[1:]-key_padding_mask.q_seqinfo.seqstart[0:-1]
            repeated_lens=torch.repeat_interleave(lens, lens)

            ## repeat v aswell
            tot_v_dim=0
            for cur_id, cur_len in enumerate(lens):
                tot_v_dim+=cur_len*cur_len

            ## repeated q
            repeated_q=torch.repeat_interleave(q, repeated_lens.to(device=q.device), dim=1)


            ## repeated k
            """
            repeated_k=[]
            offset=0
            for cur_id, cur_len in enumerate(lens):
               
                repeated_k.append(k[:,offset:cur_len+offset,:,:].repeat(1,cur_len, 1,1))
                offset=offset+cur_len
            repeated_k=torch.cat(repeated_k, dim=1)
            """
            repeated_k=torch.zeros((v.shape[0], tot_v_dim, v.shape[2], v.shape[3])).to(v)
            offset=0
            single_offset=0
            for cur_id, cur_len in enumerate(lens):
                
                repeated_k[:,offset:offset+cur_len*cur_len,:,:]=k[:,single_offset:cur_len+single_offset,:,:].repeat(1,cur_len, 1,1)
                offset+=cur_len*cur_len
                single_offset+=cur_len

            ## softmax
            bare_weights=(repeated_q*repeated_k).sum(dim=-1)*numpy.sqrt(1.0 / float(q.shape[-1]))
            #index_scatter=torch.repeat_interleave(torch.arange(len(repeated_lens)).to(device=repeated_lens.device), repeated_lens)
            index_scatter=torch.repeat_interleave(torch.arange(len(repeated_lens)), repeated_lens).to(device=q.device)
           
            softmax_result=scatter_softmax(bare_weights, index_scatter, dim=1)[...,None]

            
              
            """
            repeated_v=[]
            offset=0
            for cur_id, cur_len in enumerate(lens):
                repeated_v.append(v[:,offset:cur_len+offset,:,:].repeat(1,cur_len, 1,1))
                offset=offset+cur_len
            repeated_v=torch.cat(repeated_v, dim=1)
            """

            repeated_v=torch.zeros((v.shape[0], tot_v_dim, v.shape[2], v.shape[3])).to(v)
            offset=0
            single_offset=0
            for cur_id, cur_len in enumerate(lens):
                
                repeated_v[:,offset:offset+cur_len*cur_len,:,:]=v[:,single_offset:cur_len+single_offset,:,:].repeat(1,cur_len, 1,1)
                offset+=cur_len*cur_len
                single_offset+=cur_len


            summed_vs=scatter_add(softmax_result*repeated_v, index_scatter, dim=1)

            torch.cuda.empty_cache()
            out=summed_vs.reshape(q.shape[0], q.shape[1], -1)

            ## out projection not really needed without dropout
            if(self.do_perlayer_out_projection==1):
                out=self.out_projector(out)

            ### add relative positional encoding aswell if desired

            if(self.rel_position_mode_value=="none"):
                
                return self.dropout1(out)

            else:
             
                
                if(self.rel_position_input_feeding_type==0):
                    ## create new diffs first
                    ## only_rel: only use relative value encoding .. weights*rel_distances
           
                    ## first index (first tesnor) - second index (second tensor)
                    ## 0 - 0
                    ## 0 - 1
                    ## 0 - 2
                    ## ...
                    ## 1 - 0
                    ## 1 - 1
                    ## 
                    ##
                    
                    ## diffs
                    #flattened_distances, used_elements=_get_tensor_of_diffs_flattened(x[0], element_list=lens)
                    recovered=key_padding_mask.split(x)
                    flattened_distances,_=_get_tensor_of_diffs_flattened(recovered, element_list=lens, reverse=False)
                    ## absolute vals
                    
                    flattened_inputs=x[0]
                   
                else:
                    assert(original_feature_input is not None)
                    print(original_feature_input)

                    sys.exit(-1)
                    flattened_inputs=positional_feature_input

                
                repeated_inputs=torch.repeat_interleave(flattened_inputs, repeated_lens.to(device=x.device), dim=0)
               
                # stacked
                stacked_input=torch.cat([repeated_inputs, flattened_distances], dim=1)

                relative_values = self.rel_pos_value_mlp(stacked_input)
                #relative_values=_recreate_list_of_matrices_padded(relative_values, element_list)
              
                summed_vs=scatter_add(softmax_result[0,:,:,0]*relative_values, index_scatter, dim=0)
                summed_vs=summed_vs[None, :, None,:]
              
                summed_vs=summed_vs.reshape(q.shape[0], q.shape[1], -1)
                
                
                if(self.rel_position_mode_value=="only_rel"):
                    # only relative encoding
                    out=summed_vs
                else:
                    # add relative + standard mha calculation
                    
                    out=out+summed_vs
                
                return self.dropout1(out), None


        elif(self.attn_package=="official_pytorch_w_weights"):

            #assert(query_input is None), "Query cross attention not implemented for official pytorch with weights!"

            identity_weight=torch.eye(total_embed_dim).to(x)
            identity_bias=None

            if(self.do_perlayer_out_projection):
                used_outproj=self.out_projector.projector.mlp[0].weight
                used_outproj_bias=self.out_projector.projector.mlp[0].bias

            else:
                # placeholder outproj tensors
                used_outproj=identity_weight#torch.eye(total_embed_dim).to(x)
                used_outproj_bias=identity_bias

            
            if key_padding_mask is not None:

                #assert(key_padding_mask.dtype != torch.bool), key_padding_mask
                assert(len(key_padding_mask.shape)==2)
                #new_attn_mask=key_padding_mask[:,None,:]*(key_padding_mask[:,:,None])
                #new_attn_mask=torch.where(~torch.isfinite(new_attn_mask), float("-inf"), new_attn_mask)
                """
                new_attn_mask = key_padding_mask.view(batch_size, 1, 1, max_num_samples).   \
                expand(-1, self.nhead, -1, -1).reshape(batch_size * self.nhead, 1, max_num_samples)
                """


            temp_x = self.in_projector(x, split_qkv=False)

            sep_q=temp_x[...,:total_embed_dim]
            sep_k=temp_x[...,total_embed_dim:2*total_embed_dim]
            sep_v=temp_x[...,2*total_embed_dim:]

            updated_v, weights=torch.nn.functional.multi_head_attention_forward(sep_q.transpose(1,0),sep_k.transpose(1,0),sep_v.transpose(1,0),
                                                          total_embed_dim,
                                                          self.nhead,
                                                          None,#identity_weight,#self.in_projector.joint_projector.mlp[0].weight,
                                                          identity_bias,#self.in_projector.joint_projector.mlp[0].bias,
                                                          None,
                                                          None,
                                                          False, # add_zero_attn,
                                                          self.dropout_p,
                                                          used_outproj,
                                                          q_proj_weight=identity_weight,
                                                          k_proj_weight=identity_weight,
                                                          v_proj_weight=identity_weight,
                                                          use_separate_proj_weight=True,
                                                          out_proj_bias=used_outproj_bias,
                                                          need_weights=True,
                                                          average_attn_weights=False,
                                                          key_padding_mask=key_padding_mask)

            query_output=None
            if(query_input is not None):

               
                query_output,_=torch.nn.functional.multi_head_attention_forward(query_input.transpose(1,0),sep_k.transpose(1,0),sep_v.transpose(1,0),
                                                          total_embed_dim,
                                                          self.nhead,
                                                          None,#identity_weight,#self.in_projector.joint_projector.mlp[0].weight,
                                                          identity_bias,#self.in_projector.joint_projector.mlp[0].bias,
                                                          None,
                                                          None,
                                                          False, # add_zero_attn,
                                                          self.dropout_p,
                                                          identity_weight,
                                                          q_proj_weight=identity_weight,
                                                          k_proj_weight=identity_weight,
                                                          v_proj_weight=identity_weight,
                                                          use_separate_proj_weight=True,
                                                          out_proj_bias=identity_bias,
                                                          need_weights=False,
                                                          average_attn_weights=False,
                                                          key_padding_mask=key_padding_mask)

              
                query_output=query_output.transpose(1,0)
                
            ## check cross attention and hack it into *multi_head_attention_forward* (just needed for cpu support currently..)

                
            if(self.rel_position_mode_value=="none"):
                
                out=updated_v.transpose(1,0)
        
            else:

                element_list=torch.LongTensor([x.shape[1] for i in range(x.shape[0])]).to(device=x.device)
                if(key_padding_mask is not None):
                    element_list=torch.LongTensor([int(sum(mitem==0).cpu().detach()) for mitem in key_padding_mask]).to(device=x.device)


                if(self.rel_position_input_feeding_type==0):
                    ## create new diffs first
                    ## only_rel: only use relative value encoding .. weights*rel_distances
           
                    ## first index (first tesnor) - second index (second tensor)
                    ## 0 - 0
                    ## 0 - 1
                    ## 0 - 2
                    ## ...
                    ## 1 - 0
                    ## 1 - 1
                    ## 
                    ##

                    ## stack embedding + embedding diff
                    
                    ## diffs
                    flattened_distances, used_elements=_get_tensor_of_diffs_flattened(x, element_list=element_list)
                    flattened_inputs,_=_get_tensors_flattened(x, element_list=element_list)
                    #flattened_inputs=positional_feature_input

                    squared_el_list=torch.repeat_interleave(used_elements, element_list)
                    repeated_inputs=torch.repeat_interleave(flattened_inputs, squared_el_list, dim=0)

                    stacked_input=torch.cat([repeated_inputs, flattened_distances], dim=1)

                elif(self.rel_position_input_feeding_type==1):
                    ## stack embedding + original diff
                    assert(original_feature_input is not None)

                    #assert(positional_feature_input is not None)

                    flattened_distances, used_elements=_get_tensor_of_diffs_flattened(original_feature_input, element_list=element_list)
                    flattened_inputs,_=_get_tensors_flattened(x, element_list=element_list)
                    #flattened_inputs=positional_feature_input

                    squared_el_list=torch.repeat_interleave(used_elements, element_list)
                    repeated_inputs=torch.repeat_interleave(flattened_inputs, squared_el_list, dim=0)

                    stacked_input=torch.cat([repeated_inputs, flattened_distances], dim=1)

                elif(self.rel_position_input_feeding_type==2):

                    ## stack original + orginal diff

                    assert(original_feature_input is not None)
                    
                    flattened_distances, used_elements=_get_tensor_of_diffs_flattened(original_feature_input, element_list=element_list)
                    flattened_inputs,_=_get_tensors_flattened(original_feature_input, element_list=element_list)
                    #flattened_inputs=positional_feature_input

                    squared_el_list=torch.repeat_interleave(used_elements, element_list)
                    repeated_inputs=torch.repeat_interleave(flattened_inputs, squared_el_list, dim=0)

                    stacked_input=torch.cat([repeated_inputs, flattened_distances], dim=1)

               
                # stacked
                

                relative_values = self.rel_pos_value_mlp(stacked_input)
                relative_values=_recreate_list_of_matrices_padded(relative_values, element_list)
                    
                
                final_rel_value_result=(weights[:, 0,:,:,None]*relative_values).sum(dim=2)
               
                out=updated_v.transpose(1,0)

                ## same dim
                if(final_rel_value_result.shape[-1]==out.shape[-1]):

                    if(self.rel_position_mode_value=="only_rel"):
                        # only relative encoding
                        out=final_rel_value_result
                    else:
                        out= final_rel_value_result+out
                ## relative output is smaller in dimensionality.. embed or add to subspace
                else:
                    if(self.rel_position_mode_value=="only_rel"):
                        # only relative encoding
                        out_res=torch.zeros_like(out)
                        out_res[...,:final_rel_value_result.shape[-1]]=out_res[...,:final_rel_value_result.shape[-1]]+final_rel_value_result
                        out=out_res
                    else:
                       
                        out[...,:final_rel_value_result.shape[-1]]=out[...,:final_rel_value_result.shape[-1]]+final_rel_value_result
                        
                """
                #print(final_rel_value_result[2])
                if(self.rel_position_mode_value=="only_rel"):
                    # only relative encoding
                    out=final_rel_value_result
                else:
                    # add relative + standard mha calculation
                    out = final_rel_value_result+out
                """

            return out, query_output

        elif(self.attn_package=="nested"):
            
            assert(query_input is None), "Query cross attention not implemented for nested tensors!"

            q = self.in_projector(x)

            if(self.projection_type=="single_self"):
                q=q
                k=q
                v=q
            else:
                q,k,v = q # q is tuple of q,k,v
        

            ## generate head dimension (head=1)

            #if(len(q.shape)==3):
            # B X SEQLEN X DIM -> B X SEQLEN X 1 (HEADDIM) X DIM
            q=q.unsqueeze(2)
            k=k.unsqueeze(2)
            v=v.unsqueeze(2)

           
            # B X 1 X SEQLEN X DIM (required for memory efficient attention input shape)
            q=q.contiguous().transpose(1,2)
            k=k.contiguous().transpose(1,2)
            v=v.contiguous().transpose(1,2)


            
            #print("input shape ..", q.shape, k.shape, v.shape)

           
            #res=buffer_from_jagged(q).squeeze(1)
            #print("first 3 nested before")
            #print(res[:3])
            # spliut up last (attn dim) into nhead sectors with (attn_dim/nhead) subdimensionality
            #q=q.reshape(q.shape[0], q.shape[1], 1, q.shape[-1])
            #k=k.reshape(k.shape[0], k.shape[1], self.nhead, -1)
            #v=v.reshape(v.shape[0], v.shape[1], self.nhead, -1)

            

            out=jagged_scaled_dot_product_attention(q,k,v)
            

            ### long hack because squeeze does not work yet
            out_t=buffer_from_jagged(out).squeeze(0)
            out=jagged_from_buffer(out_t, out._offsets, out._max_seqlen, out._min_seqlen)

            

            ## out projection not really needed without dropout
            if(self.do_perlayer_out_projection==1):
                out=self.out_projector(out)

           
            return self.dropout1(out), None
            ## switch to nhead


        elif(self.attn_package=="xformer"):

            q = self.in_projector(x)

            if(self.projection_type=="single_self"):
                q=q
                k=q
                v=q
            else:
                q,k,v = q # q is tuple of q,k,v
           
            # spliut up last (attn dim) into nhead sectors with (attn_dim/nhead) subdimensionality
            q=q.reshape(q.shape[0], q.shape[1], self.nhead, -1)
            k=k.reshape(k.shape[0], k.shape[1], self.nhead, -1)
            v=v.reshape(v.shape[0], v.shape[1], self.nhead, -1)


            if(query_input is not None):
                query_input=query_input.reshape(query_input.shape[0], query_input.shape[1], self.nhead, -1)

            ## probs*v

            prev_type=q.dtype
            if(self.force_sdpa_precision is not None):
                q=q.type(self.force_sdpa_precision)
                k=k.type(self.force_sdpa_precision)
                v=v.type(self.force_sdpa_precision)

                if(query_input is not None):
                    query_input=query_input.type(self.force_sdpa_precision)

            out=fmha.memory_efficient_attention(q, k, v, attn_bias=key_padding_mask, op=self.xformers_operator)

            if(query_input is not None):
               
                cross_mask = fmha.BlockDiagonalMask.from_seqlens([1 for _ in datalens], [ int(dl.item()) for dl in datalens])
                cross_mask._batch_sizes=[1 for _ in datalens]

                query_output=fmha.memory_efficient_attention(query_input, k, v, attn_bias=cross_mask, op=self.xformers_operator)
               
            if(self.force_sdpa_precision is not None):
                out=out.type(prev_type)
                query_output=query_output.type(prev_type)

            out=out.reshape(q.shape[0], q.shape[1], -1)

            ## go back to full dim
            if(query_input is not None):
                query_output=query_output.reshape(query_output.shape[0], query_output.shape[1], -1)

            ## out projection not really needed without dropout
            if(self.do_perlayer_out_projection==1):
                out=self.out_projector(out)

            return self.dropout1(out), query_output
            ## switch to nhead

        else:
            raise Exception("Unknown package ", self.attn_package)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    # feed forward block .. we never use dropout so dropout only added pro-forma
    def _ff_block_query(self, x: Tensor) -> Tensor:
        x = self.query_linear2(self.query_dropout(self.query_activation(self.query_linear1(x))))
        return x



class overall_input_projector(nn.Module):

    def __init__(self, **kwargs):

        """
        An overall input projector that can handle different token types.

        Parameters:

        input_dim (int): input dimension
        output_dim (int): output dimension
        mlp_hidden_dims (str): Hidden dim structure, i.e. "128-256" or "" for just linear mapping
        projection_type (str): Project similar into same space for q,k,v ("single_self")
                               Project differently for q,k,v with one mapping for each ("single_qkv")
                               Project with a joint mapping for q,k,v ("joint_qkv") into 3*embedding_dim space
                               
        num_token_types (int): Number of different token types, each getting its own MLP.
        """

        super(overall_input_projector, self).__init__()

        cfg_parser=config_parser.config_parser()

        cfg_parser.add_default_kwarg("settings", "input_dim", 50, int)
        cfg_parser.add_default_kwarg("settings", "output_dim", 50, int)
        cfg_parser.add_default_kwarg("settings", "mlp_hidden_dims", "", str)
        cfg_parser.add_default_kwarg("settings", "add_skip_connection", 0, int)

        cfg_parser.add_default_kwarg("settings", "num_token_types", 1, int) ## differnt number of token types as input

        _, self.settings_kwargs=cfg_parser.parse_cfg(kwargs, "settings")

        if(self.settings_kwargs["num_token_types"]<=1):
            self.projector=SkipMLP(self.settings_kwargs["input_dim"], self.settings_kwargs["mlp_hidden_dims"], self.settings_kwargs["output_dim"], add_skip_connection=self.settings_kwargs["add_skip_connection"])
        else:
            # make a projector list (one for each type)

            token_list=[]

            for cur_ind in range(self.settings_kwargs["num_token_types"]):
                token_list.append(SkipMLP(self.settings_kwargs["input_dim"], self.settings_kwargs["mlp_hidden_dims"], self.settings_kwargs["output_dim"], add_skip_connection=self.settings_kwargs["add_skip_connection"]))
            
            self.projector_list=torch.nn.ModuleList(token_list)

    def forward(self, x, token_identifiers=None):
        """
        x: The tensor of input tokens in appropriate format (flattened over all batches of size G X dim)
        token_identifiers: Used if *num_token_types* > 1. Of shape G, each token must be identified by an index starting at 0 up to *num_token_types*-1.
        """
        if(self.settings_kwargs["num_token_types"]<=1):
            ret=self.projector(x)
        else:
            assert(token_identifiers is not None)
           
            prev_shape=x.shape
            new_x=x.view(-1,x.shape[-1])
            cat_ids=torch.cat(token_identifiers)
            
            ret=torch.zeros(new_x.shape[0], self.settings_kwargs["output_dim"]).to(x)

            # loop through all token types and apply type-specific MLP
            for ind in range(self.settings_kwargs["num_token_types"]):
                
                this_mask=cat_ids==ind

                if(this_mask.sum()>0):
                    
                    ret[this_mask,:]=self.projector_list[ind](new_x[this_mask,:])

            ret=ret.view(prev_shape[0], prev_shape[1], self.settings_kwargs["output_dim"])
           
        return ret


class qkv_projector(nn.Module):

    def __init__(self, **kwargs):

        """
        Parameters:

        input_dim (int): input dimension
        output_dim (int): output dimension
        mlp_hidden_dims (str): Hidden dim structure, i.e. "128-256" or "" for just linear mapping
        projection_type (str): Project similar into same space for q,k,v ("single_self")
                               Project differently for q,k,v with one mapping for each ("single_qkv")
                               Project with a joint mapping for q,k,v ("joint_qkv") into 3*embedding_dim space
    
        """

        super(qkv_projector, self).__init__()

        cfg_parser=config_parser.config_parser()

        cfg_parser.add_default_kwarg("settings", "input_dim", 50, int)
        cfg_parser.add_default_kwarg("settings", "output_dim", 50, int)
        cfg_parser.add_default_kwarg("settings", "mlp_hidden_dims", "", str)
        cfg_parser.add_default_kwarg("settings", "projection_type", "single_self", str, choices=["single_self", "single_qkv", "joint_qkv"]) ## project to embedding space
        cfg_parser.add_default_kwarg("settings", "add_skip_connection", 0, int)

        _, self.settings_kwargs=cfg_parser.parse_cfg(kwargs, "settings")

        self.projection_type=self.settings_kwargs["projection_type"]


        if(self.projection_type=="single_self"):

            self.projector=SkipMLP(self.settings_kwargs["input_dim"], self.settings_kwargs["mlp_hidden_dims"], self.settings_kwargs["output_dim"], add_skip_connection=self.settings_kwargs["add_skip_connection"])

        elif(self.projection_type=="single_qkv"):

            self.q_projector=SkipMLP(self.settings_kwargs["input_dim"], self.settings_kwargs["mlp_hidden_dims"], self.settings_kwargs["output_dim"], add_skip_connection=self.settings_kwargs["add_skip_connection"])
            self.k_projector=SkipMLP(self.settings_kwargs["input_dim"], self.settings_kwargs["mlp_hidden_dims"], self.settings_kwargs["output_dim"], add_skip_connection=self.settings_kwargs["add_skip_connection"])
            self.v_projector=SkipMLP(self.settings_kwargs["input_dim"], self.settings_kwargs["mlp_hidden_dims"], self.settings_kwargs["output_dim"], add_skip_connection=self.settings_kwargs["add_skip_connection"])

        elif(self.projection_type=="joint_qkv"):

            self.joint_projector=SkipMLP(self.settings_kwargs["input_dim"], self.settings_kwargs["mlp_hidden_dims"], 3*self.settings_kwargs["output_dim"], add_skip_connection=self.settings_kwargs["add_skip_connection"])

        else:
            raise Exception("Hmm this should not happen, unknown projection type ... ", self.projection_type)

    def forward(self, x, split_qkv=True):

        if(self.projection_type=="single_self"):

            q=self.projector(x)

            return q

        elif(self.projection_type=="single_qkv"):

            q=self.q_projector(x)

            k=self.k_projector(x)

            v=self.v_projector(x)

            return q,k,v

        elif(self.projection_type=="joint_qkv"):

            joint=self.joint_projector(x)

            # split qkv for most applications (default)
            if(split_qkv):

                embed_dim=self.settings_kwargs["output_dim"]

                q=joint[..., :embed_dim]
                k=joint[..., embed_dim:2*embed_dim]
                v=joint[..., 2*embed_dim:3*embed_dim]
                
                return q,k,v
            else:
                # some implementations (flash_attn) require merged qkv for fastest mode
                return joint
         

class almagate_multihead_attention(nn.Module):
    def __init__(self, **kwargs):#encoder_layers=1, dropout=0.0, rnn_type="lstm", encoder_hidden_dim=10, num_encoder_mlp_layers=0, nonlinearity="tanh"):
        super(almagate_multihead_attention, self).__init__()
        

        cfg_parser=config_parser.config_parser()

        # "Token input dimension."
        cfg_parser.add_default_kwarg("settings", "input_dim", 5, int) 

        # "Final output dimension (after aggregation+final mapping)"
        cfg_parser.add_default_kwarg("settings", "output_dim", 50, int) 

        # "MLP hidden dim structure used for input/output MLPs - str with "-" separator, i.e. "64-64"
        cfg_parser.add_default_kwarg("settings", "io_mlp_hidden_dims", "128", str) 

        # add skip connection into Input/Output MLPs?
        cfg_parser.add_default_kwarg("settings", "io_add_skip_connection", 0, int) 

        ## LEGACY PARAMETER -> hould always be "single_self"
        cfg_parser.add_default_kwarg("settings", "attn_io_projection_type", "single_self", str, choices=["single_self", "single_qkv", "joint_qkv"])

        ## LEGACY PARAMETER: just here for backwards compatability (not actually used)
        cfg_parser.add_default_kwarg("settings", "io_attn_projection_type", "single_self", str, choices=["single_self", "single_qkv", "joint_qkv"])

        cfg_parser.add_default_kwarg("settings", "attn_do_perlayer_out_projection", 1, int, choices=[0,1]) # outprojection after MHA application? -> default is 1

        cfg_parser.add_default_kwarg("settings", "skip_input_projection", 0, int) ## skipping the input projection step

        # -1 means computation dim is the same as input. Cmputational dim is divided by number of heads in the usual way.
        cfg_parser.add_default_kwarg("settings", "attn_computational_dim", -1, int)
        
        # how many attention+MLP layers
        cfg_parser.add_default_kwarg("settings", "attn_num_layers", 2, int)

        # num heads
        cfg_parser.add_default_kwarg("settings", "attn_num_heads_per_layer", 1, int)

        # layer norm position 1
        cfg_parser.add_default_kwarg("settings", "attn_use_layer_norm_1", 1, int)

        # layer norm position 2
        cfg_parser.add_default_kwarg("settings", "attn_use_layer_norm_2", 1, int)

        # use extra layer norm at the end
        cfg_parser.add_default_kwarg("settings", "attn_use_extra_layer_norm", 0, int)

        # pre-layer norm
        cfg_parser.add_default_kwarg("settings", "attn_layer_norm_first", 1, int)

        # residual setting. 0: no residual connection, 1 (regular transformer residual), 2 (slightly different residual connectivity), 3 (Dual residual connections "ResiDual" )
        cfg_parser.add_default_kwarg("settings", "attn_use_residual_addition", 1, int)

        # dtype for whole transformer encoding function
        cfg_parser.add_default_kwarg("settings", "dtype", "float32", str, choices=["float64", "float32", "float16", "bfloat16"])
        
        # every mha layer usually projects linearly into qkv space ("single_kqv"). Joint qkv (without hidden units) is the same. Joint qkv *with*
        # hidden units allows to nonlinearly project into qkv space.
        cfg_parser.add_default_kwarg("settings", "attn_projection_type", "joint_qkv", str, choices=["single_self", "single_qkv", "joint_qkv"])
        
        # mlp hidden dimensionality structure for QKV projection (default = "", "128-128" would mean 2 hidden layers of 128-d each, i.e. nonlinear qkv projection.)
        cfg_parser.add_default_kwarg("settings", "attn_inprojection_mlp_dims", "", str)

        # add a skip connection to the qkv projection?
        cfg_parser.add_default_kwarg("settings", "attn_inprojection_add_skip", 0, int)

        # add difference to mean for the qkv projection?
        cfg_parser.add_default_kwarg("settings", "attn_inprojection_add_mean_diff", 0, int)

        # perform a final output mapping after all transformer layers?
        cfg_parser.add_default_kwarg("settings", "attn_perform_final_mapping", 1, int, choices=[0,1])

        # which multi-head-attention implementation to use?
        cfg_parser.add_default_kwarg("settings", "attn_package", "custom_pytorch", str, choices=["custom_pytorch", "official_pytorch_w_weights", "xformer", "geometric_scatter", "nested", "flash_attn"])

        # use dropout?
        cfg_parser.add_default_kwarg("settings", "attn_dropout", 0.1, float)

        # the intermal MLP dimensionality of the "MLP part" in a single transformer layer (Multi-head-attention followed by MLP)
        cfg_parser.add_default_kwarg("settings", "attn_internal_mlp_dim", 512, int)

        # use a weighted mean aggregation? Only works with aggregation mode "mean"
        cfg_parser.add_default_kwarg("settings", "attn_use_weighted_mean", 0, int)

        ## aggreagtion mode? Only used if no "class token is used, which would be read out instead". 
        ## mean -> mean
        ## mean_n_diagonal + concat mean + diagonal variance of tokens
        ## mean_add_absolute -> concat mean + absolute token sum
        ## mean_n_diagonal_add_absolute -> concat mean+absolute token sum, then add diagonal variance 
        cfg_parser.add_default_kwarg("settings", "attn_aggregation_mode", "mean", str, choices=["mean", "mean_n_diagonal", "mean_add_absolute", "mean_n_diagonal_add_absolute"])
        
        ## computational class token: an auxiliary token which in the end is read out only (no aggregation anymore)
        ## 0 -> no such token (normal aggregation, e.g. "mean", defined by attn_aggregation_mode)
        ## 1 -> prepend extra auxiliary token, everything the same
        ## 2 -> first layer is just bias (no bias+projection learned for class token in first layer)
        ## 3 -> cross attention (token is readout only (cross attention), and only bias in first layer)
        ## remark: in combiatnion with "dual residual connection", cross attention token gets its own similar dual residual stream 
        cfg_parser.add_default_kwarg("settings", "attn_use_computational_class_token", 0, int, choices=[0,1,2,3]) 
        
        # NOT USED (xformer specific operators)
        cfg_parser.add_default_kwarg("settings", "attn_operator", "none", str)

        ## Use positional encoding? 
        # -1 -> no
        # positive integer n -> used first *n* dimensions of token for positional encoding 
        cfg_parser.add_default_kwarg("settings", "attn_original_input_position_feature_number", -1, int)

        # define a range for positional indices (CURRENTLY NOT USED/NO OP, first indices must encode positional features)
        cfg_parser.add_default_kwarg("settings", "attn_original_input_position_feature_range", "", str)
        
        # in each layer concatenate original token input to current token representation?
        cfg_parser.add_default_kwarg("settings", "attn_add_original_input_to_feature_input", 0, int, choices=[0,1])
        
        ## absolute position encoding mode
        # none
        # sinusodial -> default
        # roformer (not supported currently)
        cfg_parser.add_default_kwarg("settings", "attn_abs_position_mode", "none", str, choices=["none", "sinusoidal", "roformer"])
        
        # which layers to apply positional encoding on?
        # -1 -> all layers
        # integers separated by ",", i.e. 0 (only first layer) or 0,1,2 (first three layers)
        cfg_parser.add_default_kwarg("settings", "attn_abs_position_layer_indices", "0", str)

        # relative positional encoding options - NOT USED
        cfg_parser.add_default_kwarg("settings", "attn_rel_position_encoding", "feature_mlp", str, choices=["feature_mlp"])

        # indices of transformer layers which apply relative positional encodings
        # e.g. 0,2,3 would apply rel positional encoding in layer 0,2,3. Also allowed e.g.: 0i,1v,2
        # where "i" after an index overwrites rel_position feeding type to "0", while "v" will ovrwrite input
        # feeding type to 2. See below for definition of input feeding type
        cfg_parser.add_default_kwarg("settings", "attn_rel_position_layer_indices", "0", str) # layer indices of layers that use relative positioning (-1 (all), comma separated indices, e.g. "1,2")
        
        # How relative positional encodings work.
        # 0: take absolute valus + differences from previous layer
        # 1: take absolute values from previous layer, diffs from original input
        # 2: take absolute values + difference from original input 
        cfg_parser.add_default_kwarg("settings", "attn_rel_position_input_feeding_type", 0, int, choices=[0,1,2]) 
        
        # which rel position mode? 
        # none -> no "value" relative positional encoding
        # rel_only -> only value relative positional encoding, but no ordinary value updating
        # both -> add both ordinary attention + value relative positional output
        cfg_parser.add_default_kwarg("settings", "attn_rel_position_mode_value", "none", str, choices=["none", "only_rel", "both"])#, "split_in_heads_2"])
        
        # hidden dimensionality of the relative position MLP
        cfg_parser.add_default_kwarg("settings", "attn_rel_position_hidden_dim", 64, int)#, "split_in_heads_2"])
        
        # use skip connection in relative positional mlp?
        cfg_parser.add_default_kwarg("settings", "attn_rel_position_use_skip_connection", 1, int)
        
        # add relative positional encoding as a "parallel computational track
        # to a normal transformer block and result afterwards (essentially two parallel MHA passes)
        cfg_parser.add_default_kwarg("settings", "attn_rel_position_as_parallel_to_normal_track", 0, int)

        # the computational dim for relative positional encoding.. only used when used as "parallel" track
        cfg_parser.add_default_kwarg("settings", "attn_rel_position_max_computational_dim", 100, int) 

        # force a specific precision for the multi-head-attention (SDPA - scaled dot product attention) part
        # -> typically used for relative positional encoding, which requires fast bfloat16
        cfg_parser.add_default_kwarg("settings", "force_sdpa_precision", "", str, choices=["", "float16", "bfloat16", "float32"])

        # how many different "token types" in input? For distinct types, use distinct embedding input mappings in the very beginning. 
        cfg_parser.add_default_kwarg("settings", "attn_num_token_types", 1, int) 

        
        settings_args, settings_kwargs=cfg_parser.parse_cfg(kwargs, "settings", check_passed_params_are_configured=True)

        for k in settings_kwargs:
            #print(k, settings_kwargs[k])
            if(k!="dtype"):
                setattr(self, k, settings_kwargſ[k])

        ## modify num token types if using class token
        if(settings_kwargs["attn_use_computational_class_token"]>0):

            ## increase number of "token types" by 1 if using class token
            if(settings_kwargs["attn_use_computational_class_token"]<3): # 1/2 -> handle class token just as additional token
                self.attn_num_token_types+=1

                ## 1/3 -> zero vector
                ## 2/4 -> randomized (trainable) parameters
                if(settings_kwargs["attn_use_computational_class_token"]==1):
                    self.class_token_input=torch.zeros(1,1,self.input_dim)
                else:
                    self.class_token_input=torch.nn.Parameter(torch.randn(1,1,self.input_dim))
            
        # -1 means full feature vec
        if(self.attn_original_input_position_feature_number==-1):
            self.attn_original_input_position_feature_number=None
            assert(self.attn_abs_position_mode=="none"), "If you use absolute position encoding.. you must define the *positional dim dimensionality* via *attn_original_input_position_feature_number*!"

        if(self.attn_abs_position_mode!="none"):
            assert(self.attn_original_input_position_feature_range !=""), "absolute positional encoding is switched on, but no position range is defined.. define *attn_original_input_position_feature_range* as string *low_high*!"
            test_range=self.attn_original_input_position_feature_range.split("_")
            
            assert(len(test_range)==2), "Define range as *low_high*"


        #print("------------------------")

        if(self.attn_operator is not None):
            if(self.attn_operator=="None" or self.attn_operator=="none"):
                self.attn_operator=None

        ## size B X NUM ITEMS X INPUT DIM
        #self.h0=nn.Parameter(torch.randn((1, 1, self.input_dim)))

        if(self.attn_computational_dim==-1):
            self.attn_computational_dim=self.output_dim

            if(self.attn_use_weighted_mean):
                self.attn_computational_dim+=1

        if(self.attn_aggregation_mode!="mean"):
            assert(self.attn_use_weighted_mean==False)
            assert(self.attn_computational_dim!=-1)
            #assert(self.attn_perform_final_mapping)


        self.attn_dtype=settings_kwargs["dtype"]

        if(self.attn_dtype=="float64"):
            self.attn_dtype=torch.float64
        elif(self.attn_dtype=="float32"):
            self.attn_dtype=torch.float32
        elif(self.attn_dtype=="float16"):
            self.attn_dtype=torch.float16
        elif(self.attn_dtype=="bfloat16"):
            self.attn_dtype=torch.bfloat16

        assert(self.attn_computational_dim%self.attn_num_heads_per_layer == 0), ("Embedding / Computational dim must be divisible by number of attention heads!", self.attn_computational_dim, self.attn_num_heads_per_layer)


        ## first and last layer use similar transformer implementation by default
        self.attn_package_first=self.attn_package
        self.attn_package_last=self.attn_package

        if(settings_kwargs["skip_input_projection"]):
            ## we skip the input projeciton .. dimensions must match
            assert(self.attn_computational_dim==self.input_dim) 

            self.input_projector=lambda x: x
        else:

            ## TODO: dont need a qkv projector here
            self.input_projector=overall_input_projector(input_dim=self.input_dim, 
                                               output_dim=self.attn_computational_dim, 
                                               mlp_hidden_dims=settings_kwargs["io_mlp_hidden_dims"],
                                               num_token_types=settings_kwargs["attn_num_token_types"],
                                               #projection_type=settings_kwargs["attn_io_projection_type"],
                                               dtype=self.attn_dtype,
                                               add_skip_connection=settings_kwargs["io_add_skip_connection"]
                                               )
        ## use official pytorch encoder layer if parameters are right
        use_official_encoder=False
        if(("custom_pytorch" in self.attn_package) 
            and (self.attn_use_layer_norm_1==1)
            and (self.attn_use_layer_norm_2==1)
            and (self.attn_use_residual_addition==1)
            and self.attn_projection_type=="joint_qkv"
            and self.attn_rel_position_mode_value=="none"
            and self.attn_do_perlayer_out_projection==1
            and self.attn_abs_position_mode=="none"
            and self.attn_add_original_input_to_feature_input==0
            and self.attn_inprojection_mlp_dims==""
            and self.attn_inprojection_add_skip==0
            and self.attn_inprojection_add_mean_diff==0
            and self.attn_use_computational_class_token==0):

           
            encoder_layer = nn.TransformerEncoderLayer(self.attn_computational_dim, 
                                                           self.attn_num_heads_per_layer, 
                                                           dim_feedforward=self.attn_internal_mlp_dim, 
                                                           dropout=self.attn_dropout,
                                                            batch_first=True,
                                                            dtype=self.attn_dtype,
                                                            norm_first=self.attn_layer_norm_first)

            use_official_encoder=True

        elif(("pytorch" in self.attn_package) and self.attn_rel_position_mode_value=="none" and self.attn_use_computational_class_token!=3):
            encoder_layer = CustomTransformerEncoderLayer(self.attn_computational_dim, 
                                                          self.attn_num_heads_per_layer, 
                                                          dim_feedforward=self.attn_internal_mlp_dim, 
                                                          dropout=self.attn_dropout,
                                                          norm_first=self.attn_layer_norm_first,
                                                          use_layer_norm_1=self.attn_use_layer_norm_1,
                                                          use_layer_norm_2=self.attn_use_layer_norm_2,
                                                          use_residual_addition=self.attn_use_residual_addition,
                                                          attn_package=settings_kwargs["attn_package"], # pytorch / xformers / flash
                                                          projection_hidden_dims=self.attn_inprojection_mlp_dims,
                                                          projection_add_skip_connection=self.attn_inprojection_add_skip,
                                                          projection_type=settings_kwargs["attn_projection_type"],
                                                          projection_add_mean_diff=self.attn_inprojection_add_mean_diff,
                                                          dtype=self.attn_dtype,
                                                          xformers_operator=self.attn_operator,
                                                          do_perlayer_out_projection=self.attn_do_perlayer_out_projection)
        
        else:
            encoder_layer=None
        
        
        self.extra_layer_norm=None
        if(self.attn_use_extra_layer_norm):
            self.extra_layer_norm=nn.LayerNorm(self.attn_computational_dim, dtype=self.attn_dtype)
        

        if(use_official_encoder):
            assert(encoder_layer is not None)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.attn_num_layers, norm=self.extra_layer_norm)
        else:

            # we use relative positional encodings in some layers.. have to define how
            base_encoder_args=[self.attn_computational_dim, self.attn_num_heads_per_layer]
            base_encoder_kwargs=dict()
            base_encoder_kwargs["dim_feedforward"]=self.attn_internal_mlp_dim
            base_encoder_kwargs["dropout"]=self.attn_dropout
            base_encoder_kwargs["norm_first"]=self.attn_layer_norm_first
            base_encoder_kwargs["use_layer_norm_1"]=self.attn_use_layer_norm_1
            base_encoder_kwargs["use_layer_norm_2"]=self.attn_use_layer_norm_2
            base_encoder_kwargs["use_residual_addition"]=self.attn_use_residual_addition
            base_encoder_kwargs["attn_package"]=settings_kwargs["attn_package"]
            base_encoder_kwargs["projection_hidden_dims"]=self.attn_inprojection_mlp_dims
            base_encoder_kwargs["projection_add_skip_connection"]=self.attn_inprojection_add_skip
            base_encoder_kwargs["projection_add_mean_diff"]=self.attn_inprojection_add_mean_diff

            base_encoder_kwargs["projection_type"]=settings_kwargs["attn_projection_type"]
            base_encoder_kwargs["dtype"]=self.attn_dtype
            base_encoder_kwargs["xformers_operator"]=self.attn_operator
            base_encoder_kwargs["do_perlayer_out_projection"]=self.attn_do_perlayer_out_projection

            ## general information about original input
            base_encoder_kwargs["original_input_feature_dim"]=self.input_dim
            base_encoder_kwargs["original_input_position_feature_number"]=self.attn_original_input_position_feature_number
            base_encoder_kwargs["add_original_input_to_feature_input"]=self.attn_add_original_input_to_feature_input

            ## absolute positional encoding
            base_encoder_kwargs["abs_position_mode"]=self.attn_abs_position_mode
            
            ## relative positional encoding
            base_encoder_kwargs["rel_position_mode_value"]=self.attn_rel_position_mode_value
            base_encoder_kwargs["rel_position_input_feeding_type"]=self.attn_rel_position_input_feeding_type
            base_encoder_kwargs["rel_pos_mlp_hidden_dims"]="%d" % self.attn_rel_position_hidden_dim
            base_encoder_kwargs["rel_pos_mlp_use_skip_connection"]=self.attn_rel_position_use_skip_connection


            base_encoder_kwargs["force_sdpa_precision"]=None if self.force_sdpa_precision == "" else self.force_sdpa_precision

            ## only use global query input if class_token option=3 (cross attention via query)
            base_encoder_kwargs["use_query_class_token"]=self.attn_use_computational_class_token==3

            self.transformer_encoder = CustomTransformerEncoder(base_encoder_args,
                                                                base_encoder_kwargs, 
                                                                self.attn_num_layers,
                                                                self.input_dim, 
                                                                extra_layer_norm=self.extra_layer_norm, 
                                                                rel_position_layer_indices=self.attn_rel_position_layer_indices,
                                                                abs_position_layer_indices=self.attn_abs_position_layer_indices,
                                                                abs_position_range=self.attn_original_input_position_feature_range,
                                                                #abs_position_scale=self.attn_abs_position_scale,
                                                                rel_position_as_parallel_to_normal_track=self.attn_rel_position_as_parallel_to_normal_track,
                                                                rel_position_max_computational_dim=self.attn_rel_position_max_computational_dim,
                                                                use_global_query_input=self.attn_use_computational_class_token==3)
            

            ## only do anything about the encdoing structure if we have positional encoding enabled!
            if(self.attn_rel_position_mode_value!="none"):
                assert(self.attn_rel_position_layer_indices!=""), "Chose relative positional encoding for value, but no indices for layers defined!"
                
                for li in self.attn_rel_position_layer_indices.split(","):
                    if(li[-1]=="v" or li[-1]=="i"):
                        cur_li=int(li[:-1])
                    else:
                        cur_li=int(li)

                    #list_indices=[int(i) for i in self.attn_rel_position_layer_indices.split(",")]
                    if(cur_li==0):
                        ## first one is pytorch 
                        if("pytorch" in self.attn_package):
                            self.attn_package_first="official_pytorch_w_weights"
                        else:
                            self.attn_package_first="flash_attn"

                    if(cur_li==(self.attn_num_layers-1)):
                        if("pytorch" in self.attn_package):
                            self.attn_package_last="official_pytorch_w_weights"
                        else:
                            self.attn_package_last="flash_attn"

           
            """
            absolute_input_dim,
             norm=None, 
             enable_nested_tensor=True, 
             mask_check=True, 
             rel_position_layer_indices="0",
             rel_position_mode_value="none",
             rel_position_input_feeding_type=0attn_aggregation_mode
            """

        if(settings_kwargs["attn_perform_final_mapping"]==1):
            if(settings_kwargs["attn_computational_dim"]!=-1):
                aggregation_dim=self.attn_computational_dim

                if(self.attn_use_computational_class_token==0):
                    if(self.attn_aggregation_mode=="mean"):
                        if(self.attn_use_weighted_mean):
                            aggregation_dim-=1
                    else:
                        if("n_diagonal" in self.attn_aggregation_mode):
                            aggregation_dim*=2

                        if("add_absolute" in self.attn_aggregation_mode):
                            aggregation_dim*=2
                

                self.attention_to_output_mlp=SkipMLP(aggregation_dim, settings_kwargs["io_mlp_hidden_dims"], self.output_dim, dtype=self.attn_dtype, add_skip_connection=settings_kwargs["io_add_skip_connection"])

            else:

                if(self.attn_use_weighted_mean):
                    aggregation_dim=self.attn_computational_dim-1
                    ## a linear mapping to the output space...
                    self.attention_to_output_mlp=torch.nn.Linear(aggregation_dim, self.output_dim, dtype=self.attn_dtype)
                else:
                    self.attention_to_output_mlp= lambda x: x
        else:
            ## not output mapping mlp
            self.attention_to_output_mlp=lambda x: x


    def _create_pytorch_datarep_and_mask(self, datavecs, datalens):

        data=[]

        maxlen=max(datalens)
        input_dim=datavecs[0].shape[-1]

        if( (maxlen==datalens).sum()<len(datalens)):
            for ind, dvec in enumerate(datavecs):

                assert(len(dvec.shape)==3)

                diff=maxlen-dvec.shape[1]

                if(diff>0):
                    data.append(torch.cat([ dvec, torch.zeros((1,diff, input_dim)).to(dvec) ] , dim=1) )
                else:
                    data.append(dvec)

            
            data=torch.cat(data)

            padding_mask=torch.zeros_like(data[:,:,0]).to(data[0])

            for b_ind, datalen in enumerate(datalens):
                padding_mask[b_ind,datalen:]=1

            padding_mask=padding_mask.type(torch.bool)
            
        else:

            if(type(datavecs)==torch.Tensor or type(datavecs)==torch.nn.parameter.Parameter):
                data=datavecs
            elif(type(datavecs)==list):
                data=torch.cat(datavecs)
            else:
                raise Exception("Unknown datavecs format", datavecs, type(datavecs))

            
            padding_mask=None

        return data, padding_mask

    def _create_xformers_datarep_and_mask(self, datavecs, datalens):

        if(type(datavecs)==torch.Tensor):
            ## TODO: change this to directly use tensor
            maxlen=max(datalens)

            ## all datalens must be the same
            assert((maxlen==datalens).sum()==len(datalens))

            data=[vec.unsqueeze(0) for vec in datavecs]
        
        else:
            assert(type(datavecs)==list), "Datavecs must be a list of tensors"

            data=datavecs

       
        attn_bias, x = fmha.BlockDiagonalMask.from_tensor_list(data)

        return x, attn_bias

    def _create_nested_datarep_and_mask(self, datavecs, datalens):

        if(type(datavecs)==torch.Tensor):
            ## TODO: change this to directly use tensor
            maxlen=max(datalens)

            ## all datalens must be the same
            assert((maxlen==datalens).sum()==len(datalens))
            nested_tensor=torch.nested.nested_tensor([i for i in datavecs], layout=torch.jagged)
           
        
        else:
            assert(type(datavecs)==list), "Datavecs must be a list of tensors"

            data=[i.squeeze(0) for i in datavecs]
            #nested_tensor,_=jagged_from_list(data, datalens)
            nested_tensor=torch.nested.nested_tensor(data, layout=torch.jagged)
          
        return nested_tensor, None

    def _create_datarep_and_mask(self, datavecs, datalens, nhead, attn_package=None, relative_encoding_indices=[]):

        """
        Returns:
        input_vecs: Depending on attn package, can be list of vecs or a padded tensor.
        """
            
        assert(type(datalens)==torch.Tensor), "Require datalens as tensor"
        
        default_attn_package=self.attn_package
        
        if(self.attn_package_first==self.attn_package_last):
            if("pytorch" in self.attn_package_first):
                data, mask = self._create_pytorch_datarep_and_mask(datavecs, datalens)
            elif(("xformer" in self.attn_package_first) or ("geometric_scatter" in self.attn_package_first) or ("flash_attn" in self.attn_package_first)):
                data, mask = self._create_xformers_datarep_and_mask(datavecs, datalens)
            else:
                data, mask = self._create_nested_datarep_and_mask(datavecs, datalens)

            return data, mask, mask
        else:
            if("pytorch" in self.attn_package_first):
                data, mask_first = self._create_pytorch_datarep_and_mask(datavecs, datalens)
            elif(("xformer" in self.attn_package_first) or ("geometric_scatter" in self.attn_package_first) or ("flash_attn" in self.attn_package_first)):
                data, mask_first = self._create_xformers_datarep_and_mask(datavecs, datalens)
            else:
                data, mask_first = self._create_nested_datarep_and_mask(datavecs, datalens)


            if("pytorch" in self.attn_package_last):
                _, mask_last = self._create_pytorch_datarep_and_mask(datavecs, datalens)
            elif(("xformer" in self.attn_package_last) or ("geometric_scatter" in self.attn_package_last) or ("flash_attn" in self.attn_package_last)):
                _, mask_last = self._create_xformers_datarep_and_mask(datavecs, datalens)
            else:
                _, mask_last = self._create_nested_datarep_and_mask(datavecs, datalens)


            return data, mask_first, mask_last
        

    def _final_summation(self, result_matrix, mask, datalens, attn_package=None, add_weights=False, perform_final_aggregation=True, aggregation_mode="mean"):

        used_attn_package=self.attn_package
        if(attn_package is not None):
            used_attn_package=attn_package

        if("pytorch" in used_attn_package):

            if(self.attn_use_computational_class_token>0):
                if(self.attn_use_computational_class_token<3):
                    return result_matrix[:,0,:]
                else:
                    raise NotImplementedError()
            
            if(add_weights):
                raise Exception("Weighting not implemented for pytorch .. implement it!")

            if(perform_final_aggregation==False):
                raise Exception("Pytorch impl for non-aggregation still needs to be checked!")

            if(mask is not None):

                attn_result=result_matrix*((~mask).type(torch.int).unsqueeze(-1))
            else:
                attn_result=result_matrix

            if( ("mean" in aggregation_mode) and (not "diagonal" in aggregation_mode)):
                #print("attn result pytorch", attn_result[0])

                abs_sum=attn_result.sum(axis=1)
                attn_result=abs_sum/datalens.unsqueeze(-1)

                if("add_absolute" in aggregation_mode):
                    attn_result=torch.cat([attn_result, abs_sum], dim=-1)

            else:
                abs_sum=attn_result.sum(axis=1)
                means=abs_sum/datalens.unsqueeze(-1)

                if(mask is not None):
                    abs_variances=( ((means[:,None,:]-attn_result)**2)*((~mask).type(torch.int).unsqueeze(-1)) ).sum(dim=1)
                else:
                    abs_variances=( ((means[:,None,:]-attn_result)**2) ).sum(dim=1)
                variances=abs_variances/datalens.unsqueeze(-1)

                attn_result=torch.cat([means, variances], dim=-1)

                if("add_absolute" in aggregation_mode):
                    attn_result=torch.cat([attn_result, abs_sum, abs_variances], dim=-1)
           
            return attn_result
        if("nested" in used_attn_package):

            assert(not "add_absolute" in aggregation_mode), "add absolute not supported in nested"
            if(add_weights):
                raise Exception("Weighting not implemented for pytorch .. implement it!")

            if(perform_final_aggregation==False):
                raise Exception("Pytorch impl for non-aggregation still needs to be checked!")


            if(aggregation_mode=="mean"):
                #res=torch.nested.to_padded_tesresult_matrix.to_padded(1)
                #print(result_matrix.shape)
                # first is head dim, and we only have 1 head for now
                res=buffer_from_jagged(result_matrix).squeeze(1)
              

                indices=torch.repeat_interleave(torch.arange(len(datalens)).to(device=datalens.device), datalens)
                
                attn_result=scatter_mean(res, indices, dim=0)
               
            else:
                
                sys.exit(-1)

                variances=((means[:,None,:]-attn_result)**2).mean(dim=1)
                attn_result=torch.cat([means, variances], dim=-1)
           
            return attn_result

        elif(used_attn_package=="xformer" or used_attn_package=="geometric_scatter" or used_attn_package=="flash_attn"):

            out=mask.split(result_matrix)

            if(self.attn_use_computational_class_token>0):
                if(self.attn_use_computational_class_token<3):
                    return torch.cat([o[:,0,:] for o in out], dim=0)
                else:
                    raise NotImplementedError()
         
            if(perform_final_aggregation==False):
                # return list of
                return out

            if(add_weights):

                #weights=[torch.nn.functional.softmax(i[:,:,-1:], dim=1) for i in out]
                
                attn_result=torch.cat([ (torch.nn.functional.softmax(i[:,:,-1:], dim=1)*i[:,:,:-1] ).sum(dim=1) for i in out])

            else:

                if( ("mean" in aggregation_mode) and (not "diagonal" in aggregation_mode)):
                    
                    if("add_absolute" in aggregation_mode):
                        attn_result=torch.cat([ torch.cat([i.mean(dim=1), i.sum(dim=1) ],dim=-1) for i in out])
                    else:
                        attn_result=torch.cat([i.mean(dim=1) for i in out])
                else:
                    means_list=[i.mean(dim=1) for i in out]

                    variances=[ ((means_list[ind][:,None,:]-out[ind])**2).mean(dim=1) for ind in range(len(out))]
                    variances=torch.cat(variances)
                    means=torch.cat(means_list)

                    attn_result=torch.cat([means, variances], dim=-1)

                    if("add_absolute" in aggregation_mode):
                        sums=torch.cat([i.sum(dim=1) for i in out])
                        abs_variances=torch.cat([ ((means_list[ind][:,None,:]-out[ind])**2).sum(dim=1) for ind in range(len(out))])
                        
                        attn_result=torch.cat([attn_result, sums, abs_variances], dim=-1)

            return attn_result

        else:
            raise Exception("Unsupported package for aggregation", used_attn_package)

    def forward(self, 
                datavecs, 
                datalens, 
                perform_final_aggregation=True, 
                perform_final_mapping=True, 
                token_identifiers=None):

        """
        datavecs (Tensor/list of tensors): input batch
        datalens (Tensor): length per batch
        perform_final_aggregation (bool): Aggregate at the end?
        perform_final_mapping (bool): Perform the final mapping?
        token_identifiers (None/integer array): An array indicating if every item is of the same class.
        """

        ## add a "class" token in the beginning with an additional token identifier
        if(self.attn_use_computational_class_token>0):
            ## add the token either as 0-vec (and then use a simple bias in the projector) (case 1), or as a full vec
            ## and use a normal projector matrix+bias (case 2)
            ## 3/4 is then using cross attention through the whole forward pass, but same as 1/2 othewise
            ## extend datavecs/datalens
            
            ## modify datavecs
            if(self.attn_use_computational_class_token<3):

                if(type(datavecs)==list):
                    datavecs=[ torch.cat([self.class_token_input.to(dv), dv], dim=1) for dv in datavecs]
                else:
                    # B X L X D
                    datavecs=torch.cat([self.class_token_input.to(datavecs).repeat(datavecs.shape[0],1,1), datavecs], dim=1)

                ## add/extend token identifiers
                if(token_identifiers is None):
                    ## 
                    assert(self.attn_num_token_types==2), (self.attn_num_token_types, "token identifiers is None, but num token types is larger than 2 and class token is used... you should pass token_identifiers from dataloading!")
                    ## make a list of token identifiers
                    token_identifiers=[torch.LongTensor([1]+dl*[0]).to(device=datavecs[0].device) for dl in datalens]

                else:
                    
                    token_identifiers=[torch.cat([torch.Tensor([self.attn_num_token_types-1]).to(ti), ti]) for ti in token_identifiers]

                # modify datalens
                datalens=datalens+1

            

        # package-dependent processing
        
        # data and datalen preparation dependent on package
        padded_tensor_first, padding_mask_first, padding_mask_last=self._create_datarep_and_mask(datavecs, datalens, self.attn_num_heads_per_layer)
        
        """
        print("types forward ...---------_> ")
        print(type(padded_tensor_first))
        print(type(padding_mask_first))
        print(type(padding_mask_last))
        print("padding mask first")
        print(padding_mask_first)
        print(self.attn_package_first, self.attn_package_last)
        print(self.attn_rel_position_layer_indices)
        print("---------------------------")
        """

        used_token_identifiers=token_identifiers
        if("pytorch" in self.attn_package_first and token_identifiers is not None):
            ## modulate token identifiers to work with slow pytorch masking
            if(padding_mask_first is not None):
                assert(type(padding_mask_first)==torch.Tensor), type(padding_mask_first)

            used_token_identifiers=[torch.zeros(padded_tensor_first.shape[1]).to(token_identifiers[0]) for ii in range(padded_tensor_first.shape[0])]
            
            for ind in range(len(used_token_identifiers)):
                used_token_identifiers[ind][:datalens[ind]]=token_identifiers[ind]

        # input projection D -> C (computational dim)
        computational_input=self.input_projector(padded_tensor_first, token_identifiers=used_token_identifiers)
        
        # transformer layers

        if(self.attn_rel_position_mode_value=="none" 
            and self.attn_abs_position_mode=="none" 
            and self.attn_add_original_input_to_feature_input==0
            and self.attn_inprojection_add_mean_diff==0
            and self.attn_use_computational_class_token<3):
            ## in non-relative mode we take potentially standard transformer and do not pass global_relative_input
            result=self.transformer_encoder(computational_input, src_key_padding_mask=padding_mask_first)
        else:

            positional_features=None

            ## use positional features for absolute position encoding
            if(self.attn_abs_position_mode!="none"):
                positional_features=padded_tensor_first
                if(self.attn_original_input_position_feature_number!=-1):
                    positional_features=positional_features[..., :self.attn_original_input_position_feature_number]

            ## add original feature input also if desired
            #original_feature_input=None
            #if(self.attn_add_original_input_to_feature_input):

            ## always add original_feature_input for relative positional encoding and/or adding original input at every stage
            original_feature_input=padded_tensor_first

            result=self.transformer_encoder(computational_input, 
                                            src_key_padding_mask=padding_mask_first, 
                                            original_input_feature_vecs=original_feature_input, 
                                            positional_features=positional_features,
                                            datalens=datalens)

        if(self.attn_use_computational_class_token>0):
            assert(self.attn_use_weighted_mean==0), "Weighted mean only supported in standard aggregation!"

            #print(self.attn_use_computational_class_token)

            #if(self.attn_use_computational_class_token<3):

            if("pytorch" in self.attn_package_last):
                #print(self.attn_package_last)
                #print("SHAPE....", result.shape)
                result=result[:,0,:]
            else:

                if(self.attn_use_computational_class_token==3):
                    result=result[0]
                else:
                    result=padding_mask_last.split(result)
                    result=torch.cat([r[0,0:1] for r in result])
            
           

        else:
            result=self._final_summation(result, padding_mask_last,datalens, attn_package=self.attn_package_last, add_weights=self.attn_use_weighted_mean, perform_final_aggregation=perform_final_aggregation, aggregation_mode=self.attn_aggregation_mode)


        if(perform_final_mapping==False):
            return result

       
        #assert(self.attention_to_output_mlp is not None), "Choose to perform final mapping, but attention_to_output_mlp is None... have to define Encoder with that flag on!"
        ret_val = self.attention_to_output_mlp(result)
        
        return ret_val

        
