from .. import config_parser
from .. import helper_fns
from jammy_flows.amortizable_mlp import AmortizableMLP

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy

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

import subprocess

try:
    import xformers
except:
        print("cannot import xformers.. install package via pip!")
else:
    globals()["xformers"] = xformers
    from xformers.ops import fmha#, MemoryEfficientAttentionFlashAttentionOp,MemoryEfficientAttentionOp,TritonFlashAttentionOp

from .skip_mlp import SkipMLP


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_tensor_of_diffs_flattened(list_of_tensors, element_list=None):
    ## B X L X DIM

    if(type(list_of_tensors)==torch.Tensor):
        # this is a padded tensor.. element list must be given
        assert(element_list is not None)
        input_dim=list_of_tensors.shape[-1]
        distances_flattened=torch.cat([ (t[None, :element_list[tind],None,:]-t[None, None,:element_list[tind],:]).reshape(-1,input_dim) for tind,t in enumerate(list_of_tensors)])

    else:
        assert(len(list_of_tensors[0].shape)==3)

        input_dim=list_of_tensors[0].shape[-1]
        element_list=[t.shape[1] for t in list_of_tensors]
        distances_flattened=torch.cat([ (t[:, :,None,:]-t[:, None,:,:]).reshape(-1,input_dim) for t in list_of_tensors])

    
    return distances_flattened, element_list

def _recreate_list_of_matrices(flat_matrixarray, side_lengths):

    input_dim=flat_matrixarray.shape[-1]

    spl=torch.split(flat_matrixarray, split_size_or_sections=[s**2 for s in side_lengths])
    return [t.reshape(1,side_lengths[ind],side_lengths[ind],input_dim) for ind,t in enumerate(spl)]

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
                 encoder_layer, 
                 num_layers, 
                 norm=None, 
                 enable_nested_tensor=True, 
                 mask_check=True, 
                 rel_position_mode_value="none",
                 rel_position_feeding_type=0):

        super().__init__()
       
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
        self.rel_position_mode_value=rel_position_mode_value
        self.rel_position_feeding_type=rel_position_feeding_type
        
        enc_layer = "encoder_layer"
  

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
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
                
        rel_position_tensor=None
        if(self.rel_position_mode_value!="none"):
            rel_position_tensor=None
            ## make relative position base input

            element_list=[src.shape[1] for i in range(src.shape[0])]
            if(src_key_padding_mask is not None):
                element_list=[int(sum(mitem==0).cpu().detach()) for mitem in src_key_padding_mask]

            flattened_distances, used_elements=_get_tensor_of_diffs_flattened(src, element_list=element_list)
            
            

        output = src
        
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

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
                 use_residual_addition=True,
                 attn_package="pytorch", # pytorch / xformers / flash
                 projection_hidden_dims="",
                 projection_type="single_self",
                 xformers_operator=None,
                 rel_position_mode_value="none",
                 rel_position_input_feeding_type=0,
                 do_perlayer_out_projection=1) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(CustomTransformerEncoderLayer, self).__init__()

        self.in_projector=qkv_projector(input_dim=embed_dim, 
                                        mlp_hidden_dims=projection_hidden_dims, 
                                        output_dim=embed_dim,
                                        projection_type=projection_type)

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
                                        mlp_hidden_dims=projection_hidden_dims, 
                                        output_dim=embed_dim,
                                        projection_type="single_self")
        else:
            assert(use_residual_addition!=1), "Residual addition of 1 (default transformer) requires out projection.. otherwise it is too restricting"

        # Implementation of Feedforward model
        self.linear1 = Linear(embed_dim, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, embed_dim, **factory_kwargs)

        self.norm_first = norm_first


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

        if(self.attn_package!="official_pytorch_w_weights"):
            assert(self.rel_position_mode_value=="none"), "Relative position mode currently only supported in *official_pytorch_w_weights* mode!"

        if(self.rel_position_mode_value!="none"):
            ## 
            ### value_mlp_1 "concat_value_to_diff", "split_in_heads_2"
            if(self.rel_position_mode_value=="concat_abs_and_rel_and_add"):
                if(self.rel_position_input_feeding_type==0):## feed value embed dim + diffs of value

                    self.rel_pos_value_projector=torch.nn.Linear(2*embed_dim, embed_dim)
                elif(self.rel_position_input_feeding_type==1):
                    raise Exception()
                else:
                    raise NotImplementedError()
        
    def __setstate__(self, state):
        super(CustomTransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, is_causal=False,
                src_key_padding_mask: Optional[Tensor] = None,
                rel_pos_tensor = None) -> Tensor:
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

        x = src

        ## project relative position tensor to local embedding

        if(self.rel_position_mode_value!="none"):
            assert(rel_pos_tensor is not None)
       
        ## only use residual addition on SA block?
        if(self.use_residual_addition==2):
            if self.norm_first:
                
                sa_result = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, orig_rel_pos_tensor=rel_pos_tensor)
                x = x + self._ff_block(self.norm2(sa_result))
              
            else:
                sa_result = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, orig_rel_pos_tensor=rel_pos_tensor))
                
                x = self.norm2(x + self._ff_block(sa_result))

        ## DEFAULT -- both SA and FF blocks use residual addition
        elif(self.use_residual_addition==1):
            if self.norm_first:
                
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, orig_rel_pos_tensor=rel_pos_tensor)
               
                x = x + self._ff_block(self.norm2(x))
              
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, orig_rel_pos_tensor=rel_pos_tensor))
                
                x = self.norm2(x + self._ff_block(x))
        elif(self.use_residual_addition==0):
            if self.norm_first:
                x = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, orig_rel_pos_tensor=rel_pos_tensor)
                x = self._ff_block(self.norm2(x))
            else:
                x = self.norm1(self._sa_block(x, src_mask, src_key_padding_mask, orig_rel_pos_tensor=rel_pos_tensor))
                x = self.norm2(self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], 
                  key_padding_mask: Optional[Tensor],
                  orig_rel_pos_tensor=None) -> Tensor:

        assert(attn_mask is None), "No support for causal masks right now."
        ## in projection

        ## Q, k, V shape: B X num_length X Num_heads*dim_per_head
        batch_size=x.shape[0]
        max_num_samples=x.shape[1]
        total_embed_dim=self.nhead*self.dim_per_head
       
        if(self.attn_package=="custom_pytorch"):

            q = self.in_projector(x)

            if(self.projection_type=="single_self"):
                k=q
                v=q
            else:
                q,k,v = q # q is tuple of q,k,v
            

            ## but batch size at 1st pos
            q=q.transpose(1,0)
            k=k.transpose(1,0)
            v=v.transpose(1,0)

            ## reshape to make qkv dimension work nicely with scaled dot product attention
            ## shape (S, batch_size * num_heads, per_head _dim) instead of (S, batch_size, num_heads*per_head_dim=embed_dim)
            q = q.contiguous().view(q.shape[0], q.shape[1] * self.nhead, self.dim_per_head).transpose(1, 0)
            k = k.contiguous().view(k.shape[0], k.shape[1] * self.nhead, self.dim_per_head).transpose(1, 0)
            v = v.contiguous().view(v.shape[0], v.shape[1] * self.nhead, self.dim_per_head).transpose(1, 0)

            new_attn_mask=key_padding_mask
            
            ## require mask with inf istead of bool for default dot product attn
            if key_padding_mask is not None:
                
                assert(key_padding_mask.dtype != torch.bool)
                assert(len(key_padding_mask.shape)==2)
                new_attn_mask = key_padding_mask.view(batch_size, 1, 1, max_num_samples).   \
                expand(-1, self.nhead, -1, -1).reshape(batch_size * self.nhead, 1, max_num_samples)

                #new_attn_mask = torch.zeros_like(temp_mask, dtype=q.dtype)
                #new_attn_mask.masked_fill_(temp_mask, float("-inf"))
            
            x  = F.scaled_dot_product_attention(q, k, v, new_attn_mask, self.dropout_p)

            x = x.transpose(1, 0).contiguous().view(-1, self.embed_dim)
                
            ## out projection not really needed without dropout
            if(self.do_perlayer_out_projection==1):
                x=self.out_projector(x)

            x = x.view(max_num_samples, batch_size, self.embed_dim).transpose(1, 0)

            return self.dropout1(x)

        elif(self.attn_package=="official_pytorch_w_weights"):

            if(self.do_perlayer_out_projection):
                used_outproj=self.out_projector.projector.mlp[0].weight
                used_outproj_bias=self.out_projector.projector.mlp[0].bias

            else:
                used_outproj=torch.eye(total_embed_dim).to(q).type_as(q)
                used_outproj_bias=None

        
            if key_padding_mask is not None:

                assert(key_padding_mask.dtype != torch.bool), key_padding_mask
                assert(len(key_padding_mask.shape)==2)
                #new_attn_mask=key_padding_mask[:,None,:]*(key_padding_mask[:,:,None])
                #new_attn_mask=torch.where(~torch.isfinite(new_attn_mask), float("-inf"), new_attn_mask)
                """
                new_attn_mask = key_padding_mask.view(batch_size, 1, 1, max_num_samples).   \
                expand(-1, self.nhead, -1, -1).reshape(batch_size * self.nhead, 1, max_num_samples)
                """



            updated_v, weights=torch.nn.functional.multi_head_attention_forward(x.transpose(1,0),x.transpose(1,0),x.transpose(1,0),
                                                          total_embed_dim,
                                                          self.nhead,
                                                          self.in_projector.joint_projector.mlp[0].weight,
                                                          self.in_projector.joint_projector.mlp[0].bias,
                                                          None,
                                                          None,
                                                          False, # add_zero_attn,
                                                          self.dropout_p,
                                                          used_outproj,
                                                          out_proj_bias=used_outproj_bias,
                                                          need_weights=True,
                                                          average_attn_weights=False,
                                                          key_padding_mask=key_padding_mask)
                
            if(self.rel_position_mode_value=="none"):

                out=updated_v.transpose(1,0)

            else:

                if(self.rel_position_input_feeding_type==0):
                    ## create new diffs first

                    print(orig_rel_pos_tensor)

                    sys.exit(-1)
                    _get_tensor_of_diffs_flattened()
                    ## concat_abs_and_rel_and_add
                    ## concat_abs_and_rel_no_add
                    ## only_rel_and_add
                    ## only_rel_no_add

                    if(self.rel_position_mode_value=="concat_abs_and_rel_and_add"):
                        ### key padding mask has lengths


                        q = self.rel_pos_value_projector(x)

              

            return out
            

        elif(self.attn_package=="xformer"):

            q = self.in_projector(x)

            if(self.projection_type=="single_self"):
                k=q
                v=q
            else:
                q,k,v = q # q is tuple of q,k,v

            # spliut up last (attn dim) into nhead sectors with (attn_dim/nhead) subdimensionality
            q=q.reshape(q.shape[0], q.shape[1], self.nhead, -1)
            k=k.reshape(k.shape[0], k.shape[1], self.nhead, -1)
            v=v.reshape(v.shape[0], v.shape[1], self.nhead, -1)
            
            ## probs*v
            out=fmha.memory_efficient_attention(q, k, v, attn_bias=key_padding_mask, op=self.xformers_operator)

            out=out.reshape(q.shape[0], q.shape[1], -1)

            ## out projection not really needed without dropout
            if(self.do_perlayer_out_projection==1):
                out=self.out_projector(out)

            return self.dropout1(out)
            ## switch to nhead

        else:
            raise Exception("Unknown package ", self.attn_package)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



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

        _, settings_kwargs=cfg_parser.parse_cfg(kwargs, "settings")

        self.projection_type=settings_kwargs["projection_type"]

        if(self.projection_type=="single_self"):

            self.projector=SkipMLP(settings_kwargs["input_dim"], settings_kwargs["mlp_hidden_dims"], settings_kwargs["output_dim"], add_skip_connection=settings_kwargs["add_skip_connection"])

        elif(self.projection_type=="single_qkv"):

            self.q_projector=SkipMLP(settings_kwargs["input_dim"], settings_kwargs["mlp_hidden_dims"], settings_kwargs["output_dim"], add_skip_connection=settings_kwargs["add_skip_connection"])
            self.k_projector=SkipMLP(settings_kwargs["input_dim"], settings_kwargs["mlp_hidden_dims"], settings_kwargs["output_dim"], add_skip_connection=settings_kwargs["add_skip_connection"])
            self.v_projector=SkipMLP(settings_kwargs["input_dim"], settings_kwargs["mlp_hidden_dims"], settings_kwargs["output_dim"], add_skip_connection=settings_kwargs["add_skip_connection"])

        elif(self.projection_type=="joint_qkv"):

            self.joint_projector=SkipMLP(settings_kwargs["input_dim"], settings_kwargs["mlp_hidden_dims"], 3*settings_kwargs["output_dim"], add_skip_connection=settings_kwargs["add_skip_connection"])

        else:
            raise Exception("Hmm this should not happen, unknown projection type ... ", self.projection_type)

    def forward(self, x):

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

            embed_dim=x.shape[-1]

            q=joint[..., :embed_dim]
            k=joint[..., embed_dim:2*embed_dim]
            v=joint[..., 2*embed_dim:3*embed_dim]

            return q,k,v

class almagate_multihead_attention(nn.Module):
    def __init__(self, **kwargs):#encoder_layers=1, dropout=0.0, rnn_type="lstm", encoder_hidden_dim=10, num_encoder_mlp_layers=0, nonlinearity="tanh"):
        super(almagate_multihead_attention, self).__init__()
        

        cfg_parser=config_parser.config_parser()

        cfg_parser.add_default_kwarg("settings", "input_dim", 5, int)
        cfg_parser.add_default_kwarg("settings", "output_dim", 50, int)

        cfg_parser.add_default_kwarg("settings", "io_mlp_hidden_dims", "128", str)
        cfg_parser.add_default_kwarg("settings", "io_add_skip_connection", 0, int)

        ##TODO: this one can be removed probably .. should always be "single_self" .. i.e. default single MLP w/ skips
        cfg_parser.add_default_kwarg("settings", "attn_io_projection_type", "single_self", str, choices=["single_self", "single_qkv", "joint_qkv"])

        ## TODO: just here for backwards compatability (not actually used)
        cfg_parser.add_default_kwarg("settings", "io_attn_projection_type", "single_self", str, choices=["single_self", "single_qkv", "joint_qkv"])

        cfg_parser.add_default_kwarg("settings", "attn_do_perlayer_out_projection", 1, int, choices=[0,1])




        cfg_parser.add_default_kwarg("settings", "skip_input_projection", 0, int) ## skipping the input projection step

        
        cfg_parser.add_default_kwarg("settings", "attn_computational_dim", -1, int)
       
        cfg_parser.add_default_kwarg("settings", "attn_num_layers", 2, int)
        cfg_parser.add_default_kwarg("settings", "attn_num_heads_per_layer", 1, int)
        cfg_parser.add_default_kwarg("settings", "attn_use_layer_norm_1", 1, int)
        cfg_parser.add_default_kwarg("settings", "attn_use_layer_norm_2", 1, int)
        cfg_parser.add_default_kwarg("settings", "attn_use_extra_layer_norm", 0, int)
        cfg_parser.add_default_kwarg("settings", "attn_layer_norm_first", 1, int)
        cfg_parser.add_default_kwarg("settings", "attn_use_residual_addition", 1, int)
        cfg_parser.add_default_kwarg("settings", "dtype", "float32", str, choices=["float64", "float32", "float16", "bfloat16"])
        cfg_parser.add_default_kwarg("settings", "attn_projection_type", "joint_qkv", str, choices=["single_self", "single_qkv", "joint_qkv"])
        
        cfg_parser.add_default_kwarg("settings", "attn_perform_final_mapping", 1, int, choices=[0,1])


        cfg_parser.add_default_kwarg("settings", "attn_package", "custom_pytorch", str, choices=["custom_pytorch", "official_pytorch_w_weights", "xformer"])


        cfg_parser.add_default_kwarg("settings", "attn_dropout", 0.1, float)
        cfg_parser.add_default_kwarg("settings", "attn_internal_mlp_dim", 512, int)

        cfg_parser.add_default_kwarg("settings", "attn_use_weighted_mean", 0, int)

        cfg_parser.add_default_kwarg("settings", "attn_operator", "none", str)

        ####
        ## 
        cfg_parser.add_default_kwarg("settings", "attn_rel_position_encoding", "feature_mlp", str, choices=["feature_mlp"])
        cfg_parser.add_default_kwarg("settings", "attn_rel_position_input_feeding_type", 0, int, choices=[0,1,2]) ## how to feed input? 0 normal (values), 1 original first input, both if possible 
        cfg_parser.add_default_kwarg("settings", "attn_rel_position_mode_value", "none", str, choices=["none", "concat_abs_and_rel_and_add", "concat_abs_and_rel_no_add", "split_in_heads_2"])
        #cfg_parser.add_default_kwarg("settings", "attn_rel_position_mode_value", "none", str, choices=["none", "add_key_add_value", "sep_key_add_value", "split_in_heads_2", "split_in_heads_4"])



        settings_args, settings_kwargs=cfg_parser.parse_cfg(kwargs, "settings", check_passed_params_are_configured=True)

        print("Configured attantion settings....")
        for k in settings_kwargs:
            print(k, settings_kwargs[k])
            if(k!="dtype"):
                setattr(self, k, settings_kwargſ[k])

        print("------------------------")

        if(self.attn_operator is not None):
            if(self.attn_operator=="None" or self.attn_operator=="none"):
                self.attn_operator=None

        ## size B X NUM ITEMS X INPUT DIM
        #self.h0=nn.Parameter(torch.randn((1, 1, self.input_dim)))

        if(self.attn_computational_dim==-1):
            self.attn_computational_dim=self.output_dim

            if(self.attn_use_weighted_mean):
                self.attn_computational_dim+=1

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

        if(settings_kwargs["skip_input_projection"]):
            ## we skip the input projeciton .. dimensions must match
            assert(self.attn_computational_dim==self.input_dim) 

            self.input_projector=lambda x: x
        else:

            ## TODO: dont need a qkv projector here
            self.input_projector=qkv_projector(input_dim=self.input_dim, 
                                               output_dim=self.attn_computational_dim, 
                                               mlp_hidden_dims=settings_kwargs["io_mlp_hidden_dims"],
                                               projection_type=settings_kwargs["attn_io_projection_type"],
                                               dtype=self.attn_dtype,
                                               add_skip_connection=settings_kwargs["io_add_skip_connection"]
                                               )
        ## use official pytorch encoder layer if parameters are right
        if(("pytorch" in self.attn_package) 
            and (self.attn_use_layer_norm_1==1)
            and (self.attn_use_layer_norm_2==1)
            and (self.attn_use_residual_addition==1)
            and self.attn_projection_type=="joint_qkv"
            and self.attn_rel_position_mode_value=="none"
            and self.do_perlayer_out_projection==1):

           
            encoder_layer = nn.TransformerEncoderLayer(self.attn_computational_dim, 
                                                           self.attn_num_heads_per_layer, 
                                                           dim_feedforward=self.attn_internal_mlp_dim, 
                                                           dropout=self.attn_dropout,
                                                            batch_first=True,
                                                            dtype=self.attn_dtype,
                                                            norm_first=self.attn_layer_norm_first)
        else:
            encoder_layer = CustomTransformerEncoderLayer(self.attn_computational_dim, 
                                                          self.attn_num_heads_per_layer, 
                                                          dim_feedforward=self.attn_internal_mlp_dim, 
                                                          dropout=self.attn_dropout,
                                                          norm_first=self.attn_layer_norm_first,
                                                          use_layer_norm_1=self.attn_use_layer_norm_1,
                                                          use_layer_norm_2=self.attn_use_layer_norm_2,
                                                          use_residual_addition=self.attn_use_residual_addition,
                                                          attn_package=settings_kwargs["attn_package"], # pytorch / xformers / flash
                                                          projection_hidden_dims="",
                                                          projection_type=settings_kwargs["attn_projection_type"],
                                                          dtype=self.attn_dtype,
                                                          xformers_operator=self.attn_operator,
                                                          rel_position_mode_value=self.attn_rel_position_mode_value,
                                                          rel_position_input_feeding_type=self.attn_rel_position_input_feeding_type,
                                                          do_perlayer_out_projection=self.attn_do_perlayer_out_projection)
        

        
        self.extra_layer_norm=None
        if(self.attn_use_extra_layer_norm):
            self.extra_layer_norm=nn.LayerNorm(self.attention_input_dim, dtype=self.attn_dtype)
        
        if("pytorch" in self.attn_package and self.attn_rel_position_mode_value=="none"):
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.attn_num_layers, norm=self.extra_layer_norm)
        else:
            self.transformer_encoder = CustomTransformerEncoder(encoder_layer, 
                                                                self.attn_num_layers, 
                                                                norm=self.extra_layer_norm, 
                                                                rel_position_mode_value=self.attn_rel_position_mode_value)
        
        if(settings_kwargs["attn_perform_final_mapping"]==1):
            if(settings_kwargs["attn_computational_dim"]!=-1):
                aggregation_dim=self.attn_computational_dim
                if(self.attn_use_weighted_mean):
                    aggregation_dim-=1

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
            self.attention_to_output_mlp=None
    def _create_datarep_and_mask(self, datavecs, datalens, nhead, attn_package=None):

        """
        Returns:
        input_vecs: Depending on attn package, can be list of vecs or a padded tensor.
        """
            
        assert(type(datalens)==torch.Tensor), "Require datalens as tensor"
        
        used_attn_package=self.attn_package
        if(attn_package is not None):
            used_attn_package=attn_package
       
        if("pytorch" in used_attn_package):

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

        elif(used_attn_package=="xformer"):

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

        else:
            raise Exception("Unknown package ", used_attn_package)


    def _final_summation(self, result_matrix, mask, datalens, attn_package=None, add_weights=False, perform_final_aggregation=True):

        used_attn_package=self.attn_package
        if(attn_package is not None):
            used_attn_package=attn_package

    
        if("pytorch" in used_attn_package):

            if(add_weights):
                raise Exception("Weighting not implemented for pytorch .. implement it!")

            if(perform_final_aggregation==False):
                raise Exception("Pytorch impl for non-aggregation still needs to be checked!")

            if(mask is not None):
                attn_result=result_matrix*(~mask).type(torch.int).unsqueeze(-1)
            else:
                attn_result=result_matrix

            attn_result=attn_result.sum(axis=1)/datalens.unsqueeze(-1)
           
            return attn_result

        elif(used_attn_package=="xformer"):

            out=mask.split(result_matrix)

            if(perform_final_aggregation==False):
                # return list of
                return out

            if(add_weights):

                #weights=[torch.nn.functional.softmax(i[:,:,-1:], dim=1) for i in out]
                
                attn_result=torch.cat([ (torch.nn.functional.softmax(i[:,:,-1:], dim=1)*i[:,:,:-1] ).sum(dim=1) for i in out])

            else:

                
                attn_result=torch.cat([i.mean(dim=1) for i in out])
            
            return attn_result

        else:
            raise Exception("Unknown package ", used_attn_package)

    def forward(self, datavecs, datalens, perform_final_aggregation=True, perform_final_mapping=True):

        # package-dependent processing
      
        # data and datalen preparation dependent on package
        padded_tensor, padding_mask=self._create_datarep_and_mask(datavecs, datalens, self.attn_num_heads_per_layer, attn_package=self.attn_package)
      
        # input projection
        computational_input=self.input_projector(padded_tensor)
        
        # transformer layers
        result=self.transformer_encoder(computational_input, src_key_padding_mask=padding_mask)

        #if( (perform_final_aggregation == False) and (perform_final_mapping == False)):
        #    return result

        # aggregation
        
        result=self._final_summation(result, padding_mask,datalens, attn_package=self.attn_package, add_weights=self.attn_use_weighted_mean, perform_final_aggregation=perform_final_aggregation)

        if(perform_final_mapping==False):
            return result

        assert(self.attention_to_output_mlp is not None), "Choose to perform final mapping, but attention_to_output_mlp is None... have to define Encoder with that flag on!"
        ret_val = self.attention_to_output_mlp(result)
            
        return ret_val

        
