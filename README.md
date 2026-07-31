# Overview

A framework to do supervised neural network training, in particular neural posterior estimation with conditional normalizing flows. Utilizes pytorch, pytorch-lightning, ray and [jammy-flows](https://github.com/thoglu/jammy_flows/) for normalizing flows.

Includes some data encoders, in particular a transformer encoding used in the paper
[Neural posterior estimation of the neutrino direction in IceCube](https://arxiv.org/abs/2604.19846).

Grew historically from baseline pytorch implementation, added more alternative multi-head-attention (MHA) algorithms over time. Uses modified re-implementations of some pytorch classes (like TransformerEncoder) under the hood.

Supports settings for different types of aggregation/class tokens, absolute and relative positional encodings, various residual-flow options (including [resi-dual/dual residual connections](https://arxiv.org/abs/2304.14802)).

# Transformer encoder

Contains a tranformer encoder Module in [mh_attention_encoder_new.py](./learning_ground/encoders/mh_attention_encoder_new.py). It allows to run various implementations of soft-max attention, in particular:
- xformers ("xformer") -> recommended for generic cpu/gpus, utilizes memory-efficiet attention / works with float32 -> default
- flash-attention ("flash_attn") -> useful for newer cards / requires bfloat16/float16
- an older pytorch variant ("custom_pytorch")
- a second pytorch variant supporting softmax biases + relative value positional encoding ("official_pytorch_w_weights")
- experimental implementations based on pytoch_geometric and nested tensors (experimental / not working)

Unit tests are included in [./tests/test_mha_encoder](./tests/test_mha_encoder.py). It compares that the various soft-attention implementations are consistent, also versus some older MHA implementation.

## Settings Overview

The following gives an explanation of the settings (excerpt from code).
```
cfg_parser.add_default_kwarg("settings", "input_dim", 5, int) 

# "Final output dimension (after aggregation+final mapping)"
cfg_parser.add_default_kwarg("settings", "output_dim", 50, int) 

# "MLP hidden dim structure used for input/output MLPs - str with "-" separator, i.e. "64-64"
cfg_parser.add_default_kwarg("settings", "io_mlp_hidden_dims", "128", str) 

# add skip connection into Input/Output MLPs?
cfg_parser.add_default_kwarg("settings", "io_add_skip_connection", 0, int) 

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

## Use positional encoding? 
# -1 -> no
# positive integer n -> used first *n* dimensions of token for positional encoding 
cfg_parser.add_default_kwarg("settings", "attn_original_input_position_feature_number", -1, int)

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
```



