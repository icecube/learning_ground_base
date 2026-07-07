import unittest
import sys
import os
import torch
import numpy
import pylab
import random
import xformers
import itertools
import glob


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import learning_ground.encoders.mh_attention_encoder as mha
import learning_ground.encoders.mh_attention_encoder_new as mha_new

def seed_everything(seed_no):
    random.seed(seed_no)
    numpy.random.seed(seed_no)
    torch.manual_seed(seed_no)

def get_choices_from_param_dict(param_dict):

    pnames=[]
    contents=[]

    for p in param_dict.keys():
        pnames.append(p)
        contents.append(param_dict[p])

    combinations=list(itertools.product(*contents))
    
    choices=[]

    for this_comb in combinations:
        this_dict=dict()

        for ind,entry in enumerate(this_comb):
            this_dict[pnames[ind]]=entry

        choices.append(this_dict)

    
    return choices

def compare_two_models(res1, res2, model1, model2, do_grad=True, abs_allowed_eps=5e-5, grad_allowed_eps=5e-4):

    abs_diff=(res2-res1).abs().max()
                    
    assert(abs_diff<abs_allowed_eps), ("abs diff violated .. ", model1, model2, abs_diff," allowed ..", abs_allowed_eps, (res2-res1), res1, res2)
    print("ABS DIFF ", abs_diff)
    ## also contain derivatives

    if(do_grad):
        grad_res_base=torch.autograd.grad(res1.mean(), model1.parameters(), allow_unused=True, retain_graph=True)
        grad_res_other=torch.autograd.grad(res2.mean(), model2.parameters(), allow_unused=True, retain_graph=True)
        
        named_parameters=[np for np in model1.named_parameters()]
        named_parameters2=[np for np in model2.named_parameters()]

        grad_diffs=[]
        got_none_deriv=False
        for cur_grad_index, cur_grad_item in enumerate(grad_res_base):

            if(cur_grad_item is not None and grad_res_other[cur_grad_index] is not None):
                assert(cur_grad_item.numel()==grad_res_other[cur_grad_index].numel())
                grad_diff=cur_grad_item-grad_res_other[cur_grad_index]
                grad_diff_max=grad_diff.abs().max()

                assert(grad_diff_max<grad_allowed_eps), ("grad max violated .. for parameter ", grad_diff_max, " allowed .. ", grad_allowed_eps, cur_grad_item-grad_res_other[cur_grad_index])
                
                grad_diffs.append(grad_diff_max)
            else:
                print("############ NONE 'GRAD ####################")
                print(named_parameters[cur_grad_index][0])
                print(named_parameters2[cur_grad_index][0])
                got_none_deriv=True

        print("ABS GRAD diff ", max(grad_diffs))

       
#def _check_opts_compatible_with_pytorch(opts):

def trafo_comp_old_and_new(old_mha_object, new_mha_object, option_choices, do_grad=True, device="cuda:0"):


    seed_everything(0)

    used_device=torch.device(device)

    used_dtype=["float32"]

    output_dim=32

    num_layers=4

    # equal len
    test_input_equal=torch.rand(size=(5,12,2)).to(used_device)
    data_lens_equal=torch.ones(5)*12
    data_lens_equal=data_lens_equal.type(torch.int64).to(used_device)
    test_input_equal_pair=(test_input_equal, data_lens_equal)

    # unequal len
    test_input_unequal=[]
    data_lens_unequal=[]

    for l in range(5):
        num_elems=numpy.random.randint(15)+5
        data_lens_unequal.append(num_elems)

        this_t=torch.rand(size=(1,num_elems,2)).to(used_device)
        test_input_unequal.append(this_t)

    data_lens_unequal=torch.Tensor(data_lens_unequal).type(torch.int64).to(used_device)
    test_input_unequal_pair=(test_input_unequal, data_lens_unequal)

   
    ####

    for inp_pair in [test_input_unequal_pair]:#, test_input_equal_pair]:

        # inp_pair holds tensor (0) and datalens (1)
        test_input=inp_pair[0]
        data_lens=inp_pair[1]

        for cur_dtype in used_dtype:
            for cur_opt in option_choices:

                print("---------- testing options ...............")
                print(cur_opt)
                print("------------------------------------------")
                    
                ### BASE PYTORCH - WITH CORRECT SETTINGS DEFAULTS TO STANDARD PYTORCH IMPLEMENTATION
                base_model_pytorch=old_mha_object.almagate_multihead_attention(input_dim=2, 
                                                                    output_dim=output_dim,
                                                                    attn_package="xformer",
                                                                    attn_dropout=0.0,
                                                                    attn_num_layers=num_layers,
                                                                    **cur_opt)
               
                base_model_pytorch.float()
                base_model_pytorch.to(used_device)

                base_result=base_model_pytorch(test_input, data_lens)

                ### XFORMER ##############
                other_models=[]
                xformer_trafo=new_mha_object.almagate_multihead_attention(input_dim=2,
                                                               output_dim=output_dim, 
                                                               attn_package="xformer",
                                                               attn_dropout=0.0,
                                                               attn_num_layers=num_layers,
                                                               **cur_opt)
              

                xformer_trafo.float()
                xformer_trafo.to(used_device)
           
                other_models.append(("xformer_new", xformer_trafo))
                ######################

                ## check number of params are the same
                numel_base=0
                for p in base_model_pytorch.named_parameters():
                   
                    numel_base+=p[1].numel()
                   
                for o in other_models:
                    numel_other=0
                    for p2 in o[1].named_parameters():
                        
                        numel_other+=p2[1].numel()
                       

                    assert(numel_base==numel_other)

                parname_list=[]
                ## then copy params
                for p in base_model_pytorch.named_parameters():
                    parname_list.append(p[0])
                    
                    for o in other_models:
                        found_param=False
                        for p2 in o[1].named_parameters():
                            if(p[0]==p2[0]):
                                p2[1].data=p[1].data
                                found_param=True
                        if(found_param==False):
                            
                            raise Exception()
                            #print(p[0])
                            assert(".self_attn" in p[0])
                            
                            if(".self_attn.in_proj_weight" in p[0]):
                                new_name=p[0].replace(".self_attn.in_proj_weight",".in_projector.joint_projector.mlp.0.weight")
                            elif(".self_attn.in_proj_bias" in p[0]):
                                new_name=p[0].replace(".self_attn.in_proj_bias",".in_projector.joint_projector.mlp.0.bias")
                            elif(".self_attn.out_proj.weight" in p[0]):
                                new_name=p[0].replace(".self_attn.out_proj.weight",".out_projector.projector.mlp.0.weight")
                            elif(".self_attn.out_proj.bias" in p[0]):
                                new_name=p[0].replace(".self_attn.out_proj.bias",".out_projector.projector.mlp.0.bias")
                            else:
                                raise Exception("Unknown param to handle ", p[0])

                            for p2 in o[1].named_parameters():
                                if(p2[0]==new_name):
                                    
                                    assert(p2[1].data.shape==p[1].data.shape), (p[1].data.shape, p2[1].data.shape)
                                    
                                    p2[1].data=p[1].data
                                    found_param=True
                                    break
                            assert(found_param==True), new_name

                
                ##################

            
                base_result=base_model_pytorch(test_input, data_lens)
                
             
                #print("BASE", base_result)

                

                for o in other_models:
                    res=o[1](test_input, data_lens)
                    
                    print("Comparing ... ", "Base with ...", o[0])
                    print("types .. ", type(base_model_pytorch), type(o[0]))
                    
                    compare_two_models(base_result, res, base_model_pytorch, o[1],do_grad=do_grad)
                

def trafo_test(mha_object, option_choices, do_grad=True, device="cuda:0", model_types=[]):


    seed_everything(0)

    used_device=torch.device(device)

    used_dtype=["half", "bfloat16", "float32"]

    output_dim=32

    num_layers=3

    orig_input_dim=2

    ####

    ## loop over dtypes

    for cur_dtype in used_dtype:
        print("STARTING NEW DTYPE ", cur_dtype)
        # equal len
        # check compute compataibility
        if(cur_dtype=="bfloat16" or cur_dtype=="half"):
            if(torch.cuda.get_device_properties(0).major<8):
                print("GPU does not support half precision... skipping ", cur_dtype)
                continue
        test_input_equal=torch.rand(size=(5,12,orig_input_dim)).to(used_device)
        if(cur_dtype=="bfloat16"):
            test_input_equal=test_input_equal.type(torch.bfloat16)
        elif(cur_dtype=="half"):
            test_input_equal=test_input_equal.type(torch.half)

        data_lens_equal=torch.ones(5)*12
        data_lens_equal=data_lens_equal.type(torch.int64).to(used_device)
        input_identifiers=[torch.randint(2, (dle,)) for dle in data_lens_equal]
        print("################################################")
        print("tinput equl shape ",test_input_equal.shape)

        test_input_equal_pair=(test_input_equal, data_lens_equal, input_identifiers)
        
        # unequal len
        test_input_unequal=[]
        data_lens_unequal=[]
        input_identifiers=[]

        for l in range(5):
            num_elems=numpy.random.randint(15)+5
            data_lens_unequal.append(num_elems)
            
            this_t=torch.rand(size=(1,num_elems,orig_input_dim)).to(used_device)
            if(cur_dtype=="bfloat16"):
                this_t=this_t.type(torch.bfloat16)
            elif(cur_dtype=="half"):
                this_t=this_t.type(torch.half)

            test_input_unequal.append(this_t)

            input_identifiers.append(torch.randint(2,(num_elems,)))

        data_lens_unequal=torch.Tensor(data_lens_unequal).type(torch.int64).to(used_device)
        test_input_unequal_pair=(test_input_unequal, data_lens_unequal, input_identifiers)

        for inp_pair in [test_input_unequal_pair, test_input_equal_pair]:

            # inp_pair holds tensor (0) and datalens (1)
            
            test_input=inp_pair[0]
            data_lens=inp_pair[1]
            inp_identifiers=inp_pair[2]

            print("YO USAED ", inp_identifiers)
            
            for cur_opt in option_choices:
                
                print("---------- testing options ...............")
                print(cur_opt)
                print("------------------------------------------")
                    
                ### BASE PYTORCH - WITH CORRECT SETTINGS DEFAULTS TO STANDARD PYTORCH IMPLEMENTATION
                base_model_pytorch=mha_object.almagate_multihead_attention(input_dim=orig_input_dim, 
                                                                    output_dim=output_dim,
                                                                    attn_package="custom_pytorch",
                                                                    attn_dropout=0.0,
                                                                    attn_num_layers=num_layers,
                                                                    **cur_opt)

                if(cur_dtype=="float32"):
                    base_model_pytorch.float()
                elif(cur_dtype=="bfloat16"):
                    base_model_pytorch.bfloat16()
                elif(cur_dtype=="half"):
                    base_model_pytorch.half()
                else:
                    raise NotImplemented()

                base_model_pytorch.to(used_device)


                other_models=[]

                for cur_model_type in model_types:
                    
                    # flash attention only works with half precision (half/bfloat16)
                    if(cur_model_type=="flash_attn" and cur_dtype=="float32"):
                        continue

                    other_trafo=mha_object.almagate_multihead_attention(input_dim=orig_input_dim,
                                                                output_dim=output_dim, 
                                                                attn_package=cur_model_type,
                                                                attn_dropout=0.0,
                                                                attn_num_layers=num_layers,
                                                                **cur_opt)
                
                    if(cur_dtype=="float32"):
                        other_trafo.float()
                    elif(cur_dtype=="bfloat16"):
                        other_trafo.bfloat16()
                    elif(cur_dtype=="half"):
                        other_trafo.half()
                    else:
                        raise NotImplemented()

                    other_trafo.to(used_device)
            
                    other_models.append((cur_model_type, other_trafo))

                    """
                    ### GEOMETRIC SCATTER ##############
                    other_models=[]
                    xformer_trafo=mha_object.almagate_multihead_attention(input_dim=2,
                                                                output_dim=output_dim, 
                                                                attn_package="geometric_scatter",
                                                                attn_dropout=0.0,
                                                                attn_num_layers=num_layers,
                                                                **cur_opt)

                    other_models.append(("geometric_scatter", xformer_trafo))

                    ### nested tensors ##############
                    
                    xformer_trafo=mha_object.almagate_multihead_attention(input_dim=2,
                                                                output_dim=output_dim, 
                                                                attn_package="nested",
                                                                attn_dropout=0.0,
                                                                attn_num_layers=num_layers,
                                                                **cur_opt)
                

                    xformer_trafo.float()
                    xformer_trafo.to(used_device)
            
                    other_models.append(("nested", xformer_trafo))
                    """
                ######################
                """
                ## NEW PYTORCH
                second_pytorch=mha_object.almagate_multihead_attention(input_dim=2, 
                                                                output_dim=output_dim,
                                                                attn_package="official_pytorch_w_weights",
                                                                attn_dropout=0.0,
                                                                attn_num_layers=num_layers,
                                                                **cur_opt)

            
        
                second_pytorch.float()
                second_pytorch.to(used_device)

                other_models.append(("custom_pytorch", second_pytorch))
                """
                #######################

                ## check number of params are the same
                numel_base=0
                for p_ind,p in enumerate(base_model_pytorch.named_parameters()):
                    print("base ", p_ind, p[0],p[1].numel(), numel_base)
                    numel_base+=p[1].numel()
                
                for o in other_models:
                    numel_other=0

                    for p2_ind,p2 in enumerate(o[1].named_parameters()):
                        print("other ",o[0], p2_ind, p2[0],p2[1].numel(), numel_other)
                        numel_other+=p2[1].numel()
                    

                    assert(numel_base==numel_other), (numel_base, numel_other)

                parname_list=[]
                ## then copy params
                for p in base_model_pytorch.named_parameters():
                    parname_list.append(p[0])
                    
                    for o in other_models:
                        found_param=False
                        for p2 in o[1].named_parameters():
                            if(p[0]==p2[0]):
                                p2[1].data=p[1].data
                                found_param=True
                        if(found_param==False):
                            
                            #print(p[0])
                            assert(".self_attn" in p[0])
                            
                            if(".self_attn.in_proj_weight" in p[0]):
                                new_name=p[0].replace(".self_attn.in_proj_weight",".in_projector.joint_projector.mlp.0.weight")
                            elif(".self_attn.in_proj_bias" in p[0]):
                                new_name=p[0].replace(".self_attn.in_proj_bias",".in_projector.joint_projector.mlp.0.bias")
                            elif(".self_attn.out_proj.weight" in p[0]):
                                new_name=p[0].replace(".self_attn.out_proj.weight",".out_projector.projector.mlp.0.weight")
                            elif(".self_attn.out_proj.bias" in p[0]):
                                new_name=p[0].replace(".self_attn.out_proj.bias",".out_projector.projector.mlp.0.bias")
                            else:
                                raise Exception("Unknown param to handle ", p[0])

                            for p2 in o[1].named_parameters():
                                if(p2[0]==new_name):
                                    
                                    assert(p2[1].data.shape==p[1].data.shape), (p[1].data.shape, p2[1].data.shape)
                                    
                                    p2[1].data=p[1].data
                                    found_param=True
                                    break
                            assert(found_param==True), new_name

                
                ##################
                print("BEFORE base result ", inp_identifiers)


                ## no input identifiers first

                num_token_types=1

                if("attn_num_token_types" in cur_opt.keys()):
                    num_token_types=cur_opt["attn_num_token_types"]

                if(num_token_types==1):
                    base_result=base_model_pytorch(test_input, data_lens)
                    
                    print("AFTER base result ", inp_identifiers)
                    abs_allowed_eps=5e-5
                    grad_allowed_eps=2e-3
                    if(cur_dtype=="bfloat16"):
                        abs_allowed_eps=8e-2
                        grad_allowed_eps=4e-2
                    elif(cur_dtype=="half"):
                        abs_allowed_eps=15e-2
                        grad_allowed_eps=3e-2
                
                    #print("BASE", base_result)
                    for o in other_models:
                        print("BEF OTHER RESULT ", inp_identifiers)
                        res=o[1](test_input, data_lens)
                        print("AFTER OTHER result ", inp_identifiers)
                        print("Comparing ... ", "Base with ...", o[0])
                        print("types .. ", type(base_model_pytorch.transformer_encoder), type(o[1].transformer_encoder))
                        
                        compare_two_models(base_result, res, base_model_pytorch, o[1], do_grad=do_grad, abs_allowed_eps=abs_allowed_eps, grad_allowed_eps=grad_allowed_eps)

                else:

                    base_result=base_model_pytorch(test_input, data_lens, token_identifiers=inp_identifiers)
                    
                    print("AFTER base result ", inp_identifiers)
                    abs_allowed_eps=5e-5
                    grad_allowed_eps=2e-3
                    if(cur_dtype=="bfloat16"):
                        abs_allowed_eps=8e-2
                        grad_allowed_eps=4e-2
                    elif(cur_dtype=="half"):
                        abs_allowed_eps=15e-2
                        grad_allowed_eps=3e-2
                
                    #print("BASE", base_result)
                    for o in other_models:
                        print("BEF OTHER RESULT ", inp_identifiers)
                        res=o[1](test_input, data_lens, token_identifiers=inp_identifiers)
                        print("AFTER OTHER result ", inp_identifiers)
                        print("Comparing ... ", "Base with ...", o[0])
                        print("types .. ", type(base_model_pytorch.transformer_encoder), type(o[1].transformer_encoder))
                        
                        compare_two_models(base_result, res, base_model_pytorch, o[1], do_grad=do_grad, abs_allowed_eps=abs_allowed_eps, grad_allowed_eps=grad_allowed_eps)
              
class Test(unittest.TestCase):

    def setUp(self):
        
        ####

        param_dict=dict()
        param_dict["attn_use_layer_norm_1"]=[0,1]
        param_dict["attn_use_layer_norm_2"]=[0,1]
        param_dict["attn_use_residual_addition"]=[1]
        param_dict["attn_layer_norm_first"]=[1]
        param_dict["attn_projection_type"]=["joint_qkv"]
        param_dict["attn_use_weighted_mean"]=[0]
        param_dict["attn_computational_dim"]=[-1,40]
        param_dict["attn_num_heads_per_layer"]=[1]
        param_dict["attn_do_perlayer_out_projection"]=[1]
        param_dict["attn_rel_position_mode_value"]=["none"]

        
        self.option_choices_old=get_choices_from_param_dict(param_dict)

        ####

        # normal encoding
        param_dict=dict()
        param_dict["attn_use_layer_norm_1"]=[1,0]
        param_dict["attn_use_layer_norm_2"]=[1,0]
        param_dict["attn_use_residual_addition"]=[0,1,2]
        param_dict["attn_layer_norm_first"]=[1,0]
        param_dict["attn_projection_type"]=["joint_qkv"]
        param_dict["attn_use_weighted_mean"]=[0]
        param_dict["attn_computational_dim"]=[-1,40]
        param_dict["attn_num_heads_per_layer"]=[1]
        param_dict["attn_do_perlayer_out_projection"]=[1]
        param_dict["attn_rel_position_mode_value"]=["none"]
        param_dict["force_sdpa_precision"]=["", "float16"]

        self.option_choices_new=get_choices_from_param_dict(param_dict)

        # normal encoding
        param_dict=dict()
        param_dict["attn_use_layer_norm_1"]=[1]
        param_dict["attn_use_layer_norm_2"]=[1]
        param_dict["attn_use_residual_addition"]=[3]
        param_dict["attn_layer_norm_first"]=[1]
        param_dict["attn_projection_type"]=["joint_qkv"]

        param_dict["attn_aggregation_mode"]=["mean_n_diagonal_add_absolute", "mean_n_diagonal", "mean", "mean_add_absolute"]

        param_dict["attn_inprojection_mlp_dims"]=["128"]
        param_dict["attn_inprojection_add_skip"]=[0]
        param_dict["attn_inprojection_add_mean_diff"]=[1]

        param_dict["attn_use_weighted_mean"]=[0]
        param_dict["attn_computational_dim"]=[128]
        param_dict["attn_use_extra_layer_norm"]=[0]
        param_dict["attn_num_heads_per_layer"]=[1]
        param_dict["attn_do_perlayer_out_projection"]=[0,1]
        param_dict["attn_rel_position_mode_value"]=["none"]
        param_dict["force_sdpa_precision"]=[""]
        #param_dict["attn_abs_position_layer_indices"]=["0,2"]
        param_dict["attn_add_original_input_to_feature_input"]=[0]
        param_dict["attn_original_input_position_feature_number"]=[2]
        param_dict["attn_abs_position_mode"]=["none", "sinusoidal"]
        param_dict["attn_original_input_position_feature_range"]=["-1.0_1.0"]
        param_dict["attn_num_token_types"]=[1,2]

        param_dict["attn_use_computational_class_token"]=[0,1,3]

        #param_dict["attn_abs_position_scale"]=["logarithmic"]

        self.option_choices_absolute_and_input_stuff=get_choices_from_param_dict(param_dict)

        ## relative encoding
        param_dict=dict()
        param_dict["attn_use_layer_norm_1"]=[1,0]
        param_dict["attn_use_layer_norm_2"]=[1,0]
        param_dict["attn_use_residual_addition"]=[0,1,2,3]
        param_dict["attn_layer_norm_first"]=[1,0]
        param_dict["attn_projection_type"]=["joint_qkv"]
        param_dict["attn_use_weighted_mean"]=[0]
        param_dict["attn_computational_dim"]=[-1,40]
        param_dict["attn_num_heads_per_layer"]=[4]
        param_dict["attn_do_perlayer_out_projection"]=[0,1]
        param_dict["attn_rel_position_mode_value"]=["both"]
        param_dict["attn_rel_position_input_feeding_type"]=[0,1,2]
        param_dict["attn_original_input_position_feature_number"]=[1]
        param_dict["attn_abs_position_mode"]=["none", "sinusoidal"]
        param_dict["attn_rel_position_layer_indices"]=["0v,1i,2"]
        param_dict["attn_original_input_position_feature_range"]=["-1.0_1.0"]
        param_dict["attn_use_extra_layer_norm"]=[0,1]

        param_dict["attn_use_computational_class_token"]=[1,2]
        param_dict["attn_num_token_types"]=[1,2]

        param_dict["attn_rel_position_as_parallel_to_normal_track"]=[0,1]
        param_dict["attn_rel_position_max_computational_dim"]=[24]

        param_dict["attn_internal_mlp_dim"]=[10]

        self.option_choices_new_relative=get_choices_from_param_dict(param_dict)


        ## relative encoding
        param_dict=dict()
        param_dict["attn_use_layer_norm_1"]=[1,0]
        param_dict["attn_use_layer_norm_2"]=[1,0]
        param_dict["attn_use_residual_addition"]=[0,1,2,3]
        param_dict["attn_layer_norm_first"]=[1,0]
        param_dict["attn_projection_type"]=["joint_qkv"]
        param_dict["attn_use_weighted_mean"]=[0]
        param_dict["attn_computational_dim"]=[-1,40]
        param_dict["attn_num_heads_per_layer"]=[1]
        param_dict["attn_do_perlayer_out_projection"]=[0]
        param_dict["attn_rel_position_mode_value"]=["both"]
        param_dict["attn_aggregation_mode"]=["mean"]

        param_dict["attn_add_"]=["mean"]

        self.option_choices_new_relative_no_out_projection=get_choices_from_param_dict(param_dict)

        print(self.option_choices_new_relative)


        ## relative encoding
        param_dict=dict()
        param_dict["attn_use_layer_norm_1"]=[1]
        param_dict["attn_use_layer_norm_2"]=[1]
        param_dict["attn_use_residual_addition"]=[0]
        param_dict["attn_layer_norm_first"]=[1]
        param_dict["attn_projection_type"]=["joint_qkv"]
        param_dict["attn_use_weighted_mean"]=[0]
        param_dict["attn_computational_dim"]=[40]
        param_dict["attn_num_heads_per_layer"]=[1]
        param_dict["attn_do_perlayer_out_projection"]=[0]
        param_dict["attn_rel_position_mode_value"]=["both"]
        param_dict["attn_aggregation_mode"]=["mean"]

        self.option_choices_rel_debug=get_choices_from_param_dict(param_dict)

    def test_crosscheck_xformer_pytorch_transformer(self):
        print("-> Testing self consistency of sampling <-")

        devices=["cuda:0"]

        for device in devices:
            # do a comparison with the old implementation
            trafo_comp_old_and_new(mha, mha_new, self.option_choices_old)#

            ## absolute position encoding and stuff
            trafo_test(mha_new, self.option_choices_absolute_and_input_stuff, do_grad=True, device=device, model_types=["flash_attn","custom_pytorch","xformer", "official_pytorch_w_weights"])
            
            ## non-relative positional encoding
            
            ## relative positional encoding on (no out projection)
            #trafo_test(mha_new, self.option_choices_new_relative, do_grad=True, device=device, model_types=["xformer"])  

            ## relative position encoding on (with out projection)   
            #trafo_test(mha_new, self.option_choices_new_relative_no_out_projection, do_grad=True,model_types=["flash_attn"])
        
if __name__ == '__main__':
    unittest.main()