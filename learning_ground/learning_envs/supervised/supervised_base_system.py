import pytorch_lightning

import jammy_flows
import jammy_flows.helper_fns
from jammy_flows.amortizable_mlp import AmortizableMLP

import torch
#from ray.tune.integration.pytorch_lightning import TuneReportCallback

from .. import learning_env_base, system_base
from ... import config_parser
from ... import helper_fns

from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from lightning_fabric.utilities.cloud_io import _load as pl_load
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector as CheckpointConnector
import ray.tune as tune

from pytorch_lightning import seed_everything
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pylab
import gc

from torch.nn.utils import convert_parameters

import itertools

import numpy
import scipy.special
import copy

import os
import sys
import warnings
import glob
import collections
import time
import datetime

import functorch

import healpy
import collections

def _free_gradients_all_params(pdf):

    for p in pdf.parameters():
        p.requires_grad=True

def _check_allowed_names_for_blockage(pdf):

    ## some names are not allowed in the last layer
    ##
    for subdim_index, sub_pdf_type in enumerate(pdf.pdf_defs_list):
        this_type=sub_pdf_type[0]

        if(this_type=="s"):
            assert(pdf.flow_defs_list[subdim_index][-1]=="f")
            for flow_key in pdf.flow_opts[subdim_index][-1]:
                if("add_vertical_rq_spline_flow" == flow_key):
                    assert(pdf.flow_opts[subdim_index][-1][flow_key]==0), "No appropriate last flow in line found for second moment scaling! Need f flow with correct settings!"
                elif("add_circular_rq_spline_flow" == flow_key):
                    assert(pdf.flow_opts[subdim_index][-1][flow_key]==0), "No appropriate last flow in line found for second moment scaling! Need f flow with correct settings!"
                elif("add_corrlated_rq_spline_flow" == flow_key):
                    assert(pdf.flow_opts[subdim_index][-1][flow_key]==0), "No appropriate last flow in line found for second moment scaling! Need f flow with correct settings!"
                elif("inverse_z_scaling" == flow_key):
                    assert(pdf.flow_opts[subdim_index][-1][flow_key]==1), "No appropriate last flow in line found for second moment scaling! Need f flow with correct settings!"

        elif(this_type=="e"):
            for flow_key in pdf.flow_opts[subdim_index][-1]:
                if("cov_type" in pdf.flow_opts[subdim_index][-1]):
                    assert(pdf.flow_defs_list[subdim_index][-1][flow_key]=="full")
        
def _attach_second_moment_hook(pdf):

    print("IN ATTACH SECOND MOMENT HOOK")
    def get_specific_hook(zero_indices):
        def forward_hook(module, input, output):
            if(output.requires_grad):
                if getattr(pdf, 'use_second_moment_hook'):
                    def backward_hook(grad):
                        # Modify the gradient in some way
                       
                        grad[:, zero_indices]=0.0
                        
                        return grad

                    output.register_hook(backward_hook)

        return forward_hook

    for subdim_index, ll in enumerate(pdf.layer_list):

        tot_num_params_zero=0
        for l in ll[:-1]:
            tot_num_params_zero+=l.get_total_param_num()
        
        if(tot_num_params_zero>0):
            
            these_zero_indices=slice(0, tot_num_params_zero)
            this_forward_hook=get_specific_hook(these_zero_indices)

            pdf.mlp_predictors[subdim_index].register_forward_hook(this_forward_hook)


def _contour_regularization_terms(pdf, batch_labels, sample_size_per_item=100, conditional_input=None, regularization_space="base"):

    dtype, device=pdf.obtain_current_dtype_n_device()

    batch_size=batch_labels.shape[0]

    data_summary_repeated=None
    if(conditional_input is not None):
        if(type(conditional_input)==list):  
            data_summary_repeated=[ci.repeat_interleave(sample_size_per_item, dim=0) for ci in conditional_input]
        else:
            data_summary_repeated=conditional_input.repeat_interleave(sample_size_per_item, dim=0)


    if(regularization_space=="target"):

        raise Exception("check target space regularization")
        ## calc mean
        xyz_positions,_,_,_=pdf.sample(samplesize=1000, force_embedding_coordinates=True, allow_gradients=True)
        xyz_mean=torch.mean(xyz_positions, dim=0)
        xyz_mean=xyz_mean/torch.sqrt(torch.sum(xyz_mean**2))
        angle_mean,_=pdf.transform_target_space(xyz_mean[None,:], transform_from="embedding", transform_to="intrinsic")

        ## have to be slightly away from zero to allow for gradient to flow - (0,0) is caught and fixed to handle singularity
        reg_input=torch.Tensor([[0.01, 0.01]]).type(dtype).to(device)

        #target_reg,_,_,_=pdf._obtain_sample(predefined_target_input=reg_input)
        target_reg_xyz,_,_,_=pdf._obtain_sample(predefined_target_input=reg_input, force_embedding_coordinates=True)
        reg_target_angles,_=pdf.transform_target_space(target_reg_xyz, transform_from="embedding", transform_to="intrinsic")

        print("xyz reg ",target_reg_xyz, "xyz mean", xyz_mean)
        print("angle reg ",reg_target_angles, " angle mean ", angle_mean)
        similarity_COGs=-(target_reg_xyz*xyz_mean).sum()+1.0
        
        ### 0.9 centering term

        npts=20
        o5_terms=get_gauss_contour_input(npts, 0.5)
        reg_input=torch.from_numpy(o5_terms).type(dtype).to(device)
        target_reg_xyz,_,_,_=pdf._obtain_sample(predefined_target_input=reg_input, force_embedding_coordinates=True)
            
        #great_circle_dist_to_mean=great_circle_distance(target_reg, angle_mean.repeat(npts,1))
        great_circle_dist_to_mean=-(target_reg_xyz*xyz_mean.repeat(npts,1)).sum(dim=-1)
        o5_variance=torch.var(great_circle_dist_to_mean)
        
        symmetry_50_contour_term=0.5*o5_variance
    else:
        # perform regularization calculation in base space

        ## calc mean
        xyz_positions,_,_,_=pdf.sample(samplesize=batch_size*sample_size_per_item, conditional_input=data_summary_repeated, force_embedding_coordinates=True, allow_gradients=False)
     
        xyz_positions=xyz_positions.view(batch_size, sample_size_per_item, 3)
       
        xyz_mean=torch.mean(xyz_positions, dim=1)
        xyz_mean=xyz_mean/torch.sqrt(torch.sum(xyz_mean**2, dim=-1, keepdims=True))

        angle_mean,_=pdf.transform_target_space(xyz_mean, transform_from="embedding", transform_to="intrinsic")


        ## have to be slightly away from zero to allow for gradient to flow - (0,0) is caught and fixed to handle singularity
        reg_target=torch.Tensor([[0.0, 0.0]]).type(dtype).to(device)

        _,_,base_space_mean=pdf.forward(angle_mean, conditional_input=conditional_input)
        
        similarity_COGs=0.5*torch.sum((reg_target-base_space_mean)**2, dim=1)
        
        ### 0.5 centering term does not make sense in base space

        """
        npts=20
        o5_terms=get_gauss_contour_input(npts, 0.5)
        reg_target_circle=torch.from_numpy(o5_terms).type(dtype).to(device)
        #target_reg_xyz,_,_,_=pdf._obtain_sample(predefined_target_input=reg_input, force_embedding_coordinates=True)
            
        #great_circle_dist_to_mean=great_circle_distance(target_reg, angle_mean.repeat(npts,1))
       
        tot_diffs=((base_space_mean-reg_target_circle)**2).sum(dim=-1).sqrt()
        o5_variance=torch.var(tot_diffs)
        symmetry_50_contour_term=0.5*o5_variance
        """
        symmetry_50_contour_term=0.0
        reg_target_angles=None

    
    return similarity_COGs  


def kappa_regularization(pdf, conditional_input, regularization_indices, threshold_dict):
    ## regularization_indices: dict of index arrays, each array indexes a certain sub pdf for kappa params
    ## assumes the regularization indices have been found out before

    used_cinput=conditional_input
    if(type(conditional_input)!=list):
        used_cinput=[conditional_input]

    vec_of_all_kappas=[]

    for ind, pdf_def in enumerate(pdf.pdf_defs_list):
        if(pdf_def=="s2"):
            predictor_result=pdf.mlp_predictors[ind](used_cinput[ind])
            vec_of_all_kappas.append(predictor_result[:, regularization_indices[ind]])
            assert(vec_of_all_kappas[-1].shape[-1]==threshold_dict[ind].shape[-1])
       
            this_threshold=threshold_dict[ind].to(used_cinput[ind])
            vec_of_all_kappas[-1]=torch.where(vec_of_all_kappas[-1]<this_threshold, 0.5*(vec_of_all_kappas[-1]-this_threshold)**2, 0.0)
           
    reg_loss=torch.cat(vec_of_all_kappas, dim=1).mean()

    return reg_loss

def width_height_fvm_regularization(pdf, conditional_input, regularization_indices, threshold_dict):
    ## regularization_indices: dict of index arrays, each array indexes a certain sub pdf for kappa params
    ## assumes the regularization indices have been found out before
    print("--------- WIDTH / HEIGHT REGS ---------------")
    used_cinput=conditional_input
    if(type(conditional_input)!=list):
        used_cinput=[conditional_input]

    vec_of_abs_logwidths_logheights=[]

    for ind, pdf_def in enumerate(pdf.pdf_defs_list):
        ## only look at the reg terms if it is necessary
        cur_flow_defs_minus_last=pdf.flow_defs_list[ind][:-1]
        
        if(pdf_def=="s2" and (ind in regularization_indices.keys()) and "f" in cur_flow_defs_minus_last):
            predictor_result=pdf.mlp_predictors[ind](used_cinput[ind])

            vec_of_abs_logwidths_logheights.append(torch.abs(predictor_result[:, regularization_indices[ind]]))
            
            #assert(vec_of_all_kappas[-1].shape[-1]==threshold_dict[ind].shape[-1])
            
            this_threshold=threshold_dict[ind]
            vec_of_abs_logwidths_logheights[-1]=torch.where(torch.abs(vec_of_abs_logwidths_logheights[-1])>this_threshold, 0.5*(vec_of_abs_logwidths_logheights[-1]-this_threshold)**2, 0.0)

    reg_loss=torch.cat(vec_of_abs_logwidths_logheights, dim=1).mean()

    print("####################################")
    return reg_loss

def _check_label_defs(label_defs, label_ordering, pdf_defs):
    """
    Ensures that label definitions (as given by dict label_defs) agrees with the label ordering and the pdf_def.
    """
   
    compressed_pdf_defs=[]
    last_was_euclidean=False
    cumulative_def_dim=0

    for p in pdf_defs.split("+"):

        if("e" in p):

            cumulative_def_dim+=int(p[1:])

            last_was_euclidean=True

        else:

            if(last_was_euclidean):
                ##finalize previous euclidean

                compressed_pdf_defs.append("e%d" % cumulative_def_dim)
                cumulative_def_dim=0

            compressed_pdf_defs.append(p)

            last_was_euclidean=False

    if(last_was_euclidean):
        ## append final euclidean sub pdf
        compressed_pdf_defs.append("e%d" % cumulative_def_dim)


    cur_compressed_pdf_index=0
    cur_compressed_pdf_def_dim_counter=0

    for label_ind, cur_label in enumerate(label_ordering):

        assert(cur_compressed_pdf_index<len(compressed_pdf_defs))

        assert(label_defs[cur_label]["sub_manifold"] in compressed_pdf_defs[cur_compressed_pdf_index])

        cur_compressed_pdf_def_dim_counter+=label_defs[cur_label]["dim"]

        if(cur_compressed_pdf_def_dim_counter < int(compressed_pdf_defs[cur_compressed_pdf_index][1:])):

            continue

        else:
            # we saturate the compressed sub dim .. go to next one
            assert(cur_compressed_pdf_def_dim_counter == int(compressed_pdf_defs[cur_compressed_pdf_index][1:]))

            cur_compressed_pdf_index+=1
            cur_compressed_pdf_def_dim_counter=0

    assert(cur_compressed_pdf_index==len(compressed_pdf_defs)), ("Disagreement in labels: ", "Label defs: ", label_defs, "label_ordering: ", label_ordering, " pdf_defs: ", pdf_defs, "compressed_pdf_count: ", cur_compressed_pdf_index, " len(compressed_pdf_defs): ", len(compressed_pdf_defs))


class supervised_base_system(system_base.system_base):

    def __init__(self, config=dict()):
        
        super().__init__(config=config)

        if("direct_init" in config.keys()):
            ## direct initalization from genetic units
            
            init_dict=config["direct_init"]

            for k in init_dict.keys():
                setattr(self, k, init_dict[k])

            #self.encoder=init_dict["encoder"]
            #self.pdf=init_dict["pdf"]

        else:

            self._setup(config)

        ## obtain parameter specs that can be passed to the optimizer .. must have all children defined by this point
        self.parameter_specs=self.obtain_parameter_specs_for_optimization(config)   

    def _setup(self, config):
        
        self.cfg_parser=config_parser.config_parser()

        flow_options_overwrite=dict()
        if("flow_options_overwrite" in config.keys()):
            flow_options_overwrite=config["flow_options_overwrite"].copy()

        ########################
        ## optimizer config


        """
        self.loss_add_chi2_constraint=0
        if("loss.add_chi2_constraint" in config.keys()):
            self.loss_add_chi2_constraint=config["loss.add_chi2_constraint"]
        """
       
        ########################
        ## encoder config

        ## maybe eventualyl use preselection based on MLP / GRU / etc type
        #self.cfg_parser.add_default_kwarg("enctype", "encoder.type", "gru", str)
        #enc_type_args, enc_type_kwargs=self.cfg_parser.parse_cfg(config, "enctype")

        self._init_encoder(config)

        #########################
        ## pdf config

        self.cfg_parser.add_default_arg("pdf", "pdf.pdf_defs", "e2", str)
        self.cfg_parser.add_default_arg("pdf", "pdf.flow_defs", "gg", str)
        self.cfg_parser.add_default_kwarg("pdf", "pdf.amortization_mlp_dims", "128", str)
        self.cfg_parser.add_default_kwarg("pdf", "pdf.amortization_mlp_use_custom_mode", 0, int)
        self.cfg_parser.add_default_kwarg("pdf", "pdf.amortization_mlp_highway_mode", 0, int)
        self.cfg_parser.add_default_kwarg("pdf", "pdf.amortization_mlp_ranks", 0, int)
        self.cfg_parser.add_default_kwarg("pdf", "pdf.conditional_input_dim", 20, int)

        #self.cfg_parser.add_default_kwarg("pdf", "pdf.data_summary_dim", 20, int)
        #self.cfg_parser.add_default_kwarg("pdf", "pdf.input_encoder", "passthrough", str)

        self.pdf_args, self.pdf_kwargs=self.cfg_parser.parse_cfg(config, "pdf", drop_name_piece=True)

        ## check label defs

        assert("pdf.label_defs" in config.keys()), "Require dict with label_defs in config!"
        assert("pdf.label_ordering" in config.keys()), "Require str with label ordering in config!"
        print(self.pdf_kwargs)
        ## crosscheck that label definitions work
        _check_label_defs(config["pdf.label_defs"], config["pdf.label_ordering"], self.pdf_args[0])

        #self.pdf_kwargs["options_overwrite"]=self.flow_options_overwrite
        
        print("configured supervised pdf args..", self.pdf_kwargs)

        #######
        second_parser=config_parser.config_parser()
        second_parser.add_default_kwarg("pdf", "pdf.consider_all_permutations", 0, int)
        second_parser.add_default_kwarg("pdf", "pdf.fully_amortized_flag", 0, int)
        second_parser.add_default_kwarg("pdf", "pdf.independent_input_streams", 0, int)

        second_parser.add_default_kwarg("pdf", "pdf.second_moment_prefit_num_epochs", 0, int)

        #0 (no moment reg), 1 (hook), 2 (manually set to 0), 3 (only_last fit)
        second_parser.add_default_kwarg("pdf", "pdf.second_moment_prefit_mode", 0, int, choices=[0,1,2,3]) # 0 no prefit, 1 
        
        second_parser.add_default_kwarg("pdf", "pdf.base_centering_prefactor", 0.0, float)
        second_parser.add_default_kwarg("pdf", "pdf.add_final_second_moment_flow", 0, int)
        second_parser.add_default_kwarg("pdf", "pdf.add_pure_second_moment_in_loss", 0, int)

        second_parser.add_default_kwarg("pdf", "pdf.kappa_regularization", -1.0, float)
        second_parser.add_default_kwarg("pdf", "pdf.fvm_width_height_regularization", 3.0, float)

        second_parser.add_default_kwarg("pdf", "pdf.use_summary_statistic_bottleneck_prior", 0, int, choices=[0,1])

        _, self.supervised_kwargs=second_parser.parse_cfg(config, "pdf", drop_name_piece=True)

        if(self.supervised_kwargs["second_moment_prefit_mode"]==2):
            assert(self.automatic_optimization==False), "Second moment prefit = 2 (but using autoamtic gradients)... however requires manual gradient calc.!"
        
        if(self.supervised_kwargs["second_moment_prefit_mode"]>0):
            assert(self.supervised_kwargs["second_moment_prefit_num_epochs"]>0), "Using prefit mode but num_epochs for prefit set to 0!"
            #assert(self.supervised_kwargs["add_pure_second_moment_in_loss"]==0), "We use second moment prefit mode, so we cannot add second moment during training to normal loss.. either use prefit or add second moment, but not both!"
        else:
            assert(self.supervised_kwargs["second_moment_prefit_num_epochs"]==0), "Using NO prefit mode but num_epochs for prefit set to >0!"
        
        if(self.supervised_kwargs["add_pure_second_moment_in_loss"]>0):
            assert(self.supervised_kwargs["add_final_second_moment_flow"]), "We want to use 2nd moment in loss but not final second moment flow is added automatically .. this is required!"
        #if(self.supervised_kwargs["second_moment_prefit_num_epochs"]>0):
        #    assert(self.automatic_optimization==False), "Second moment fitting in the beginning requires manual optimization! set *optimizer.automatic* to 0"

        #self.cfg_parser.add_default_kwarg("pdf", "pdf.data_summary_dim", 20, int)
        #self.cfg_parser.add_default_kwarg("pdf", "pdf.input_encoder", "passthrough", str)

        if(self.supervised_kwargs["fully_amortized_flag"]):
            pdf_class=jammy_flows.fully_amortized_pdf
        else:
            pdf_class=jammy_flows.pdf

        if(self.pdf_kwargs["conditional_input_dim"]!=self.encoder_kwargs["output_dim"]):
            raise Exception("Encoder output dim %d does not fit together with PDF input conditional input dim %d. " % (self.encoder_kwargs["output_dim"], self.pdf_kwargs["conditional_input_dim"]))

        ## check for independent encoding streams
        num_sub_pdfs=len(self.pdf_args[0].split("+"))
        
        if(self.supervised_kwargs["independent_input_streams"] and num_sub_pdfs>1):
            

            self.pdf_kwargs["conditional_input_dim"]=num_sub_pdfs*[self.pdf_kwargs["conditional_input_dim"]]


        used_pdf_args=self.pdf_args
        if(self.supervised_kwargs["add_final_second_moment_flow"]):
            used_pdf_args=self.pdf_args[0].split("+")
            used_flow_defs=self.pdf_args[1].split("+")

            new_flow_defs=[]
            for cur_sub_index, cur_submanifold in enumerate(used_pdf_args):
                cur_flow_def=used_flow_defs[cur_sub_index]

                if(cur_submanifold[0]=="e"):
                    # add g
                    flow_options_overwrite[(0,len(cur_flow_def))]=dict()
                    flow_options_overwrite[(0,len(cur_flow_def))]["t"]=dict()
                    flow_options_overwrite[(0,len(cur_flow_def))]["t"]["cov_type"]="full"
                    cur_flow_def=cur_flow_def+"t"
                    
                elif(cur_submanifold=="s2"):

                    flow_options_overwrite[(0,len(cur_flow_def))]=dict()
                    flow_options_overwrite[(0,len(cur_flow_def))]["f"]=dict()
                    flow_options_overwrite[(0,len(cur_flow_def))]["f"]["add_vertical_rq_spline_flow"]=0
                    flow_options_overwrite[(0,len(cur_flow_def))]["f"]["add_circular_rq_spline_flow"]=0
                    flow_options_overwrite[(0,len(cur_flow_def))]["f"]["add_rotation"]=1
                    ## copy kappa modelling from options overwrite, otherwise take default
                    if("f" in flow_options_overwrite):
                        if("kappa_prediction" in flow_options_overwrite["f"]):
                            flow_options_overwrite[(0,len(cur_flow_def))]["f"]["kappa_prediction"]=flow_options_overwrite["f"]["kappa_prediction"]
                        
                        if("rotation_mode" in flow_options_overwrite["f"]):
                            flow_options_overwrite[(0,len(cur_flow_def))]["f"]["rotation_mode"]=flow_options_overwrite["f"]["rotation_mode"]

                        if("min_kappa" in flow_options_overwrite["f"]):
                            flow_options_overwrite[(0,len(cur_flow_def))]["f"]["min_kappa"]=flow_options_overwrite["f"]["min_kappa"]
                        
                        if("kappa_clamping" in flow_options_overwrite["f"]):
                            flow_options_overwrite[(0,len(cur_flow_def))]["f"]["kappa_clamping"]=flow_options_overwrite["f"]["kappa_clamping"]
                        
                        if("num_householder_iter" in flow_options_overwrite["f"]):
                            flow_options_overwrite[(0,len(cur_flow_def))]["f"]["num_householder_iter"]=flow_options_overwrite["f"]["num_householder_iter"]
                             

                    cur_flow_def=cur_flow_def+"f"
                else:
                    raise Exception("sub manifold ", cur_submanifold, " does currently not support second moment final flow!")

                new_flow_defs.append(cur_flow_def)

            new_flow_defs="+".join(new_flow_defs)

            self.pdf_args=[self.pdf_args[0], new_flow_defs]

        ## the main PDF
        self.pdf=pdf_class(*self.pdf_args, options_overwrite=flow_options_overwrite, **self.pdf_kwargs)

        ## find kappa indices
        ## x=ln(exp(y)-1)
        if(self.supervised_kwargs["kappa_regularization"]>0.0):
            self.kappa_indices=dict()
            self.kappa_thresholds=dict()

            f_found=False

            for ind, layer_list in enumerate(self.pdf.layer_list):
                all_but_last_flows=self.pdf.flow_defs_list[ind][:-1]
                if("f" in all_but_last_flows):
                    f_found=True

                if("s2" in self.pdf.pdf_defs_list[ind]):
                    index_list=[]
                    threshold_list=[]

                    cur_offset=0
                    for sub_pdf_ind, sub_pdf_str in enumerate(self.pdf.flow_defs_list[ind]):
                        this_layer=layer_list[sub_pdf_ind]

                        if(sub_pdf_str=="f"):
                            index_list.append(this_layer.num_householder_params+cur_offset)

                            ## find out which kind of kappa we have here
                            if(this_layer.kappa_prediction=="direct_log_real_bounded"):
                                this_log_kappa_par=numpy.log(self.supervised_kwargs["kappa_regularization"])
                            elif(this_layer.kappa_prediction=="softplus_real_bounded"):
                                this_log_kappa_par=numpy.log(numpy.exp(self.supervised_kwargs["kappa_regularization"])-1.0)
                            else:   
                                raise Exception("Require either log or softplus kappa parametrization!")

                            threshold_list.append(this_log_kappa_par)

                        cur_offset+=this_layer.total_param_num


                    index_list=torch.LongTensor(index_list)
                    self.kappa_indices[ind]=index_list

                    ## require shape (1,num_kappas) here to broadcast later
                    self.kappa_thresholds[ind]=torch.Tensor(threshold_list)[None,:]

            assert(f_found), "No f flow found but kappa regularization is on!"

        if(self.supervised_kwargs["fvm_width_height_regularization"]>0.0):
            self.width_height_indices=dict()
            self.width_height_thresholds=dict()

            f_found=False
            for ind, layer_list in enumerate(self.pdf.layer_list):
                all_but_last_flows=self.pdf.flow_defs_list[ind][:-1]
                if("f" in all_but_last_flows):
                    f_found=True

                if("s2" in self.pdf.pdf_defs_list[ind]):
                    index_list=[]
                    #threshold_list=[]

                    cur_offset=0
                    for sub_pdf_ind, sub_pdf_str in enumerate(self.pdf.flow_defs_list[ind]):
                        this_layer=layer_list[sub_pdf_ind]

                        if(sub_pdf_str=="f"):
                            if(this_layer.total_num_vertical_params>0):

                                # +1 because of skipping kappa, which comes from first (and after householder params)
                                this_start_index=cur_offset+this_layer.num_householder_params+1
                                this_end_index=this_start_index+this_layer.total_num_vertical_params
                                # have to start at +1 because of kappa, which comes before vertical params
                                this_slice=torch.arange(this_start_index, this_end_index)
                                index_list.append(this_slice)

                               
                        cur_offset+=this_layer.total_param_num

                    ## only add if necessary
                    if(len(index_list)>0):
                        self.width_height_indices[ind]=torch.cat(index_list)

                        ## just use a scaler (same threshold for all)
                        self.width_height_thresholds[ind]=self.supervised_kwargs["fvm_width_height_regularization"]

            assert(f_found), "Width height regularization switched on but no appropriate f flow found!"
        if(self.supervised_kwargs["consider_all_permutations"]):
            ## also add permuted versions

            num_subpdfs=len(self.pdf_args[0].split("+"))

            permuted_pdf_defs=["+".join(l) for l in list(itertools.permutations(self.pdf_args[0].split("+")))]
            permuted_flow_defs=["+".join(l) for l in list(itertools.permutations(self.pdf_args[1].split("+")))]
            permuted_indices=list(itertools.permutations(range(len(self.pdf_args[0].split("+")))))

            print("more than 1 PDF .. consider all permutations (%d in total)" % len(permuted_flow_defs))
        else:

            permuted_pdf_defs=[self.pdf_args[0]]
            permuted_flow_defs=[self.pdf_args[1]]
            permuted_indices=[range(len(self.pdf_args[0].split("+")))]
        
        self.permuted_indices=permuted_indices
        self.permuted_flow_defs=permuted_flow_defs
        self.permuted_pdf_defs=permuted_pdf_defs

        ## store the remaining n-1 permutations in module dict

        self.permuted_pdfs=torch.nn.ModuleDict()

        for perm_index, pdf_perm in enumerate(permuted_pdf_defs):
            if(perm_index>=1):
                self.permuted_pdfs[pdf_perm]=pdf_class(pdf_perm, permuted_flow_defs[perm_index], **self.pdf_kwargs)
            
        ## set diffeq layers ## TODO: somehow solve this in a better way
        for subflow_index, subflow_description in enumerate(self.pdf.pdf_defs_list):

            layer_descr=self.pdf.flow_defs_list[subflow_index]

            pdf_obj=self.pdf
            ## get the internal object
            if(type(pdf_obj)==jammy_flows.fully_amortized_pdf):
                pdf_obj=pdf_obj.pdf_to_amortize
            for layer_index, layer in enumerate(pdf_obj.layer_list[subflow_index]):

                ## sphere charts
                if(layer_descr[layer_index]=="c"):

                    layer.set_variables_from_parent(self)

            ## also do the same for other permuted pdfs
            for k in self.permuted_pdfs.keys():
                layer_descr=self.permuted_pdfs[k].flow_defs_list[subflow_index]

                pdf_obj=self.permuted_pdfs[k]
                if(type(pdf_obj)==jammy_flows.fully_amortized_pdf):
                    pdf_obj=pdf_obj.pdf_to_amortize

                for layer_index, layer in enumerate(pdf_obj.layer_list[subflow_index]):

                    ## sphere charts
                    if(layer_descr[layer_index]=="c"):

                        layer.set_variables_from_parent(self)


        self.pdf.count_parameters(verbose=True)

    
    def _forward(self, data_summary, target_labels, batch_index=None):
        # in lightning, forward defines the prediction/inference actions
        #data_summary=self._apply_encoder(batch, batch_index=batch_index)

        if(type(data_summary)==list):
            if(data_summary[0].dtype!=torch.float64):
                data_summary=[di.type(torch.float64) for di in data_summary]
        else:
            if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)
                
        log_pdf,_,_=self.pdf(target_labels, conditional_input=data_summary)

        return log_pdf
    
    """
    def _forward(self, batch_item, batchitem_length, target_label):
        return self._forward_single(batch_item, batchitem_length, target_label)
    """
    def _forward_single(self, batch_item, batchitem_length, target_label):
        # in lightning, forward defines the prediction/inference actions
        
        data_summary=self.encoder(batch_item)

        if(type(data_summary)==list):
            if(data_summary[0].dtype!=torch.float64):
                data_summary=[di.type(torch.float64) for di in data_summary]
        else:
            if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)
                
        log_pdf,_,_=self.pdf(target_label, conditional_input=data_summary)

        return log_pdf

    def _set_training_settings(self, config, datamodule=None, current_epoch=0, current_total_iter=0):
        # check regularization settings
        print("IN SET TRAINING SETTINGS")
        
        if(self.supervised_kwargs["second_moment_prefit_mode"]>0):
            if(self.supervised_kwargs["second_moment_prefit_num_epochs"]>0):
                _check_allowed_names_for_blockage(self.pdf)

                # use the hook if hook-mode, mode==1
                if(self.supervised_kwargs["second_moment_prefit_mode"]==1):
                    if(current_epoch<self.supervised_kwargs["second_moment_prefit_num_epochs"]):
                        setattr(self.pdf, "use_second_moment_hook", True)
                        _attach_second_moment_hook(self.pdf)


        self._set_specific_training_settings(config, datamodule=datamodule, current_epoch=current_epoch, current_total_iter=current_total_iter)

    def _set_specific_training_settings(self, config, datamodule, current_epoch=0, current_total_iter=0):
        """
        Overwrite by supervised subclasses
        """
        print("---> NO SPECIFIC TRAINING SETTINGS FOR THIS SUPERVISED SUBCLASS!!")
        return True

    def _training_step(self, batch, batch_idx):
            
        #######
        ## HACK to print current cluster id in stderr
        #if(self.global_step%10==0):
        #    cur_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #    #warnings.warn("W__%s__JOBID:__%s"% (cur_time,os.getenv('CONDOR_JOB_ID')))

        #    print("p__%s__JOBID:__%s"% (cur_time,os.getenv('CONDOR_JOB_ID')), file=sys.stderr)

        #######
        if(self.automatic_optimization==False):
            opti=self.optimizers()
            opti.zero_grad()

        tbeg=time.time()
        data_summary=self._apply_encoder(batch)
        
        ### TODO change hardcoded float64 for NFs
        if(type(data_summary)==list):
            if(data_summary[0].dtype!=torch.float64):
                data_summary=[di.type(torch.float64) for di in data_summary]
        else:
            if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)

                ## data summary constraint
            
        print("data encoding took .. ", time.time()-tbeg)

        labels=batch["labels"]

        #### LOG PDF bias

        self.log("s.train_kappa_bias", self.pdf.mlp_predictors[0].u_v_b_pars[0,-1].detach())

        embedded_labels=self.pdf.transform_target_into_returnable_params(labels)

        default_perm_str=self.permuted_pdf_defs[0]

        ## merge two losses over 100 steps to slowly fade from one to the other
        ## either when starting with a pure 2nd moment loss or when starting with a mixed loss
        ## Use a 10th of an epoch for merging
        num_steps_merging=int(1.0*self.trainer.num_training_batches)
        current_step_this_epoch=self.global_step-self.current_epoch*self.trainer.num_training_batches
        p_mixed=float(current_step_this_epoch)/float(num_steps_merging)
        ################################

        tbef=time.time()

        ## kappa regularization term

        
        only_last=False

        log_pdf,_,base_samples=self.pdf(embedded_labels, conditional_input=data_summary, force_embedding_coordinates=True)
        loss_batch=-log_pdf
        mean_loss=loss_batch.mean()

        if( (self.supervised_kwargs["second_moment_prefit_mode"]==3) or (self.supervised_kwargs["add_pure_second_moment_in_loss"]>0)):
            log_pdf_only_second,_,_=self.pdf(embedded_labels, conditional_input=data_summary, force_embedding_coordinates=True, only_last=True)
            loss_batch_only_second=-log_pdf_only_second
            mean_loss_only_second=loss_batch_only_second.mean()

            ## log second moment loss
            self.log("s.train_loss_perm_%s_only_sec_moment_last_flow" % default_perm_str, mean_loss_only_second.detach())
            #self.log("s.train_loss_perm_%s_diff_second_to_full" % default_perm_str, mean_loss_only_second.detach()-mean_loss.detach())
            self.log("s.train_loss_perm_%s_diff_second_to_full_per_bitem" % default_perm_str, (loss_batch_only_second-loss_batch).mean().detach())
            ## check for merging interval or mean loss modification

            if(self.current_epoch < self.supervised_kwargs["add_pure_second_moment_in_loss"]):
                ## actually add second moment loss to normal loss
                mean_loss=0.5*mean_loss+0.5*mean_loss_only_second

            ## base centering only possible when we use extra second moment
            if(False):#self.supervised_kwargs["base_centering_prefactor"]>0.0):
                reg_target=(torch.ones(size=(labels.shape[0], self.pdf.total_target_dim_intrinsic)).type_as(mean_loss).to(mean_loss))*0.001

                target_space_only_last,_,_,_=self.pdf._obtain_sample(predefined_target_input=reg_target, conditional_input=data_summary, force_embedding_coordinates=True,only_last=True)
                target_space_only_last=target_space_only_last.detach()
               
                _,_,base_space_mean=self.pdf.forward(target_space_only_last, conditional_input=data_summary, force_embedding_coordinates=True)

                base_centering_term=0.5*torch.sum((reg_target-base_space_mean)**2, dim=1).mean()

                mean_loss=mean_loss+self.supervised_kwargs["base_centering_prefactor"]*base_centering_term
            ## first for prefit
            if(self.supervised_kwargs["second_moment_prefit_mode"]==3):
                # we take the full second moment loss
                print("make decision for merging...",current_step_this_epoch,num_steps_merging)
                if(self.trainer.current_epoch<self.supervised_kwargs["second_moment_prefit_num_epochs"]):
                    mean_loss=mean_loss_only_second
                    print("only sec...")
                # we merge
                elif( (self.trainer.current_epoch==self.supervised_kwargs["second_moment_prefit_num_epochs"]) and (current_step_this_epoch<num_steps_merging)):
                    
                    mean_loss=(1.0-p_mixed)*(mean_loss_only_second)+p_mixed*mean_loss
                    print("merging..")
                

            ## second check for adding second moment loss
            """
            if(self.current_epoch<self.supervised_kwargs["add_pure_second_moment_in_loss"]):
                ## average
                
                
            elif((self.current_epoch==self.supervised_kwargs["add_pure_second_moment_in_loss"]) and (current_step_this_epoch<num_steps_merging)):
                ## merging

                mean_loss=(1.0-p_mixed)*(0.5*mean_loss+0.5*mean_loss_only_second)+p_mixed*mean_loss
            """

        self.log("s.train_loss_perm_%s" % default_perm_str, mean_loss.detach())

        ### kappa regularization
        kappa_reg_loss=0.0

        if(self.supervised_kwargs["kappa_regularization"]>0.0):

            kappa_reg_loss=kappa_regularization(self.pdf, data_summary, self.kappa_indices, self.kappa_thresholds)
            self.log("s.kappa_regularization_loss", kappa_reg_loss.detach())
            mean_loss=mean_loss+0.1*kappa_reg_loss

        ### kappa regularization
        width_min_max_loss=0.0

        if(self.supervised_kwargs["fvm_width_height_regularization"]>0.0):

            width_min_max_loss=width_height_fvm_regularization(self.pdf, data_summary, self.width_height_indices, self.width_height_thresholds)
            self.log("s.widthheight_regularization_loss", width_min_max_loss.detach())
            mean_loss=mean_loss+0.1*width_min_max_loss
            
        
        ## add other pdfs
        permuted_losses=0.0

        ind=1
        for k, permuted_pdf in self.permuted_pdfs.items():
           
            label_permutation=[]

            ## use intrinsic coordinates
            
            for cur_index in self.permuted_indices[ind]:
                inp_indices=self.pdf.target_dim_indices_intrinsic[cur_index]
               

                label_permutation.append(labels[:,inp_indices[0]:inp_indices[1]])
              
            label_permutation=torch.cat(label_permutation, dim=1)

            

            log_pdf_permuted,_,_=permuted_pdf(label_permutation, conditional_input=data_summary)

            this_loss_batch=-log_pdf_permuted.mean()

            self.log("s.train_loss_perm_%s" % self.permuted_pdf_defs[ind], this_loss_batch.detach())

            ## add to total

            permuted_losses=permuted_losses+this_loss_batch

            ind+=1

        if(permuted_losses!=0.0):
            # also add mean loss of "default permutation"
            permuted_losses=permuted_losses+mean_loss
            permuted_losses=permuted_losses/ind

            total_loss=permuted_losses
        else:
            total_loss=mean_loss

        ## add regularization if required


        if(self.supervised_kwargs["use_summary_statistic_bottleneck_prior"]):
            
            assert(type(data_summary)!=list), "summary stat bottleneck prior only supports tensor summary stat currently!"
            
            st_constraint_fac=0.1

            summary_stat_constraint=st_constraint_fac*0.5*(data_summary**2).mean()

            self.log("s.train_summary_stat_prior", summary_stat_constraint.detach())

            total_loss=total_loss+summary_stat_constraint

        ###############################################
        
        print("pdf eval took ", time.time()-tbef)

        if(self.automatic_optimization==False):
            
            tbef=time.time()

            ## check parameter values for non-finite or grad
            for nparam in self.named_parameters():
                non_fin_mask=~torch.isfinite(nparam[1].data)
                if(  non_fin_mask.sum()>0):
                    print("#<#<#<#<#<#<#<#<#<#<#<#< NON FINITE DETECTOR IN PAR VALUE (BEFORE BACKWARD)")
                    print(nparam[0])
                    print(nparam[1].data)
                    print(nparam[1].data[non_fin_mask])
                    print("#<#<#<#<#<#<#<#<#<#<#<#< NON FINITE DETECTOR IN PAR VALUE (BEFORE BACKWARD)")

                    nan_mask=torch.isnan(nparam[1].data)
                    if(nan_mask.sum()>0):

                        print("#<#<#<#<#<#<#<#<#<#<#<#< NAN DETECTED IN PAR VALUE (BEFORE BACKWARD)")
                        print(nparam[1].data)
                        print(nparam[1].data[nan_mask])
                        print("#<#<#<#<#<#<#<#<#<#<#<#< NAN DETECTED IN PAR VALUE (BEFORE BACKWARD)")

                        raise Exception("NAN detected in PAR VALUE")


            #print("returned optimizers: ", opt)
            self.manual_backward(total_loss, retain_graph=True)
            #print("backward time ... ", time.time()-tbef)

            ## check grads for non-finite entries
            for nparam in self.named_parameters():
                if(nparam[1].grad is None):
                    raise Exception("Param ", nparam[0], " none in grad=?!")
                non_fin_mask=~torch.isfinite(nparam[1].grad.data)
                if(  non_fin_mask.sum()>0):
                    print("#<#<#<#<#<#<#<#<#<#<#<#< NON FINITE DETECTOR IN GRAD")
                    print(nparam[0])
                    print(nparam[1].grad.data)
                    print(nparam[1].grad.data[non_fin_mask])
                    print("#<#<#<#<#<#<#<#<#<#<#<#< NON FINITE DETECTOR IN GRAD")

                    nan_mask=torch.isnan(nparam[1].grad.data)
                    if(nan_mask.sum()>0):

                        print("#<#<#<#<#<#<#<#<#<#<#<#< NAN DETECTED IN GRAD")
                        print(nparam[1].grad.data)
                        print(nparam[1].grad.data[nan_mask])
                        print("#<#<#<#<#<#<#<#<#<#<#<#< NAN DETECTED IN GRAD")
                        
                        

                    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< SETTING GRAD IN NON-FIN TO ZERO")
                    nparam[1].grad.data[non_fin_mask]=0.0

            ## log grad norm
            grads = [param.grad.detach().flatten() for param in self.parameters()]
            cur_grad_vec=torch.cat(grads)
            grad_norm = torch.linalg.vector_norm(cur_grad_vec)
            self.log("s.train_grad_norm", grad_norm)

            if(self.optimizer_config["optimizer.name"]=="adam"):
                exp_avg_grad_norm=0.0
                if(self.trainer.global_step>3):
                    opti=self.optimizers()
                    num_groups=0
                    exp_avg=[]
                    exp_avg_sq=[]
                    for group in opti.param_groups:
                        num_groups+=1
                        for p in group["params"]:
                            exp_avg.append(opti.state[p]["exp_avg"].flatten())
                            exp_avg_sq.append(opti.state[p]["exp_avg_sq"].flatten())
                  
                    exp_avg=torch.cat(exp_avg)
                    exp_avg_sq=torch.cat(exp_avg_sq)
                    approx_sigmas=torch.sqrt((exp_avg_sq+1e-8))

                    ## log the chi value
                    grad_chi_val=torch.sqrt(torch.sum(((cur_grad_vec-exp_avg)/approx_sigmas)**2)).detach()
                    self.log("s.train_grad_chi_value", grad_chi_val)
                    self.log("s.train_grad_chi_dof", len(approx_sigmas))
                    assert(num_groups==1), "Gradient exp avg norm is only working for a single combined param group!"

                    ## we got running mean and running cov -> calculate mean and CDF values of corresponding  
                    exp_avg_grad_norm = torch.linalg.vector_norm(exp_avg)

                    ## calculate cos(alpha) between normed grad and normed "exp averaged" grad
                    self.log("s.cos_alpha_grad_exp_grad", ((cur_grad_vec/grad_norm)*(exp_avg/exp_avg_grad_norm)).sum().detach())

                    ### constrain grad norm to exp_avg_grad_norm*10
                    if(self.optimizer_config["optimizer.relative_chi_trust_region"]>0.0):

                        desired_max_chi=self.optimizer_config["optimizer.relative_chi_trust_region"]*numpy.sqrt(float(len(approx_sigmas)))
                        ratio_to_desired=float(grad_chi_val/desired_max_chi)
                        
                        ## scale grads by relative down scaling
                        if(ratio_to_desired>1.0):
                            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_norm/ratio_to_desired, norm_type=2.0)

                    elif(self.optimizer_config["optimizer.relative_grad_norm_factor"]>0.0):

                        torch.nn.utils.clip_grad_norm_(self.parameters(), exp_avg_grad_norm*self.optimizer_config["optimizer.relative_grad_norm_factor"], norm_type=2.0)
                        
                self.log("s.train_exp_avg_grad_norm", exp_avg_grad_norm)

            if(self.optimizer_config["optimizer.absolute_grad_norm_allowed"]>0.0):
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.optimizer_config["optimizer.absolute_grad_norm_allowed"], norm_type=2.0)

            ## grad norm after clippinf
            grads = [param.grad.detach().flatten() for param in self.parameters() if param.grad is not None]
            grad_norm = torch.linalg.vector_norm(torch.cat(grads))
            self.log("s.train_grad_norm_clipped", grad_norm)

            batch_size=labels.shape[0]
            
            optimizer_kwargs=dict()
            """
            if(self.optimizer_config["optimizer.fisher_mode"]=="fisher_function"):
                raise Exception("Fisher function not supported for flows")
                optimizer_kwargs["fisher_functions"]=(self.fisher_fn_for_optimization,)
                optimizer_kwargs["batch_data"]=batch
            else:

            """
            
            
            ## only in prefit mode 2 (manual zero gradient)
            if(self.supervised_kwargs["second_moment_prefit_mode"]==2):
                if(self.trainer.current_epoch<self.supervised_kwargs["second_moment_prefit_num_epochs"]):
                    
                    ## swtich off relevant gradients
                    for subpdf_ind, mlp_obj in enumerate(self.pdf.mlp_predictors):
                       
                        number_last_flow_trafo=self.pdf.layer_list[subpdf_ind][-1].get_total_param_num()

                        if(type(mlp_obj)==torch.nn.Sequential):
                            
                            assert(type(mlp_obj[-1]==torch.nn.Linear))
                           
                            # set relevant grads to 0
                            
                            mlp_obj[-1].weight.grad[:-number_last_flow_trafo,:]=0.0
                            mlp_obj[-1].bias.grad[:-number_last_flow_trafo]=0.0
                        elif(type(mlp_obj)==AmortizableMLP):
                            if(mlp_obj.highway_mode>1):
                                raise NotImplementedError()
                            
                            uvb_tensor=mlp_obj.u_v_b_pars
                            
                            ## go through MLP first
                            uvb_index=0
                            for mlp_index, mlp_def in enumerate(mlp_obj.sub_mlp_structures["mlp_list"]):
                                assert(mlp_def["svd_mode"]=="smart")  
                                for linear_index, inp in enumerate(mlp_def["inputs"]):
                                   
                                    out=mlp_def["outputs"][linear_index]
                                    assert(mlp_def["used_ranks"][linear_index]==mlp_def["max_ranks"][linear_index])
                                    assert(mlp_def["full_weight_matrix_flags"][linear_index]==1)

                                    this_num=mlp_def["num_u_s"][linear_index]
                                    
                                    ## set grads to 0 if we are in the last index
                                    if(linear_index == (len(mlp_def["inputs"])-1)):
                                        weights_grad=uvb_tensor.grad[:,uvb_index:uvb_index+this_num].view(1, out, inp)
                                        
                                        weights_grad[0,:-number_last_flow_trafo,:]=0.0

                                        
                                    uvb_index+=this_num

                                    
                                    this_num=mlp_def["num_b_s"][linear_index]
                                    
                                    # check if bias is used in this linear layer
                                    if(this_num>0):
                                            
                                        # check if we are in last layer
                                        if(linear_index == (len(mlp_def["inputs"])-1)):
                                            bias_grad=uvb_tensor.grad[:,uvb_index:uvb_index+this_num].view(1,out)

                                            bias_grad[0, :-number_last_flow_trafo]=0.0
                                            
                                        uvb_index+=this_num
                                    
                                    
                            ## extra linear layer comes afterwards
                            if("linear_highway" in mlp_obj.sub_mlp_structures):
                               
                                linear_mapping=mlp_obj.sub_mlp_structures["linear_highway"]

                                assert(linear_mapping["svd_mode"]=="smart")
                                assert(linear_mapping["used_ranks"][0]==linear_mapping["max_ranks"][0])
                                assert(linear_mapping["full_weight_matrix_flags"][0]==1)

                                out=linear_mapping["outputs"][0]
                                inp=linear_mapping["inputs"][0]

                                this_num=linear_mapping["num_u_s"][0]

                                weights_grad=uvb_tensor.grad[:,uvb_index:uvb_index+this_num].view(1, out, inp)
                                weights_grad[0,:-number_last_flow_trafo,:]=0.0
                                
                                uvb_index+=this_num

                                this_num=linear_mapping["num_b_s"][0]
                                assert(this_num>0)
                                        
                                bias_grad=uvb_tensor.grad[:,uvb_index:uvb_index+this_num].view(1,out)
                                bias_grad[0, :-number_last_flow_trafo]=0.0
                                
                                uvb_index+=this_num

                             
                                
                        else:


                            raise NotImplementedError("Using prefit mode 2, but MLP type not supported currently")
            
            opti.step(**optimizer_kwargs)
            print("manual opti step took ... ", time.time()-tbef)
    

        self.log("s.train_loss", total_loss.detach())

        #return {"mean_accuracy": mean_accuracy}

        return total_loss
    
    def _validation_step(self, batch, batch_idx, validation_name, log=True):
        # loss calculation
        #torch.use_deterministic_algorithms(False)
        #batch_loss, base_samples=self.batch_loss_fn(batch, self)

        data_summary=self._apply_encoder(batch)

        labels=batch["labels"]

        loss_dict=dict()

        ### TODO change hardcoded float64 for NFs
        if(type(data_summary)==list):
            if(data_summary[0].dtype!=torch.float64):
                data_summary=[di.type(torch.float64) for di in data_summary]
        else:
            if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)
    
        embedded_labels=self.pdf.transform_target_into_returnable_params(labels)

        log_pdf,log_pdf_base,base_samples=self.pdf(embedded_labels, conditional_input=data_summary, force_embedding_coordinates=True)
        per_item_neg_log_loss=-log_pdf

        if(self.supervised_kwargs["add_final_second_moment_flow"]):
            ## save second moment loss as a test

            log_pdf_2nd,_,_=self.pdf(embedded_labels, conditional_input=data_summary, force_embedding_coordinates=True, only_last=True)
            loss_batch_pure_second=-log_pdf_2nd
            mean_loss_pure_second=loss_batch_pure_second.mean().detach()
            this_name="s.val_loss_only_sec_moment_last_flow_%s" % validation_name
            if(log):
                self.log(this_name, mean_loss_pure_second, add_dataloader_idx=False)
                
            loss_dict[this_name]=mean_loss_pure_second

            ## difference second moment to full
            this_name="s.val_diff_second_to_full_per_bitem_%s" % validation_name
            diff_val=(loss_batch_pure_second-per_item_neg_log_loss).mean().detach()
            if(log):
                self.log(this_name, diff_val , add_dataloader_idx=False)
                
            loss_dict[this_name]=diff_val
       
        if(log):
            self.log("s.val_loss_perm_%s_%s" % (self.permuted_pdf_defs[0], validation_name), per_item_neg_log_loss.mean(), add_dataloader_idx=False)

            ## add other pdfs

            ind=1
            for k, permuted_pdf in self.permuted_pdfs.items():
               
                label_permutation=[]

                ## use intrinsic coordinates
                
                for cur_index in self.permuted_indices[ind]:
                    inp_indices=self.pdf.target_dim_indices_intrinsic[cur_index]
                  
                    label_permutation.append(labels[:,inp_indices[0]:inp_indices[1]])
                    
                label_permutation=torch.cat(label_permutation, dim=1)

                log_pdf_permuted,_,_=permuted_pdf(label_permutation, conditional_input=data_summary)

                this_per_item_neg_log_loss=-log_pdf_permuted

                self.log("s.val_loss_perm_%s_%s" % (self.permuted_pdf_defs[ind], validation_name), this_per_item_neg_log_loss.mean().detach(), add_dataloader_idx=False)

                ## add to total

                per_item_neg_log_loss=per_item_neg_log_loss+this_per_item_neg_log_loss

                ind+=1

            per_item_neg_log_loss=per_item_neg_log_loss/ind

        

        mean_total_loss=per_item_neg_log_loss.mean().detach()
        total_loss_name="s.val_loss_total_%s" % validation_name
        loss_dict[total_loss_name]=mean_total_loss
       
        #loss_dict["total"]=mean_total_loss

        if(log):
            self.log(total_loss_name, mean_total_loss, add_dataloader_idx=False)

            ## approx 2*(log(p0)-log(p_label) over batch
            
            if(len(self.pdf.pdf_defs_list)==1):
                moments=self.pdf.marginal_moments(samplesize=1000, conditional_input=data_summary)
                max_log_pdfs,_,_=self.pdf(torch.from_numpy(moments["argmax_0"]).to(log_pdf), conditional_input=data_summary, force_embedding_coordinates=True)
                

                avg_approx_chi2_target=(2*(max_log_pdfs-log_pdf)).mean()
                this_name="s.avg_chi2_target_%s" % validation_name
                self.log(this_name, avg_approx_chi2_target, add_dataloader_idx=False)

                std_approx_chi2_target=(2*(max_log_pdfs-log_pdf)).std()
                this_name="s.std_chi2_target_%s" % validation_name
                self.log(this_name, std_approx_chi2_target, add_dataloader_idx=False)

            _,avg_approx_chi2_base,_=jammy_flows.helper_fns.coverage.calculate_approximate_coverage(log_pdf_base.cpu().detach(), self.pdf.total_target_dim_intrinsic, numpy.array([0.5]))
            
            this_name="s.avg_chi2_base_%s" % validation_name
            self.log(this_name, avg_approx_chi2_base.mean(), add_dataloader_idx=False)
            
            this_name="s.std_chi2_base_%s" % validation_name
            self.log(this_name, avg_approx_chi2_base.std(), add_dataloader_idx=False)
            
        ###################

        ## calculate regularization here

        center_regularization_terms=_contour_regularization_terms(self.pdf, labels, sample_size_per_item=100, conditional_input=data_summary, regularization_space="base")
        center_reg_loss=center_regularization_terms.mean().detach()
        this_name="s.val_contour_center_regularization_%s" % validation_name
        if(log):
            center_regularization_terms=_contour_regularization_terms(self.pdf, labels, sample_size_per_item=100, conditional_input=data_summary, regularization_space="base")
            self.log(this_name, center_reg_loss, add_dataloader_idx=False)

        per_item_entropy=dict()

        if(hasattr(self.pdf, "entropy")):
            ### TODO.. implement entropy for amoritzed pdf
            ########################################

            with torch.no_grad():

                """
                if(type(data_summary)==list):
                    smaller_data_summary=[s[:20] for s in data_summary]
                else:
                    smaller_data_summary=data_summary[:20]
                """
                try:
                    if(len(self.pdf.pdf_defs_list)==1):
                        if(self.pdf.pdf_defs_list[0]=="s2"):
                            ent_dict=self.pdf.entropy(samplesize=100,conditional_input=data_summary, sub_manifolds=[-1])
                            per_item_entropy["val_dir_entropy"]=ent_dict["total"].detach()
                             
                except:
                    raise Exception("Entropy calculation in validation failed!?")

                
                for k in per_item_entropy.keys():
                    this_name=k+"_%s" % validation_name
                    loss_dict[this_name]=per_item_entropy[k].mean()
                    if(log):
                        self.log(this_name, per_item_entropy[k].mean(), add_dataloader_idx=False)

                    ### diff of entropy to negative cross entropy
                    this_name=k+"_%s_minus_crossent" % validation_name
                    loss_dict[this_name]=per_item_entropy[k].mean()-loss_dict[total_loss_name]

                    if(log):
                        self.log(this_name, loss_dict[this_name], add_dataloader_idx=False)
        
        ## calculate extra custom calculations for specific use cases
        custom_returns=self._validation_custom_calculations(batch, per_item_neg_log_loss, per_item_entropy, validation_name, log=log)
        for c in custom_returns:
            loss_dict[c]=custom_returns[c]

        ## check if only second moment is running
        if(log):
            if(self.trainer.current_epoch>=self.supervised_kwargs["second_moment_prefit_num_epochs"]):
                if(hasattr(self.pdf, "use_second_moment_hook")):
                    if(self.pdf.use_second_moment_hook):
                        self.pdf.use_second_moment_hook=False
                        print(">>>>>>>>>> SWITCHED OFF MOMENT HOOK <<<<<<<<<")

        return loss_dict

    def entropy(self, batch, samples_per_event=50, sub_manifolds=[-1]):
        """
        Inference function.
        """

        with torch.no_grad():

            data_summary=self._apply_encoder(batch)

            if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)

            result=self.pdf.entropy(conditional_input=data_summary, samplesize=samples_per_event, sub_manifolds=sub_manifolds)

            return result

    

    def obtain_flow_params(self, batch, batch_idx):
        
        data_summary=self._apply_encoder(batch, batch_index=batch_idx)

        if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)

        flow_structure=self.pdf.obtain_flow_param_structure(conditional_input=data_summary)
        
        d=collections.OrderedDict()

        d["pdf"]=flow_structure

        return d

    def _init_pdfs(self, labels):

        ## initialize pdf with prior
        self.pdf.init_params(data=labels)

        ind=1
        for k, permuted_pdf in self.permuted_pdfs.items():
            
            label_permutation=[]

            ## use intrinsic coordinates
            
            for cur_index in self.permuted_indices[ind]:
                inp_indices=self.pdf.target_dim_indices_intrinsic[cur_index]
               
                label_permutation.append(labels[:,inp_indices[0]:inp_indices[1]])
              
            label_permutation=torch.cat(label_permutation, dim=1)

            #############################

            permuted_pdf.init_params(label_permutation)

            ind+=1

        

    def _obtain_parameter_specs_for_optimization(self, optimizer_kwargs):

        cfg_parser=config_parser.config_parser()

        cfg_parser.add_default_kwarg("optimizer", "optimizer.fisher_for_known_posterior", 1, int)
        
        _, fisher_kwargs=cfg_parser.parse_cfg(optimizer_kwargs, "optimizer", drop_name_piece=True)

        param_structure=collections.OrderedDict()

        if(fisher_kwargs["fisher_for_known_posterior"]):

            union_param_indices, union_group_indices=self.helper_get_relative_indices_by_names(["encoder", "pdf"])

            param_structure["pdf"]=dict()
            param_structure["pdf"]["param_indices"]=union_param_indices
            #param_structure["pdf"]["shifted_param_indices_relative_to_single"]=shifted_param_indices
            param_structure["pdf"]["group_indices"]=union_group_indices
            
        return param_structure

    ###########################

    def marginal_moments_and_others(self, 
                                    batch, 
                                    samples_per_event=50, 
                                    mises_abs_precision=1e-7, 
                                    iterative_samplesize=10, 
                                    max_iterative_batchsize=20, 
                                    use_gpu=False, 
                                    failsafe_crosscheck_tolerance=None, 
                                    s2_entropy_scanning=0, 
                                    s2_entropy_scan_nside=128, 
                                    calc_kl_diff_and_entropic_quantities=True,
                                    save_pdf_scan=False,
                                    exact_coverage_calculation=False,
                                    coverage_num_percentile_points=100,
                                    calculate_MAP=False,
                                    save_summary_statistic=False,
                                    save_flow_params=False):
        """
        Inference function for marginal moments.
        """
        print("iterative batchsize ", max_iterative_batchsize)
        
        with torch.no_grad():

            data_summary=self._apply_encoder(batch)

            if(type(data_summary)==list):
                if(data_summary[0].dtype!=torch.float64):
                    data_summary=[di.type(torch.float64) for di in data_summary]

                assert(save_summary_statistic==False)
            else:
                if(data_summary.dtype!=torch.float64):
                    data_summary=data_summary.type(torch.float64)
            


            tbef=time.time()

            #self.pdf.set_use_embedding_parameters_flag(True)
            
            moment_dict=self.pdf.marginal_moments(conditional_input=data_summary, 
                                               samplesize=samples_per_event, 
                                               iterative_samplesize=iterative_samplesize, 
                                               max_iterative_batchsize=max_iterative_batchsize, 
                                               mises_abs_precision=mises_abs_precision, 
                                               failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                                               calc_kl_diff_and_entropic_quantities=calc_kl_diff_and_entropic_quantities,
                                               s2_entropy_scanning=s2_entropy_scanning,
                                               s2_entropy_scan_nside=s2_entropy_scan_nside)

            
            if(save_summary_statistic):
                 moment_dict["summary_statistic"]=data_summary

            if(save_flow_params):
                flow_params=self.pdf.mlp_predictors[0](data_summary).detach().cpu().numpy()
                moment_dict["flow_params"]=flow_params
                 
            used_labels=None
            if("labels" in batch):
                used_labels=batch["labels"]
                

            
            if(used_labels is not None):
                assert(data_summary.shape[0]==used_labels.shape[0]), "Label and data summary shapes do not match!"

            coverage_dict=self.pdf.coverage_and_or_pdf_scan(conditional_input=data_summary,
                                                            labels=used_labels,
                                                            exact_coverage_calculation=exact_coverage_calculation,
                                                            coverage_num_percentile_points=coverage_num_percentile_points,
                                                            save_pdf_scan=save_pdf_scan,
                                                            calculate_MAP=calculate_MAP)


            moment_dict.update(coverage_dict)
            
        print(">>>> moment calculation took ", time.time()-tbef, " secs")
        return moment_dict



    def _init_encoder(self, config):

        ## has to define encoder
        raise NotImplementedError

    def _apply_encoder(self, batch, batch_index=None):
        """
        Subclasses can define various ways how to encode data.
        """
        raise NotImplementedError
    
    def _validation_custom_calculations(self, batch, per_batchitem_loss, validation_name):
        ## implementes custom logging and calculations based on per-batch items
        raise NotImplementedError("Implement function for custom calculations in validation loss.")
    

