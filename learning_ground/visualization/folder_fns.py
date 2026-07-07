import sys
import glob
import os
import yaml
import awkward
import numpy
import pandas
import importlib

from learning_ground.data.datamodule_base import obtain_further_configs

def _generate_inference_file(inference_file_filename, base_config, extra_config_options=dict(), only_include_keys=[]):

    ## first look for data.test.* strings in params.json
    ## If not found, copy data.train and overwrite with extra config options.
    ## Throws an error if no extra config options are given, since we dont want to test on train events
    ## by default.

    dataset_kwargs=dict()

    if("data.test." in base_config):
        sys.exit(-1)
        new_configs=obtain_further_configs(base_config, config_descriptor="val")
    else:
        ## take data.train
        #for k in sorted(base_config.keys()):
        #    print(k, base_config[k])
        #assert("data.train" in base_config.keys()), [kk for kk in base_config.keys()]

        new_configs=obtain_further_configs(base_config, config_descriptor="val")

        for opt_string in new_configs[0].keys():
            if("data.val." in opt_string):
                dataset_kwargs[opt_string[9:]]=new_configs[0][opt_string]
            elif("data.val0." in train_opt):
                dataset_kwargs[opt_string[10:]]=new_configs[0][opt_string]

    
    for k in extra_config_options:
        assert(k in dataset_kwargs.keys()), ("Key ", k, " to overwrite is not found in standard settings...")
        dataset_kwargs[k]=extra_config_options[k]
        print("overwrite ", extra_config_options[k])

    print("intermediate kwargs")
    all_preliminary_keys=[k for k in dataset_kwargs.keys()]
    ### HACK for icecube defs.. should be done differently from the start, cleanly separating data.train and data.val
    ### but for some vars it wasnt done
    for k in all_preliminary_keys:
        if("add_cog_diff" in k):
            del dataset_kwargs["add_cog_diff"]

        if("batch_size" in k):
            del dataset_kwargs["batch_size"]

        if("correct_for_domefficiency_and_remove_domtype" in k):
            del dataset_kwargs["correct_for_domefficiency_and_remove_domtype"]

        ## name is arg, not kwargs
        if("name" in k):
            del dataset_kwargs["name"]

        if("num_workers" in k):
            del dataset_kwargs["num_workers"]

        if("only_max_q_doms" in k):
            del dataset_kwargs["only_max_q_doms"]

        if("shifting_mode_pos" in k):
            del dataset_kwargs["shifting_mode_pos"]

        if("shifting_mode_time" in k):
            del dataset_kwargs["shifting_mode_time"]
        
        if("shuffle_times" in k):
            del dataset_kwargs["shuffle_times"]

        if("use_mc_diffs" in k):
            del dataset_kwargs["use_mc_diffs"]

        if("enforce_equal_no_in_energy_decades" in k):
            del dataset_kwargs["enforce_equal_no_in_energy_decades"]

        if("cog_diff_ignore_saturated" in k):
            del dataset_kwargs["cog_diff_ignore_saturated"]

        if("randomized_xyz_jitter" in k):
            del dataset_kwargs["randomized_xyz_jitter"]
    
    """
    ### write standard inference dict + some extra default options to inference file
    print("ONLY INCLUDE KEYS ", only_include_keys)
    if(len(only_include_keys)>0):
        for cur_ds_kwarg in [kk for kk in dataset_kwargs.keys()]:
            if(cur_ds_kwarg not in only_include_keys):
                del dataset_kwargs[cur_ds_kwarg]
    """

    print("final ds kwargs")
    print(dataset_kwargs)
    ## TODO: change hardcodings of dataloader/env
    inference_yaml_dict=dict()
    inference_yaml_dict["dataloader"]="icecube_learning_ground.data.loading.generic_datamodule.IceCubeSingleDatamodule"
    inference_yaml_dict["dataset_kwargs"]=dataset_kwargs
    inference_yaml_dict["dataset_name"]="test"
    inference_yaml_dict["env"]="icecube_learning_ground.envs.supervised.sl_neutrino_env.sl_neutrino_env"

    inf_functions=dict()
    inf_functions["marginal_moments_and_others"]=dict()
    inf_functions["marginal_moments_and_others"]["exact_coverage_calculation"]=True
    inf_functions["marginal_moments_and_others"]["samples_per_event"]=10000
    inf_functions["marginal_moments_and_others"]["save_pdf_scan"]=False
    inf_functions["marginal_moments_and_others"]["save_summary_statistic"]=True

    inference_yaml_dict["inference_functions"]=inf_functions

    ## write inference file
    with open(inference_file_filename, "w") as yfile:
        yaml.dump(inference_yaml_dict, yfile)


def _obtain_extra_config_options(arg_strings):
    print("in ob")
    opts_dict=dict()

    only_include_keys=[]

    for arg_string in arg_strings:
        splits=arg_string.split("=")
        this_key=splits[0]
        if(this_key=="only_include"):
            ## add only include keys
            only_include_keys=splits[1].split(",")
            print("the only include keys ", only_include_keys)
            continue
        print("this key ", this_key)

        this_val=float(splits[1])

        opts_dict[this_key]=this_val
    print("final only_include_keys", only_include_keys)
    return opts_dict, only_include_keys

def _obtain_cp_dirs_and_init_state(tot_folder, app_state):

    if 'open_run_tabs' in app_state:
        return

    top_folders=glob.glob(os.path.join(tot_folder, "trainable*"))

    top_folders.sort()

    ret_dict=dict()
    print("Initializing STATE.............")
    print("################################")
   
    ## run ids that identify runs

    ## open tab information for specific runs
    app_state.open_run_tabs = {}

    app_state.open_run_plotting_tabs = {}

    ## open tab information for models
    app_state.open_model_tabs = {}

    app_state.preloaded_inference_models = {}
    
    #app_state.preloaded_data_images = {}
    #app_state.preloaded_posterior_images = {}

    for t in top_folders:

        basename_t=os.path.basename(t)
        print("basename ", basename_t)
        basename_t="_".join(basename_t.split("_")[1:3])
        ret_dict[basename_t]=dict()

        further_subfolds=glob.glob(os.path.join(t, "checkpoint_epoch*"))

        associated_steps=[int(f.split("=")[-1]) for f in further_subfolds]

        sorted_subfolds = [x for _,x in sorted(zip(associated_steps,further_subfolds))]
        sorted_steps=[x for x,_ in sorted(zip(associated_steps,further_subfolds))]

        ret_dict[basename_t]["params_json_file"]=os.path.join(t, "params.json")
            
        ## load params.json
        #loaded_params_config=learning_env()._load_inference_config(ret_dict[t]["params_json_file"])

        ret_dict[basename_t]["steps"]=sorted_steps
        ret_dict[basename_t]["folders"]=sorted_subfolds
        ret_dict[basename_t]["checkpoints"]=[os.path.join(ii, "cp_validation_end") for ii in ret_dict[basename_t]["folders"]]

        for find, f in enumerate(ret_dict[basename_t]["folders"]):
            
            res=glob.glob(os.path.join(f, "*/*.awk"))

            if(len(res)==1):
                ## inference result file found
               
                ## create tab
                ## update state of processing
                step_to_process=sorted_steps[find]
                this_key=(basename_t, step_to_process)
                step_index=sorted_steps.index(step_to_process)

                app_state.open_run_tabs[this_key]=dict()

                ak_array=awkward.from_parquet(res[0])
                first_field=ak_array.fields[0]
                first_field_npy=ak_array[first_field].to_numpy()
               
                slices = [slice(None)]  # Start with [:]
                for _ in range(1, first_field_npy.ndim):
                    slices.append(0)
                
                slices = tuple(slices)
                nan_mask=numpy.isnan(first_field_npy[slices])
               
                num_nan=nan_mask.sum()
                num_tot=len(ak_array[first_field].to_numpy()) 
                

                print(awkward.to_dataframe(ak_array[~nan_mask]))

                app_state.open_run_tabs[this_key]["num_tot"]=num_tot
                app_state.open_run_tabs[this_key]["num_processed"]=num_tot-num_nan

                data, summary_statistic=awk_to_dataframe(ak_array[~nan_mask])
                app_state.open_run_tabs[this_key]["data"]=data
                app_state.open_run_tabs[this_key]["summary_statistic"]=summary_statistic
                app_state.open_run_tabs[this_key]["checkpoint"]=ret_dict[basename_t]["checkpoints"][step_index]
                print("mena angles")
                print(awkward.to_dataframe(ak_array[~nan_mask])["mean_0_angles"])

        print("processed ... ", len(ret_dict[basename_t]["folders"]))
        ## only create inference file when necessary!
        """
        for f in ret_dict[t]["folders"]:
            this_cfg_filename=os.path.join(f, "inference_config.yaml")
            inference_configs.append(this_cfg_filename)

            if(not os.path.exists(this_cfg_filename)):
                ## make inference config
                _generate_inference_file(this_cfg_filename, loaded_params_config, extra_config_options=extra_config_options)

        """

        
        #ret_dict[basename_t]["inference_configs"]=[os.path.join(ii, "cp_validation_end") for ii in ret_dict[basename_t]["folders"]]

        # Initialize session state

    ## run ids that identify runs
    app_state.run_ids = [k for k in ret_dict.keys()]  # Example initial folders
 
    ## log structure dict based on run_ids
    app_state.log_structure=ret_dict

 

    app_state.selected_left=app_state.run_ids[0]
    app_state.selected_right=app_state.log_structure[app_state.selected_left]["steps"][0]

    app_state.plot_id_key=None


def awk_to_dataframe(awk_arr):

    ff=awk_arr.fields[0]
    np_first=awk_arr[ff].to_numpy()
    desired_shape=(len(np_first),)


    # Initialize a dictionary to collect the fields with the desired shape
    filtered_data = {}

    # Loop through the fields and check their shapes
    for field in awk_arr.fields:
        # Get the field array
        # Check if the shape matches the desired shape
        field_array = awkward.to_numpy(awk_arr[field])

        if (field_array.shape == desired_shape) and (field_array.dtype.kind in {'i', 'f'}):
            
            filtered_data[field] = field_array
        else:
            ## hacks for string based fields .. should be automatically given in integer
           
            if(field=="containment" or field=="interaction_type"):
                unique_values=set(field_array)

                new_integer_array=numpy.zeros(field_array.shape)
                for this_new_id,unique_val in enumerate(unique_values):
                    this_mask=field_array==unique_val
                    new_integer_array[this_mask]=this_new_id

                filtered_data[field]=new_integer_array

                print("new integer array ", new_integer_array)



    # Create a DataFrame from the filtered fields
    df = pandas.DataFrame(filtered_data)

    ## also extract summary stats in extra array

    summary_stats=None
    if("summary_statistic" in awk_arr.fields):
        summary_stats=awkward.to_numpy(awk_arr["summary_statistic"])

    return df, summary_stats

def get_inference_model_from_inference_file(checkpoint_filename, inference_filename, json_model_config):

    with open(inference_filename, "r") as stream:
        try:
            res=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_module=None

    assert("dataloader" in res.keys()), "Require *dataloader* entry in inference .yaml file to know how to read data. Specfiy module as x.y.dataloaderclass"

    module_name, attr_name=res["dataloader"].rsplit(".", 1)
    data_module=getattr(importlib.import_module(module_name), attr_name)

    overwrite_config=dict()
    if("overwrite_config" in res):
        overwrite_config=res["overwrite_config"]

    assert("env" in res.keys()), "Require *env* entry in inference.yaml to know which env to use."

    module_name, attr_name=res["env"].rsplit(".", 1)
    env_module=getattr(importlib.import_module(module_name), attr_name)

    ## TODO: use gpu by default?
    model_container=env_module(inference_config=json_model_config, averaging_mode="swa", inference_model_path=checkpoint_filename, inference_batch_size=50, use_gpu=1, overwrite_config=overwrite_config)
    
    dataset_kwargs=dict()
    if("dataset_kwargs" in res):
        dataset_kwargs=res["dataset_kwargs"]
    model_container.add_dataset(data_module, res["dataset_name"], extra_dataset_params=dataset_kwargs)

    return model_container.inference_model, res["dataset_name"],model_container