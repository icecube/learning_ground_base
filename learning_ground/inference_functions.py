import sys
import os
import numpy
import awkward
import yaml
import importlib

from tqdm import tqdm


def full_inference_with_saving(model_container, 
                              inference_functions, 
                              inference_filename="./inference_casc_c.pd",
                              dataset_name="cascades",
                              sweeping_batchsize=50,
                              cap_num_inference_events_to=-1,
                              use_gpu=False,
                              model_file="",
                              model_cfg_file="",
                              inference_cfg_file="",
                              **kwargs):

    
    dataset_len=model_container.dataset_sizes[dataset_name]

    # list of indices in the dataset in batches
    index_list=numpy.arange(dataset_len).reshape(-1, sweeping_batchsize)

    if(not os.path.exists(inference_filename)):
        print("INF FILENAME DOES NOT EXSIT .. .create new one")
        ## get the first index to test structure
        test_res=model_container.inference(inference_functions, batch_indices=[0], dataset_name=dataset_name, **kwargs)

        ## then generate a panda file that fills everyting with infs

        pre_structure=dict()
        for k in test_res:
            
            init_shape=(dataset_len,)+test_res[k].shape[1:]

            if(test_res[k].dtype==numpy.int64 or test_res[k].dtype==numpy.float64):
                pre_structure[k]=numpy.ones(init_shape, dtype=test_res[k].dtype)*numpy.nan


        ak_array = awkward.Array(pre_structure)

        ak_array["__model_file"]=numpy.array([os.path.abspath(model_file)])
        if(model_cfg_file is None):
            ak_array["__model_cfg_file"]=None
        else:
            ak_array["__model_cfg_file"]=numpy.array([os.path.abspath(model_cfg_file)])

        if(inference_cfg_file is None):
            ak_array["__inference_cfg_file"]=None
        else:
            ak_array["__inference_cfg_file"]=numpy.array([os.path.abspath(inference_cfg_file)])

        awkward.to_parquet(ak_array, inference_filename)

    ak_array=awkward.from_parquet(inference_filename)

    ## find an array for checking

    first_field=ak_array.fields[0]
    
    total_num_inference_thistime=0
    for cur_index_list in tqdm(index_list):

        if( numpy.isnan(ak_array[first_field][cur_index_list].to_numpy()).sum() == 0):
            #print(first_field)
            #print("not nan?!", ak_array[first_field][cur_index_list].to_numpy())
            continue
     
       
        res=model_container.inference(inference_functions, batch_indices=cur_index_list, dataset_name=dataset_name, use_gpu=use_gpu, **kwargs)
        
        for k in res:
         
            if(k in ak_array.fields):

                numpy.asarray(ak_array[k])[cur_index_list]=res[k]
        
        awkward.to_parquet(ak_array, inference_filename)
        
        print("saved file....")

        total_num_inference_thistime+=len(cur_index_list)

        if(cap_num_inference_events_to>0):
            if(total_num_inference_thistime>=cap_num_inference_events_to):
                break


def perform_inference(model_file, 
                      inference_data_file, 
                      model_cfg_file=None, 
                      inference_batch_size=50, 
                      cap_num_inference_events_to=-1,
                      use_gpu=0,
                      iterative_samplesize=50,
                      averaging_mode="single",
                      target_folder=""):

    with open(inference_data_file, "r") as stream:
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

    model_container=env_module(inference_config=model_cfg_file, averaging_mode=averaging_mode, inference_model_path=model_file, inference_batch_size=inference_batch_size, use_gpu=use_gpu, overwrite_config=overwrite_config)
    
    dataset_kwargs=dict()
    if("dataset_kwargs" in res):
        dataset_kwargs=res["dataset_kwargs"]
    
    assert("dataset_name" in res), "Require a dataset_name in config"

    model_container.add_dataset(data_module, res["dataset_name"], extra_dataset_params=dataset_kwargs)

    if(target_folder==""):
        inf_folder=os.path.join(os.path.dirname(model_file), "inf_results_"+os.path.basename(inference_data_file).replace(".yaml", ""))
        if(not os.path.exists(inf_folder)):
            os.makedirs(inf_folder)
    else:
        inf_folder=target_folder

    inference_filename=os.path.join(inf_folder, "results_avgmode_%s.awk" % averaging_mode)

    inference_names=[]
    inference_opts=[]

    for fn_name in res["inference_functions"]:

        inference_names.append(fn_name)

        extra_fn_kwargs=dict()
        for extra_kwarg_name in res["inference_functions"][fn_name]:
            extra_fn_kwargs[extra_kwarg_name]=res["inference_functions"][fn_name][extra_kwarg_name]

        inference_opts.append(extra_fn_kwargs)

    full_inference_with_saving(model_container, 
                              inference_names, 
                              inference_filename=inference_filename,
                              dataset_name=res["dataset_name"],
                              sweeping_batchsize=inference_batch_size,
                              cap_num_inference_events_to=cap_num_inference_events_to,
                              iterative_samplesize=iterative_samplesize,
                              max_iterative_batchsize=200,
                              inference_function_options=inference_opts,
                              use_gpu=use_gpu,
                              model_file=model_file,
                              model_cfg_file=model_cfg_file,
                              inference_cfg_file=inference_data_file)

