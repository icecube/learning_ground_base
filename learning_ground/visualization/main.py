import streamlit as st 
from streamlit_elements import elements, dashboard, mui
from streamlit_tensorboard import st_tensorboard
import os 
import glob
import sys
import time
import argparse
import numpy
import yaml
import awkward
import pylab

from streamlit_plotly_events import plotly_events



from streamlit.components.v1 import html

from ..learning_envs.learning_env_base import learning_env
from learning_ground import inference_functions
from .folder_fns import  _obtain_extra_config_options, _obtain_cp_dirs_and_init_state, _generate_inference_file, awk_to_dataframe, get_inference_model_from_inference_file
from . import plotting, it_fns
from .callbacks import update_step_dropdown

def _wrap_cb_fn(cb_fn, sess_obj, *args, **kwargs):

    def new_cb_fn():
        return cb_fn(sess_obj, *args, **kwargs)

    return new_cb_fn


# CSS to align divs
# CSS to align divs
css = """
<style>
#left-align-tab .stButton, #left-align-tab .stSelectbox, #left-align-tab .stTextInput {
    display: block;
    text-align: left;
    margin: 0;
}

#center-align-tab .stButton, #center-align-tab .stSelectbox, #center-align-tab .stTextInput {
    display: flex;
    justify-content: center;
    text-align: center;
    margin: 0 auto;
}
</style>
"""


def process_data(folder_to_process, step_to_process, state, extra_config_options, only_include_keys, max_events_processed=500):
    """
    Checks if folder is already processing.. if not add to processing tabs.

    In adding to processing tabs, either starts a new processing, or loads the inference results onto webpage.
    """

    this_key=(folder_to_process, step_to_process)
    step_index=state.log_structure[folder_to_process]["steps"].index(step_to_process)

    if(not this_key in state.open_run_tabs.keys()):
        
        state.open_run_tabs[this_key]=dict()

        print("step index ", step_index)

        state.open_run_tabs[this_key]["checkpoint"]=state.log_structure[folder_to_process]["checkpoints"][step_index]
    cp_folder=state.log_structure[folder_to_process]["folders"][step_index]
    ## make inference config if not present
    this_cfg_filename=os.path.join(cp_folder, "inference_config.yaml")

    #print("checking ..", this_cfg_filename)
    #if(not os.path.exists(this_cfg_filename)):
        
    loaded_params_config=learning_env()._load_inference_config(state.log_structure[folder_to_process]["params_json_file"])
    _generate_inference_file(this_cfg_filename, loaded_params_config, extra_config_options=extra_config_options, only_include_keys=only_include_keys)

    #print("generated config inference file ... ", this_cfg_filename)

    ## check if the target inference file exists

    print("Folder to check ", )
    inf_results_files=glob.glob(os.path.join(state.log_structure[folder_to_process]["folders"][step_index], "*/*.awk"))
   
    assert(len(inf_results_files)<2), "something strange.. more than 2 result files?!"

    num_nan=1
    if(len(inf_results_files)==1):
        ak_array=awkward.from_parquet(inf_results_files[0])

        first_field=ak_array.fields[0]
        this_arr=ak_array[first_field].to_numpy()
        slices = [slice(None)]  # Start with [:]
        for _ in range(1, this_arr.ndim):
            slices.append(0)
        
        slices = tuple(slices)
        
        num_nan=numpy.isnan(this_arr[slices]).sum()
        num_tot=len(ak_array[first_field].to_numpy()) 
        
    if(num_nan>0):
        use_gpu=1
        averaging_mode="swa"

        print("trying to start inference....")
        inference_functions.perform_inference(state.open_run_tabs[this_key]["checkpoint"], 
          this_cfg_filename, 
          model_cfg_file=state.log_structure[folder_to_process]["params_json_file"],
          inference_batch_size=50, 
          cap_num_inference_events_to=max_events_processed,
          use_gpu=use_gpu, 
          averaging_mode=averaging_mode)

        ## get folder again if we just created file
        inf_results_files=glob.glob(os.path.join(state.log_structure[folder_to_process]["folders"][step_index], "*/*.awk"))
        assert(len(inf_results_files)==1)
        ## created file.. load it up
        ak_array=awkward.from_parquet(inf_results_files[0])
        first_field=ak_array.fields[0]
        
        this_arr=ak_array[first_field].to_numpy()
        slices = [slice(None)]  # Start with [:]
        for _ in range(1, this_arr.ndim):
            slices.append(0)
        
        slices = tuple(slices)
        nan_mask=numpy.isnan(this_arr[slices])
        num_nan=nan_mask.sum()

        data, summary_statistic=awk_to_dataframe(ak_array[~nan_mask])

        state.open_run_tabs[this_key]["data"]=data
        state.open_run_tabs[this_key]["summary_statistic"]=summary_statistic

        num_tot=len(ak_array[first_field].to_numpy()) 

    ## update state of processing
    state.open_run_tabs[this_key]["num_processed"]=num_tot-num_nan
    state.open_run_tabs[this_key]["num_tot"]=num_tot
    
def main(logdir, extra_args=[]):

    print("extra args ", extra_args)
    extra_config_options, only_include_keys=_obtain_extra_config_options(extra_args)
    log_structure=_obtain_cp_dirs_and_init_state(logdir, st.session_state)

    ####

    st.set_page_config(layout='wide')

    
    ####################

    st.sidebar.title("base folder: ", logdir)
    option = st.sidebar.radio("Go to", ['TensorBoard', 'Processor'])

    #st.markdown(css, unsafe_allow_html=True)  # Inject CSS

    ## side bar configs based on option

    # Bottom part of the sidebar for additional configurations
    if option == 'TensorBoard':
        with st.sidebar:
            st.write("No additional configurations available.")
    elif option == 'Processor':
        with st.sidebar:
            st.header("Loaded Runs...")
            
            # Create two columns
            col1, col2 = st.columns(2)

            # Left dropdown menu
            with col1:
                selected_left = st.selectbox(
                    "Select Folder:", st.session_state.run_ids, key='selected_left', on_change=_wrap_cb_fn(update_step_dropdown, st.session_state)
                )

            # Right dropdown menu
            with col2:
                selected_right = st.selectbox(
                    "Select a step:", st.session_state.log_structure[selected_left]["steps"], key='selected_right'
                )

            col3, col4 = st.columns(2)
            with col3:
                process_button_clicked=st.button("Process Data", key="button_process")
                    
                #button_clicked = st.button("Calculate")

            # Add an integer input in the second column
            with col4:
                max_events_processed = st.number_input("Number max. events processed:", value=500, step=1)

            if(process_button_clicked):
                process_data(selected_left, selected_right, st.session_state, extra_config_options, only_include_keys, max_events_processed=max_events_processed)

            #if st.button("Process Data", key="button_process"):
            #process_data(selected_left, selected_right, st.session_state, extra_config_options, only_include_keys)

            # Display data editor tabs
            
            if(len(st.session_state.open_run_tabs)>0):
                tabs = st.tabs([ (k[0][:12]+"_%.9d" % k[1]) for k in st.session_state.open_run_tabs.keys()])
                print("NUM TABS", len(tabs))
                for i, (tab_key, data) in enumerate(st.session_state.open_run_tabs.items()):
                    with tabs[i]:
                        st.title("%d / %d processed" % (data["num_processed"], data["num_tot"]))

                        st.dataframe(data["data"])

                        if st.button("Update Cornerplots...", key="button_plot_%s_%d" % (tab_key[0], tab_key[1])):
                            ## create a big cornerplot
                            this_arr=data["data"].to_numpy()
                            labels=data["data"].columns

                            num_samples = this_arr.shape[0]
                            num_dimensions = this_arr.shape[1]

                            mpl_fig = plotting.create_corner_plot_matplotlib(this_arr, labels=labels)
                            high_res_image_base64 = plotting.fig_to_base64(mpl_fig, dpi=200)
                            pylab.close(mpl_fig)

                            st.session_state["open_run_plotting_tabs"][tab_key]=dict()
                            st.session_state["open_run_plotting_tabs"][tab_key]["base64_image"]=high_res_image_base64
                            st.session_state["open_run_plotting_tabs"][tab_key]["open_interactive_plots"]=dict()


    ## main screen configs base on option

    if option == 'TensorBoard':
        with st.container():
            #st.markdown('<div class="left-align">', unsafe_allow_html=True)
            st.header("%s" % logdir)
            st_tensorboard(logdir=logdir, port=6006, height=1024)
            #st.markdown('</div>', unsafe_allow_html=True)
        

    elif option == 'Processor':
        with st.container():
            top_container=st.container()

            with top_container:
                if(len(st.session_state.open_run_plotting_tabs)>0):
                    tabs = st.tabs([ (k[0][:12]+"_%.9d" % k[1]) for k in st.session_state.open_run_plotting_tabs.keys()])
                    for i, (tab_key, data) in enumerate(st.session_state.open_run_plotting_tabs.items()):
                        with tabs[i]:

                          

                            html = f"""
<div id="openseadragon1" style="width: 100%; height: 600px;"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/openseadragon.min.js"></script>
<script type="text/javascript">
    var viewer = OpenSeadragon({{ 
        id: "openseadragon1",
        prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/images/",
        tileSources: {{
            type: 'image',
            url: 'data:image/png;base64,{data["base64_image"]}'
        }}
    }});
</script>
"""

             
                            
                            with st.expander("Click to view high resolution version"):
                                st.components.v1.html(html, height=800, scrolling=True)

                            col1, col2, col3 = st.columns([1, 1, 1])

                            with col1:
                                selected_x = st.selectbox("Select X-axis", st.session_state["open_run_tabs"][tab_key]["data"].columns,key=("sel1"+tab_key[0][:5]+"_%.9d" % tab_key[1]))
                            
                            with col2:
                                selected_y = st.selectbox("Select Y-axis",st.session_state["open_run_tabs"][tab_key]["data"].columns, key=("sel2"+tab_key[0][:5]+"_%.9d" % tab_key[1]))
                            
                            with col3:
                                if st.button("Add", key="add_%s" % ((tab_key[0][:5]+"_%.9d" % tab_key[1]))):

                                    vs_key="%s__vs__%s" % (selected_x, selected_y)
                                    if(vs_key not in st.session_state["open_run_plotting_tabs"][tab_key]["open_interactive_plots"].keys()):
                                        interactive_fig=plotting.create_basic_scatter_plot(st.session_state["open_run_tabs"][tab_key]["data"], selected_x, selected_y)
                                        st.session_state["open_run_plotting_tabs"][tab_key]["open_interactive_plots"][vs_key]=interactive_fig

                            if(len(data["open_interactive_plots"])>0):
                                subplot_tabs = st.tabs([k for k in data["open_interactive_plots"].keys()])

                                for i, (subplot_tab_name, fig) in enumerate(data["open_interactive_plots"].items()):
                                    with subplot_tabs[i]:
                                        # Add a "Remove" button above each chart
                                        if st.button(f"Remove {subplot_tab_name}", key=f"remove_{subplot_tab_name}"):
                                            data["open_interactive_plots"].pop(subplot_tab_name)
                                            st.experimental_rerun()

                                        selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=True, key=f"plot_{subplot_tab_name}")
                                        
                                        # Button to execute another function with selected points
                                        if st.button("Visualize Event", key=f"execute_{subplot_tab_name}"):
                                            if(selected_points):
                                                selected_index=selected_points[0]["pointIndex"]
                                           
                                                plotting.check_n_add_new_model_visualization(tab_key, st.session_state, selected_index)

                                                
                                                ## add the visualization to visualization 

                                                #visualize_event(st.session_state.selected_data.get(subplot_tab_name))
                                        


                   
            bottom_container=st.container()
            with bottom_container:
                st.header("Lower")

                if(len(st.session_state.preloaded_inference_models)>0):
                    ## go through preloaded models and plot them if not already done
                    print(type)
                    tabs = st.tabs([(str(k[0])+"__"+str(k[1])+"__"+str(k[2])) for k in st.session_state.preloaded_inference_models.keys()])
                    for ii, (tab_key, data) in enumerate(st.session_state.preloaded_inference_models.items()):
                        with tabs[ii]:

                            original_model_key=data["original_model_key"]
                            all_summary_stats_this_model=st.session_state.open_run_tabs[original_model_key]["summary_statistic"]

                            num_columns = 8
                            num_rows = len(data["current_summary_statistic"]) // num_columns + (1 if len(data["current_summary_statistic"]) % num_columns != 0 else 0)

                            max_mi_value=-999.0
                            ## do one round for mutual information calculation first to buffer it
                            for i in range(num_rows):
                                with st.container():
                                    columns = st.columns(num_columns)
                                    for j in range(num_columns):
                                        summ_ind = i * num_columns + j
                                        if(summ_ind < len(data["current_summary_statistic"])):

                                            if(not ("mi_val_%d" % summ_ind) in st.session_state.preloaded_inference_models[tab_key].keys()):
                                                ## make histogram of summary stats

                                                ## create mutual information of this event
                                                ## "draw" samples from p(x)
                                                ##

                                                ## mutual information
                                                mi_val=it_fns.compute_mutual_information(st.session_state.preloaded_inference_models[tab_key]["model"].pdf, 
                                                                           st.session_state.preloaded_inference_models[tab_key]["current_summary_statistic"],
                                                                           all_summary_stats_this_model[:,summ_ind],
                                                                           summ_ind)

                                                
                                                print("MI ", summ_ind, mi_val)
                                                st.session_state.preloaded_inference_models[tab_key]["mi_val_%d" % summ_ind]=mi_val

                                            ## obtain maximum mutual information for relative importance weighting
                                            if(st.session_state.preloaded_inference_models[tab_key]["mi_val_%d" % summ_ind]>max_mi_value):
                                                max_mi_value=st.session_state.preloaded_inference_models[tab_key]["mi_val_%d" % summ_ind]
                            ## now go through all the rows/columns agian and create plots
                            print("num rows/cols", num_rows, num_columns)
                            for i in range(num_rows):
                                with st.container():
                                    columns = st.columns(num_columns)
                                    for j in range(num_columns):
                                        summ_ind = i * num_columns + j
                                        if(summ_ind < len(data["current_summary_statistic"])):

                                            with columns[j]:

                                                cur_min=min(all_summary_stats_this_model[:,summ_ind])
                                                cur_max=max(all_summary_stats_this_model[:,summ_ind])

                                                cur_stat_value=data["current_summary_statistic"][summ_ind]
                                                default_stat_value=data["default_summary_statistic"][summ_ind]      

                                                prev_val=st.session_state.preloaded_inference_models[tab_key]["current_summary_statistic"][summ_ind]
                                                print("prev slider vlau", st.session_state.preloaded_inference_models[tab_key]["current_summary_statistic"][summ_ind])
                                                # Create a slider for each subplot
                                                after_val = st.slider(f"Summary Stat {summ_ind + 1}", 
                                                                                          min_value=cur_min, max_value=cur_max, value=cur_stat_value, step=0.1, key=f"slider_%s_%d_%d_{summ_ind}" % (tab_key[0], tab_key[1], tab_key[2]))
                                                
                                                print("new slider values", st.session_state.preloaded_inference_models[tab_key]["current_summary_statistic"][summ_ind])


                                                if(after_val!=prev_val):
                                                    ## change in slider.. adapt plot
                                                    st.session_state.preloaded_inference_models[tab_key]["current_summary_statistic"][summ_ind]=after_val
                                                    plotting.check_n_add_new_model_visualization((tab_key[0], tab_key[1]), st.session_state, tab_key[2])
                                                    st.experimental_rerun()

                                                # Create the Plotly histogram plot
                                                if(not ("global_summary_fig_%d" % summ_ind) in st.session_state.preloaded_inference_models[tab_key].keys()):
                                                    
                                                    #st.session_state.preloaded_inference_models[tab_key]["mi_val_%d" % summ_ind]=mi_val
                                                    buffered_image = plotting.create_histogram_with_line(cur_stat_value,default_stat_value, all_summary_stats_this_model[:,summ_ind], summ_ind, st.session_state.preloaded_inference_models[tab_key]["mi_val_%d" % summ_ind], max_mi_value)
                                                    st.session_state.preloaded_inference_models[tab_key]["global_summary_fig_%d" % summ_ind]=buffered_image

                                                # Display the plot
                                                st.image(st.session_state.preloaded_inference_models[tab_key]["global_summary_fig_%d" % summ_ind], use_column_width=True)
                                                #st.plotly_chart(fig, use_container_width=True)

                            st.image(data["image"], use_column_width=True)

 
