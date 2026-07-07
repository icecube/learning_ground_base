import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import corner
import pylab
from matplotlib.gridspec import GridSpec
from io import BytesIO
import base64
import plotly.express as px
import os
import numpy

from .folder_fns import get_inference_model_from_inference_file

import jammy_flows.helper_fns.plotting.spherical as spherical_plotting
import jammy_flows.helper_fns.contours as contours
import torch


def create_histogram_with_line(position, default_position, data, index, mi_val, max_mi_val):
    """
    Histogram with line indicating summary stats values.
    """
    print(mi_val, max_mi_val)
    fig=pylab.figure(figsize=(3,3))

    gs = GridSpec(4, 1)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.barh(y=0, width=mi_val, color='blue')
    # Set the x-axis limit
    ax0.set_xlim(0, max_mi_val)

    ax1=fig.add_subplot(gs[1:4,0])
    ax1.hist(data, color="k",density="normed")
    ax1.axvline(default_position, color="black")


    # create buffered image for streamlit
    buffered_image=fig_to_image(fig)
    pylab.close(fig)

    return buffered_image
    ## BASE-ORDERED CONTOURS

    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(x=data, nbinsx=30, opacity=0.75))
    fig.add_shape(
        type="line",
        x0=position,
        y0=0,
        x1=position,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color='red', width=2, dash='dash')
    )
    fig.add_shape(
        type="line",
        x0=default_position,
        y0=0,
        x1=default_position,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color='black', width=2, dash='dash')
    )
    fig.update_layout(title=f'Dim {index + 1}', bargap=0.2)

    return fig
    """

    

def plot_jammyflows_pdf(fig, pdf, data_summary, gridspec_full, gridspec_zoom, alternative_summary=None):

    #triple_gridspec = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=total_gridspec[-2, :])

    used_type, used_device=pdf.obtain_current_dtype_n_device()

    with torch.no_grad():
        data_summary=torch.from_numpy(data_summary).to(device=used_device, dtype=used_type)
        alternative_summary=torch.from_numpy(alternative_summary).to(device=used_device, dtype=used_type)
        print("ds ",data_summary)
        # Add axes in the nested GridSpec
        ax1 = fig.add_subplot(gridspec_full)
        ax2 = fig.add_subplot(gridspec_zoom)

        ## BASE-ORDERED CONTOURS
        gaussian_contours=[]
        used_contour_probs=[0.68,0.95]
        used_contour_colors=["black", "orange"]
        used_contour_colors_alternative=["red", "pink"]

        used_base_contour_linestyles=["-", "--"]
        used_base_contour_colors=["red", "red"]

        """
        for contour_ind,contour_prob in enumerate(used_contour_probs):

            vals=numpy.linspace(0,2*numpy.pi, 10000)
            gauss_contour=numpy.concatenate([numpy.cos(vals)[:,None],numpy.sin(vals)[:,None]],axis=1)
            gauss_contour*=numpy.sqrt(-2*numpy.log(1.0-contour_prob))
            
            gaussian_contours.append([gauss_contour])
        """

        ## plot alternative contours
        if(alternative_summary is not None):
            eval_positions, _, pdf_evals, eval_areas, moc_map=spherical_plotting.get_multiresolution_evals(pdf, 
                                                                                 sub_pdf_index=0,
                                                                        samplesize=10000,
                                                                        conditional_input=alternative_summary,
                                                                        max_entries_per_pixel=5,
                                                                        use_density_if_possible=True)



            ## make contours


        # BASE CONTOURS AT TARGET
        #base_contours_at_target=transform_contours(system.pdf, gaussian_contours, backward=False, force_embedding_coordinates=False, conditional_input=data_summary)

        

        ## plot base, skymap, and zoomed skymap

        # plot skymap
        new_ax=spherical_plotting.plot_multiresolution_healpy(pdf, 
                                                       fig=fig, 
                                                       ax_to_plot=ax1, 
                                                       conditional_input=data_summary,
                                                       contour_colors=used_contour_colors, 
                                                       contour_probs=used_contour_probs,
                                                       draw_contours=True, 
                                                       zoom=False, 
                                                       log_scale=True,
                                                       graticule_kwargs={}, 
                                                       cbar_kwargs={}, 
                                                       show_grid=False)

        if(alternative_summary is not None):
            ret=contours.ContourGenerator(new_ax, 
                                     "zen_azi",
                                           eval_positions[:,0], 
                                           eval_positions[:,1], 
                                           pdf_evals, 
                                           eval_areas,
                                           levels=used_contour_probs[::-1],
                                           colors=used_contour_colors_alternative[::-1],
                                           zorder=1
                                           )
        
        #new_ax.proj_plot([labels_to_visualize[0].cpu().numpy()], [labels_to_visualize[1].cpu().numpy()], color="magenta", marker="o", ms=5.0)


        """
        ret=contours.ContourGenerator(new_ax, 
                                  "zen_azi",
                                 used_contour_probs,
                                       base_contours_at_target,
                                       levels=used_contour_probs[::-1],
                                       colors=used_base_contour_colors,
                                       linestyles=used_base_contour_linestyles[::-1],
                                       zorder=1
                                       )
        """

        
        # zoomed skymap
        new_ax=spherical_plotting.plot_multiresolution_healpy(pdf, 
                                                       fig=fig, 
                                                       ax_to_plot=ax2, 
                                                       conditional_input=data_summary,
                                                       contour_colors=["black", "orange"],
                                                       contour_probs=[0.68,0.95],
                                                       draw_contours=True, 
                                                       zoom=True, 
                                                       log_scale=True,
                                                       graticule_kwargs={}, 
                                                       cbar_kwargs={}, 
                                                       show_grid=False)

        if(alternative_summary is not None):
            ret=contours.ContourGenerator(new_ax, 
                                     "zen_azi",
                                           eval_positions[:,0], 
                                           eval_positions[:,1], 
                                           pdf_evals, 
                                           eval_areas,
                                           levels=used_contour_probs[::-1],
                                           colors=used_contour_colors_alternative[::-1],
                                           zorder=1
                                           )

        """
        ret=contours.ContourGenerator(new_ax, 
                                  "zen_azi",
                                 used_contour_probs,
                                       base_contours_at_target,
                                       levels=used_contour_probs[::-1],
                                       colors=used_base_contour_colors,
                                       linestyles=used_base_contour_linestyles[::-1],
                                       zorder=1
                                       )
        """
        #new_ax.proj_plot([labels_to_visualize[0].cpu().numpy()], [labels_to_visualize[1].cpu().numpy()], color="magenta", marker="o", ms=5.0)


def _update_event_image_in_buffer(session_state, key, selected_index):
    
    model=session_state["preloaded_inference_models"][key]["model"]
    dataset_name=session_state["preloaded_inference_models"][key]["dataset_name"]
    env_object=session_state["preloaded_inference_models"][key]["env_object"]

    used_dataset_size=env_object.dataset_sizes[dataset_name]

    absolute_indices_numpy=numpy.array([selected_index])

    tot_events=0

    new_batch=[env_object.data_loaders[dataset_name].dataset.__getitem__(selected_index)]
    collated_batch=env_object.data_loaders[dataset_name].collate_fn(new_batch)

    # Create a figure
    fig = pylab.figure(figsize=(12, 6))

    # Create a GridSpec with 1 row and 3 columns
    outer_grid = GridSpec(2, 4)
   
    env_object.mock_data_modules[dataset_name]._visualize_data(env_object.data_loaders[dataset_name].dataset, fig, outer_grid[0,:], collated_batch, 0,  dataset_identifier=0)

    this_pdf=session_state["preloaded_inference_models"][key]["model"].pdf
    gridspec_full=outer_grid[1,0:2]
    gridspec_zoom=outer_grid[1,2:]
    plot_jammyflows_pdf(fig, this_pdf, session_state["preloaded_inference_models"][key]["default_summary_statistic"], gridspec_full, gridspec_zoom, alternative_summary=session_state["preloaded_inference_models"][key]["current_summary_statistic"])
    ## generate plot

    # create buffered image for streamlit
    buffered_image=fig_to_image(fig)
    pylab.close(fig)

    session_state["preloaded_inference_models"][key]["image"]=buffered_image

def check_n_add_new_model_visualization(tab_key, session_state, selected_index):

    selected_row=session_state["open_run_tabs"][tab_key]["data"].iloc[selected_index]

    #new_key=tab_key[0]+"__%d__%d" % (tab_key[1],selected_index)
    new_key=(tab_key[0], tab_key[1], selected_index)

    print("new key")
    print(type(tab_key[0]), tab_key[0])
    print(type(tab_key[1]), tab_key[1])
   
    #session_state["visualized_tabs"][new_key]=dict()
    if(not new_key in session_state["preloaded_inference_models"]):
        ## add the preloaded_model
        
        print(tab_key)
        print(tab_key[0])
        print(tab_key[1])
        step_index=session_state.log_structure[tab_key[0]]["steps"].index(tab_key[1])
        cp_folder=session_state.log_structure[tab_key[0]]["folders"][step_index]
        this_cfg_filename=os.path.join(cp_folder, "inference_config.yaml")
        checkpoint_file=os.path.join(cp_folder, "cp_validation_end")
        json_file=session_state.log_structure[tab_key[0]]["params_json_file"]

        session_state["preloaded_inference_models"][new_key]=dict()

        model, dataset_name, env_object =get_inference_model_from_inference_file(checkpoint_file, this_cfg_filename, json_file)
        
        session_state["preloaded_inference_models"][new_key]["model"]=model
        session_state["preloaded_inference_models"][new_key]["dataset_name"]=dataset_name
        session_state["preloaded_inference_models"][new_key]["env_object"]=env_object
        session_state["preloaded_inference_models"][new_key]["original_model_key"]=tab_key

        #session_state["preloaded_inference_models"][tab_key]["dataloader"]=dataloader
        session_state["preloaded_inference_models"][new_key]["default_summary_statistic"]=session_state["open_run_tabs"][tab_key]["summary_statistic"][selected_index]
        session_state["preloaded_inference_models"][new_key]["current_summary_statistic"]=session_state["open_run_tabs"][tab_key]["summary_statistic"][selected_index].copy()
        
    ## make a data image
    _update_event_image_in_buffer(session_state, new_key, selected_index)
    

def create_corner_plot_matplotlib(samples, labels=None):
    ## if samples are unique, exclude them
    print("sampl shape")
    print(samples.shape)

    new_samples=[]
    new_labels=[]

    for sind in range(samples.shape[1]):
        unique_samps=numpy.unique(samples[:,sind])
        if(len(unique_samps)>1):
            new_samples.append(samples[:,sind:sind+1])
            new_labels.append(labels[sind])
        else:
            print("exclude ", labels[sind], "from cornerplot since only single value...")

    new_samples=numpy.concatenate(new_samples,axis=1)

    size=samples.shape[1]*1.0
    fig_used=pylab.figure(figsize=(size,size))
    fig = corner.corner(new_samples, labels=new_labels, show_titles=True, fig=fig_used)
    return fig

def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

def fig_to_base64(fig, dpi=100):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode()
    return base64_image

def create_basic_scatter_plot(full_dataframe, x, y):
    fig = px.scatter(full_dataframe, x=x, y=y)
    fig.update_layout(clickmode='event+select')
    return fig


# Function to create a corner plot using Plotly
def create_corner_plot_new(samples, labels=None):
    num_variables = samples.shape[1]
    fig = make_subplots(rows=num_variables, cols=num_variables,
                        shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.02, vertical_spacing=0.02)

    for i in range(num_variables):
        for j in range(num_variables):
            if i == j:
                # Diagonal: histogram
                fig.add_trace(go.Histogram(x=samples[:, i], nbinsx=20, marker=dict(color='blue', opacity=0.7)),
                              row=i+1, col=j+1)
            else:
                # Off-diagonal: scatter plot with hover info
                fig.add_trace(go.Scatter(x=samples[:, j], y=samples[:, i], mode='markers',
                                         marker=dict(color='blue', opacity=0.7),
                                         text=[f'Point {k}' for k in range(samples.shape[0])],
                                         hovertemplate='Point %{text}<br>x: %{x}<br>y: %{y}<extra></extra>',
                                         showlegend=False),
                              row=i+1, col=j+1)
            print("create .. ", i,j)

            # Set axis labels if labels are provided
            if labels:
                if j == 0:
                    fig.update_yaxes(title_text=labels[i], row=i+1, col=j+1)
                if i == num_variables - 1:
                    fig.update_xaxes(title_text=labels[j], row=i+1, col=j+1)
                

    fig.update_layout(height=300*num_variables, width=300*num_variables, title_text="Corner Plot")
    return fig

# Function to create a corner plot using Plotly
def create_corner_plot(samples, labels=None):
    num_dimensions = samples.shape[1]

    # Create a subplot grid
    fig = make_subplots(
        rows=num_dimensions, cols=num_dimensions,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.02, vertical_spacing=0.02,
    )

    # Initialize axis range storage
    axis_ranges = {i: (samples[:, i].min(), samples[:, i].max()) for i in range(num_dimensions)}

    # Create histogram plots on the diagonal
    for i in range(num_dimensions):
        hist_data = samples[:, i]
        fig.add_trace(go.Histogram(x=hist_data, nbinsx=20, name=f'Dim {i+1}',
                                   marker=dict(color='blue', opacity=0.7)),
                      row=i+1, col=i+1)
        fig.update_xaxes(range=axis_ranges[i], row=i+1, col=i+1)

    # Create scatter plots for the off-diagonal subplots
    for i in range(num_dimensions):
        for j in range(i):
            fig.add_trace(go.Scatter(
                x=samples[:, j], y=samples[:, i],
                mode='markers', name=f'Dim {i+1} vs Dim {j+1}',
                marker=dict(color='blue', opacity=0.7),
            ), row=i+1, col=j+1)
            fig.update_xaxes(range=axis_ranges[j], row=i+1, col=j+1)
            fig.update_yaxes(range=axis_ranges[i], row=i+1, col=j+1)
            print("create .. ", i,j)

    # Adjust the domains to ensure proper alignment
    for i in range(num_dimensions):
        for j in range(num_dimensions):
            x_domain_start = j / num_dimensions
            x_domain_end = (j + 1) / num_dimensions
            y_domain_start = 1 - (i + 1) / num_dimensions
            y_domain_end = 1 - i / num_dimensions

            fig.update_xaxes(domain=[x_domain_start, x_domain_end], row=i+1, col=j+1)
            fig.update_yaxes(domain=[y_domain_start, y_domain_end], row=i+1, col=j+1)
            print("align .. ", i,j)

    fig.update_layout(dragmode='select')  # Enable selection tools
    return fig
