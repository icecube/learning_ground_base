from .. import plotting_callback_base
from ... import helper_fns

import torch
import pylab
import numpy
import copy

import collections

import jammy_flows
import jammy_flows.helper_fns.plotting.spherical as spherical_plotting
import jammy_flows.helper_fns.contours as contours

import os
import glob

import torch.autograd.functional
import torch
import time
from matplotlib.patches import Ellipse
import matplotlib

#from umap.parametric_umap import UMAP
try:
    import umap
except:
    print("no umap imported...")

def transform_contours(pdf, input_contours, backward=True, force_embedding_coordinates=False, conditional_input=None):
    """
    Default is to transform "backward" from target to base space.  
    """

    ## require list of lists (one level can have multiple disconnected components)

    data_type, used_device=pdf.obtain_current_dtype_n_device()

    assert(type(input_contours)==list)
    input_contours_used=[]
    for l in input_contours:
        assert(type(l)==list)

        input_contours_used.append([torch.from_numpy(c).type(data_type).to(used_device) for c in l])

    base_list=[]
    for contour_ind, inp_contour_list in enumerate(input_contours_used):

        cur_base_list=[]

        for inp_contour in inp_contour_list:
            #new_input=numpy.concatenate( [zen_contour[:,None],azi_contours[contour_ind][:,None]], axis=1)
            new_input=inp_contour

            tmp=0.0
            if(backward):
                base_pts,_=pdf.all_layer_inverse(new_input,
                                  tmp,
                                  conditional_input,
                                  amortization_parameters=None, 
                                  force_embedding_coordinates=force_embedding_coordinates, 
                                  force_intrinsic_coordinates=False)

            else:
                base_pts,_=pdf.all_layer_forward(new_input,
                                  tmp,
                                  conditional_input,
                                  amortization_parameters=None, 
                                  force_embedding_coordinates=force_embedding_coordinates, 
                                  force_intrinsic_coordinates=False)

            cur_base_list.append(base_pts.cpu().numpy())

        base_list.append(cur_base_list)

    return base_list

class supervised_base_plotting(plotting_callback_base.val_plot_callback_base):

    def __init__(self, val_dataset_indices=None, plotting_dataset_indices=None, indices=None, data_on_the_left=0, plot_interval=100, constraints=None, **kwargs):
        super().__init__(indices=indices, val_dataset_indices=val_dataset_indices, plotting_dataset_indices=plotting_dataset_indices, data_on_the_left=data_on_the_left, plot_interval=plot_interval, constraints=constraints)

        self.plot_name="basic"

        ## set extra plot options as part of the name to differntiate different plots

        self.config=collections.OrderedDict()

        self.config["s2_mode"]="standard"

        ## update config
        for k in kwargs.keys():
            if(k=="s2_mode"):
                self.config[k]=kwargs[k]
            elif(k=="constraints"):
                ## we set constraints in the parent already
                assert(self.constraints is not None)
            else:
                raise Exception("Unknown supervised draw option", k)

        ## modify plot name with config
        for k in self.config.keys():
            self.plot_name=self.plot_name+"__"+k+"="+str(self.config[k])

        if(self.constraints is not None):
            self.plot_name=self.plot_name+"__constraints="+self.constraints
        
        if( len(self.indices)==0):
            raise Exception("Basic plotting requires a list of batch indices to plot")

        ## we want 2 plots .. one visualizing data, the other the pdfs
        self.num_horizontal_gridspecs=3
    #visualize_model(fig, total_gridspec, system, batch_info, uncoll_batch_info, bounds=bounds, batch_index=batch_index, data_vis_returns=data_vis_returns)
    
    def make_fig_and_layout(self, system, batch_info):
        
        # 2 known
        self.num_horizontal_gridspecs=2
        
        if(self.data_on_the_left):
            self.num_horizontal_gridspecs+=1
            self.num_vertical_gridspecs=1
        else:
            self.num_vertical_gridspecs=2
       
        stretch_factor=6.0

        fig=pylab.figure(figsize=(self.num_horizontal_gridspecs*stretch_factor,self.num_vertical_gridspecs*stretch_factor))

        ### one label gridspec
        self.num_vertical_gridspecs+=1

        total_gridspec=fig.add_gridspec(self.num_vertical_gridspecs, self.num_horizontal_gridspecs, height_ratios=[1,1,0.2])
       
        return fig, total_gridspec

    def _visualize_model(self, fig, total_gridspec, system, batch_info, uncoll_batch_info, bounds=None, batch_index=None, data_vis_returns=None):

        ## just do the normal plotting of the PDF
        ## data is everything starting at indexc 1 onwards
        #input_data_vec=[i[batch_index] for i in uncoll_batch_info[1:]]
        
        text_size=12
        #pylab.rc('font', family='serif', serif="cm", size=text_size)

        ### 
        data_plot_kwargs=data_vis_returns

        ### label is 0th entry
        true_labels=batch_info["labels"][batch_index]

        horizontal_gs_offset=-1

        if(self.data_on_the_left==1):
            horizontal_gs_offset=0

        
        
        """
        if(self.data_on_the_left):
        gridspec_fixed=total_gridspec[0,1] # with bounds
        gridspec_zoom=total_gridspec[0,2] # zoomed in version
        """
        total_num_samples=50000
        
        data_summary=system._apply_encoder(batch_info, batch_index=batch_index) 
        
        print("after encoder application")

        s2_plotting_type=self.config["s2_mode"]

        if(bounds is not None):
            bounds_to_visualize=copy.deepcopy(bounds)
            ## replace bounds of angle with -2,2 if necessary here

            if(s2_plotting_type=="lambert"):
                num_changes=0
                ## manually change all angle ranges to -2/2
                new_bounds=[]
                for b in bounds_to_visualize:
                    if(b[0]==0.0 and (numpy.fabs(b[1]-numpy.pi)<1e-6)):
                        num_changes+=1
                        new_bounds.append([-2.0,2.0])
                    elif(b[0]==0.0 and (numpy.fabs(b[1]-2*numpy.pi)<1e-6)):
                        num_changes+=1
                        new_bounds.append([-2.0,2.0])
                    else:
                        new_bounds.append(b)

                assert(num_changes%2==0), "Something weird happened, num angle changes not divisible by 2!"

                bounds_to_visualize=new_bounds
          

        else:
            bounds_to_visualize=bounds

        labels_to_visualize=copy.deepcopy(true_labels)

        
        ## for fully amortized PDFs, we need to split sampling up in iterative batches
        num_iterative_steps=10
        if(type(system.pdf)==jammy_flows.main.fully_amortized.fully_amortized_pdf):
            num_iterative_steps=500

        total_eval_pts=50
        if(len(bounds)<=2):
            ## increase eval pts if only in 2 d
            total_eval_pts=50000

        if(type(data_summary)==list):
            if(data_summary[0].dtype!=torch.float64):
                data_summary=[di.type(torch.float64) for di in data_summary]
        else:
            if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)

        print("num num_iterative_steps ", num_iterative_steps)

        if(len(system.pdf.pdf_defs_list)==1 and system.pdf.pdf_defs_list[0]=="s2"):

            ## make a subplotspec from the overall gridspec
            gridspec_fixed=total_gridspec[-2,1+horizontal_gs_offset]
            gridspec_zoom=total_gridspec[-2,2+horizontal_gs_offset]

            #triple_gridspec = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=total_gridspec[-2, :])

            # Add axes in the nested GridSpec
            ax1 = fig.add_subplot(gridspec_fixed)
            ax2 = fig.add_subplot(gridspec_zoom)

            
            ## BASE-ORDERED CONTOURS
            gaussian_contours=[]
            used_contour_probs=[0.68,0.95]
            used_base_contour_linestyles=["-", "--"]
            used_base_contour_colors=["red", "red"]

            for contour_ind,contour_prob in enumerate(used_contour_probs):

                vals=numpy.linspace(0,2*numpy.pi, 10000)
                gauss_contour=numpy.concatenate([numpy.cos(vals)[:,None],numpy.sin(vals)[:,None]],axis=1)
                gauss_contour*=numpy.sqrt(-2*numpy.log(1.0-contour_prob))
                
                gaussian_contours.append([gauss_contour])

            # BASE CONTOURS AT TARGET
            base_contours_at_target=transform_contours(system.pdf, gaussian_contours, backward=False, force_embedding_coordinates=False, conditional_input=data_summary)

            

            ## plot base, skymap, and zoomed skymap

            # plot skymap
            new_ax=spherical_plotting.plot_multiresolution_healpy(system.pdf, 
                                                           fig=fig, 
                                                           ax_to_plot=ax1, 
                                                           conditional_input=data_summary,
                                                           contour_colors=["black", "orange"], 
                                                           contour_probs=[0.68,0.95],
                                                           draw_contours=True, 
                                                           zoom=False, 
                                                           log_scale=True,
                                                           graticule_kwargs={}, 
                                                           cbar_kwargs={}, 
                                                           show_grid=False)
            
            new_ax.proj_plot([labels_to_visualize[0].cpu().numpy()], [labels_to_visualize[1].cpu().numpy()], color="magenta", marker="o", ms=5.0)



            ret=contours.ContourGenerator(new_ax, 
                                      "zen_azi",
                                     used_contour_probs,
                                           base_contours_at_target,
                                           levels=used_contour_probs[::-1],
                                           colors=used_base_contour_colors,
                                           linestyles=used_base_contour_linestyles[::-1],
                                           zorder=1
                                           )


            
            # zoomed skymap
            new_ax=spherical_plotting.plot_multiresolution_healpy(system.pdf, 
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

            ret=contours.ContourGenerator(new_ax, 
                                      "zen_azi",
                                     used_contour_probs,
                                           base_contours_at_target,
                                           levels=used_contour_probs[::-1],
                                           colors=used_base_contour_colors,
                                           linestyles=used_base_contour_linestyles[::-1],
                                           zorder=1
                                           )

            new_ax.proj_plot([labels_to_visualize[0].cpu().numpy()], [labels_to_visualize[1].cpu().numpy()], color="magenta", marker="o", ms=5.0)





        else:

            gridspec_fixed=total_gridspec[-2,1+horizontal_gs_offset]
            gridspec_zoom=total_gridspec[-2,2+horizontal_gs_offset]

            _, _, _=jammy_flows.helper_fns.visualize_pdf(system.pdf, 
                                                         fig, 
                                                         gridspec=gridspec_fixed, 
                                                         nsamples=total_num_samples, 
                                                         total_pdf_eval_pts=total_eval_pts, 
                                                         bounds=bounds_to_visualize, 
                                                         conditional_input=data_summary, 
                                                         true_values=labels_to_visualize, 
                                                         s2_norm=s2_plotting_type,
                                                         num_iterative_steps=num_iterative_steps,
                                                         show_relative_std=1)
           
            _, _, _=jammy_flows.helper_fns.visualize_pdf(system.pdf, 
                                                         fig, 
                                                         gridspec=gridspec_zoom, 
                                                         nsamples=total_num_samples, 
                                                         total_pdf_eval_pts=total_eval_pts, 
                                                         bounds=None, 
                                                         conditional_input=data_summary, 
                                                         true_values=labels_to_visualize, 
                                                         s2_norm=s2_plotting_type,
                                                         num_iterative_steps=num_iterative_steps,
                                                         show_relative_std=1)
                
        index_to_pdf_def=dict()

        
        for ind,pdf_def in enumerate(system.pdf.pdf_defs_list):
            index_to_pdf_def[ind]=pdf_def

        if(hasattr(system.pdf, "entropy")):
            all_entropies=system.pdf.entropy(samplesize=100, conditional_input=data_summary, sub_manifolds=[-1]+list(range(len(system.pdf.pdf_defs_list))))
            overview_gridspec=total_gridspec[-1,:]
            entropy_ax=fig.add_subplot(overview_gridspec)

            tot_text="Entropies: "
            for key_index, k in enumerate(all_entropies.keys()):
                if(k=="total"):
                    tot_text+="tot: %.2f" % (all_entropies[k])
                else:
                    if(index_to_pdf_def[k]=="s2"):
                        deg_sigma_equiv=numpy.exp( (all_entropies[k].cpu().numpy()-1.0-numpy.log(2*numpy.pi))/2.0 )*180.0/numpy.pi
                        tot_text+="%d (%s): %.2f (~%.2f deg sigma)" % (k, index_to_pdf_def[k], all_entropies[k], deg_sigma_equiv)
                    else:
                        tot_text+="%d (%s): %.2f" % (k, index_to_pdf_def[k], all_entropies[k])

                if(key_index<(len(all_entropies.keys())-1)):
                    tot_text += " / "
            entropy_ax.set_title(tot_text)
            entropy_ax.axis("off")
        #fig.suptitle("test123")
        fig.tight_layout()

        
class vis_coverage(plotting_callback_base.val_plot_callback_base):
    """
    Find correlations between summary dims and target label dims.

    """
    def __init__(self, **kwargs):
        super().__init__(indices=None, **kwargs)

        self.plot_name="coverage"

    def make_fig_and_layout(self,system, batch_info):

        self.num_horizontal_gridspecs=1
        
        self.constraint_descriptions=["all"]
        true_mask=None
        for k in batch_info["event_properties"].keys():
            true_mask=numpy.array([True]*len(batch_info["event_properties"][k]))
            break

        assert(true_mask is not None)

        self.constraint_masks=[true_mask]

        if(self.constraints is not None):

            for parent_constraint in self.constraints.split("&&"):

                constraint_mask=true_mask
                for c in parent_constraint.split("&"):
                    if(">" in c):
                        s=c.split(">")
                        ar=batch_info["event_properties"][s[0]]
                        if(type(ar)==list):
                            ar=numpy.array(ar)

                        constraint_mask=constraint_mask & (ar>float(s[1]))
                    elif("<" in c):
                        s=c.split("<")
                        ar=batch_info["event_properties"][s[0]]
                        if(type(ar)==list):
                            ar=numpy.array(ar)
                       
                        constraint_mask=constraint_mask & (ar<float(s[1]))
                    else:
                        raise Exception("Unknown constraint operation .. ", c)

                self.constraint_descriptions.append(parent_constraint)
                self.constraint_masks.append(constraint_mask)


        num_items=len(self.constraint_masks)

        pix_sidelength=4.0
        fig=pylab.figure(figsize=(num_items*pix_sidelength,pix_sidelength))

        total_gridspec=fig.add_gridspec(1, self.num_horizontal_gridspecs)
       
        return fig, total_gridspec

  
    def _visualize_model(self, fig, total_gridspec, system, batch_info, uncoll_batch_info, bounds=None, batch_index=None, data_vis_returns=None):

        data_summary=system._apply_encoder(batch_info)

        nsamples_per_item=10

        num_pdfs=len(system.pdf.flow_defs_list)

        if(type(data_summary)==list):
            if(data_summary[0].dtype!=torch.float64):
                data_summary=[di.type(torch.float64) for di in data_summary]
        else:
            if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)
        
        subgridspec = total_gridspec[:].subgridspec(1, len(self.constraint_masks))

        colors=["green", "red", "blue", "orange", "magenta"]

        for constr_index, constr_mask in enumerate(self.constraint_masks):

            if(constr_mask.sum()==0):
                continue
                
            if(type(data_summary)==list):
                constr_data_summary=[ds[constr_mask] for ds in data_summary]
            else:
                constr_data_summary=data_summary[constr_mask]
                
            ax = fig.add_subplot(subgridspec[0, constr_index])
            cov_dict=system.pdf.approximate_coverage(batch_info["labels"][constr_mask], conditional_input=constr_data_summary, sub_manifolds=numpy.arange(-1,num_pdfs))

            for k in cov_dict["true"].keys():
                if(type(k)==int):
                    ax.plot(cov_dict["expected"], cov_dict["true"][k], label="%d - %s" % (k, system.pdf.pdf_defs_list[k]), color=colors[k], lw=2.0)
               
            ax.plot(cov_dict["expected"], cov_dict["true"]["total"], label="total", color="black", lw=2.0)
            ax.legend(loc="upper right")

            ax.set_xlabel("exp. coverage")
            ax.set_ylabel("true coverage")

            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            
            if(constr_index==0):
                ax.set_title(self.constraint_descriptions[constr_index].replace("&", "\n") + "   " + "+".join(system.pdf.pdf_defs_list))
            else:
                ax.set_title(self.constraint_descriptions[constr_index].replace("&", "\n"))
      
        ## now got *batchsize* means and stds, for each embedding dim

        ## create grid x axis = means/std , y axis = data summary dims

        fig.tight_layout()

class vis_compressed_correlations(plotting_callback_base.val_plot_callback_base):
    """
    Find correlations between summary dims and target label dims.

    """
    def __init__(self, **kwargs):
        super().__init__(indices=None, **kwargs)

        self.plot_name="vis_compressed_correlations"

    def make_fig_and_layout(self,system, batch_info):

        self.num_horizontal_gridspecs=1
        
        used_input_dim=system.pdf.conditional_input_dim
        if(type(used_input_dim)==list):
            tot_dim=0
            for d in used_input_dim:
                tot_dim+=d
            used_input_dim=tot_dim

        pix_sidelength=0.6
        fig=pylab.figure(figsize=( int(system.pdf.total_target_dim_embedded*pix_sidelength*4),used_input_dim*pix_sidelength))

        total_gridspec=fig.add_gridspec(1, self.num_horizontal_gridspecs)
       
        return fig, total_gridspec

  
    def _visualize_model(self, fig, total_gridspec, system, batch_info, uncoll_batch_info, bounds=None, batch_index=None, data_vis_returns=None):

        data_summary=system._apply_encoder(batch_info)


        nsamples_per_item=10

        if(type(data_summary)==list):
            if(data_summary[0].dtype!=torch.float64):
                data_summary=[di.type(torch.float64) for di in data_summary]

            batch_size=data_summary[0].shape[0]

            cinput=[ds.repeat_interleave(nsamples_per_item, dim=0) for ds in data_summary]
            

        else:
            if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)

            batch_size=data_summary.shape[0]

            cinput=data_summary.repeat_interleave(nsamples_per_item, dim=0)


        samples,_,_,_=system.pdf.sample(conditional_input=cinput)

     
        ret=system.pdf.transform_target_into_returnable_params(samples).unsqueeze(-1).reshape(batch_size, nsamples_per_item,system.pdf.total_target_dim_embedded)
        
        data_summaries=data_summary
        if(type(data_summaries)!=list):
            data_summaries=[data_summaries]

        overall_latent_dim_offset=0

        total_latant_sum=sum(l.shape[1] for l in data_summaries)
        subgridspec = total_gridspec[:].subgridspec(total_latant_sum, system.pdf.total_target_dim_embedded*2)

        for overall_ds_index, this_data_summary in enumerate(data_summaries):
            dim_this_ds=this_data_summary.shape[1]

            this_data_summary=this_data_summary.cpu().numpy()
            means=ret.mean(dim=1).cpu().numpy() # B X embedding dim
            stds=ret.std(dim=1).cpu().numpy() # B X embedding dim

            

            for ds_dim in range(this_data_summary.shape[1]):

                ## only get 1st and 2nd central moment for each dim currently
                for moment_dim in range(means.shape[1]):

                    ax = fig.add_subplot(subgridspec[ds_dim+overall_ds_index*dim_this_ds, moment_dim])

                    if(moment_dim==0):
                        ## plot y symbol
                        ax.set_ylabel("d %d_%d" % (overall_ds_index,ds_dim))
                    else:
                        ax.set_yticklabels([])

                    if( (overall_ds_index==(len(data_summaries)-1)) and (ds_dim==(this_data_summary.shape[1]-1))):

                        prev_total=0
                        for test_subpdf_ind in range(len(system.pdf.target_dims_embedded)):
                            if(moment_dim<system.pdf.target_dims_embedded[test_subpdf_ind]+prev_total):
                                used_index=test_subpdf_ind
                                break
                            prev_total+=system.pdf.target_dims_embedded[test_subpdf_ind]

                        difference_within=moment_dim-prev_total
                        ## ds_dim is between prev_total and prev_total+system.pdf.target_dims_embedded[used_index]
                        ax.set_xlabel("mean %s - %d" % (system.pdf.pdf_defs_list[used_index], difference_within))
                    else:
                        ax.set_xticklabels([])

                    ax.hist2d(means[:,moment_dim],this_data_summary[:,ds_dim])

                ## only get 1st and 2nd central moment for each dim currently
                for std_dim in range(stds.shape[1]):

                    ax = fig.add_subplot(subgridspec[ds_dim+overall_ds_index*dim_this_ds, std_dim+system.pdf.total_target_dim_embedded])

                    ax.set_yticklabels([])

                    if((overall_ds_index==(len(data_summaries)-1)) and (ds_dim==(this_data_summary.shape[1]-1))):

                        prev_total=0
                        for test_subpdf_ind in range(len(system.pdf.target_dims_embedded)):
                            if(std_dim<system.pdf.target_dims_embedded[test_subpdf_ind]+prev_total):
                                used_index=test_subpdf_ind
                                break
                            prev_total+=system.pdf.target_dims_embedded[test_subpdf_ind]

                        difference_within=std_dim-prev_total
                        ## ds_dim is between prev_total and prev_total+system.pdf.target_dims_embedded[used_index]
                        ax.set_xlabel("std %s - %d" % (system.pdf.pdf_defs_list[used_index], difference_within))
                    else:

                        ax.set_xticklabels([])
                    ax.hist2d(stds[:,std_dim],this_data_summary[:,ds_dim])


          
            ## now got *batchsize* means and stds, for each embedding dim

            ## create grid x axis = means/std , y axis = data summary dims

        fig.tight_layout()


def _get_medians(x_binedges, xvals, yvals):

    medians=[]

    for ind, lower_bin in enumerate(x_binedges[:-1]):
        upper_bin=x_binedges[ind+1]

        cur_mask=(xvals>=lower_bin) & (xvals < upper_bin)


        medians.append(numpy.median(yvals[cur_mask]))

    midpoints=0.5*(x_binedges[1:]+x_binedges[:-1])

    return midpoints, numpy.array(medians)

class vis_per_item_loss_and_entropy(plotting_callback_base.val_plot_callback_base):
    """
    Visualize loss and entropy per item vs observables.

    """
    def __init__(self, observables="", **kwargs):
        super().__init__(indices=None, **kwargs)

        self.plot_name="vis_per_item_loss_and_entropy"

        self.observable_dict=dict()
        print("OBSERVABLES ", observables)
        for o in observables.split("?"):
            osplits=o.split("#")

            assert(len(osplits)==4), "Require 4 strings separated by *_* to define observable settings. *name_lowend_highend_numbins*"
            print("osplits ", osplits)
            name=osplits[0]
            lowend=float(osplits[1])
            highend=float(osplits[2])
            numbins=int(osplits[3])
            self.observable_dict[name]=numpy.linspace(lowend, highend, numbins+1)

        assert(len(self.observable_dict)>0), "Chose to plot *vis_per_item_loss_and_entropy*, but no observables keyword defined. Define observables as observables=name1_lowen1_highen1_numbins1+name2_lowen2_highen2_numbins2+..."

    def make_fig_and_layout(self,system, batch_info):

        ## 3 plots .. one for 
        self.num_horizontal_gridspecs=1
        
        used_input_dim=system.pdf.conditional_input_dim
        if(type(used_input_dim)==list):
            tot_dim=0
            for d in used_input_dim:
                tot_dim+=d
            used_input_dim=tot_dim

        ## check if we have v flow.. v flow is computationally intractable for entropy calculation, so skip it
        self.num_h_plots=3
        has_v_flow=False
        for cur_flow_list in system.pdf.flow_defs_list:
            if("v" in cur_flow_list):
                self.num_h_plots=1
                has_v_flow=True
                break

        self.has_v_flow=has_v_flow


        # one row per observable
        self.num_v_plots=len(self.observable_dict)

        
  
        pix_sidelength=3.0
        fig=pylab.figure(figsize=( pix_sidelength*self.num_h_plots*1.4,pix_sidelength*self.num_v_plots))
        print(pix_sidelength*self.num_v_plots,pix_sidelength*self.num_h_plots)
        total_gridspec=fig.add_gridspec(1, self.num_horizontal_gridspecs)
        
        return fig, total_gridspec

  
    def _visualize_model(self, fig, total_gridspec, system, batch_info, uncoll_batch_info, bounds=None, batch_index=None, data_vis_returns=None):

        data_summary=system._apply_encoder(batch_info)


        nsamples_per_item=10

        if(type(data_summary)==list):
            if(data_summary[0].dtype!=torch.float64):
                data_summary=[di.type(torch.float64) for di in data_summary]

        else:
            if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)

        # get previous embedding situation
        previous_embedding_flags=system.pdf.get_embedding_flags()
        transformed_labels=system.pdf.transform_target_into_returnable_params(batch_info["labels"])

        # set to embedding space
        system.pdf.set_embedding_flags(True)

        log_pdfs,_,_=system.pdf(transformed_labels, conditional_input=data_summary)
        neg_log_pdfs=-log_pdfs.detach().cpu().numpy()

        entropies=None
        if(self.num_h_plots>1):

            entropy_dict=system.pdf.entropy(conditional_input=data_summary, samplesize=1000)
            entropies=entropy_dict["total"].detach().cpu().numpy()

        # reset to previous embedding situation
        for emb_index in range(len(previous_embedding_flags)):
            system.pdf.set_embedding_flags(previous_embedding_flags[emb_index], sub_pdf_index=emb_index)



        subgridspec = total_gridspec[:].subgridspec(self.num_v_plots, self.num_h_plots)

        for obs_ind, obs in enumerate(self.observable_dict):
            assert(obs in batch_info["event_properties"])

            x_obs=batch_info["event_properties"][obs]
            used_x_bins=self.observable_dict[obs]
            used_y_bins=100

            # normal logp
            ax = fig.add_subplot(subgridspec[obs_ind, 0])
            ax.hist2d(x_obs, neg_log_pdfs, bins=[used_x_bins, used_y_bins], density=True, norm=matplotlib.colors.LogNorm()) #cmin=1e-4, 
            midpoints,this_meds=_get_medians(used_x_bins, x_obs, neg_log_pdfs)

            ax.plot(midpoints, this_meds, color="red", marker="o")
            ax.set_xlabel(obs)
            ax.set_ylabel("logp")

            if(self.num_h_plots>1):
                ## plot entropy and diff
                ax = fig.add_subplot(subgridspec[obs_ind, 1])
                ax.hist2d(x_obs, entropies, bins=[used_x_bins, used_y_bins], density=True, norm=matplotlib.colors.LogNorm()) #cmin=1e-4, 
                midpoints,this_meds=_get_medians(used_x_bins, x_obs, entropies)
                ax.plot(midpoints, this_meds, color="red", marker="o")
                ax.set_xlabel(obs)
                ax.set_ylabel("entropy")

                ax = fig.add_subplot(subgridspec[obs_ind, 2])
                ax.hist2d(x_obs, entropies-neg_log_pdfs, bins=[used_x_bins, used_y_bins], density=True, norm=matplotlib.colors.LogNorm()) #cmin=1e-4, 
                midpoints,this_meds=_get_medians(used_x_bins, x_obs, entropies-neg_log_pdfs)
                ax.plot(midpoints, this_meds, color="red", marker="o")
                ax.set_xlabel(obs)
                ax.set_ylabel("entropy-logp")

        fig.tight_layout()



class vis_per_item_loss_and_entropy_fixed_yaxis(plotting_callback_base.val_plot_callback_base):
    """
    Visualize loss and entropy per item vs observables.

    """
    def __init__(self, observables="", **kwargs):
        super().__init__(indices=None, **kwargs)

        self.plot_name="vis_per_item_loss_and_entropy_fixed_yaxis"

        self.observable_dict=dict()
        print("OBSERVABLES ", observables)
        for o in observables.split("?"):
            osplits=o.split("#")

            assert(len(osplits)==4), "Require 4 strings separated by *_* to define observable settings. *name_lowend_highend_numbins*"
            print("osplits ", osplits)
            name=osplits[0]
            lowend=float(osplits[1])
            highend=float(osplits[2])
            numbins=int(osplits[3])
            self.observable_dict[name]=numpy.linspace(lowend, highend, numbins+1)

        assert(len(self.observable_dict)>0), "Chose to plot *vis_per_item_loss_and_entropy*, but no observables keyword defined. Define observables as observables=name1_lowen1_highen1_numbins1+name2_lowen2_highen2_numbins2+..."

    def make_fig_and_layout(self,system, batch_info):

        ## 3 plots .. one for 
        self.num_horizontal_gridspecs=1
        
        used_input_dim=system.pdf.conditional_input_dim
        if(type(used_input_dim)==list):
            tot_dim=0
            for d in used_input_dim:
                tot_dim+=d
            used_input_dim=tot_dim

        ## check if we have v flow.. v flow is computationally intractable for entropy calculation, so skip it
        self.num_h_plots=3
        has_v_flow=False
        for cur_flow_list in system.pdf.flow_defs_list:
            if("v" in cur_flow_list):
                self.num_h_plots=1
                has_v_flow=True
                break

        self.has_v_flow=has_v_flow


        # one row per observable
        self.num_v_plots=len(self.observable_dict)

        
  
        pix_sidelength=3.0
        fig=pylab.figure(figsize=( pix_sidelength*self.num_h_plots*1.4,pix_sidelength*self.num_v_plots))
        print(pix_sidelength*self.num_v_plots,pix_sidelength*self.num_h_plots)
        total_gridspec=fig.add_gridspec(1, self.num_horizontal_gridspecs)
        
        return fig, total_gridspec

  
    def _visualize_model(self, fig, total_gridspec, system, batch_info, uncoll_batch_info, bounds=None, batch_index=None, data_vis_returns=None):

        data_summary=system._apply_encoder(batch_info)


        nsamples_per_item=10

        if(type(data_summary)==list):
            if(data_summary[0].dtype!=torch.float64):
                data_summary=[di.type(torch.float64) for di in data_summary]

        else:
            if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)

        print(batch_info.keys())

        # get previous embedding situation
        previous_embedding_flags=system.pdf.get_embedding_flags()
        transformed_labels=system.pdf.transform_target_into_returnable_params(batch_info["labels"])

        # set to embedding space
        system.pdf.set_embedding_flags(True)

        log_pdfs,_,_=system.pdf(transformed_labels, conditional_input=data_summary)
        neg_log_pdfs=-log_pdfs.detach().cpu().numpy()

        entropies=None
        if(self.num_h_plots>1):

            entropy_dict=system.pdf.entropy(conditional_input=data_summary, samplesize=1000)
            entropies=entropy_dict["total"].detach().cpu().numpy()

        # reset to previous embedding situation
        for emb_index in range(len(previous_embedding_flags)):
            system.pdf.set_embedding_flags(previous_embedding_flags[emb_index], sub_pdf_index=emb_index)



        subgridspec = total_gridspec[:].subgridspec(self.num_v_plots, self.num_h_plots)

        for obs_ind, obs in enumerate(self.observable_dict):
            assert(obs in batch_info["event_properties"])

            x_obs=batch_info["event_properties"][obs]
            used_x_bins=self.observable_dict[obs]
            used_y_bins=100

            # normal logp
            ax = fig.add_subplot(subgridspec[obs_ind, 0])
            ax.hist2d(x_obs, neg_log_pdfs, bins=[used_x_bins, used_y_bins], density=True, norm=matplotlib.colors.LogNorm()) #cmin=1e-4, 
            midpoints,this_meds=_get_medians(used_x_bins, x_obs, neg_log_pdfs)

            ax.plot(midpoints, this_meds, color="red", marker="o")
            ax.set_xlabel(obs)
            ax.set_ylabel("logp")
            ax.set_ylim(-7,3)

            if(self.num_h_plots>1):
                ## plot entropy and diff
                ax = fig.add_subplot(subgridspec[obs_ind, 1])
                
                ax.hist2d(x_obs, entropies, bins=[used_x_bins, used_y_bins], density=True, norm=matplotlib.colors.LogNorm()) #cmin=1e-4, 
                midpoints,this_meds=_get_medians(used_x_bins, x_obs, entropies)
                ax.plot(midpoints, this_meds, color="red", marker="o")
                ax.set_xlabel(obs)
                ax.set_ylabel("entropy")
                ax.set_ylim(-7,3)

                ax = fig.add_subplot(subgridspec[obs_ind, 2])
                ax.hist2d(x_obs, entropies-neg_log_pdfs, bins=[used_x_bins, used_y_bins], density=True, norm=matplotlib.colors.LogNorm()) #cmin=1e-4, 
                midpoints,this_meds=_get_medians(used_x_bins, x_obs, entropies-neg_log_pdfs)
                ax.plot(midpoints, this_meds, color="red", marker="o")
                ax.set_xlabel(obs)
                ax.set_ylabel("entropy-logp")
                ax.set_ylim(-2,2)

        fig.tight_layout()


      

class vis_labels_umap(plotting_callback_base.val_plot_callback_base):

    def __init__(self, names="", **kwargs):
        super().__init__(indices=None, **kwargs)
       
        split_names=names.split(",")
       
        assert(len(names)>0), "Specify names to visualize in umap latent visualization."
        self.label_names=split_names
        assert(len(split_names)>0), "Please specify names keyword (label names sperated by commas), which should be visualized via umap"
        assert(split_names[0]!=""), "Split names seems to be empty.. "
        self.plot_name="vis_labels_umap"

    def make_fig_and_layout(self,system, batch_info):

        self.num_horizontal_gridspecs=1

        if(len(self.label_names)==1):

            self.num_horizontal_plots=1
            self.num_vertical_plots=1

        elif(len(self.label_names)==2):

            self.num_vertical_plots=1
            self.num_horizontal_plots=2

        elif(len(self.label_names)<=4):

            self.num_vertical_plots=2
            self.num_horizontal_plots=2

        elif(len(self.label_names)<=6):

            self.num_vertical_plots=3
            self.num_horizontal_plots=2

        elif(len(self.label_names)<=9):

            self.num_vertical_plots=3
            self.num_horizontal_plots=3
        else:
            raise Exception("More than 9 labels not supported atm .. change plotting")



        pix_sidelength=3.0
        fig=pylab.figure(figsize=(self.num_vertical_plots*pix_sidelength,self.num_horizontal_plots*pix_sidelength))

        total_gridspec=fig.add_gridspec(1, self.num_horizontal_gridspecs)
       
        return fig, total_gridspec

  
    def _visualize_model(self, fig, total_gridspec, system, batch_info, uncoll_batch_info, bounds=None, batch_index=None, data_vis_returns=None):

        for name in self.label_names:
            if(name != "entropy"):
                
                assert(name in batch_info["event_properties"].keys()), ("Label name ", name, " not found in event_properties.", [i for i in batch_info["event_properties"].keys()])

        data_summary=system._apply_encoder(batch_info)

        if(type(data_summary)==list):
            if(data_summary[0].dtype!=torch.float64):
                data_summary=[di.type(torch.float64) for di in data_summary]
        else:
            if(data_summary.dtype!=torch.float64):
                data_summary=data_summary.type(torch.float64)

        if(type(data_summary)==list):

            data_summaries=data_summary
        else:
            data_summaries=[data_summary]

        ## take first for now
        
        subgridspec = total_gridspec[:].subgridspec(self.num_vertical_plots, self.num_horizontal_plots*len(data_summaries))

        for ds_index, data_summary in enumerate(data_summaries):

            #embedder = umap.ParametricUMAP(parametric_embedding=1, verbose=True, n_training_epochs=2, batch_size=data_summary.shape[0])

            low_dim_embedding = umap.UMAP().fit_transform(data_summary.cpu().numpy())

            ###############

            total_label_ind=0
            breakout=False
            for vertical_dim in range(self.num_vertical_plots):

                ## only get 1st and 2nd central moment for each dim currently
                for horizontal_dim in range(self.num_horizontal_plots):

                    if(self.label_names[total_label_ind]=="entropy"):
                        if(hasattr(system.pdf, "entropy")):
                            these_label_values=system.pdf.entropy(conditional_input=data_summary, samplesize=8)["total"].cpu().numpy()
                        else:
                            total_label_ind+=1
                            continue
                    else:
                        these_label_values=numpy.array(batch_info["event_properties"][self.label_names[total_label_ind]])
                    min_val=min(these_label_values)
                    max_val=max(these_label_values)

                    colorscale=(these_label_values-min_val)/(max_val-min_val)

                    ax = fig.add_subplot(subgridspec[vertical_dim, horizontal_dim+self.num_horizontal_plots*ds_index])

                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                    ax.scatter(low_dim_embedding[:,0], low_dim_embedding[:,1], c=colorscale)

                    ax.set_title(self.label_names[total_label_ind])

                    total_label_ind+=1

                    if(total_label_ind>=len(self.label_names)):
                        breakout=True
                        break


                if(breakout):
                    break

        fig.tight_layout()
