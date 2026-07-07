import numpy
import torch

def select_random_items(array, N):
    L = len(array)
    
    random_indices = numpy.random.choice(L, size=N, replace=False)
    random_items = array[random_indices]

    return random_items

def compute_mutual_information(pdf, default_summary, all_values, index_to_check, samplesize=200):
    """
    Computes mutual information for summary statistic index *index_to_check* given a distribution of values *all_values*.
    """
    random_values=select_random_items(all_values, samplesize)
    
    repeated_summary=default_summary[None,:].repeat(samplesize, axis=0)
    repeated_summary[:,index_to_check]=random_values
    

    used_dtype, used_device=pdf.obtain_current_dtype_n_device()

    with torch.no_grad():
        cinput=torch.from_numpy(repeated_summary).to(device=used_device, dtype=used_dtype)
        samples,_,log_pdfs,_=pdf.sample(conditional_input=cinput,force_embedding_coordinates=True)

        rep_samples=samples.repeat_interleave(samplesize, dim=0)
        repeated_input=cinput.repeat(samplesize, 1)

        log_prior,_,_=pdf(rep_samples, conditional_input=repeated_input,force_embedding_coordinates=True)

        print(log_pdfs[0], " from ", cinput[0])
        print("log priors... ", log_prior[:samplesize])
        print(torch.logsumexp(log_prior[:samplesize],0)-numpy.log(samplesize), " from ... ", repeated_input[:100,:2])

        prior_log_pdfs=torch.logsumexp(log_prior.reshape(samplesize, -1), dim=-1)-numpy.log(samplesize)

        
    return float((log_pdfs-prior_log_pdfs).mean().cpu())
