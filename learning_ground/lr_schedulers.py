import torch
from torch import inf
from torch.optim import Optimizer
import collections
import numpy
import sys

import copy
from typing import Callable, Optional, Union

from torch import nn
from torch.optim.swa_utils import SWALR

import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import _load as pl_load

from lightning.pytorch.callbacks import Callback
#from pytorch_lightning.callbacks.base import Callback
#from pytorch_lightning.trainer.optimizers import _get_default_scheduler_config
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import glob
import os
import pickle

import fnmatch

_AVG_FN = Callable[[torch.Tensor, torch.Tensor, torch.LongTensor], torch.FloatTensor]

def make_reducing_and_averaging_scheduler(scheduler_class):

    class red_avg_scheduler(scheduler_class):

        def __init__(self, *args, 
                             red_min_lr=0.0,
                             red_averaging_steps=None, 
                             red_reducing_steps=None, 
                             red_cumulative_averaging_value=0.9,
                             red_num_anomaly_backwards_steps=20,
                             red_backwards_multiple_lr_scheduling=2,
                             red_upside_fluctuation_regulation_threshold=3,
                             # reduction pars
                             red_mode="min",
                             red_factor=0.5, 
                             #red_eps=1e-8,
                             red_patience=10,
                             red_threshold=1e-4, 
                             red_threshold_mode='abs',
                             tracking_metric="",
                             force_reduction_in_fixed_intervals=0,
                             warmup_period_steps=-1,
                             **kwargs):

            self.red_min_lr=red_min_lr
            self.red_averaging_steps=red_averaging_steps
            self.red_reducing_steps=red_reducing_steps
            self.red_cumulative_averaging_value=red_cumulative_averaging_value
            self.red_num_anomaly_backwards_steps=red_num_anomaly_backwards_steps
            self.red_backwards_multiple_lr_scheduling=red_backwards_multiple_lr_scheduling
            self.red_upside_fluctuation_regulation_threshold=red_upside_fluctuation_regulation_threshold

            self.red_factor=red_factor
            self.red_patience=red_patience
            self.red_threshold=red_threshold
            self.red_threshold_mode=red_threshold_mode
            self.red_mode=red_mode

            self.tracking_metric=tracking_metric

            self.force_reduction_in_fixed_intervals=force_reduction_in_fixed_intervals
            self.warmup_period_steps=warmup_period_steps
            #self.red_eps=red_eps
            
            self.effective_beta=None
            if(red_cumulative_averaging_value>0.0):
                assert(red_averaging_steps is not None), "Please specify *red_averaging_steps*"
                self.effective_beta=(1.0-red_cumulative_averaging_value)**(1.0/float(red_averaging_steps))
            print("SCHEDULER EFFECTIVE BETA ", self.effective_beta)
            print("SET MIN LR IN INIT ", self.red_min_lr)
            self.red_past_metrics=None
            self.red_reducing_steps = red_reducing_steps

            self.in_sampling_regime=False
            self.large_outlier_in_last_step=False

            self._reset_reduced_and_averaging_vars()

            super(red_avg_scheduler, self).__init__(*args, **kwargs)


        def _reset_reduced_and_averaging_vars(self):
            """Resets num_bad_epochs counter and cooldown counter."""
            #print("LR RESET")
            
            self.past_metrics = None
            self.outlier_issues=[]

            self.averaged_param_vector=None
            self.averaged_param_counter=0

            if(self.red_reducing_steps is not None):
                ## the bakcwards list is *backwards_multiple* as long as the amount of averaging
                self.past_metrics=collections.deque(maxlen=self.red_backwards_multiple_lr_scheduling*self.red_reducing_steps)

            print("LEN OF VALIDATION STEPS OF PAST METRICS ... ", len(self.past_metrics), "indep .. ", self.red_backwards_multiple_lr_scheduling, " numred ..", self.red_reducing_steps)
            self.past_anomaly_detection_metrics=dict()#collections.deque(maxlen=self.num_anomaly_backwards_steps)

        def _check_lrs_and_reduce(self, metrics):

            

            ## outliers for resetting .. not implemented for now
            """
            for key in metrics.keys():
                if(key != "total"):

                    if(key not in self.past_anomaly_detection_metrics.keys()):
                        self.past_anomaly_detection_metrics[key]=collections.deque(maxlen=self.num_anomaly_backwards_steps)

                    current = float(metrics[key])

                    

                    self.past_anomaly_detection_metrics[key].append(current)

                    

                    ## need at least 3 points to calculate std of difference
                    if(len(self.past_anomaly_detection_metrics[key])>2):

                        np_past=numpy.array(self.past_anomaly_detection_metrics[key])
                        past_diffs=(np_past[1:]-np_past[:-1])

                        mean_of_diffs=numpy.mean(past_diffs)
                        std_of_diffs=numpy.std(past_diffs)
                        
                        last_relative_deviation=(past_diffs[-1]-mean_of_diffs)/std_of_diffs
                        print("past %d diffs ... " % len(past_diffs), past_diffs)
                        print("mean / std:" , mean_of_diffs, std_of_diffs)
                        print("-------------_> last relative deviation vs mean/std", key,  last_relative_deviation)
                        print("mean / std ", mean_of_diffs, std_of_diffs)
                        # only register significant fluctuations to the upside
                        if(last_relative_deviation>self.upside_fluctuation_regulation_threshold):
                            print("### upwards fluctuation in ", key, " of over ", last_relative_deviation, " sigmas")
                            print("mean/std of diffs")
                            print(mean_of_diffs, std_of_diffs)

                            print("last deviation ", past_diffs[-1])
                            print("last relative deviation ", last_relative_deviation)
                         
                            self.large_outlier_in_last_step=True

                            ## log the step of optimizre and the loss which had the issue

                           
                            self.outlier_issues.append(key)

                            #self.relative_deviations=last_relative_deviation
            """

            ###########################################
            ## now normal scheduling

            assert(len(metrics.keys())==1)
            metric_name=[i for i in metrics.keys()][0]
            current = float(metrics[metric_name])

            if(self.past_metrics is not None):
                self.past_metrics.append(current)

                print("threshold ", self.red_threshold)
                print("past metrics ..", len(self.past_metrics), " avg step to check .. ", self.red_backwards_multiple_lr_scheduling*self.red_reducing_steps)

                #################################

                if(len(self.past_metrics)==self.red_backwards_multiple_lr_scheduling*self.red_reducing_steps):
                    # we reach full length - average!
                    print("reached final point")
                    ## check if we should start sampling
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        cur_lr=float(self.base_lrs[i])
                        target_lr=self.red_min_lr

                        ## no final hvp averaging anymore - pure sampling
                        if(cur_lr==target_lr):
                            print("matching target lr ..")
                            if(self.in_sampling_regime == False):
                                print("going to sampling regime..")
                                ## set betas for sampling regime

                                ## SWITCH OFF BETAS
                                """
                                if("betas" in param_group.keys()):
                                    ## set beta1 (gradient avg) and beta3 (hvp average) to 0
                                    if(len(param_group['betas'])==3):
                                        param_group['betas']=(0.0, param_group['betas'][1], 0.0)
                                    elif(len(param_group["betas"])==2):
                                        param_group['betas']=(0.0, param_group['betas'][1])
                                """
                                self.in_sampling_regime=True

                                
                    ##############################################
                    ######## HACK HACK HACK
                    #### safety here... cannot go with red_min_lr below eta_min...
                    if(hasattr(self, "eta_min")):
                        if(self.red_min_lr<self.eta_min):
                            self.red_min_lr=self.eta_min*1.00001

                    if(self.force_reduction_in_fixed_intervals>0):

                        ## automatically reduce max LR after fixed period
                        
                        for ind in range(len(self.base_lrs)):
                            self.base_lrs[ind]*=self.red_factor
                            if(self.base_lrs[ind]<self.red_min_lr):
                                self.base_lrs[ind]=self.red_min_lr

                        self.past_metrics.clear()

                    else:
                        #if(self.in_sampling_regime == False):

                        mean_first=numpy.mean(list(self.past_metrics)[:self.red_reducing_steps])
                        mean_last=numpy.mean(list(self.past_metrics)[self.red_reducing_steps:])
                        print("--------------->")
                        print("checking metric .. first:" , mean_first, ".. later.. ", mean_last)
                        print("---------------->")
                        
                        if(not self.is_better(mean_last, mean_first)):
                            ## if the last one is not better, do something about it
                            ## put a fake epoch of -1

                            ## reduce base LR
                            for ind in range(len(self.base_lrs)):
                                self.base_lrs[ind]*=self.red_factor
                                if(self.base_lrs[ind]<self.red_min_lr):
                                    self.base_lrs[ind]=self.red_min_lr

                            self.past_metrics.clear()

                            print("-----------> updating learning rate ....")




        def is_better(self, a, best):
            if self.red_mode == 'min' and self.red_threshold_mode == 'rel':
                if(a>0 and best>0):
                    rel_epsilon = 1. - self.red_threshold
                    return a < best * rel_epsilon
                else:
                    raise Exception("This should strictly only work if the loss isnt negative or becomes negative!")

            elif self.red_mode == 'min' and self.red_threshold_mode == 'abs':
                return a < best - self.red_threshold

            elif self.red_mode == 'max' and self.red_threshold_mode == 'rel':
                rel_epsilon = self.red_threshold + 1.
                return a > best * rel_epsilon

            else:  # mode == 'max' and epsilon_mode == 'abs':
                return a > best + self.red_threshold



        def get_lr(self):

            ### get the actual LR by parent class
            original_vals=super(red_avg_scheduler, self).get_lr()

            if(self._step_count<=self.warmup_period_steps):
                print("step count / warmup ", self._step_count, self.warmup_period_steps)
                print(self.base_lrs)
                cur_lr=self.base_lrs[0]*float(self._step_count)/float(self.warmup_period_steps)

                new_lrs=[cur_lr for i in range(len(self.base_lrs))]
             
                return new_lrs

            if(hasattr(self, "latest_metrics")):
                print("got latest metrics now...")
                # start with extra reducing/averaging once we have metric keys defined (after step 1)

                ## overwrite with warmup if still below startup step

                self._check_lrs_and_reduce(self.latest_metrics)

                ## check if now the new base LR is lower than the intended lr .. if so use that
                if(self.base_lrs[0]<original_vals[0]):
                    return copy.copy(self.base_lrs)
                else:
                    return original_vals

            else:
                print("not yet lateset metrics")
                return original_vals
            
        def step(self, epoch=None, metrics=None):


            ## there is an initial call to *step*, which has _step_count=0 and involves no metrics etc .. skip metric check and init then
            total_loss_name=""
            if(self._step_count>0):
                if(metrics is None):

                    return

                assert(type(metrics)==dict), ("This is an reducing and averaging scheduler.. requires a metric (a dict of metrics) to work with!", metrics)
                found_total_loss_counter=0
                total_loss_name=""
                for mk in metrics.keys():
                    print("checing ..", mk)
                    if(self.tracking_metric==""):
                        print("check va loss.. none")
                        if("val_loss_total" in mk and "averaged" in mk):
                            print("... taking")
                            found_total_loss_counter+=1
                            total_loss_name=mk
                            
                    else:
                        if(fnmatch.fnmatch(mk, self.tracking_metric)):
                            found_total_loss_counter+=1
                            total_loss_name=mk
                            print("... taking")

                print("tracking metric ", self.tracking_metric)
                print("found metric ", total_loss_name)
                print("counter ", found_total_loss_counter)
                assert(found_total_loss_counter==1), (self.tracking_metric, metrics, "No suitable metric found. Either define metric via *scheduler.tracking_metric* or define as ''")#A key containing 'val_loss_total' key must be part of the metrics dict."
                        
               
                ## save metric as latest metric
                ##if(not hasattr(self, "latest_metrics")):
                self.latest_metrics=dict()

                #for k in metrics.keys():
                self.latest_metrics[total_loss_name]=float(metrics[total_loss_name])
            try:
                original_vals=super(red_avg_scheduler, self).get_lr()[0]
            except:
                original_vals=-1.0

            if("nue_" in total_loss_name):
                print("fixing to high LR for nue....")
                self.red_min_lr=4e-5
                self.eta_min=4e-5
                self.base_lrs[0]=4e-5
                
            print("STEP... red min lr ", self.red_min_lr, " eta min ", self.eta_min, " base lr ", self.base_lrs[0], "orignal vals ", original_vals, file=sys.stderr)

            super(red_avg_scheduler, self).step(epoch=epoch)


        def update_running_parameters(self, param_vector):

            if(self.averaged_param_vector==None):

                self.averaged_param_vector=param_vector.cpu().detach().clone()
                
            else:

                used_beta=(1.0-0.99)**(1.0/float(self.red_averaging_steps))
                if(self.effective_beta is not None):
                    used_beta=self.effective_beta
                
                print("avg. with eff. beta .. ", used_beta, " over %d steps" % self.red_averaging_steps,"... bw check red steps.. ", self.red_reducing_steps, "metric ", self.tracking_metric, "MIN LR ", self.red_min_lr)
                bias = 1 - used_beta ** (self.averaged_param_counter+1)
                prev_bias=1 - used_beta ** (self.averaged_param_counter)

                correction=prev_bias/bias
                if(correction==0.0):
                    correction=1.0/bias

                self.averaged_param_vector=self.averaged_param_vector*used_beta*correction+((1.0-used_beta)/bias)*param_vector.cpu().detach()

                
                """
                avg_fac=1.0/float(self.averaged_param_counter+1)

                self.averaged_param_vector=float(self.averaged_param_counter)*avg_fac*self.averaged_param_vector+avg_fac*param_vector.cpu().detach()
                """

            self.averaged_param_counter+=1
            
            return self.averaged_param_vector

    return red_avg_scheduler

class ModifiedPlateauReducer(torch.optim.lr_scheduler._LRScheduler):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    
    --

    A modified version of the ReduceLROnPlateau LR scheduler to implement a "reducing" scheduler that 
    adaptively averages over last items to not be dependent on full epochs. 

    new params: averaging_steps - if not defined, this acts like the normal reduceLROnPlateau
    see https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau for original code


    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, 
                 optimizer, 
                 mode='min', 
                 factor=0.1, 
                 patience=10,
                 threshold=1e-4, 
                 threshold_mode='rel', 
                 cooldown=0,
                 min_lr=0, 
                 eps=1e-8, 
                 verbose=False, 
                 averaging_steps=None, 
                 reducing_steps=None, 
                 cumulative_averaging_value=0.9,
                 num_anomaly_backwards_steps=20,
                 backwards_multiple_lr_scheduling=2,
                 upside_fluctuation_regulation_threshold=3):
        """
        upside_fluctuation_regulation_threshold (int): How many sigmas for an overfluctuation of the last validation difference should the model/optimizer be reset to last checkpoint?
        """
        super().__init__(optimizer)

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self.num_anomaly_backwards_steps=num_anomaly_backwards_steps
        self.backwards_multiple_lr_scheduling=backwards_multiple_lr_scheduling
        self.upside_fluctuation_regulation_threshold=upside_fluctuation_regulation_threshold
        self.averaging_steps=averaging_steps

        self.effective_beta=None
        if(cumulative_averaging_value>0.0):
            assert(averaging_steps is not None)
            self.effective_beta=(1.0-cumulative_averaging_value)**(1.0/float(self.averaging_steps))
        print("SCHEDULER EFFECTIVE BETA ", self.effective_beta)
        self.past_metrics=None
      
        self.reducing_steps = reducing_steps

        self.in_sampling_regime=False
        self.large_outlier_in_last_step=False

        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

        #print("LR SCHEDULER INIT END--------------------")
    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        #print("LR RESET")
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.past_metrics = None
        self.outlier_issues=[]

        self._last_lr=[]

        self.averaged_param_vector=None
        self.averaged_param_counter=0

        if(self.reducing_steps is not None):
            ## the bakcwards list is twice the amount of averaging
            self.past_metrics=collections.deque(maxlen=self.backwards_multiple_lr_scheduling*self.reducing_steps)

        self.past_anomaly_detection_metrics=dict()#collections.deque(maxlen=self.num_anomaly_backwards_steps)

    def set_optimizer(self, opti):
        self.optimizer=opti
        
    def update_running_parameters(self, param_vector):

        if(self.averaged_param_vector==None):

            self.averaged_param_vector=param_vector.cpu().detach().clone()
            
        else:

            ## TEMP FIX: hardcode averaging steps!
            if(not(hasattr(self, "averaging_steps"))):
                print("fixing averaging stpes to approximately 10000")
                setattr(self, "averaging_steps", 10000)
            ## exponential averaging with 0.99 avg sum by default
            used_beta=(1.0-0.99)**(1.0/float(self.averaging_steps))
            if(self.effective_beta is not None):
                used_beta=self.effective_beta
            
            print("averaging with effective beta .. ", used_beta)
            bias = 1 - used_beta ** (self.averaged_param_counter+1)
            prev_bias=1 - used_beta ** (self.averaged_param_counter)

            correction=prev_bias/bias
            if(correction==0.0):
                correction=1.0/bias

            self.averaged_param_vector=self.averaged_param_vector*used_beta*correction+((1.0-used_beta)/bias)*param_vector.cpu().detach()

            """
            else:
                print("averaging over total dataset (should only be done at end of training!)")
                avg_fac=1.0/float(self.averaged_param_counter+1)

                self.averaged_param_vector=float(self.averaged_param_counter)*avg_fac*self.averaged_param_vector+avg_fac*param_vector.cpu().detach()
            """

        self.averaged_param_counter+=1
        
        return self.averaged_param_vector


    def step(self, metrics=None, epoch=None):

        ## this is a hack to be compatible with _LR_Scheduler and pytorch-lightning in automatic optimization.
        if(metrics is None):
            return None

        if(type(metrics)!=dict):
            return None


        #assert(type(metrics)==dict), "Metrics must be given as dictionary"
        assert("total" in metrics.keys()), "At least the 'total' key must be part of the metrics dict."
        # convert `metrics` to float, in case it's a zero-dim Tensor

        print(">>>>> Scheduler step <<<<<")
        print("--------------------------->----------------------->")

        ############ anomaly detection ##########

        self.large_outlier_in_last_step=False

        for key in metrics.keys():

            if(key != "total"):

                if(key not in self.past_anomaly_detection_metrics.keys()):
                    self.past_anomaly_detection_metrics[key]=collections.deque(maxlen=self.num_anomaly_backwards_steps)

                current = float(metrics[key])

                

                self.past_anomaly_detection_metrics[key].append(current)

                

                ## need at least 3 points to calculate std of difference
                if(len(self.past_anomaly_detection_metrics[key])>2):

                    np_past=numpy.array(self.past_anomaly_detection_metrics[key])
                    past_diffs=(np_past[1:]-np_past[:-1])

                    mean_of_diffs=numpy.mean(past_diffs)
                    std_of_diffs=numpy.std(past_diffs)
                    
                    last_relative_deviation=(past_diffs[-1]-mean_of_diffs)/std_of_diffs
                    print("past %d diffs ... " % len(past_diffs), past_diffs)
                    print("mean / std:" , mean_of_diffs, std_of_diffs)
                    print("-------------_> last relative deviation vs mean/std", key,  last_relative_deviation)
                    print("mean / std ", mean_of_diffs, std_of_diffs)
                    # only register significant fluctuations to the upside
                    if(last_relative_deviation>self.upside_fluctuation_regulation_threshold):
                        print("### upwards fluctuation in ", key, " of over ", last_relative_deviation, " sigmas")
                        print("mean/std of diffs")
                        print(mean_of_diffs, std_of_diffs)

                        print("last deviation ", past_diffs[-1])
                        print("last relative deviation ", last_relative_deviation)
                     
                        self.large_outlier_in_last_step=True

                        ## log the step of optimizre and the loss which had the issue

                       
                        self.outlier_issues.append(key)

                        #self.relative_deviations=last_relative_deviation
                   
        ###########################################
        ## now normal scheduling

        current = float(metrics["total"])

        if(self.past_metrics is not None):
            self.past_metrics.append(current)

            print("scheduler eps ", self.eps)
            print("threshold ", self.threshold)
            print("past metrics ..", len(self.past_metrics), " avg step to check .. ", self.backwards_multiple_lr_scheduling*self.reducing_steps)

            #################################

            if(len(self.past_metrics)==self.backwards_multiple_lr_scheduling*self.reducing_steps):
                # we reach full length - average!
                print("reached final point")
                ## check if we should start sampling
                for i, param_group in enumerate(self.optimizer.param_groups):
                    cur_lr=float(param_group['lr'])
                    target_lr=self.min_lrs[i]

                    ## no final hvp averaging anymore - pure sampling
                    if(cur_lr==target_lr):
                        print("matching target lr ..")
                        if(self.in_sampling_regime == False):
                            print("going to sampling regime..")
                            ## set betas for sampling regime

                            ## TODO: For now switch off beta change in sampling regime
                            """
                            if("betas" in param_group.keys()):
                                ## set beta1 (gradient avg) and beta3 (hvp average) to 0
                                if(len(param_group['betas'])==3):
                                    param_group['betas']=(0.0, param_group['betas'][1], 0.0)
                                elif(len(param_group["betas"])==2):
                                    param_group['betas']=(0.0, param_group['betas'][1])
                            """

                            self.in_sampling_regime=True

                            
                ##############################################

                if(self.in_sampling_regime == False):

                    mean_first=numpy.mean(list(self.past_metrics)[:self.reducing_steps])
                    mean_last=numpy.mean(list(self.past_metrics)[self.reducing_steps:])
                    print("--------------->")
                    print("checking metric .. first:" , mean_first, ".. later.. ", mean_last)
                    print("---------------->")
                    
                    if(not self.is_better(mean_last, mean_first)):
                        ## if the last one is not better, do something about it
                        ## put a fake epoch of -1
                        self._reduce_lr(-1)

                        self.past_metrics.clear()

                        print("-----------> updating learning rate ....")


        ###########

        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
            
        self.last_epoch = epoch


        """

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        """

        ## update last lr

        #print("OPTIMIZER and optimizer LR in SCHED step")
        #print(self.optimizer)
        #print([group['lr'] for group in self.optimizer.param_groups])
        #print("--------------------")
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        """
        check_sampling=True
        for ind, last_lr in enumerate(self._last_lr):
            if(self.min_lrs[ind]!=last_lr):
                check_sampling=False
                break
                
        if(check_sampling):

            self.in_sampling_regime=True
        """

    def _reduce_lr(self, epoch, force=False):
        for i, param_group in enumerate(self.optimizer.param_groups):

            old_lr = float(param_group['lr'])

            if(force):
                ## force new reduction, disregarding minimum target LR
                param_group['lr'] = old_lr * self.factor
            else:
                
                new_lr = max(old_lr * self.factor, self.min_lrs[i])
                if old_lr - new_lr > self.eps:
                    param_group['lr'] = new_lr
                    if self.verbose:
                        print('Epoch {:5d}: reducing learning rate'
                              ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

        for i, param_group in enumerate(self.optimizer.param_groups):
            print("param group ", i)
            print("reduced LR ... new LR ", param_group["lr"])

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            if(a>0 and best>0):
                rel_epsilon = 1. - self.threshold
                return a < best * rel_epsilon
            else:
                raise Exception("This should strictly only work if the loss isnt negative or becomes negative!")

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)





class SchedulerCallback(Callback):
    def __init__(
        self,
        swa_epoch_start: Union[int, float] = 0.8,
        swa_lrs: Optional[Union[float, list]] = None,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
        avg_fn: Optional[_AVG_FN] = None,
        device: Optional[Union[torch.device, str]] = torch.device("cpu"),
    ):
        r"""

        Implements the Stochastic Weight Averaging (SWA) Callback to average a model.

        Stochastic Weight Averaging was proposed in ``Averaging Weights Leads to
        Wider Optima and Better Generalization`` by Pavel Izmailov, Dmitrii
        Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
        (UAI 2018).

        This documentation is highly inspired by PyTorch's work on SWA.
        The callback arguments follow the scheme defined in PyTorch's ``swa_utils`` package.

        For a SWA explanation, please take a look
        `here <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging>`_.

        .. warning:: ``StochasticWeightAveraging`` is in beta and subject to change.

        .. warning:: ``StochasticWeightAveraging`` is currently not supported for multiple optimizers/schedulers.

        .. warning:: ``StochasticWeightAveraging`` is currently only supported on every epoch.

        SWA can easily be activated directly from the Trainer as follow:

        .. code-block:: python

            Trainer(stochastic_weight_avg=True)

        Arguments:

            swa_epoch_start: If provided as int, the procedure will start from
                the ``swa_epoch_start``-th epoch. If provided as float between 0 and 1,
                the procedure will start from ``int(swa_epoch_start * max_epochs)`` epoch

            swa_lrs: the learning rate value for all param groups together or separately for each group.

            annealing_epochs: number of epochs in the annealing phase (default: 10)

            annealing_strategy: Specifies the annealing strategy (default: "cos"):

                - ``"cos"``. For cosine annealing.
                - ``"linear"`` For linear annealing

            avg_fn: the averaging function used to update the parameters;
                the function must take in the current value of the
                :class:`AveragedModel` parameter, the current value of :attr:`model`
                parameter and the number of models already averaged; if None,
                equally weighted average is used (default: ``None``)

            device: if provided, the averaged model will be stored on the ``device``.
                When None is provided, it will infer the `device` from ``pl_module``.
                (default: ``"cpu"``)

        """

        if device is not None and not isinstance(device, (torch.device, str)):
            raise MisconfigurationException(f"device is expected to be a torch.device or a str. Found {device}")

        self._device = device
      

    """
    @staticmethod
    def pl_module_contains_batch_norm(pl_module: "pl.LightningModule"):
        return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in pl_module.modules())

    def on_before_accelerator_backend_setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # copy the model before moving it to accelerator device.
        with pl_module._prevent_trainer_and_dataloaders_deepcopy():
            self._average_model = deepcopy(pl_module)
    

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        optimizers = trainer.optimizers
        lr_schedulers = trainer.lr_schedulers

        if len(optimizers) != 1:
            raise MisconfigurationException("SWA currently works with 1 `optimizer`.")

        if len(lr_schedulers) > 1:
            raise MisconfigurationException("SWA currently not supported for more than 1 `lr_scheduler`.")

        if isinstance(self._swa_epoch_start, float):
            self._swa_epoch_start = int(trainer.max_epochs * self._swa_epoch_start)

        self._model_contains_batch_norm = self.pl_module_contains_batch_norm(pl_module)

        self._max_epochs = trainer.max_epochs
        if self._model_contains_batch_norm:
            # virtually increase max_epochs to perform batch norm update on latest epoch.
            trainer.fit_loop.max_epochs += 1
    """

    #def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

    #    self._check_scheduler(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        
        self._check_scheduler(trainer, pl_module)

    def _check_scheduler(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        if(pl_module.scheduler_config["scheduler.name"]=="bayes_reduce"):
            
            # obtain scheduler
            current_scheduler=pl_module.lr_schedulers()

            # obtain current learning rate
            old_lrs=[]

            assert(len(trainer.optimizers)==1)
            
            for i, param_group in enumerate(trainer.optimizers[0].param_groups):
                
                old_lrs.append(float(param_group['lr']))

            # check if we are (usually above) different than minmum LR .. if we are smaller, than we also go back up
            if(old_lrs[-1]!=current_scheduler.min_lrs[-1]):

                if(current_scheduler.large_outlier_in_last_step):
                    
                    print("large outlier in check.....................")
                    ### reload module with last checkpoint ...
                    
                    possible_cp_dirs=glob.glob(os.path.join(trainer.logger.log_dir, "checkpoint_epoch=*"))
                    possible_cp_dirs.sort()

                    final_ids=numpy.array([int(i.split("=")[-1]) for i in possible_cp_dirs])

                    real_sort_mask=numpy.argsort(final_ids)
                    final_str=numpy.array(possible_cp_dirs)[real_sort_mask][-1]

                    possible_cp_files=glob.glob(os.path.join(final_str, "cp_*"))

                    assert(len(possible_cp_files)>0), ("Could not find any checkpoint files in last cp dir ", final_str, " while trying to re-load older checkpoint for large outlier correction in scheduler.")
                        
                    ## take one of the checkpoints in cp dir, they are identical
                    final_str=possible_cp_files[-1]
                    # reload weights from last checkpoint on current model

                    ## check if going back two is better
                    if(len(possible_cp_files)>1):
                        final_str=possible_cp_files[-2]

                    print("Resetting model based on scheduler overfluctiation .. loading model from ", final_str)
                    
                    ckpt = pl_load(
                            final_str,
                            map_location=lambda storage, loc: storage)

                    with open(os.path.join(trainer.logger.log_dir, "params.pkl"), "rb") as f:
                        inference_config=pickle.load(f)

                    # copy fixed choices over for correct model def
                    if("fixed_choices" in inference_config.keys()):
                        for k in inference_config["fixed_choices"]:
                            inference_config[k]=inference_config["fixed_choices"][k]
                    
                

                    new_module=next(trainer.lightning_module.modules())


                    #print("HAS ATTR BEFORE LOADING ", hasattr(new_module, "num_validation_steps_per_sweep"))
                    #pl_module=pl_module ._load_model_state( ckpt, config=inference_config)
                    trainer.strategy.load_model_state_dict(ckpt)

                 
            
                    ## overwrite optimizer 
                    ## also check that optimier overwrite works
                    trainer.strategy.load_optimizer_state_dict(ckpt)
                    #trainer.training_type_plugin.load_optimizer_state_dict(ckpt)

                   
                    # set the optimizers for the scheduler
                    current_scheduler.set_optimizer(trainer.optimizers[0])
                    
                    # reset outlier flag
                    current_scheduler.large_outlier_in_last_step=False

                    ## reduce lr in standard way
                    print("reducing LR due to overfluctuation ...")
                    current_scheduler._reduce_lr(-1, force=False)
                    current_scheduler.past_metrics.clear()

                    print("reset all past metrics and continuing with lower LR....")
             