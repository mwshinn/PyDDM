'''
Simulation code for Drift Diffusion Model
Author: Norman Lam (norman.lam@yale.edu)
'''

import numpy as np
from scipy.optimize import minimize
import copy

from DDM_parameters import *


########################################################################################################################
### Defined functions.

# Function that simulates one trial of the time-varying drift/bound DDM
def DDM_pdf_general(mu, mudep, sigma, sigmadep, B, bounddep, task=None, IC=None):
    '''
    Now assume f_mu, f_sigma, f_Bound are callable functions
    '''
    ### Initialization
    mu_base = mu # Constant component of drift rate mu.
    sigma = sigma # Constant (and currently only) component of noise sigma.
    bound_base = B # Constant component of the bound.

    # We convert `setting_index`, an indicator of which predefined
    # settings to use for mu/sigma/bound dependence on x and t, into a
    # list containing the actual specifications of those.  See
    # DDM_parameters.py for definitions of these presets.

    # Control the type of model we use for mu/sigma dependence on x/t.
    # Here, `x` and `t` are coefficients to x and t, the particular
    # use of which depends on the model (specified in `name`).
    mudep = mudep
    sigmadep = sigmadep
    bounddep = bounddep

    if IC == None:
        IC = "point_source_center"
    
    ### Initialization: Lists
    pdf_curr = f_initial_condition(IC, x_list) # Initial condition
    pdf_prev = np.zeros((len(x_list)))
    Prob_list_corr = np.zeros(len(t_list)) # Probability flux to correct choice
    Prob_list_err = np.zeros(len(t_list)) # Probability flux to erred choice

    # If we are in a task, define task specific parameters as
    # `param_task`.  If not, `param_task` is undefined.
    if task == None:
        task = Dependence("Fixed_Duration")
    # if len(params) == 9:
    #     task.add_parameter(param=params[8])
    #     if task_list[task_index] == 'Duration_Paradigm':
    #         task.add_parameter(mu_base=mu_base)
            

    ##### Looping through time and updating the pdf.
    for i_t, t in enumerate(t_list[:-1]): # NOTE I translated this directly from "for i_t in range(len(t_list_temp)-1):" but I think the -1 was a bug -MS
        # Update Previous state. To be frank pdf_prev could be
        # removed for max efficiency. Leave it just in case.
        pdf_prev = copy.copy(pdf_curr)

        # If we are in a task, adjust mu according to the task
        mu = mu_base + f_mu1_task(t, task)

        if sum(pdf_curr[:])>0.0001: # For efficiency only do diffusion if there's at least some densities remaining in the channel.
            ## Define the boundaries at current time.
            bound = f_bound_t(bound_base, bounddep, t) # Boundary at current time-step. Can generalize to assymetric bounds

            ## Now figure out which x positions are still within the
            # (collapsing) bound.
            assert bound_base >= bound, "Invalid change in bound" # Ensure the bound didn't expand
            bound_shift = bound_base - bound
            # Note that we linearly approximate the bound by the two surrounding grids sandwiching it.
            x_index_inner = int(np.ceil(bound_shift/dx)) # Index for the inner bound (smaller matrix)
            x_index_outer = int(bound_shift/dx) # Index for the outer bound (larger matrix)
            weight_inner = (bound_shift - x_index_outer*dx)/dx # The weight of the upper bound matrix, approximated linearly. 0 when bound exactly at grids.
            weight_outer = 1. - weight_inner # The weight of the lower bound matrix, approximated linearly.
            x_list_inbounds = x_list[x_index_outer:len(x_list)-x_index_outer] # List of x-positions still within bounds.
            
            ## Define the diffusion matrix for implicit method
            # Diffusion Matrix for Implicit Method. Here defined as
            # Outer Matrix, and inder matrix is either trivial or an
            # extracted submatrix.
            diffusion_matrix = np.eye(len(x_list_inbounds)) + \
                               f_mu_matrix(mu, mudep, x_list_inbounds, t) + \
                               f_sigma_matrix(sigma, sigmadep, x_list_inbounds, t)
            
            ### Compute Probability density functions (pdf)
            # PDF for outer matrix
            pdf_outer = np.linalg.solve(diffusion_matrix, pdf_prev[x_index_outer:len(x_list)-x_index_outer])
            # PDF for inner matrix (with optimization)
            if x_index_inner == x_index_outer: # Optimization: When bound is exactly at a grid, use the most efficient method.
                pdf_inner = copy.copy(pdf_outer)
            else:
                pdf_inner = np.linalg.solve(diffusion_matrix[1:-1, 1:-1], pdf_prev[x_index_inner:len(x_list)-x_index_inner])

            # Pdfs out of bound is consideered decisions made.
            Prob_list_err[i_t+1] += weight_outer * np.sum(pdf_prev[:x_index_outer]) \
                                    + weight_inner * np.sum(pdf_prev[:x_index_inner])
            Prob_list_corr[i_t+1] += weight_outer * np.sum(pdf_prev[len(x_list)-x_index_outer:]) \
                                     + weight_inner * np.sum(pdf_prev[len(x_list)-x_index_inner:])
            pdf_curr = np.zeros((len(x_list))) # Reconstruct current proability density function, adding outer and inner contribution to it.
            pdf_curr[x_index_outer:len(x_list)-x_index_outer] += weight_outer*pdf_outer
            pdf_curr[x_index_inner:len(x_list)-x_index_inner] += weight_inner*pdf_inner

        else:
            break #break if the remaining densities are too small....

        ### Increase current, transient probability of crossing either
        ### bounds, as flux.  Corr is a correct answer, err is an
        ### incorrect answer
        _inner_bound_corr = x_list[len(x_list)-1-x_index_inner]
        _outer_bound_corr = x_list[len(x_list)-1-x_index_outer]
        _inner_bound_err = x_list[x_index_inner]
        _outer_bound_err = x_list[x_index_outer]
        Prob_list_corr[i_t+1] += weight_outer * pdf_outer[-1] * ( f_mu_flux(_outer_bound_corr, mu, mudep, t) +
                                                                  f_sigma_flux(_outer_bound_corr, sigma, sigmadep, t)) \
                              +  weight_inner * pdf_inner[-1] * ( f_mu_flux(_inner_bound_corr, mu, mudep, t) +
                                                                  f_sigma_flux(_inner_bound_corr, sigma, sigmadep, t))
        Prob_list_err[i_t+1]  += weight_outer * pdf_outer[0] * ( f_mu_flux(_outer_bound_err, mu, mudep, t) +
                                                                 f_sigma_flux(_outer_bound_err, sigma, sigmadep, t)) \
                              +  weight_inner * pdf_inner[0] * ( f_mu_flux(_inner_bound_err, mu, mudep, t) +
                                                                 f_sigma_flux(_inner_bound_err, sigma, sigmadep, t ))

        if bound < dx: # Renormalize when the channel size has <1 grid, although all hell breaks loose in this regime.
            Prob_list_corr[i_t+1] *= (1+ (1-bound/dx))
            Prob_list_err[i_t+1] *= (1+ (1-bound/dx))

    return Prob_list_corr, Prob_list_err # Only return pdf for correct and erred choices. Add more ouputs if needed


### Matrix terms due to different parameters (mu, sigma, bound)
# Diffusion Matrix containing drift=mu related terms.  Reminder: The
# general definition is (mu*p)_(x_{n+1},t_{m+1}) -
# (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the same
# x,t with p(x,t) (probability distribution function). Hence we use
# x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
def f_mu_matrix(mu, mudep, x, t):
    if mudep.name == 'linear_xt': # If dependence of mu on x & t is at most linear (or constant):
        return np.diag( 0.5*dt/dx * (mu + mudep.x*x[1:]  + mudep.t*t), 1) \
             + np.diag(-0.5*dt/dx * (mu + mudep.x*x[:-1] + mudep.t*t),-1)
    elif mudep.name == 'sinx_cost': # Weird dependence for testing. Remove at will.
        return np.diag( 0.5*dt/dx * (mu + mudep.x*np.sin(x[1:])  + mudep.t*np.cos(t)), 1) \
             + np.diag(-0.5*dt/dx * (mu + mudep.x*np.sin(x[:-1]) + mudep.t*np.cos(t)),-1)
    # Add f_mu_setting definitions as needed...
    else:
        print 'Incorrect mu dependency'

# Diffusion Matrix containing noise=sigma related terms
def f_sigma_matrix(sigma, sigmadep, x, t):
    if sigmadep.name == 'linear_xt': # If dependence of mu on x & t is at most linear (or constant):
        return np.diag(1.0*(sigma + sigmadep.x*x      + sigmadep.t*t)**2 * dt/dx**2, 0) \
             - np.diag(0.5*(sigma + sigmadep.x*x[1:]  + sigmadep.t*t)**2 * dt/dx**2, 1) \
             - np.diag(0.5*(sigma + sigmadep.x*x[:-1] + sigmadep.t*t)**2 * dt/dx**2,-1)
    elif sigmadep.name == 'sinx_cost': # Weird dependence for testing. Remove at will.
        return np.diag(1.0*(sigma + sigmadep.x*np.sin(x)      + sigmadep.t*np.cos(t))**2 * dt/dx**2, 0) \
             - np.diag(0.5*(sigma + sigmadep.x*np.sin(x[1:])  + sigmadep.t*np.cos(t))**2 * dt/dx**2, 1) \
             - np.diag(0.5*(sigma + sigmadep.x*np.sin(x[:-1]) + sigmadep.t*np.cos(t))**2 * dt/dx**2,-1)
    # Add f_sigma_setting definitions as needed...
    else:
        print'Invalid sigma dependency'

## Second effect of Collapsing Bounds: Collapsing Center: Positive and Negative states are closer to each other over time.
def f_bound_t(bound, bounddep, t):
    if bounddep.name == 'constant':
        return bound
    elif bounddep.name == 'collapsing_linear':
        return max(bound - bounddep.t*t, 0.)
    elif bounddep.name == 'collapsing_exponential':
        return bound * np.exp(-bounddep.t*t)
    # Add f_bound_setting definitions as needed...
    else:
        print'Wrong/unspecified f_bound_setting for f_bound_t function'

### Amount of flux from bound/end points to correct and erred response probabilities, due to different parameters (mu, sigma, bound)
## Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) -
## (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the same
## x,t with p(x,t) (probability distribution function). Hence we use
## x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
def f_mu_flux(x_bound, mu, mudep, t): # Diffusion Matrix containing drift=mu related terms
    if mudep.name == 'linear_xt': # If dependence of mu on x & t is at most linear (or constant):
        return 0.5*dt/dx * np.sign(x_bound) * (mu + mudep.x*x_bound + mudep.t*t)
    elif mudep.name == 'sinx_cost':
        return 0.5*dt/dx * np.sign(x_list[index_bound]) * (mu + mudep.x*np.sin(x_bound) + mudep.t*np.cos(t))
    # Add f_mu_setting definitions as needed...
    else:
        print'Invalid mu dependency'

def f_sigma_flux(x_bound, sigma, sigmadep, t): # Diffusion Matrix containing noise=sigma related terms
    ## Similar to f_sigma_flux
    if sigmadep.name == 'linear_xt': # If dependence of sigma on x & t is at most linear (or constant):
        return 0.5*dt/dx**2 * (sigma + sigmadep.x*x_bound + sigmadep.t*t)**2
    elif sigmadep.name == 'sinx_cost': # If dependence of sigma on x & t is at most linear (or constant):
        return 0.5*dt/dx**2 * (sigma + sigmadep.x*np.sin(x_bound) + sigmadep.t*np.cos(t))**2
    # Add f_sigma_setting definitions as needed...
    else:
        print'Invalid sigma dependency'

def f_initial_condition(f_IC_setting, x): # Returns the pdf distribution at time 0 of the simulation
        ## Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) - (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the same x,t with p(x,t) (probability distribution function). Hence we use x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
    pdf_IC = np.zeros(len(x))
    if f_IC_setting == 'point_source_center':
        pdf_IC[int((len(x)-1)/2)] = 1. # Initial condition at x=0, center of the channel.
    elif f_IC_setting == 'uniform': # Weird dependence for testing. Remove at will.
        pdf_IC = 1/(len(x))*np.ones((len(x)))
    # Add f_mu_setting definitions as needed...
    else:
        print'Wrong/unspecified f_mu_setting for f_mu_matrix function'
    return pdf_IC








### Fit Functions: Largely overlapping...modify from one...
def MLE_model_fit_over_coh(params, y_2_fit_setting_index, y_fit2): # Fit the final/total probability for correct, erred, and undecided choices.
    Prob_corr_err_undec_temp = np.zeros(3,len(coherence_list))
    for i_coh in range(coherence_list):
        (Prob_list_corr_coh_MLE_temp, Prob_list_err_coh_MLE_temp)     = DDM_pdf_general(params, y_2_fit_setting_index, 0)
        Prob_list_sum_corr_coh_MLE_temp  = np.sum(Prob_list_corr_coh_MLE_temp)
        Prob_list_sum_err_coh_MLE_temp   = np.sum(Prob_list_err_coh_MLE_temp)
        Prob_list_sum_undec_coh_MLE_temp = 1. - Prob_list_sum_corr_coh_MLE_temp - Prob_list_sum_err_coh_MLE_temp
        Prob_corr_err_undec_temp[:,i_coh]   = [Prob_list_sum_corr_coh_MLE_temp, Prob_list_sum_err_coh_MLE_temp, Prob_list_sum_undec_coh_MLE_temp] # Total probability for correct, erred and undecided choices.
    to_min = -np.log(np.sum((y_fit2*Prob_corr_err_undec_temp)**0.5 /dt**1)) # Bhattacharyya distance
    return to_min
# Other minimizers
    # to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2) # MLE
    # to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0) # KL divergence

def MSE_model_fit_RT(params, y_2_fit_setting_index, y_fit2): # Fit the probability density functions of both correct and erred choices. TO BE VERIFIED.
    (Prob_list_corr_temp, Prob_list_err_temp)     = DDM_pdf_general(params, y_2_fit_setting_index, 0)
    to_min = -np.log(np.sum((y_fit2*np.column_stack(Prob_list_corr_temp, Prob_list_err_temp))**0.5)) # Bhattacharyya distance
    return to_min
#    to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2)                                                           # MLE
#    to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0)     #KL divergence












########################################################################################################################
## Functions for Analytical Solutions.
### Analytical form of DDM. Can only solve for simple DDM or linearly collapsing bound... From Anderson1960
# Note that the solutions are automatically normalized.
def DDM_pdf_analytical(mu, mudep=None, sigma=None, sigmadep=None, B=None, bounddep=None, task=None, IC=None):
    '''
    Now assume f_mu, f_sigma, f_Bound are callable functions
    See DDM_pdf_general for nomenclature
    '''

    assert mu != None and sigma != None and B != None, "Please specify mu, sigma, and B"
    ### Initialization
    if mudep != None:
        assert mudep == Dependence("linear_xt", x=0, t=0), "mu dependence not implemented"
    if sigmadep != None:
        assert sigmadep == Dependence("linear_xt", x=0, t=0), "sigma dependence not implemented"
    if bounddep != None:
        assert bounddep.name in ["constant", "collapsing_linear"], "bounddep dependence not implemented"
    if task != None:
        assert task.name == "Fixed_Duration"
    if IC != None:
        assert IC == "point_source_center"
    
    if bounddep.name == "constant": # Simple DDM
        DDM_anal_corr, DDM_anal_err = analytic_ddm(mu, sigma, B, t_list)
    elif bounddep.name == "collapsing_linear": # Linearly Collapsing Bound
        DDM_anal_corr, DDM_anal_err = analytic_ddm(mu, sigma, B, t_list, -bounddep.t) # TODO why must this be negative? -MS

    ## Remove some abnormalities such as NaN due to trivial reasons.
    DDM_anal_corr[DDM_anal_corr==np.NaN] = 0.
    DDM_anal_corr[0] = 0.
    DDM_anal_err[DDM_anal_err==np.NaN] = 0.
    DDM_anal_err[0] = 0.
    return DDM_anal_corr*dt, DDM_anal_err*dt


def analytic_ddm_linbound(a1, b1, a2, b2, teval):
    '''
    Calculate the reaction time distribution of a Drift Diffusion model
    with linear boundaries, zero drift, and sigma = 1.

    The upper boundary is y(t) = a1 + b1*t
    The lower boundary is y(t) = a2 + b2*t
    The starting point is 0
    teval is the array of time where the reaction time distribution is evaluated

    Return the reaction time distribution of crossing the upper boundary

    Reference:
    Anderson, Theodore W. "A modification of the sequential probability ratio test
    to reduce the sample size." The Annals of Mathematical Statistics (1960): 165-197.

    Code: Guangyu Robert Yang 2013
    '''
    # Avoid dividing by zero
    teval[teval==0] = 1e-30
    
    # Change of variables
    tmp = -2.*((a1-a2)/teval+b1-b2)

    # Initialization
    nMax     = 100  # Maximum looping number
    errbnd   = 1e-7 # Error bound for looping
    suminc   = 0
    checkerr = 0

    for n in xrange(nMax):
        # increment
        inc = np.exp(tmp*n*((n+1)*a1-n*a2))*((2*n+1)*a1-2*n*a2)-\
              np.exp(tmp*(n+1)*(n*a1-(n+1)*a2))*((2*n+1)*a1-2*(n+1)*a2)
        suminc += inc
        # Break when the relative increment is low for two consecutive updates
        # if(max(abs(inc/suminc)) < errbnd):
        #     checkerr += 1
        #     if(checkerr == 2):
        #         break
        # else:
        #     checkerr = 0

    # Probability Distribution of reaction time
    dist=np.exp(-(a1+b1*teval)**2./teval/2)/np.sqrt(2*np.pi)/teval**1.5*suminc;
    dist = dist*(dist>0) # make sure non-negative
    return dist

def analytic_ddm(mu, sigma, b, teval, b_slope=0):
    '''
    Calculate the reaction time distribution of a Drift Diffusion model
    Parameters
    -------------------------------------------------------------------
    mu    : Drift rate
    sigma : Noise intensity
    B     : Constant boundary
    teval : The array of time points where the reaction time distribution is evaluated
    b_slope : (Optional) If provided, then the upper boundary is B(t) = b + b_slope*t,
              and the lower boundary is B(t) = -b - b_slope*t

    Return:
    dist_cor : Reaction time distribution at teval for correct trials
    dist_err : Reaction time distribution at teval for error trials
    '''
    # Scale B, mu, and (implicitly) sigma so new sigma is 1
    b       /= sigma
    mu      /= sigma
    b_slope /= sigma

    # Get valid time points (before two bounds collapsed)
    teval_valid = teval[b+b_slope*teval>0]

    dist_cor = analytic_ddm_linbound(b, -mu+b_slope, -b, -mu-b_slope, teval_valid)
    dist_err = analytic_ddm_linbound(b,  mu+b_slope, -b,  mu-b_slope, teval_valid)

    # For invalid time points, set the probability to be a very small number
    if len(teval_valid) < len(teval):
        eps = np.ones(len(teval)-len(teval_valid)) * 1e-100
        dist_cor = np.concatenate((dist_cor,eps))
        dist_err = np.concatenate((dist_err,eps))

    return dist_cor, dist_err





## Various tasks that causes change in signals and what not, in addition to the space and time varying f_mu, f_sigma, and f_bound.
def f_mu1_task(t, task): # Define the change in drift at each time due to active perturbations, in different tasks
    if task.name == 'Fixed_Duration':                                                                                # No task
        return 0.
    elif task.name == 'PsychoPhysical_Kernel': #Note that I assumed in the DDM_general fcn for the PK case, that the input of mu_0 to be 0. Else have to counter-act the term...
        return task.param[int(t/dt_mu_PK)] ## Fix/Implement later
    ## For Duration/Pulse paradigms, param_task[0]= magnitude of pulse. param_task[1]= T_Dur_Duration/t_Mid_Pulse respectively
    elif task.name == 'Duration_Paradigm': # Vary Stimulus Duration
        T_Dur_Duration = task.param # Duration of pulse. Variable in Duration Paradigm
        ## if stimulus starts at 0s:
        if t< T_Dur_Duration:
            return 0
        else:
            return -task.mu_base #Remove pulse if T> T_Dur_duration
        ## if stimulus starts at T_dur/2 =1s, use this instead and put it above:
        # t_Mid_Duration = T_dur/2. # (Central) Onset time of pulse/duration. Arbitrary but set to constant as middle of fixed duration task.
        # if abs(t-t_Mid_Duration)< (T_Dur_Duration/2):
    elif task.name == 'Pulse_Paradigm': # Add brief Pulse to stimulus, with varying onset time.
        T_Dur_Pulse = 0.1 # Duration of pulse. 0.1s in the spiking circuit case.
        t_Pulse_onset = task.param # (Central) Onset time of pulse/duration. Variable in Pulse Paradigm
        if (t>t_Pulse_onset) and  (t<(t_Pulse_onset+T_Dur_Pulse)):
            return mu_0*0.15 # 0.15 based on spiking circuit simulations.
        else:
            return 0.
        ## If we define time as at the middle of pulse, use instead:
        # t_Mid_Pulse = param_task # (Central) Onset time of pulse/duration. Variable in Pulse Paradigm
        # if abs(t-t_Mid_Pulse)< (T_Dur_Pulse/2):
    # Add task_setting definitions as needed...
    else:
        print'Invalid task'




def Psychometric_fit_P(params_pm, pm_fit2):
    if params_pm[0] == 0:
        print params_pm
    prob_corr_fit = 0.5 + 0.5*np.sign(mu_0_list_Pulse+params_pm[2])*(1. - np.exp(-(np.abs(mu_0_list_Pulse+params_pm[2])/params_pm[0])**params_pm[1])) #Use duration paradigm and add shift parameter. Fit for both positive and negative
    to_min = np.sum((prob_corr_fit-pm_fit2)**2)                                                                         # Least Square
    return to_min
# Other possible forms of psychometric function
    # prob_corr_fit = 1./(1.+ np.exp(-params_pm[1]*(mu_0_list_Pulse+params_pm[0])))
    # prob_corr_fit = 0.5 + 0.5*np.sign(mu_0_list_Pulse)*(1. - np.exp(-np.sign(mu_0_list_Pulse)*((mu_0_list_Pulse+params_pm[2])/params_pm[0])**params_pm[1])) #Use duration paradigm and add shift parameter. Fit for both positive and negative
# Other possible minimizers
    # to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2) # Maximum Likelihood Estimator
    # to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0) # KL divergence
    # to_min = -np.log(np.sum((pm_fit2*Prob_list_corr_temp)**0.5 /dt**1)) # Bhattacharyya distance

def Psychometric_fit_D(params_pm, pm_fit2):
    prob_corr_fit = 0.5 + 0.5*(1. - np.exp(-(1e-30+mu_0_list/params_pm[0])**params_pm[1])) # Add 1e-10 to avoid divide by zero for negative powers
    # 1./(1.+ np.exp(-params_pm[1]*(mu_0_list+params_pm[0])))
    return np.sum((prob_corr_fit-pm_fit2)**2) # Least Square
# Other possible minimizers
    # to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2) # Maximum Likelihood Estimator
    # to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0) # KL divergence
    # to_min = -np.log(np.sum((pm_fit2*Prob_list_corr_temp)**0.5 /dt**1)) # Bhattacharyya distance


def Threshold_D_fit(param_Thres, pm_fit2, n_skip, t_dur_list_duration):
    prob_corr_fit = param_Thres[0] + param_Thres[3]*(np.exp(-((t_dur_list_duration[n_skip:]-param_Thres[2])/param_Thres[1])))
    return np.sum((prob_corr_fit-pm_fit2[n_skip:])**2) # Least Square
# Other possible forms of prob_corr_fit
    # prob_corr_fit = param_Thres[0] + (100.-param_Thres[0])*(np.exp(-((t_dur_list_duration-param_Thres[2])/param_Thres[1])))
    # prob_corr_fit = param_Thres[0] + (100.-param_Thres[0])*(np.exp(-((t_dur_list_duration-param_Thres[2])/param_Thres[1])))
    # prob_corr_fit = param_Thres[0] + param_Thres[2]*(np.exp(-((t_dur_list_duration)/param_Thres[1])))
        # 1./(1.+ np.exp(-params_pm[1]*(mu_0_list+params_pm[0])))
# Other Minimizers
    # Other posssible forms of minimizer
    # to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2) # MAximum Likelihood
    # to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0) # KL divergence
    # pm_fit2 /= np.sum(pm_fit2)
    # Prob_list_corr_temp /= np.sum(Prob_list_corr_temp)
    # to_min = -np.log(np.sum((pm_fit2*Prob_list_corr_temp)**0.5 /dt**1)) # Bhattacharyya distance
