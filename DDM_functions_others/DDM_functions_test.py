'''
Simulation code for Drift Diffusion Model
Author: Norman Lam (norman.lam@yale.edu)
'''
#import brian_no_units           #Note: speeds up the code
from numpy.fft import rfft,irfft
import time
import numpy as np
from scipy.special import erf
#from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
import cProfile
import re
import matplotlib.cm as matplotlib_cm
import math
#import sys
#sys.path.append('C:\Users\nhl8\Desktop\CCNSS\Drift_Diffusion_Model')
from ddm import analytic_ddm
import multiprocessing
import copy





########################################################################################################################
### Defined functions.

# Function that simulates one trial of the time-varying drift/bound DDM
#def DDM_pdf_general(mu, f_mu_setting, sigma, f_sigma_setting, B, f_bound_setting, param_mu, param_sigma):
def DDM_pdf_general(params, setting_index, task_index=0):
    '''
    Now assume f_mu, f_sigma, f_Bound are callable functions
    '''
    ### Initialization
    mu_0_temp          = params[0]                                                                                      # Constant component of drift rate mu.
    param_mu_x_temp    = params[1]                                                                                      # Parameter for x_dependence of mu. Add more if 1 param is not sufficient...
    param_mu_t_temp    = params[2]                                                                                      # Parameter for t_dependence of mu. Add more if 1 param is not sufficient...
    sigma_0_temp       = params[3]                                                                                      # Constant component of noise sigma.
    param_sigma_x_temp = params[4]                                                                                      # Parameter for t_dependence of sigma. Add more if 1 param is not sufficient...
    param_sigma_t_temp = params[5]                                                                                      # Parameter for t_dependence of sigma. Add more if 1 param is not sufficient...
    B_0_temp           = params[6]                                                                                      # Constant component of drift rate mu.
    param_B_t_temp     = params[7]                                                                                      # Parameter for t_dependence of B (no x-dep I sps?). Add more if 1 param is not sufficient...
    ## Simulation Settings
    settings = setting_list[setting_index]                                                                              # Define the condition for mu, sigma, and B, for x&t dependences.
    f_mu_setting = settings[0]
    f_sigma_setting = settings[1]
    f_bound_setting = settings[2]
    ## Lists
    # pdf_list = np.zeros((len(x_list), len(t_list_temp)))
    # pdf_list[int((len(x_list)-1)/2), 0] = 1.                                                                            # Initial condition at x=0, center of the channel.
    t_list_temp = t_list
    pdf_list_curr = np.zeros((len(x_list)))
    pdf_list_curr[int((len(x_list)-1)/2)] = 1.                                                                            # Initial condition at x=0, center of the channel.
    pdf_list_prev = np.zeros((len(x_list)))
    Prob_list_corr = np.zeros(len(t_list_temp))                                                                         # Probability flux to correct choice
    Prob_list_err = np.zeros(len(t_list_temp))                                                                          # Probability flux to erred choice
    # traj_mean_pos = np.zeros(len(t_list_temp))                                                                          # Mean position of distribution
    ## Task specific
    param_task=0
    if len(params)>=9:
        param_task = params[8]
    task_temp = task_list[task_index]
    if task_temp == 'Duration_Paradigm':
        param_task = [params[8], mu_0_temp]
    ### Looping through time and updating the pdf.
    for i_t in range(len(t_list_temp)-1):
        pdf_list_prev = copy.copy(pdf_list_curr)                                                                        # Update Previous state. To be frank probably not necessary for max efficiency. Left it just in case.
        mu_1_temp = f_mu1_task(t_list_temp[i_t], task_temp, param_task)                                                 ##### Temporary, placeholder for our method to update mu_1 based on the task and the current time.
        mu_temp = mu_0_temp + mu_1_temp
        if sum(pdf_list_curr[:])>0.0001:                                                                                # For efficiency only do diffusion if there's at least some densities remaining in the channel.
            ### Explicit form for illustration. We actually use the implicit method instead.
            # pdf_list[:,i_t+1] = pdf_list[:,i_t] + dt*( -deriv_first(f_mu(mu, x_list, t_list_temp[i_t], f_mu_setting, param_mu) * pdf_list[:,i_t] , dx) + deriv_second(0.5*f_sigma(sigma, x_list, t_list_temp[i_t], f_sigma_setting, param_sigma)**2 * pdf_list[:,i_t] , dx))
            # pdf_list[:,i_t+1] = pdf_list[:,i_t] + dt*( -deriv_first(f_mu(mu, x_list, t_list_temp[i_t], f_mu_setting, param_mu), dx)*pdf_list[:,i_t] -deriv_first(pdf_list[:,i_t], dx)*f_mu(mu, x_list, t_list_temp[i_t], f_mu_setting, param_mu)  + deriv_second(0.5*f_sigma(sigma, x_list, t_list_temp[i_t], f_sigma_setting, param_sigma)**2 * pdf_list[:,i_t] , dx))
            # Define the boundaries at current time.
            bound_temp = f_bound_shift(B_0_temp, param_B_t_temp, t_list_temp[i_t], f_bound_setting)                     # Boundary for current time-step. Can generalize to assymetric bounds
            bound_shift = B_0_temp - bound_temp                                                                         # pre-define. ASsume bound_temp < B_0_temp.
            # Note that we linearly approximate the bound by the two surrounding grids above (outer) and below (inner) it.
            index_temp_inner = int(math.ceil(bound_shift/dx))                                                           # Index for the inner bound (smaller matrix)
            index_temp_outer = int(bound_shift/dx)                                                                      # Index for the outer bound (larger matrix)
            weight_inner    = (bound_shift - index_temp_outer*dx)/dx                                                    # The weight of the upper bound matrix, approximated linearly. 0 when bound exactly at grids.
            weight_outer    =  1. - weight_inner                                                                        # The weight of the lower bound matrix, approximated linearly.

            # Define and Modify diffusion matrices if the position of bounds relative to grids has changed.
            index_temp_outer_list = np.logical_and(x_list>=x_list[index_temp_outer], x_list<=x_list[len(x_list)-1-index_temp_outer])*1.
            index_temp_inner_list = np.logical_and(x_list>=x_list[index_temp_inner], x_list<=x_list[len(x_list)-1-index_temp_inner])*1.
            # matrix_diffusion_outer = np.diag(np.ones(len(x_list))) + f_mu_matrix(mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting, index_temp_outer_list) + f_sigma_matrix(sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting, index_temp_outer_list)       #Diffusion Matrix for Implicit Method. Here defined as Outer Matrix, and inder matrix is either trivial or a submatrix extracted.
            # matrix_diffusion_inner = np.diag(np.ones(len(x_list))) + f_mu_matrix(mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting, index_temp_inner_list) + f_sigma_matrix(sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting, index_temp_inner_list)       #Diffusion Matrix for Implicit Method. Here defined as Outer Matrix, and inder matrix is either trivial or a submatrix extracted.
            matrix_diffusion = np.diag(np.ones(len(x_list))) + weight_outer*f_mu_matrix(mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting, index_temp_outer_list) + weight_outer*f_sigma_matrix(sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting, index_temp_outer_list) \
                                                             + weight_inner*f_mu_matrix(mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting, index_temp_inner_list) + weight_inner*f_sigma_matrix(sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting, index_temp_inner_list)       #Diffusion Matrix for Implicit Method. Here defined as inner Matrix, and inder matrix is either trivial or a submatrix extracted.
            ### Compute Probability density functions (pdf)
            # print np.diag(matrix_diffusion_outer)
            # print len(pdf_list_prev)
            # pdf_list_temp_outer = np.linalg.solve(matrix_diffusion_outer  , pdf_list_prev)                                      # Probability density function for outer matrix. Is np.linalg.solve_banded useful?
            # if index_temp_inner == index_temp_outer:
            #     pdf_list_temp_inner = copy.copy(pdf_list_temp_outer)                                                                                                                    # When bound is exactly at a grid, use the most efficient method.
            # else:
            #     pdf_list_temp_inner = np.linalg.solve(matrix_diffusion_inner, pdf_list_prev)                                     # Probability density function for inner matrix. Is np.linalg.solve_banded useful?
            Prob_list_err[i_t+1]  += weight_outer*np.sum(pdf_list_prev[:index_temp_outer])               + weight_inner*np.sum(pdf_list_prev[:index_temp_inner])                        # Pdfs out of bound is consideered decisions made.
            Prob_list_corr[i_t+1] += weight_outer*np.sum(pdf_list_prev[len(x_list)-index_temp_outer:]) + weight_inner*np.sum(pdf_list_prev[len(x_list)-index_temp_inner:])          # Pdfs out of bound is consideered decisions made.
            # pdf_list_curr = weight_inner*pdf_list_temp_inner + weight_outer*pdf_list_temp_outer
            pdf_list_curr = np.linalg.solve(matrix_diffusion  , pdf_list_prev)                                      # Probability density function for outer matrix. Is np.linalg.solve_banded useful?
            if index_temp_outer>0:
                pdf_list_curr[:index_temp_outer] = np.zeros((index_temp_outer))                                                                                                                                     # Reconstruct current proability density function, adding outer and inner contribution to it.
                pdf_list_curr[len(x_list)-index_temp_outer:] = np.zeros((index_temp_outer))
                pdf_list_curr[index_temp_inner] *= weight_outer
                pdf_list_curr[len(x_list)-1-index_temp_inner] *= weight_outer
        else:
            break       #break if the remaining densities are too small....

        ### Increase current, transient probability of crossing either bounds, as flux
        # Prob_list_corr[i_t+1] += weight_outer*pdf_list_temp_outer[len(x_list)-1-index_temp_outer]* ( f_mu_flux(len(x_list)-1-index_temp_outer, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(len(x_list)-1-index_temp_outer, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting)) \
        #                       +  weight_inner*pdf_list_temp_inner[len(x_list)-1-index_temp_inner]* ( f_mu_flux(len(x_list)-1-index_temp_inner, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(len(x_list)-1-index_temp_inner, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting))
        # Prob_list_err[i_t+1]  += weight_outer*pdf_list_temp_outer[              index_temp_outer]* ( f_mu_flux(              index_temp_outer, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(              index_temp_outer, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting)) \
        #                       +  weight_inner*pdf_list_temp_inner[              index_temp_inner]* ( f_mu_flux(              index_temp_inner, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(              index_temp_inner, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting))
        Prob_list_corr[i_t+1] += weight_outer*pdf_list_curr[len(x_list)-1-index_temp_outer]* ( f_mu_flux(len(x_list)-1-index_temp_outer, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(len(x_list)-1-index_temp_outer, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting)) \
                              +  weight_inner*pdf_list_curr[len(x_list)-1-index_temp_inner]* ( f_mu_flux(len(x_list)-1-index_temp_inner, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(len(x_list)-1-index_temp_inner, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting))
        Prob_list_err[i_t+1]  += weight_outer*pdf_list_curr[              index_temp_outer]* ( f_mu_flux(              index_temp_outer, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(              index_temp_outer, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting)) \
                              +  weight_inner*pdf_list_curr[              index_temp_inner]* ( f_mu_flux(              index_temp_inner, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(              index_temp_inner, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting))
        if bound_temp < dx:                                                                                             # Renormalize when the channel size has <1 grid, although all hell breaks loose in this regime.
            Prob_list_corr[i_t+1] *= (1+ (1-bound_temp/dx))
            Prob_list_err[i_t+1]  *= (1+ (1-bound_temp/dx))
        # traj_mean_pos[i_t+1] = np.sum(Prob_list_corr)*1. + np.sum(Prob_list_err[:])*-1. + np.sum(pdf_list[:,i_t+1]*x_list)        # To record the mean position.


    return Prob_list_corr, Prob_list_err#, traj_mean_pos
    # return pdf_list, Prob_list_corr, Prob_list_err


### Analytical form of DDM. Can only solve for simple DDM or linearly collapsing bound... From Anderson1960
# Note that the solutions are automatically normalized.
def DDM_pdf_analytical(params, setting_index, task_index=0):
    '''
    Now assume f_mu, f_sigma, f_Bound are callable functions
    '''
    ### Initialization
    mu = params[0]
    sigma = params[1]
    B_temp = params[3]
    param_B = params[4]
    settings = setting_list[setting_index]
    ## Settings
    f_mu_setting = settings[0]
    f_sigma_setting = settings[1]
    f_bound_setting = settings[2]
    ## Task Specifics
    task_temp = task_list[task_index]
    #Quick fix to allow us to use params[2] as tau in the case for collapsing bound... Would not work if say we also need parameters in mu.
    if settings  == ['linear_xt', 'linear_xt', 'constant']:                                                               # Simple DDM
        DDM_anal_corr, DDM_anal_err = analytic_ddm(mu, sigma, B_temp, t_list)
    elif settings  == ['linear_xt', 'linear_xt', 'collapsing_linear']:                                                    # Linearly Collapsing Bound
        DDM_anal_corr, DDM_anal_err = analytic_ddm(mu, sigma, B_temp, t_list, param_B)
    ## Remove some abnormalities such as NaN due to trivial reasons.
    DDM_anal_corr[DDM_anal_corr==np.NaN]=0.
    DDM_anal_corr[0]=0.
    DDM_anal_err[DDM_anal_err==np.NaN]=0.
    DDM_anal_err[0]=0.
    return DDM_anal_corr*dt, DDM_anal_err*dt







### Matrix terms due to different parameters (mu, sigma, bound)
def f_mu_matrix(mu_temp, param_mu_x_temp, param_mu_t_temp, x, t, f_mu_setting, index_list):                                         # Diffusion Matrix containing drift=mu related terms
    ## Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) - (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the same x,t with p(x,t) (probability distribution function). Hence we use x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
    if f_mu_setting == 'linear_xt':                                                                                     # If dependence of mu on x & t is at most linear (or constant):
        return np.diag( index_list[1:]*(0.5*dt/dx *(mu_temp + param_mu_x_temp*x[1:] + param_mu_t_temp*t)),1) + np.diag( index_list[:-1]*(-0.5*dt/dx *(mu_temp + param_mu_x_temp*x[:-1] + param_mu_t_temp*t)),-1)
    if f_mu_setting == 'sinx_cost':                                                                                     # Weird dependence for testing. Remove at will.
        return np.diag( index_list[1:]*(0.5*dt/dx *(mu_temp + param_mu_x_temp*np.sin(x[1:]) + param_mu_t_temp*np.cos(t))),1) + np.diag(index_list[:-1]*(-0.5*dt/dx *(mu_temp + param_mu_x_temp*np.sin(x[:-1]) + param_mu_t_temp*np.cos(t))),-1)
    # Add f_mu_setting definitions as needed...
    else:
        print'Wrong/unspecified f_mu_setting for f_mu_matrix function'

def f_sigma_matrix(sigma_temp, param_sigma_x_temp, param_sigma_t_temp, x, t, f_sigma_setting, index_list):                          # Diffusion Matrix containing noise=sigma related terms
    # Refer to f_mu_matrix. Same idea.
    if f_sigma_setting == 'linear_xt':                                                                                  # If dependence of mu on x & t is at most linear (or constant):
        return np.diag(index_list*((sigma_temp+ param_sigma_x_temp*x + param_sigma_t_temp*t)**2*dt/dx**2))   -   np.diag(index_list[1:]*0.5*(sigma_temp+ param_sigma_x_temp*x[1:] + param_sigma_t_temp*t)**2*dt/dx**2,1)   -   np.diag(index_list[:-1]*0.5*(sigma_temp+ param_sigma_x_temp*x[:-1] + param_sigma_t_temp*t)**2*dt/dx**2,-1)
    if f_sigma_setting == 'sinx_cost':                                                                                  # Weird dependence for testing. Remove at will.
        return np.diag(index_list*((sigma_temp+ param_sigma_x_temp*np.sin(x) + param_sigma_t_temp*np.cos(t))**2*dt/dx**2))   -   np.diag(index_list[1:]*0.5*(sigma_temp+ param_sigma_x_temp*np.sin(x[1:]) + param_sigma_t_temp*np.cos(t))**2*dt/dx**2,1)   -   np.diag(index_list[:-1]*0.5*(sigma_temp+ param_sigma_x_temp*np.sin(x[:-1]) + param_sigma_t_temp*np.cos(t))**2*dt/dx**2,-1)
    # Add f_sigma_setting definitions as needed...
    else:
        print'Wrong/unspecified f_sigma_setting for f_sigma_matrix function'

## Second effect of Collapsing Bounds: Collapsing Center: Positive and Negative states are closer to each other over time.
def f_bound_shift(bound_temp, param_B, t, f_bound_setting):
    if f_bound_setting == 'constant':
        return bound_temp
    elif f_bound_setting == 'collapsing_linear':
        return max(bound_temp - param_B*t, 0.)
    elif f_bound_setting == 'collapsing_exponential':
        return bound_temp*np.exp(-param_B*t)
    # Add f_bound_setting definitions as needed...
    else:
        print'Wrong/unspecified f_bound_setting for f_bound_shift function'
    ###And so on...
    #Can we find a collapsing bound signal comparing to autonomous ones (from the neuron itself).




## Various tasks that causes change in signals and what not, in addition to the space and time varying f_mu, f_sigma, and f_bound.
def f_mu1_task(t, task_setting, param_task=T_dur/2.):                                                                   # Define the change in drift at each time due to active perturbations, in different tasks
    if task_setting == 'Fixed_Duration':                                                                                # No task
        return 0.
    elif task_setting == 'PsychoPhysical_Kernel':                                                                       #Note that I assumed in the DDM_general fcn for the PK case, that the input of mu_0 to be 0. Else have to counter-act the term...
        return param_task[int(t/dt_mu_PK)]                                                                                                        ## Fix/Implement later
    ## For Duration/Pulse paradigms, param_task[0]= magnitude of pulse. param_task[1]= T_Dur_Duration/t_Mid_Pulse respectively
    elif task_setting == 'Duration_Paradigm':                                                                           # Vary Stimulus Duration
        T_Dur_Duration = param_task[0]                                                                                  # Duration of pulse. Variable in Duration Paradigm
        ## if stimulus starts at 0s:
        if t< T_Dur_Duration:
            return 0
        else:
            return -param_task[1]        #Remove pulse if T> T_Dur_duration
        ## if stimulus starts at T_dur/2 =1s, use this instead and put it above:
        # t_Mid_Duration = T_dur/2.                                                                                       # (Central) Onset time of pulse/duration. Arbitrary but set to constant as middle of fixed duration task.
        # if abs(t-t_Mid_Duration)< (T_Dur_Duration/2):
    elif task_setting == 'Pulse_Paradigm':                                                                              # Add brief Pulse to stimulus, with varying onset time.
        T_Dur_Pulse = 0.1                                                                                               # Duration of pulse. 0.1s in the spiking circuit case.
        t_Pulse_onset = param_task                                                                                      # (Central) Onset time of pulse/duration. Variable in Pulse Paradigm
        if (t>t_Pulse_onset) and  (t<(t_Pulse_onset+T_Dur_Pulse)):
            return mu_0*0.15                                                                                            # 0.15 based on spiking circuit simulations.
        else:
            return 0.
        ## If we define time as at the middle of pulse, use instead:
        # t_Mid_Pulse = param_task                                                                                     # (Central) Onset time of pulse/duration. Variable in Pulse Paradigm
        # if abs(t-t_Mid_Pulse)< (T_Dur_Pulse/2):
    # Add task_setting definitions as needed...
    else:
        print'Wrong/unspecified f_mu1_task task_setting'











### Amount of flux from bound/end points to correct and erred probabilities, due to different parameters (mu, sigma, bound)
# f_mu_setting = 'constant'
def f_mu_flux(index_bound, mu_temp, param_mu_x_temp, param_mu_t_temp, x, t, f_mu_setting):                                         # Diffusion Matrix containing drift=mu related terms
    ## Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) - (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the same x,t with p(x,t) (probability distribution function). Hence we use x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
    if f_mu_setting == 'linear_xt':                                                                                     # If dependence of mu on x & t is at most linear (or constant):
        return 0.5*dt/dx * np.sign(x_list[index_bound]) * (mu_temp + param_mu_x_temp*x_list[index_bound] + param_mu_t_temp*t)
    if f_mu_setting == 'sinx_cost':                                                                                     # If dependence of mu on x & t is at most linear (or constant):
        return 0.5*dt/dx * np.sign(x_list[index_bound]) * (mu_temp + param_mu_x_temp*np.sin(x_list[index_bound]) + param_mu_t_temp*np.cos(t))
    # Add f_mu_setting definitions as needed...
    else:
        print'Wrong/unspecified f_mu_setting for f_mu_flux function'

def f_sigma_flux(index_bound, sigma_temp, param_sigma_x_temp, param_sigma_t_temp, x, t, f_sigma_setting):                                         # Diffusion Matrix containing noise=sigma related terms
    ## Similar to f_sigma_flux
    if f_sigma_setting == 'linear_xt':                                                                                     # If dependence of sigma on x & t is at most linear (or constant):
        return 0.5*dt/dx**2 * (sigma_temp + param_sigma_x_temp*x_list[index_bound] + param_sigma_t_temp*t)**2
    if f_sigma_setting == 'sinx_cost':                                                                                     # If dependence of sigma on x & t is at most linear (or constant):
        return 0.5*dt/dx**2 * (sigma_temp + param_sigma_x_temp*np.sin(x_list[index_bound]) + param_sigma_t_temp*np.cos(t))**2
    # Add f_sigma_setting definitions as needed...
    else:
        print'Wrong/unspecified f_sigma_setting for f_sigma_flux function'







### Fit Functions
def MSE_model_fit(params, y_2_fit_setting_index, y_fit2):
#    y_2fit = DDM_pdf_general(mu, f_mu_setting, sigma, f_sigma_setting, B, f_bound_setting, param_mu, param_sigma)
    (Prob_list_corr_temp, Prob_list_err_temp)     = DDM_pdf_general(params, y_2_fit_setting_index, 0)
# Note that we normalize both fitting and fitted probabilities, due to a easy but unresolved normalization issue
    y_fit2 /= np.sum(y_fit2)
    Prob_list_corr_temp /= np.sum(Prob_list_corr_temp)
    to_min = -np.log(np.sum((y_fit2*Prob_list_corr_temp)**0.5 /dt**1))                                                  # Bhattacharyya distance
    return to_min
#    to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2)                                                           # MLE
#    to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0)     #KL divergence

def MLE_model_fit_over_coh(params, y_2_fit_setting_index, y_fit2):                                                # To fit to spiking circuit data.
    Prob_corr_err_undec_temp = np.zeros(3,len(coherence_list))
    for i_coh in range(coherence_list):
        (Prob_list_corr_coh_MLE_temp, Prob_list_err_coh_MLE_temp)     = DDM_pdf_general(params, y_2_fit_setting_index, 0)
        Prob_list_cumsum_corr_coh_MLE_temp  = np.cumsum(Prob_list_corr_coh_MLE_temp)
        Prob_list_cumsum_err_coh_MLE_temp   = np.cumsum(Prob_list_err_coh_MLE_temp)
        Prob_list_cumsum_undec_coh_MLE_temp = 1. - Prob_list_cumsum_corr_coh_MLE_temp - Prob_list_cumsum_err_coh_MLE_temp
        Prob_corr_err_undec_temp[:,i_coh]   = [Prob_list_cumsum_corr_coh_MLE_temp, Prob_list_cumsum_err_coh_MLE_temp, Prob_list_cumsum_undec_coh_MLE_temp]  # Total probability for correct, erred and undecided choices.
    to_min = -np.log(np.sum((y_fit2*Prob_corr_err_undec_temp)**0.5 /dt**1))     #Bhattacharyya distance
    return to_min
# Other minimizers
    # to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2)                                                          # MLE
    # to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0)                 # KL divergence

def MSE_model_fit_RT(params, y_2_fit_setting_index, y_fit2):                                                            # To be worked on and verify
    (Prob_list_corr_temp, Prob_list_err_temp)     = DDM_pdf_general(params, y_2_fit_setting_index, 0)
    to_min = -np.log(np.sum((y_fit2*np.column_stack(Prob_list_corr_temp, Prob_list_err_temp))**0.5))                                                         # Bhattacharyya distance
    return to_min
#    to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2)                                                           # MLE
#    to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0)     #KL divergence




def Psychometric_fit_P(params_pm, pm_fit2):
    prob_corr_fit = 0.5 + 0.5*np.sign(mu_0_list_Pulse+params_pm[2])*(1. - np.exp(-(np.abs(mu_0_list_Pulse+params_pm[2])/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
    to_min = np.sum((prob_corr_fit-pm_fit2)**2)                                                                         # Least Square
    return to_min
# Other possible forms of psychometric function
    # prob_corr_fit = 1./(1.+ np.exp(-params_pm[1]*(mu_0_list_Pulse+params_pm[0])))
    # prob_corr_fit = 0.5 + 0.5*np.sign(mu_0_list_Pulse)*(1. - np.exp(-np.sign(mu_0_list_Pulse)*((mu_0_list_Pulse+params_pm[2])/params_pm[0])**params_pm[1]))                                    #Use duration paradigm and add shift parameter. Fit for both positive and negative
# Other possible minimizers
    # to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2)                                                          # Maximum Likelihood Estimator
    # to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0)                 # KL divergence
    # to_min = -np.log(np.sum((pm_fit2*Prob_list_corr_temp)**0.5 /dt**1))                                               # Bhattacharyya distance


def Psychometric_fit_D(params_pm, pm_fit2):
    prob_corr_fit = 0.5 + 0.5*(1. - np.exp(-(mu_0_list/params_pm[0])**params_pm[1]))
        # 1./(1.+ np.exp(-params_pm[1]*(mu_0_list+params_pm[0])))
    to_min = np.sum((prob_corr_fit-pm_fit2)**2)                                                                         # Least Square
    return to_min
# Other possible minimizers
    # to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2)                                                          # Maximum Likelihood Estimator
    # to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0)                 # KL divergence
    # to_min = -np.log(np.sum((pm_fit2*Prob_list_corr_temp)**0.5 /dt**1))                                               # Bhattacharyya distance


def Threshold_D_fit(param_Thres, pm_fit2, n_skip):
    prob_corr_fit = param_Thres[0] + param_Thres[3]*(np.exp(-((t_dur_list_duration[n_skip:]-param_Thres[2])/param_Thres[1])))
    to_min = np.sum((prob_corr_fit-pm_fit2[n_skip:])**2)                                                                # Least Square
    return to_min
# Other possible forms of prob_corr_fit
    # prob_corr_fit = param_Thres[0] + (100.-param_Thres[0])*(np.exp(-((t_dur_list_duration-param_Thres[2])/param_Thres[1])))
    # prob_corr_fit = param_Thres[0] + (100.-param_Thres[0])*(np.exp(-((t_dur_list_duration-param_Thres[2])/param_Thres[1])))
    # prob_corr_fit = param_Thres[0] + param_Thres[2]*(np.exp(-((t_dur_list_duration)/param_Thres[1])))
        # 1./(1.+ np.exp(-params_pm[1]*(mu_0_list+params_pm[0])))
# Other Minimizers
    # Other posssible forms of minimizer
    # to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2)                                                          # MAximum Likelihood
    # to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0)                 # KL divergence
    # pm_fit2 /= np.sum(pm_fit2)
    # Prob_list_corr_temp /= np.sum(Prob_list_corr_temp)
    # to_min = -np.log(np.sum((pm_fit2*Prob_list_corr_temp)**0.5 /dt**1))                                               # Bhattacharyya distance










########################################################################################################################
########################################################################################################################
########################################################################################################################
### Functions not used anymore. Discardable but kept for record...
# Note the derivative terms are optimized for dx for now...
def deriv_first(fcn_curr, dx):       #First derivative. for x more so than for t...
    fcn_next = np.roll(fcn_curr, -1)
    fcn_prev = np.roll(fcn_curr, 1)
    deriv_first_temp     = (fcn_next    -fcn_prev   )/2./dx                                                             # Central method for most terms
    deriv_first_temp[-1] = (fcn_curr[-1]-fcn_prev[-1])  /dx                                                             # Backward method for boundary term
    deriv_first_temp[0]  = (fcn_next[0] -fcn_curr[0])   /dx                                                             # Forward method for boundary term
    return deriv_first_temp

def deriv_second(fcn_curr, dx):      # Secnd derivative
    fcn_next = np.roll(fcn_curr, -1)
    fcn_prev = np.roll(fcn_curr, 1)
    deriv_second_temp    = (fcn_next     - 2.*fcn_curr     + fcn_prev)    /dx**2                                        # Central method for most of the terms
    deriv_second_temp[-1]= (fcn_curr[-1] - 2.*fcn_curr[-2] + fcn_curr[-3])/dx**2                                        # Backward method for boundary term
    deriv_second_temp[0]=  (fcn_curr[0]  - 2.*fcn_curr[1]  + fcn_curr[2]) /dx**2                                        # Forward method for boundary term
    return deriv_second_temp


def f_bound_matrix(bound_temp, param_B_temp, x, t, f_bound_setting):                                                    # Diffusion Matrix containing boundary related terms. Used for collapsing center...
    if f_bound_setting == 'constant':
        return 0.
    elif f_bound_setting == 'collapsing_linear':
        return np.diag(0.5*dt/dx*param_B_temp*sign_fix(x[1:]),1) + np.diag(-0.5*dt/dx*param_B_temp*sign_fix(x[:-1]),-1)
    elif f_bound_setting == 'collapsing_exponential':
        return np.diag(0.5*dt/dx*(param_B_temp*np.exp(-param_B_temp*t))*sign_fix(x[1:]),1) + np.diag(-0.5*dt/dx*(param_B_temp*np.exp(-param_B_temp*t))*sign_fix(x[:-1]),-1)
        # More Exact Expression (if needed): matrix_diffusion += np.diag(0.5*dt/dx*(np.exp(-param_B_temp*t)*(1-np.exp(-param_B_temp*dt)))*sign_fix(x[1:]),1) + np.diag(-0.5*dt/dx*(np.exp(-param_B_temp*t)*(1-np.exp(-param_B_temp*dt)))*sign_fix(x[:-1]),-1)
    # Add f_bound_setting definitions as needed...
    else:
        print'Wrong/unspecified f_bound_setting for f_bound_matrix function'
    ###And so on...

def f_bound_flux(index_bound, param_B_temp, x, t, f_bound_setting):                                                     # Flux due to boundary related terms. Used for collapsing center...
    if f_bound_setting == 'constant':
        return 0.
    elif f_bound_setting == 'collapsing_linear':
        return 0.5*dt/dx * param_B_temp
    elif f_bound_setting == 'collapsing_exponential':
        return 0.5*dt/dx * (param_B_temp*np.exp(-param_B_temp*t))                                                               # Could have used x instead of actual definition of bound...
        # More Exact Expression (if needed): matrix_diffusion += np.diag(0.5*dt/dx*(np.exp(-param_B_temp*t)*(1-np.exp(-param_B_temp*dt)))*sign_fix(x[1:]),1) + np.diag(-0.5*dt/dx*(np.exp(-param_B_temp*t)*(1-np.exp(-param_B_temp*dt)))*sign_fix(x[:-1]),-1)
    # Add f_bound_setting definitions as needed...
    else:
        print'Wrong/unspecified f_bound_setting for f_mu_flux function'
    ###And so on...

def sign_fix(x_list_temp):
    y_list_temp = np.zeros(len(x_list_temp))
    for i_temp in range(len(x_list_temp)):
        if np.abs(x_list_temp[i_temp])> 10.**10:
            y_list_temp[i_temp] = np.sign(x_list_temp[i_temp])
        else:
            y_list_temp[i_temp] = 0.0
    return y_list_temp



def fcn_par(i_2fit):
    if i_2fit != i_fit2:
        #name_2fit = name_joint.join(('Prob_list_corr', str(i_2fit+1)))
    #    res = minimize(MSE_model_fit, [mu_0, 0.8*sigma_0, B, param_mu_0, param_sigma_0], args = (i_2fit, model_fit2))
        res = minimize(MSE_model_fit, [mu_0, (1.*0.8+0.*1.1)*sigma_0, param_0_list[i_2fit]], tol=10.**-6, args = (i_2fit, model_fit2))
        # print (plotlabel_list[i_fit2], plotlabel_list[i_2fit], res.x)
        (Prob_list_corr_temp_par, Prob_list_err_temp_par)     = DDM_pdf_general(res.x, i_2fit, 0)
        return(Prob_list_corr_temp_par, res.x)

def fcn_par_expt(i_2fit):
    res = minimize(MSE_model_fit, [mu_0, (1.*0.8+0.*1.1)*sigma_0, param_0_list[i_2fit]], tol=10.**-6, args = (i_2fit, model_fit2))
    # print (plotlabel_list[i_fit2], plotlabel_list[i_2fit], res.x)
    (Prob_list_corr_temp_par, Prob_list_err_temp_par)     = DDM_pdf_general(res.x, i_2fit, 0)
    return(Prob_list_corr_temp_par, res.x)





def MSE_model_fit_expt(params, y_2_fit_setting_index, y_fit2, t_fit2):
    (Prob_list_corr_temp, Prob_list_err_temp)     = DDM_pdf_general(params, y_2_fit_setting_index, 0)
    t_temp_2fit_index = np.round(t_fit2/dt).astype(int)
    to_min = -np.log(np.sum((y_fit2/np.sum(y_fit2)*Prob_list_corr_temp[t_temp_2fit_index]/np.sum(Prob_list_corr_temp[t_temp_2fit_index]))**0.5 /dt**0))     #KL divergence
    return to_min
    # Other minimizers
#    to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2)                                                           # MLE
#    to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
#    print to_min




