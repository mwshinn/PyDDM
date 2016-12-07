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
#def DDM_pdf_general(mu, f_mu_setting, sigma, f_sigma_setting, B, f_bound_setting, param_mu, param_sigma):
def DDM_pdf_general(params, setting_index, task_index=0):
    '''
    Now assume f_mu, f_sigma, f_Bound are callable functions
    '''
    ### Initialization
    mu_0_temp = params[0] # Constant component of drift rate mu.
    param_mu_x_temp = params[1] # Parameter for x_dependence of mu. Add more if 1 param is not sufficient...
    param_mu_t_temp = params[2] # Parameter for t_dependence of mu. Add more if 1 param is not sufficient...
    sigma_0_temp = params[3] # Constant component of noise sigma.
    param_sigma_x_temp = params[4] # Parameter for t_dependence of sigma. Add more if 1 param is not sufficient...
    param_sigma_t_temp = params[5] # Parameter for t_dependence of sigma. Add more if 1 param is not sufficient...
    B_0_temp = params[6] # Constant component of drift rate mu.
    param_B_t_temp = params[7] # Parameter for t_dependence of B (no x-dep I sps?). Add more if 1 param is not sufficient...
    ## Simulation Settings
    settings = setting_list[setting_index] # Define the condition for mu, sigma, and B, for x&t dependences.
    f_mu_setting = settings[0] # Declare the type of DDM model regarding mu. Need to modify f_mu_matrix and f_mu_flux.
    f_sigma_setting = settings[1] # Declare the type of DDM model regarding sigma. Need to modify f_sigma_matrix and f_sigma_flux.
    f_bound_setting = settings[2] # Declare the type of DDM model regarding boundaries. Specify on f_bound_t.
    f_IC_setting = settings[3] # Declare the type of Initial Condition. Specify on f_initial_condition
    ### Initialization: Lists
    t_list_temp = t_list
    pdf_list_curr = f_initial_condition(f_IC_setting, x_list) # Initial condition
    pdf_list_prev = np.zeros((len(x_list)))
    Prob_list_corr = np.zeros(len(t_list_temp)) # Probability flux to correct choice
    Prob_list_err = np.zeros(len(t_list_temp)) # Probability flux to erred choice
    # traj_mean_pos = np.zeros(len(t_list_temp)) # Mean position of distribution
    ### Task specific
    param_task=0 # 0 if no extra active tasks/regimes/paradigms, else modify f_mu1_task.
    if len(params)>=9: # Probably want to optimize...
        param_task = params[8]
    task_temp = task_list[task_index]
    if task_temp == 'Duration_Paradigm':
        param_task = [params[8], mu_0_temp]
    ##### Looping through time and updating the pdf.
    for i_t in range(len(t_list_temp)-1):
        pdf_list_prev = copy.copy(pdf_list_curr) # Update Previous state. To be frank pdf_list_prev could be removed for max efficiency. Leave it just in case.
        mu_1_temp = f_mu1_task(t_list_temp[i_t], task_temp, param_task) # Perturbation to mu due to various tasks specified. Trivial if param_task=0
        mu_temp = mu_0_temp + mu_1_temp # Actual mu= drift to be used.
        if sum(pdf_list_curr[:])>0.0001: # For efficiency only do diffusion if there's at least some densities remaining in the channel.
            ## Define the boundaries at current time.
            bound_temp = f_bound_t(B_0_temp, param_B_t_temp, t_list_temp[i_t], f_bound_setting) # Boundary at current time-step. Can generalize to assymetric bounds
            bound_shift = B_0_temp - bound_temp # pre-define. Assumed bound_temp < B_0_temp.
            # Note that we linearly approximate the bound by the two surrounding grids sandwiching it.
            index_temp_inner = int(np.ceil(bound_shift/dx)) # Index for the inner bound (smaller matrix)
            index_temp_outer = int(bound_shift/dx) # Index for the outer bound (larger matrix)
            weight_inner = (bound_shift - index_temp_outer*dx)/dx # The weight of the upper bound matrix, approximated linearly. 0 when bound exactly at grids.
            weight_outer = 1. - weight_inner # The weight of the lower bound matrix, approximated linearly.

            ## Define the diffusion matrix for implicit method
            x_list_temp = x_list[index_temp_outer:len(x_list)-index_temp_outer] # List of x-positions still within bounds.
            matrix_diffusion = np.diag(np.ones(len(x_list_temp))) + f_mu_matrix(mu_temp, param_mu_x_temp, param_mu_t_temp, x_list_temp, t_list_temp[i_t], f_mu_setting) + f_sigma_matrix(sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list_temp, t_list_temp[i_t], f_sigma_setting) #Diffusion Matrix for Implicit Method. Here defined as Outer Matrix, and inder matrix is either trivial or an extracted submatrix.
            ### Compute Probability density functions (pdf)
            pdf_list_temp_outer = np.linalg.solve(matrix_diffusion , pdf_list_prev[index_temp_outer:len(x_list)-index_temp_outer]) # Probability density function for outer matrix. Is np.linalg.solve_banded useful?
            if index_temp_inner == index_temp_outer:
                pdf_list_temp_inner = copy.copy(pdf_list_temp_outer) # When bound is exactly at a grid, use the most efficient method.
            else:
                pdf_list_temp_inner = np.linalg.solve(matrix_diffusion[1:-1, 1:-1], pdf_list_prev[index_temp_inner:len(x_list)-index_temp_inner]) # Probability density function for inner matrix. Is np.linalg.solve_banded useful?
            Prob_list_err[i_t+1] += weight_outer*np.sum(pdf_list_prev[:index_temp_outer])               + weight_inner*np.sum(pdf_list_prev[:index_temp_inner]) # Pdfs out of bound is consideered decisions made.
            Prob_list_corr[i_t+1] += weight_outer*np.sum(pdf_list_prev[len(x_list)-index_temp_outer:]) + weight_inner*np.sum(pdf_list_prev[len(x_list)-index_temp_inner:]) # Pdfs out of bound is consideered decisions made.
            pdf_list_curr = np.zeros((len(x_list))) # Reconstruct current proability density function, adding outer and inner contribution to it.
            pdf_list_curr[index_temp_outer:len(x_list)-index_temp_outer] += weight_outer*pdf_list_temp_outer
            pdf_list_curr[index_temp_inner:len(x_list)-index_temp_inner] += weight_inner*pdf_list_temp_inner

        else:
            break #break if the remaining densities are too small....

        ### Increase current, transient probability of crossing either bounds, as flux
        Prob_list_corr[i_t+1] += weight_outer*pdf_list_temp_outer[-1]* ( f_mu_flux(len(x_list)-1-index_temp_outer, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(len(x_list)-1-index_temp_outer, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting)) \
                              +  weight_inner*pdf_list_temp_inner[-1]* ( f_mu_flux(len(x_list)-1-index_temp_inner, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(len(x_list)-1-index_temp_inner, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting))
        Prob_list_err[i_t+1]  += weight_outer*pdf_list_temp_outer[ 0]* ( f_mu_flux(              index_temp_outer, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(              index_temp_outer, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting)) \
                              +  weight_inner*pdf_list_temp_inner[ 0]* ( f_mu_flux(              index_temp_inner, mu_temp, param_mu_x_temp, param_mu_t_temp, x_list, t_list_temp[i_t], f_mu_setting) + f_sigma_flux(              index_temp_inner, sigma_0_temp, param_sigma_x_temp, param_sigma_t_temp, x_list, t_list_temp[i_t], f_sigma_setting))
        if bound_temp < dx: # Renormalize when the channel size has <1 grid, although all hell breaks loose in this regime.
            Prob_list_corr[i_t+1] *= (1+ (1-bound_temp/dx))
            Prob_list_err[i_t+1] *= (1+ (1-bound_temp/dx))
 # traj_mean_pos[i_t+1] = np.sum(Prob_list_corr)*1. + np.sum(Prob_list_err[:])*-1. + np.sum(pdf_list[:,i_t+1]*x_list) # To record the mean position.

    return Prob_list_corr, Prob_list_err # Only return pdf for correct and erred choices. Add more ouputs if needed
    # return Prob_list_corr, Prob_list_err, traj_mean_pos
    # return pdf_list, Prob_list_corr, Prob_list_err









### Matrix terms due to different parameters (mu, sigma, bound)
def f_mu_matrix(mu_temp, param_mu_x_temp, param_mu_t_temp, x, t, f_mu_setting): # Diffusion Matrix containing drift=mu related terms
        ## Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) - (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the same x,t with p(x,t) (probability distribution function). Hence we use x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
    if f_mu_setting == 'linear_xt': # If dependence of mu on x & t is at most linear (or constant):
        return np.diag( 0.5*dt/dx *(mu_temp + param_mu_x_temp*x[1:] + param_mu_t_temp*t),1) + np.diag( -0.5*dt/dx *(mu_temp + param_mu_x_temp*x[:-1] + param_mu_t_temp*t),-1)
    elif f_mu_setting == 'sinx_cost': # Weird dependence for testing. Remove at will.
        return np.diag( 0.5*dt/dx *(mu_temp + param_mu_x_temp*np.sin(x[1:]) + param_mu_t_temp*np.cos(t)),1) + np.diag( -0.5*dt/dx *(mu_temp + param_mu_x_temp*np.sin(x[:-1]) + param_mu_t_temp*np.cos(t)),-1)
    # Add f_mu_setting definitions as needed...
    else:
        print'Wrong/unspecified f_mu_setting for f_mu_matrix function'

def f_sigma_matrix(sigma_temp, param_sigma_x_temp, param_sigma_t_temp, x, t, f_sigma_setting): # Diffusion Matrix containing noise=sigma related terms
        # Refer to f_mu_matrix. Same idea.
    if f_sigma_setting == 'linear_xt': # If dependence of mu on x & t is at most linear (or constant):
        return np.diag(((sigma_temp+ param_sigma_x_temp*x + param_sigma_t_temp*t)**2*dt/dx**2))   -   np.diag(0.5*(sigma_temp+ param_sigma_x_temp*x[1:] + param_sigma_t_temp*t)**2*dt/dx**2,1)   -   np.diag(0.5*(sigma_temp+ param_sigma_x_temp*x[:-1] + param_sigma_t_temp*t)**2*dt/dx**2,-1)
    elif f_sigma_setting == 'sinx_cost': # Weird dependence for testing. Remove at will.
        return np.diag(((sigma_temp+ param_sigma_x_temp*np.sin(x) + param_sigma_t_temp*np.cos(t))**2*dt/dx**2))   -   np.diag(0.5*(sigma_temp+ param_sigma_x_temp*np.sin(x[1:]) + param_sigma_t_temp*np.cos(t))**2*dt/dx**2,1)   -   np.diag(0.5*(sigma_temp+ param_sigma_x_temp*np.sin(x[:-1]) + param_sigma_t_temp*np.cos(t))**2*dt/dx**2,-1)
    # Add f_sigma_setting definitions as needed...
    else:
        print'Wrong/unspecified f_sigma_setting for f_sigma_matrix function'

## Second effect of Collapsing Bounds: Collapsing Center: Positive and Negative states are closer to each other over time.
def f_bound_t(bound_temp, param_B, t, f_bound_setting):
    if f_bound_setting == 'constant':
        return bound_temp
    elif f_bound_setting == 'collapsing_linear':
        return max(bound_temp - param_B*t, 0.)
    elif f_bound_setting == 'collapsing_exponential':
        return bound_temp*np.exp(-param_B*t)
    # Add f_bound_setting definitions as needed...
    else:
        print'Wrong/unspecified f_bound_setting for f_bound_t function'
    ###And so on...














### Amount of flux from bound/end points to correct and erred probabilities, due to different parameters (mu, sigma, bound)
# f_mu_setting = 'constant'
def f_mu_flux(index_bound, mu_temp, param_mu_x_temp, param_mu_t_temp, x, t, f_mu_setting): # Diffusion Matrix containing drift=mu related terms
        ## Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) - (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the same x,t with p(x,t) (probability distribution function). Hence we use x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
    if f_mu_setting == 'linear_xt': # If dependence of mu on x & t is at most linear (or constant):
        return 0.5*dt/dx * np.sign(x_list[index_bound]) * (mu_temp + param_mu_x_temp*x_list[index_bound] + param_mu_t_temp*t)
    elif f_mu_setting == 'sinx_cost': # If dependence of mu on x & t is at most linear (or constant):
        return 0.5*dt/dx * np.sign(x_list[index_bound]) * (mu_temp + param_mu_x_temp*np.sin(x_list[index_bound]) + param_mu_t_temp*np.cos(t))
    # Add f_mu_setting definitions as needed...
    else:
        print'Wrong/unspecified f_mu_setting for f_mu_flux function'

def f_sigma_flux(index_bound, sigma_temp, param_sigma_x_temp, param_sigma_t_temp, x, t, f_sigma_setting): # Diffusion Matrix containing noise=sigma related terms
    ## Similar to f_sigma_flux
    if f_sigma_setting == 'linear_xt': # If dependence of sigma on x & t is at most linear (or constant):
        return 0.5*dt/dx**2 * (sigma_temp + param_sigma_x_temp*x_list[index_bound] + param_sigma_t_temp*t)**2
    elif f_sigma_setting == 'sinx_cost': # If dependence of sigma on x & t is at most linear (or constant):
        return 0.5*dt/dx**2 * (sigma_temp + param_sigma_x_temp*np.sin(x_list[index_bound]) + param_sigma_t_temp*np.cos(t))**2
    # Add f_sigma_setting definitions as needed...
    else:
        print'Wrong/unspecified f_sigma_setting for f_sigma_flux function'





def f_initial_condition(f_IC_setting, x): # Returns the pdf distribution at time 0 of the simulation
        ## Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) - (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the same x,t with p(x,t) (probability distribution function). Hence we use x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
    pdf_list_IC = np.zeros((len(x)))
    if f_IC_setting == 'point_source_center':
        pdf_list_IC[int((len(x)-1)/2)] = 1. # Initial condition at x=0, center of the channel.
    elif f_IC_setting == 'uniform': # Weird dependence for testing. Remove at will.
        pdf_list_IC = 1/(len(x))*np.ones((len(x)))
    # Add f_mu_setting definitions as needed...
    else:
        print'Wrong/unspecified f_mu_setting for f_mu_matrix function'
    return pdf_list_IC








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
def DDM_pdf_analytical(params, setting_index, task_index=0):
    '''
    Now assume f_mu, f_sigma, f_Bound are callable functions
    See DDM_pdf_general for nomenclature
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
    if settings  == ['linear_xt', 'linear_xt', 'constant', 'point_source_center']:                                      # Simple DDM
        DDM_anal_corr, DDM_anal_err = analytic_ddm(mu, sigma, B_temp, t_list)
    elif settings  == ['linear_xt', 'linear_xt', 'collapsing_linear', 'point_source_center']:                           # Linearly Collapsing Bound
        DDM_anal_corr, DDM_anal_err = analytic_ddm(mu, sigma, B_temp, t_list, param_B)
    ## Remove some abnormalities such as NaN due to trivial reasons.
    DDM_anal_corr[DDM_anal_corr==np.NaN]=0.
    DDM_anal_corr[0]=0.
    DDM_anal_err[DDM_anal_err==np.NaN]=0.
    DDM_anal_err[0]=0.
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
def f_mu1_task(t, task_setting, param_task=T_dur/2.): # Define the change in drift at each time due to active perturbations, in different tasks
    if task_setting == 'Fixed_Duration':                                                                                # No task
        return 0.
    elif task_setting == 'PsychoPhysical_Kernel': #Note that I assumed in the DDM_general fcn for the PK case, that the input of mu_0 to be 0. Else have to counter-act the term...
        return param_task[int(t/dt_mu_PK)] ## Fix/Implement later
    ## For Duration/Pulse paradigms, param_task[0]= magnitude of pulse. param_task[1]= T_Dur_Duration/t_Mid_Pulse respectively
    elif task_setting == 'Duration_Paradigm': # Vary Stimulus Duration
        T_Dur_Duration = param_task[0] # Duration of pulse. Variable in Duration Paradigm
        ## if stimulus starts at 0s:
        if t< T_Dur_Duration:
            return 0
        else:
            return -param_task[1] #Remove pulse if T> T_Dur_duration
        ## if stimulus starts at T_dur/2 =1s, use this instead and put it above:
        # t_Mid_Duration = T_dur/2. # (Central) Onset time of pulse/duration. Arbitrary but set to constant as middle of fixed duration task.
        # if abs(t-t_Mid_Duration)< (T_Dur_Duration/2):
    elif task_setting == 'Pulse_Paradigm': # Add brief Pulse to stimulus, with varying onset time.
        T_Dur_Pulse = 0.1 # Duration of pulse. 0.1s in the spiking circuit case.
        t_Pulse_onset = param_task # (Central) Onset time of pulse/duration. Variable in Pulse Paradigm
        if (t>t_Pulse_onset) and  (t<(t_Pulse_onset+T_Dur_Pulse)):
            return mu_0*0.15 # 0.15 based on spiking circuit simulations.
        else:
            return 0.
        ## If we define time as at the middle of pulse, use instead:
        # t_Mid_Pulse = param_task # (Central) Onset time of pulse/duration. Variable in Pulse Paradigm
        # if abs(t-t_Mid_Pulse)< (T_Dur_Pulse/2):
    # Add task_setting definitions as needed...
    else:
        print'Wrong/unspecified f_mu1_task task_setting'




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
