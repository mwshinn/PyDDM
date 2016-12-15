'''
Simulation code for Drift Diffusion Model
Author: Norman Lam (norman.lam@yale.edu)
'''

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from scipy.optimize import minimize
import copy

from DDM_parameters import *
from DDM_model import *

########################################################################################################################
### Defined functions.

# Function that simulates one trial of the time-varying drift/bound DDM
def DDM_pdf_general(model):
    '''
    Now assume f_mu, f_sigma, f_Bound are callable functions
    '''
    ### Initialization
    mu_base = model.parameters['mu'] # Constant component of drift rate mu.
    sigma = model.parameters['sigma'] # Constant (and currently only) component of noise sigma.
    bound_base = model.parameters['B'] # Constant component of the bound.

    # Control the type of model we use for mu/sigma dependence on x/t.
    # Here, `x` and `t` are coefficients to x and t, the particular
    # use of which depends on the model (specified in `name`).
    mudep = model.mudep
    sigmadep = model.sigmadep
    bounddep = model.bounddep
    task = model.task
    IC = model.IC
    
    ### Initialization: Lists
    pdf_curr = IC.get_IC(x_list) # Initial condition
    pdf_prev = np.zeros((len(x_list)))
    # If pdf_corr + pdf_err + undecided probability are summed, they
    # equal 1.  So these are componets of the joint pdf.
    pdf_corr = np.zeros(len(t_list)) # Probability flux to correct choice.  Not a proper pdf (doesn't sum to 1)
    pdf_err = np.zeros(len(t_list)) # Probability flux to erred choice. Not a proper pdf (doesn't sum to 1)


    ##### Looping through time and updating the pdf.
    for i_t, t in enumerate(t_list[:-1]): # NOTE I translated this directly from "for i_t in range(len(t_list_temp)-1):" but I think the -1 was a bug -MS
        # Update Previous state. To be frank pdf_prev could be
        # removed for max efficiency. Leave it just in case.
        pdf_prev = copy.copy(pdf_curr)

        # If we are in a task, adjust mu according to the task
        mu = task.adjust_mu(mu_base, t)

        if sum(pdf_curr[:])>0.0001: # For efficiency only do diffusion if there's at least some densities remaining in the channel.
            ## Define the boundaries at current time.
            bound = bounddep.get_bound(bound_base, t) # Boundary at current time-step. Can generalize to assymetric bounds

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
                               mudep.get_matrix(mu, x_list_inbounds, t) + \
                               sigmadep.get_matrix(sigma, x_list_inbounds, t)
            
            ### Compute Probability density functions (pdf)
            # PDF for outer matrix
            pdf_outer = np.linalg.solve(diffusion_matrix, pdf_prev[x_index_outer:len(x_list)-x_index_outer])
            # PDF for inner matrix (with optimization)
            if x_index_inner == x_index_outer: # Optimization: When bound is exactly at a grid, use the most efficient method.
                pdf_inner = copy.copy(pdf_outer)
            else:
                pdf_inner = np.linalg.solve(diffusion_matrix[1:-1, 1:-1], pdf_prev[x_index_inner:len(x_list)-x_index_inner])

            # Pdfs out of bound is consideered decisions made.
            pdf_err[i_t+1] += weight_outer * np.sum(pdf_prev[:x_index_outer]) \
                              + weight_inner * np.sum(pdf_prev[:x_index_inner])
            pdf_corr[i_t+1] += weight_outer * np.sum(pdf_prev[len(x_list)-x_index_outer:]) \
                               + weight_inner * np.sum(pdf_prev[len(x_list)-x_index_inner:])
            pdf_curr = np.zeros((len(x_list))) # Reconstruct current proability density function, adding outer and inner contribution to it.
            pdf_curr[x_index_outer:len(x_list)-x_index_outer] += weight_outer*pdf_outer
            pdf_curr[x_index_inner:len(x_list)-x_index_inner] += weight_inner*pdf_inner

        else:
            break #break if the remaining densities are too small....

        ### Increase current, transient probability of crossing either
        ### bounds, as flux.  Corr is a correct answer, err is an
        ### incorrect answer
        _inner_B_corr = x_list[len(x_list)-1-x_index_inner]
        _outer_B_corr = x_list[len(x_list)-1-x_index_outer]
        _inner_B_err = x_list[x_index_inner]
        _outer_B_err = x_list[x_index_outer]
        pdf_corr[i_t+1] += weight_outer * pdf_outer[-1] * ( mudep.get_flux(_outer_B_corr, mu, t) +
                                                            sigmadep.get_flux(_outer_B_corr, sigma, t)) \
                        +  weight_inner * pdf_inner[-1] * ( mudep.get_flux(_inner_B_corr, mu, t) +
                                                            sigmadep.get_flux(_inner_B_corr, sigma, t))
        pdf_err[i_t+1]  += weight_outer * pdf_outer[0] * ( mudep.get_flux(_outer_B_err, mu, t) +
                                                           sigmadep.get_flux(_outer_B_err, sigma, t)) \
                        +  weight_inner * pdf_inner[0] * ( mudep.get_flux(_inner_B_err, mu, t) +
                                                           sigmadep.get_flux(_inner_B_err, sigma, t ))

        if bound < dx: # Renormalize when the channel size has <1 grid, although all hell breaks loose in this regime.
            pdf_corr[i_t+1] *= (1+ (1-bound/dx))
            pdf_err[i_t+1] *= (1+ (1-bound/dx))

    return pdf_corr, pdf_err # Only return jpdf components for correct and erred choices. Add more ouputs if needed







### Fit Functions: Largely overlapping...modify from one...
# coh = coherence
# Parameters: [mu, mu_x_dependence, mu_t_dependence
#              sigma, sigma_x_dependence, sigma_t_dependence
#              B, bound_t_dependence]
# Model types should be a list of five classes/forms defining the model: mu, sigma, bounds, task, IC
# TODO: if this is slow, I should separate the logic and the interface in DDM_pdf_general
def MLE_model_fit_over_coh(params, model_types, y_goal, coherence_list): # Fit the final/total probability for correct, erred, and undecided choices.
    probs_all = np.zeros((3,len(coherence_list))) # [ [prob correct, prob error, prob undecided], ... ]
    mu = params.pop(0)
    sigma = params.pop(0)
    B = params.pop(0)
    # This is a somewhat hacky way to create a new model based on a
    # list of parameters.
    mts = []
    for mt in model_types:
        dep_params = dict(zip(mt.required_parameters, params))
        params = params[len(mt.required_parameters):]
        mts.append(mt(**dep_params))
    m = Model(name="to_fit", mu=mu, sigma=sigma, B=B,
              mudep=mts[0], sigmadep=mts[1],
              bounddep=mts[2], task=mts[3], IC=mts[4])
    
    for i_coh,coh in enumerate(coherence_list):
        m.parameters['mu'] = coh
        (pdf_corr, pdf_err) = DDM_pdf_general(m)
        prob_corr  = np.sum(pdf_corr)
        prob_err   = np.sum(pdf_err)
        prob_undec = 1. - prob_corr - prob_err
        probs_all[:,i_coh]   = [prob_corr, prob_err, prob_undec] # Total probability for correct, erred and undecided choices.
    to_min = -np.log(np.sum((y_goal*probs_all)**0.5 /dt**1)) # Bhattacharyya distance
    return to_min
# Other minimizers
    # to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2) # MLE
    # to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0) # KL divergence

def MSE_model_fit_RT(params, model_types, y_goal): # Fit the probability density functions of both correct and erred choices. TO BE VERIFIED.
    mu = params.pop(0)
    sigma = params.pop(0)
    B = params.pop(0)
    # This is a somewhat hacky way to create a new model based on a
    # list of parameters.
    mts = []
    for mt in model_types:
        dep_params = dict(zip(mt.required_parameters, params))
        params = params[len(mt.required_parameters):]
        mts.append(mt(**dep_params))
    assert params == []
    m = Model(name="to_fit", mu=mu, sigma=sigma, B=B,
              mudep=mts[0], sigmadep=mts[1],
              bounddep=mts[2], task=mts[3], IC=mts[4])
    
    (pdf_corr, pdf_err) = DDM_pdf_general(m)
    to_min = -np.log(np.sum((y_goal*numpy.asarray([pdf_corr, pdf_err]))**0.5)) # Bhattacharyya distance
    return to_min
#    to_min = sum(np.log(Prob_list_cumsum_corr_temp) *y_fit2)                                                           # MLE
#    to_min = -np.sum((Prob_list_corr_temp) *y_fit2)
    # epi_log = 0.000001
    # to_min = np.sum((y_fit2) * (np.log(y_fit2+epi_log) - np.log(Prob_list_corr_temp+epi_log)) /dt**0)     #KL divergence












########################################################################################################################
## Functions for Analytical Solutions.
### Analytical form of DDM. Can only solve for simple DDM or linearly collapsing bound... From Anderson1960
# Note that the solutions are automatically normalized.
def DDM_pdf_analytical(model):
    '''
    Now assume f_mu, f_sigma, f_Bound are callable functions
    See DDM_pdf_general for nomenclature
    '''

    ### Initialization
    assert model.mudep == MuLinear(x=0, t=0), "mu dependence not implemented"
    assert model.sigmadep == SigmaLinear(x=0, t=0), "sigma dependence not implemented"
    assert type(model.bounddep) in [BoundConstant, BoundCollapsingLinear], "bounddep dependence not implemented"
    assert model.task.name == "Fixed_Duration", "No analytic solution for that task"
    assert type(model.IC) == ICPointSourceCenter, "No analytic solution for those initial conditions"
    
    if type(model.bounddep) == BoundConstant: # Simple DDM
        anal_pdf_corr, anal_pdf_err = analytic_ddm(model.parameters['mu'],
                                                   model.parameters['sigma'],
                                                   model.parameters['B'], t_list)
    elif type(model.bounddep) == BoundCollapsingLinear: # Linearly Collapsing Bound
        anal_pdf_corr, anal_pdf_err = analytic_ddm(model.parameters['mu'],
                                                   model.parameters['sigma'],
                                                   model.parameters['B'],
                                                   t_list, -model.bounddep.t) # TODO why must this be negative? -MS

    ## Remove some abnormalities such as NaN due to trivial reasons.
    anal_pdf_corr[anal_pdf_corr==np.NaN] = 0.
    anal_pdf_corr[0] = 0.
    anal_pdf_err[anal_pdf_err==np.NaN] = 0.
    anal_pdf_err[0] = 0.
    return anal_pdf_corr*dt, anal_pdf_err*dt


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

    for n in range(nMax):
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
    dist = np.exp(-(a1+b1*teval)**2./teval/2)/np.sqrt(2*np.pi)/teval**1.5*suminc;
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








def Psychometric_fit_P(params_pm, pm_fit2):
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
