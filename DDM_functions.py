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
