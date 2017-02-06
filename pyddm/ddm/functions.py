'''
Simulation code for Drift Diffusion Model
Author: Norman Lam (norman.lam@yale.edu)
'''

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
import copy

from .parameters import *
from .model import *

########################################################################################################################
### Defined functions.

def models_close(m1, m2, tol=.1):
    """Determines whether two models are similar.

    This compares the parameters of models `m1` and `m2` and checks to
    make sure that each of the parameters in model `m1` is within a
    distance of `tol` of `m2`.  Return True if this is the case,
    otherwise False."""
    p1 = m1.get_model_parameters()
    p2 = m2.get_model_parameters()
    assert len(p1) == len(p2)
    assert m1.get_model_type() == m2.get_model_type()
    for mp1, mp2 in zip(p1, p2):
        if np.abs(mp1-mp2) > tol:
            return False
    return True

def fit_model_stable(sample,
                     mu=MuConstant(mu=0),
                     sigma=SigmaConstant(sigma=1),
                     bound=BoundConstant(B=1),
                     IC=ICPointSourceCenter(),
                     task=TaskFixedDuration(),
                     dt=dt):
    """A more stable version of fit_model.

    The purpose of this function is to avoid local minima when fitting
    models.  This calls `fit_model` multiple times, saving the best
    model.  Once a second model is found which matches the best model,
    it returns this model.

    For documentation of the parameters, see "fit_model".
    """

    models = []
    min_fit_val = np.inf
    best_model = None
    while True:
        m = fit_model(sample,
                      mu=mu, sigma=sigma,
                      bound=bound, IC=IC, task=task, dt=dt)
        if (not best_model is None) and models_close(best_model, m):
            if m._fitfunval < min_fit_val:
                return m
            else:
                return best_model
        elif m._fitfunval < min_fit_val:
            best_model = m
            min_fit_val = m._fitfunval

def fit_model(sample,
              mu=MuConstant(mu=0),
              sigma=SigmaConstant(sigma=1),
              bound=BoundConstant(B=1),
              IC=ICPointSourceCenter(),
              task=TaskFixedDuration(),
              dt=dt, fitparams={},
              method="differential_evolution",
              lossfunction=SquaredErrorLoss):
    """Fit a model to reaction time data.
    
    The data `sample` should be a Sample object of the reaction times
    to fit in seconds (NOT milliseconds).  This function will generate
    a model using the `mu`, `sigma`, `bound`, `IC`, and `task`
    parameters to specify the model.  At least one of these should
    have a parameter which is a "Fittable()" instance, as this will be
    the parameter to be fit.
    
    Optionally, dt specifies the temporal resolution with which to fit
    the model.

    `method` specifies how the model should be fit.
    "differential_evolution" is the default, which seems to be able to
    accurately locate the global maximum without using a
    derivative. "simple" uses a derivative-based method to minimize,
    and just uses randomly initialized parameters and gradient
    descent.  "basin" uses "scipy.optimize.basinhopping" to find an
    optimal solution, which is much slower but also gives better
    results than "simple".  It does not appear to give better results
    than "differential_evolution".

    `fitparams` is a dictionary of kwargs to be passed directly to the
    minimization routine for fine-grained low-level control over the
    optimization.  Normally this should not be needed.  
    
    Returns a "Model()" object with the specified `mu`, `sigma`,
    `bound`, `IC`, and `task`.
    """
    
    # Loop through the different components of the model and get the
    # parameters that are fittable.  
    components_list = [mu, sigma, bound, IC, task]
    params = [] # A list of all of the Fittables that were passed.
    setters = [] # A list of functions which set the value of the corresponding parameter in `params`
    for component in components_list:
        for param_name in component.required_parameters:
            pv = getattr(component, param_name) # Parameter value in the object
            if isinstance(pv, Fittable):
                params.append(getattr(component, param_name))
                # Create a function which sets each parameter in the
                # list to some value `a` for model `x`.  Note the
                # default arguments to the lambda function are
                # necessary here to preserve scope.  Without them,
                # these variables would be interpreted in the local
                # scope, so they would be equal to the last value
                # encountered in the loop.
                setters.append(lambda x,a,component=component,param_name=param_name : setattr(x.get_dependence(component.depname), param_name, a))
    # Use the reaction time data (a list of reaction times) to
    # construct a reaction time distribution.  
    T_dur = np.ceil(max(sample)/dt)*dt
    assert T_dur < 30, "Too long of a simulation... are you using milliseconds instead of seconds?"
    # For optimization purposes, create a base model, and then use
    # that base model in the optimization routine.  First, set up the
    # model with all of the Fittables inside.  Deep copy on the entire
    # model is a shortcut for deep copying each individual component
    # of the model.
    m = copy.deepcopy(Model(mu=mu, sigma=sigma, bound=bound, IC=IC, task=task, T_dur=T_dur, dt=dt))
    # And now get rid of the Fittables, replacing them with the
    # default values.  Simultaneously, create a list to pass to the
    # solver.
    x_0 = []
    constraints = [] # List of (min, max) tuples.  min/max=None if no constraint.
    for p,s in zip(params, setters):
        default = p.default()
        s(m, default)
        minval = p.minval if p.minval > -np.inf else None
        maxval = p.maxval if p.maxval < np.inf else None
        constraints.append((minval, maxval))
        x_0.append(default)
    # Set up a loss function
    lf = lossfunction(sample, T_dur=T_dur, dt=dt)
    # A function for the solver to minimize.  Since the model is in
    # this scope, we can make use of it by using, for example, the
    # model `m` defined previously.
    def _fit_model(xs):
        print(xs)
        for x,s in zip(xs, setters):
            s(m, x)
        #to_min = -np.log(np.sum((fit_to_data*np.asarray([sol.pdf_corr(), sol.pdf_err()]))**0.5)) # Bhattacharyya distance
        return lf.loss(m)
    # Run the solver
    print(x_0)
    if method == "simple":
        x_fit = minimize(_fit_model, x_0, bounds=constraints)
        assert x_fit.success == True, "Fit failed: %s" % x_fit.message
    elif method == "basin":
        x_fit = basinhopping(_fit_model, x_0, minimizer_kwargs={"bounds" : constraints, "method" : "TNC"}, disp=True, **fitparams)
    elif method == "differential_evolution":
        x_fit = differential_evolution(_fit_model, constraints, disp=True, **fitparams)
    else:
        raise NotImplementedError("Invalid method")
    print("Params", x_fit.x, "gave", x_fit.fun)
    m._fitfunval = x_fit.fun
    for x,s in zip(x_fit.x, setters):
        s(m, x)
    return m
    

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
    to_min = -np.log(np.sum((y_goal*np.asarray([pdf_corr, pdf_err]))**0.5)) # Bhattacharyya distance
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
