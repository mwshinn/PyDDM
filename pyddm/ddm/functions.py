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
              dt=dt, dx=dx, fitparams={},
              method="differential_evolution",
              overlay=OverlayNone(),
              lossfunction=SquaredErrorLoss,
              pool=None):
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

    `lossfunction` is a subclass of LossFunction representing the
    method to use when calculating the goodness-of-fit.  Pass the
    subclass itself, NOT an instance of the subclass.
    
    If `pool` is None, then solve normally.  If `pool` is a Pool
    object from pathos.multiprocessing, then parallelize the loss
    function.  Note that `pool` must be pathos.multiprocessing.Pool,
    not multiprocessing.Pool, since the latter does not support
    pickling functions, whereas the former does.

    Returns a "Model()" object with the specified `mu`, `sigma`,
    `bound`, `IC`, and `task`.
    """
    
    # Loop through the different components of the model and get the
    # parameters that are fittable.  Save the "Fittable" objects in
    # "params".  Create a list of functions to set the value of these
    # parameters, named "setters".
    components_list = [mu, sigma, bound, IC, task, overlay]
    required_conditions = list(set([x for l in components_list for x in l.required_conditions]))
    params = [] # A list of all of the Fittables that were passed.
    setters = [] # A list of functions which set the value of the corresponding parameter in `params`
    for component in components_list:
        for param_name in component.required_parameters:
            pv = getattr(component, param_name) # Parameter value in the object
            if isinstance(pv, Fittable):
                param = pv
                # Create a function which sets each parameter in the
                # list to some value `a` for model `x`.  Note the
                # default arguments to the lambda function are
                # necessary here to preserve scope.  Without them,
                # these variables would be interpreted in the local
                # scope, so they would be equal to the last value
                # encountered in the loop.
                setter = lambda x,a,component=component,param_name=param_name : setattr(x.get_dependence(component.depname), param_name, a)
                # If we have the same Fittable object in two different
                # components inside the model, we only want the Fittable
                # object in the list "params" once, but we want the setter
                # to update both.
                if param in params:
                    pind = params.index(param)
                    oldsetter = setters[pind]
                    newsetter = lambda x,a,setter=setter,oldsetter=oldsetter : [setter(x,a), oldsetter(x,a)] # Hack way of making a lambda to run two other lambdas
                    setters[pind] = newsetter
                else: # This setter is unique (so far)
                    params.append(param)
                    setters.append(setter)
                
    # Use the reaction time data (a list of reaction times) to
    # construct a reaction time distribution.  
    T_dur = np.ceil(max(sample)/dt)*dt
    assert T_dur < 30, "Too long of a simulation... are you using milliseconds instead of seconds?"
    # For optimization purposes, create a base model, and then use
    # that base model in the optimization routine.  First, set up the
    # model with all of the Fittables inside.  Deep copy on the entire
    # model is a shortcut for deep copying each individual component
    # of the model.
    m = copy.deepcopy(Model(mu=mu, sigma=sigma, bound=bound, IC=IC, task=task, overlay=overlay, T_dur=T_dur, dt=dt, dx=dx))
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
    lf = lossfunction(sample, required_conditions=required_conditions, pool=pool, T_dur=T_dur, dt=dt)
    # A function for the solver to minimize.  Since the model is in
    # this scope, we can make use of it by using, for example, the
    # model `m` defined previously.
    def _fit_model(xs):
        for x,s in zip(xs, setters):
            s(m, x)
        #to_min = -np.log(np.sum((fit_to_data*np.asarray([sol.pdf_corr(), sol.pdf_err()]))**0.5)) # Bhattacharyya distance
        lossf = lf.loss(m)
        print(xs, "loss="+str(lossf))
        return lossf
    # Run the solver
    print(x_0)
    if method == "simple":
        x_fit = minimize(_fit_model, x_0, bounds=constraints)
        assert x_fit.success == True, "Fit failed: %s" % x_fit.message
    elif method == "simplex":
        x_fit = minimize(_fit_model, x_0, method='Nelder-Mead')
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
    
