# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

__all__ = ['models_close', 'fit_model', 'fit_adjust_model',
           'evolution_strategy', 'solve_partial_conditions',
           'hit_boundary', 'dependence_hit_boundary', 'display_model',
           'get_model_loss', 'set_N_cpus', 'solve_all_conditions',
           'gddm']

import copy
import logging
import inspect
import numpy as np
import pandas
import keyword
from scipy.optimize import minimize, basinhopping, differential_evolution, OptimizeResult

from . import parameters as param
from .model import Model, Solution, Fitted, Fittable
from .sample import Sample
from .models.drift import Drift, DriftConstant
from .models.noise import Noise, NoiseConstant
from .models.ic import InitialCondition, ICPointSourceCenter, ICPointRatio
from .models.bound import Bound, BoundConstant
from .models.overlay import Overlay, OverlayNone, OverlayChain, OverlayNonDecision, OverlayUniformMixture
from .models.loss import LossLikelihood
from .logger import logger as _logger

from paranoid.types import Boolean, Number, String, Set, Unchecked, Natural1, Maybe
from paranoid.decorators import accepts, returns, requires, ensures, paranoidconfig
from paranoid.settings import Settings as paranoid_settings
from .models.paranoid_types import Conditions

from .fitresult import FitResult, FitResultEmpty

# For parallelization support
_parallel_pool = None # Note: do not change this directly.  Call set_N_cpus() instead.
#@accepts(Natural1)
#@paranoidconfig(enabled=False)

def set_N_cpus(N):
    """Enable parallelization with N threads."""
    global _parallel_pool
    if _parallel_pool is not None:
        _parallel_pool.close()
    if N != 1:
        try:
            import pathos
            from packaging import version
            import dill
            assert version.parse(dill.__version__) > version.parse('0.3.4'), "Please update the package 'dill' to 3.5.0 or later to use multiprocessing"
        except ImportError:
            raise ImportError("Parallel support requires pathos.  Please install pathos.")
        #_parallel_pool = pathos.multiprocessing.Pool(N)
        _parallel_pool = pathos.pools._ProcessPool(N)
        _parallel_pool.n_cpus = N
    else:
        _parallel_pool = None

@accepts(Model, Model, tol=Number)
@requires("m1.get_model_type() == m2.get_model_type()")
@returns(Boolean)
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

def get_model_loss(model, sample, lossfunction=LossLikelihood, method=None):
    """A shortcut to compusing the loss of a model.

    A shortcut method to compute the loss (under loss function
    `lossfunction`) of Model `model` with respect to Sample `sample`.
    Optionaly, specificy the numerical method `method` to use
    (e.g. analytical, numerical, implicit, etc.)
    
    Note that this should not be used when performing model fits, as
    it is faster to use the optimizations implemented in
    fit_adjust_model.
    """
    assert model.choice_names == sample.choice_names, "Model and sample choice names must match, currently "+repr(model.choice_names)+" and "+repr(sample.choice_names)+".  Please specify the correct choice names when creating your Model and Sample."
    # Count parameters (for AIC/BIC), making sure not to double count
    # for repeated parameters.
    params = []
    for component in model.dependencies:
        for param_name in component.required_parameters:
            pv = getattr(component, param_name) # Parameter value in the object
            if pv not in params and isinstance(pv, Fittable):
                params.append(pv)
    lf = lossfunction(sample, required_conditions=model.required_conditions,
                      T_dur=model.T_dur, dt=model.dt, method=method,
                      nparams=len(params), samplesize=len(sample))
    return lf.loss(model)


def fit_model(sample,
              drift=DriftConstant(drift=0),
              noise=NoiseConstant(noise=1),
              bound=BoundConstant(B=1),
              IC=ICPointSourceCenter(),
              dt=param.dt, dx=param.dx, fitparams=None,
              fitting_method="differential_evolution",
              method=None,
              overlay=OverlayNone(),
              lossfunction=LossLikelihood,
              verbose=True,
              name="fit_model",
              verify=False):
    """Fit a model to reaction time data.
    
    The data `sample` should be a Sample object of the reaction times
    to fit in seconds (NOT milliseconds).  This function will generate
    a model using the `drift`, `noise`, `bound`, and `IC`
    parameters to specify the model.  At least one of these should
    have a parameter which is a "Fittable()" instance, as this will be
    the parameter to be fit.
    
    Optionally, dt specifies the temporal resolution with which to fit
    the model.

    `method` specifies how the model should be fit.
    "differential_evolution" is the default, which accurately locates
    the global maximum without using a derivative.  "simple" uses a
    derivative-based method to minimize, and just uses randomly
    initialized parameters and gradient descent.  "simplex" is the
    Nelder-Mead method, and is a gradient-free local search.  "basin"
    uses "scipy.optimize.basinhopping" to find an optimal solution,
    which is much slower but also gives better results than "simple".
    It does not appear to give better or faster results than
    "differential_evolution" in most cases.  Alternatively, a custom
    objective function may be used by setting `method` to be a
    function which accepts the "x_0" parameter (for starting position)
    and "constraints" (for min and max values).  In general, it is
    recommended you almost always use differential evolution, unless
    you have a model which is highly-constrained (e.g. only one or two
    parameters to estimate with low covariance) or you already know
    the approximate parameter values.  In practice, besides these two
    special cases, changing the method is unlikely to give faster or
    more reliable estimation.

    `fitparams` is a dictionary of kwargs to be passed directly to the
    minimization routine for fine-grained low-level control over the
    optimization.  Normally this should not be needed.  

    `lossfunction` is a subclass of LossFunction representing the
    method to use when calculating the goodness-of-fit.  Pass the
    subclass itself, NOT an instance of the subclass.
    
    `name` gives the name of the model after it is fit.

    If `verify` is False (the default), checking for programming
    errors is disabled during the fit. This can decrease runtime and
    may prevent crashes.  If verification is already disabled, this
    does not re-enable it.
    
    `verbose` enables out-of-boundaries warnings and prints the model
    information at each evaluation of the fitness function.

    Returns a "Model()" object with the specified `drift`, `noise`,
    `bound`, `IC`, and `overlay`.

    The model will include a "FitResult" object, accessed as
    m.fitresult.  This can be used to get the value of the objective
    function, as well as to access diagnostic information about the
    fit.

    This function will automatically parallelize if set_N_cpus() has
    been called.

    """
    
    # Use the reaction time data (a list of reaction times) to
    # construct a reaction time distribution.
    T_dur = np.ceil(max(sample)/dt)*dt
    assert T_dur < 30, "Too long of a simulation... are you using milliseconds instead of seconds?"
    # For optimization purposes, create a base model, and then use
    # that base model in the optimization routine.  First, set up the
    # model with all of the Fittables inside.  Deep copy on the entire
    # model is a shortcut for deep copying each individual component
    # of the model.
    m = copy.deepcopy(Model(name=name, drift=drift, noise=noise, bound=bound, IC=IC, overlay=overlay, T_dur=T_dur, dt=dt, dx=dx, choice_names=sample.choice_names))
    return fit_adjust_model(sample, m, fitparams=fitparams, fitting_method=fitting_method, 
                            method=method, lossfunction=lossfunction, verbose=verbose)


def fit_adjust_model(sample, model, fitparams=None, fitting_method="differential_evolution",
                     lossfunction=LossLikelihood, verify=False, method=None, verbose=True):
    """Modify parameters of a model which has already been fit.
    
    The data `sample` should be a Sample object of the reaction times
    to fit in seconds (NOT milliseconds).  At least one of the
    parameters for one of the components in the model should be a
    "Fitted()" instance, as these will be the parameters to fit.
    
    `fitting_method` specifies how the model should be fit.
    "differential_evolution" is the default, which accurately locates
    the global maximum without using a derivative.  "simple" uses a
    derivative-based method to minimize, and just uses randomly
    initialized parameters and gradient descent.  "simplex" is the
    Nelder-Mead method, and is a gradient-free local search.  "basin"
    uses "scipy.optimize.basinhopping" to find an optimal solution,
    which is much slower but also gives better results than "simple".
    It does not appear to give better or faster results than
    "differential_evolution" in most cases.  Alternatively, a custom
    objective function may be used by setting `fitting_method` to be a
    function which accepts the "x_0" parameter (for starting position)
    and "constraints" (for min and max values).  In general, it is
    recommended you almost always use differential evolution, unless
    you have a model which is highly-constrained (e.g. only one or two
    parameters to estimate with low covariance) or you already know
    the approximate parameter values.  In practice, besides these two
    special cases, changing the method is unlikely to give faster or
    more reliable estimation.

    `fitparams` is a dictionary of kwargs to be passed directly to the
    minimization routine for fine-grained low-level control over the
    optimization.  Normally this should not be needed.  

    `lossfunction` is a subclass of LossFunction representing the
    method to use when calculating the goodness-of-fit.  Pass the
    subclass itself, NOT an instance of the subclass.

    `name` gives the name of the model after it is fit.

    If `verify` is False (the default), checking for programming
    errors is disabled during the fit. This can decrease runtime and
    may prevent crashes.  If verification is already disabled, this
    does not re-enable it.

    `method` gives the method used to solve the model, and can be
    "analytical", "numerical", "cn", "implicit", or "explicit".

    `verbose` enables out-of-boundaries warnings and prints the model
    information at each evaluation of the fitness function.

    Returns the same model object that was passed to it as an
    argument.  However, the parameters will be modified.  The model is
    modified in place, so a reference is returned to it for
    convenience only.

    After running this function, the model will be modified to include
    a "FitResult" object, accessed as m.fitresult.  This can be used
    to get the value of the objective function, as well as to access
    diagnostic information about the fit.

    This function will automatically parallelize if set_N_cpus() has
    been called.

    """
    assert model.choice_names == sample.choice_names, "Model and sample choice names must match, currently "+repr(model.choice_names)+" and "+repr(sample.choice_names)+".  Please specify the correct choice names when creating your Model and Sample."
    # Disable paranoid if `verify` is False.
    paranoid_state = paranoid_settings.get('enabled')
    renorm_warnings_state = param.renorm_warnings
    if paranoid_state and not verify:
        paranoid_settings.set(enabled=False)
        param.renorm_warnings = False
    # Loop through the different components of the model and get the
    # parameters that are fittable.  Save the "Fittable" objects in
    # "params".  Create a list of functions to set the value of these
    # parameters, named "setters".
    m = model
    components_list = [m.get_dependence("drift"),
                       m.get_dependence("noise"),
                       m.get_dependence("bound"),
                       m.get_dependence("IC"),
                       m.get_dependence("overlay")]
    required_conditions = list(set([x for l in components_list for x in l.required_conditions]))
    assert 0 < len([1 for component in components_list
                      for param_name in component.required_parameters
                          if isinstance(getattr(component, param_name), Fittable)]), \
           "Models must contain at least one Fittable parameter in order to be fit"
    params = [] # A list of all of the Fittables that were passed.
    setters = [] # A list of functions which set the value of the corresponding parameter in `params`
    for component in components_list:
        for param_name in component.required_parameters:
            pv = getattr(component, param_name) # Parameter value in the object
            if isinstance(pv, Fittable):
                # Create a function which sets each parameter in the
                # list to some value `a` for model `x`.  Note the
                # default arguments to the function are necessary here
                # to preserve scope.  Without them, these variables
                # would be interpreted in the local scope, so they
                # would be equal to the last value encountered in the
                # loop.
                def setter(x,a,pv=pv,component=component,param_name=param_name):
                    if not isinstance(a, Fittable):
                        a = pv.make_fitted(a)
                    setattr(x.get_dependence(component.depname), param_name, a)
                    # Return the fitted instance so we can chain it.
                    # This way, if the same Fittable object is passed,
                    # the same Fitted object will be in both places in
                    # the solution.
                    return a 
                
                # If we have the same Fittable object in two different
                # components inside the model, we only want the Fittable
                # object in the list "params" once, but we want the setter
                # to update both.
                if id(pv) in map(id, params):
                    pind = list(map(id, params)).index(id(pv))
                    oldsetter = setters[pind]
                    # This is a hack way of executing two functions in
                    # a single function call while passing forward the
                    # same argument object (not just the same argument
                    # value)
                    newsetter = lambda x,a,setter=setter,oldsetter=oldsetter : oldsetter(x,setter(x,a)) 
                    setters[pind] = newsetter
                else: # This setter is unique (so far)
                    params.append(pv)
                    setters.append(setter)
                
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
    lf = lossfunction(sample, required_conditions=required_conditions,
                      T_dur=m.T_dur, dt=m.dt, method=method,
                      nparams=len(params), samplesize=len(sample))
    # A function for the solver to minimize.  Since the model is in
    # this scope, we can make use of it by using, for example, the
    # model `m` defined previously.
    def _fit_model(xs):
        for x,p,s in zip(xs, params, setters):
            # Sometimes the numpy optimizers will ignore bounds up to
            # floating point errors, i.e. if your upper bound is 1,
            # they will give 1.000000000001.  This fixes that problem
            # to make sure the model is within its domain.
            if x > p.maxval:
                if verbose:
                    _logger.warning("Optimizer went out of bounds.  Setting %f to %f" % (x, p.maxval))
                x = p.maxval
            if x < p.minval:
                if verbose:
                    _logger.warning("Optimizer went out of bounds.  Setting %f to %f" % (x, p.minval))
                x = p.minval
            s(m, x)
        lossf = lf.loss(m)
        if verbose:
            _logger.info(repr(m) + " loss="+ str(lossf))
        return lossf
    # Cast to a dictionary if necessary
    if fitparams is None:
        fitparams = {}
    # Run the solver
    if fitting_method == "simple":
        x_fit = minimize(_fit_model, x_0, bounds=constraints)
        assert x_fit.success, "Fit failed: %s" % x_fit.message
    elif fitting_method == "simplex":
        x_fit = minimize(_fit_model, x_0, method='Nelder-Mead')
    elif fitting_method == "basin":
        x_fit = basinhopping(_fit_model, x_0, minimizer_kwargs={"bounds" : constraints, "method" : "TNC"}, **fitparams)
    elif fitting_method == "differential_evolution":
        if "disp" not in fitparams.keys():
            fitparams["disp"] = verbose
        x_fit = differential_evolution(_fit_model, constraints, **fitparams)
    elif fitting_method == "hillclimb":
        x_fit = evolution_strategy(_fit_model, x_0, **fitparams)
    elif callable(fitting_method):
        x_fit = fitting_method(_fit_model, x_0=x_0, constraints=constraints)
    else:
        raise NotImplementedError("Invalid fitting method")
    res = FitResult(method=(method if method is not None else "auto"),
                    fitting_method=fitting_method, loss=lf.name, value=x_fit.fun,
                    nparams=len(params), samplesize=len(sample),
                    mess=(x_fit.message if "message" in x_fit.__dict__ else ""))
    m.fitresult = res
    _logger.info("Params " + str(x_fit.x) + " gave " + str(x_fit.fun))
    for x,s in zip(x_fit.x, setters):
        s(m, x)
    if not verify:
        paranoid_settings.set(enabled=paranoid_state)
        param.renorm_warnings = renorm_warnings_state
    return m

def evolution_strategy(fitness, x_0, mu=1, lmbda=3, copyparents=True, mutate_var=.002, mutate_prob=.5, evals=100, seed=None):
    """Optimize using the Evolution Strategy (ES) heuristic method.

    Evolution Strategy is an optimization method specified in the form
    (lambda + mu) or (lambda, mu), for some integer value of lambda
    and mu.  The algorithm will generate an intial population of
    lambda individuals.  Each individual will have an equal number of
    offspring, so that there is a total of mu organisms in the next
    generation.  In the case of the (lambda + mu) algorithm, the
    parents are also copied into the next generation.  Then, the
    lambda best organisms in this population are selected to
    reproduce.

    The starting population includes `x_0` and `lmbda`-1 other
    individuals generated by mutating `x_0`.  Mutations occur by
    perturbing `x_0` by a Gaussian-distributed variable (so-called
    "Gaussian convolution") with variance `mutate_var`.  Each element
    is changed with a probability `mutate_prob`.  The number of
    function evaluations will be approximately `evals`, as this
    algorithm will iterate `evals`/`lmbda` times.

    `lmbda` is the lambda parameter (note the spelling difference) and
    `mu` is the mu parameter for the ES.  If `copyparents` is True,
    use (`lmbda` + `mu`), and if it is False, use (`lmbda`, `mu`).
    
    `seed` allows optional seed values to be used during random number 
    generation during evolution-driven optimization. If set to None, 
    random number generation is subject to unseeded behavior. If
    convergence is difficult, setting seed=None may result in different 
    solutions between runs.

    The purpose of this is if you already have a good model, but you
    want to test the local space to see if you can make it better.
    """
    assert isinstance(lmbda, int) and isinstance(mu, int), "Bad lambda and mu"
    assert lmbda/mu == lmbda//mu, "Lambda must be a multiple of mu"
    x_0 = list(x_0) # Ensure we have a list, not an ndarray
    it = evals//lmbda
    
    # Mutation function: with a probability of `mutate_prob`, add a
    # uniform gaussian random variable multiplied by the current value
    # of the parameter, with variance `mutate_var`.
    if seed is None:
        mutate = lambda x : [e+np.random.normal(0, mutate_var) if np.random.random()<mutate_prob else e for e in x]
    else:
        assert isinstance(seed, (int, np.int32, np.int64)), "Expected seed to be <int>, got <{}>".format(type(seed))
        rng = np.random.RandomState(seed)
        mutate = lambda x : [e+rng.normal(0, mutate_var) if rng.random()<mutate_prob else e for e in x]
    
    # Set up the initial population.  We make the initial population
    # by mutating X_0.  This is not good for explorative search but is
    # good for exploitative search.
    P = [(x_0, fitness(x_0))]
    best = P[0]
    for _ in range(0, lmbda-1):
        new = mutate(x_0)
        fit = fitness(new)
        if fit < best[1]:
            best = (new, fit)
        P.append((new, fit))
    for _ in range(0, it):
        # Find the `mu` best individuals
        P.sort(key=lambda e : e[1])
        Q = P[0:mu]
        # Copy the parents if we're supposed to
        P = Q.copy() if copyparents else []
        # Create the next generation population
        for q in Q:
            for _ in range(0, lmbda//mu):
                new = mutate(q[0])
                fit = fitness(new)
                if fit < best[1]:
                    best = (new, fit)
                P.append((new, fit))
    return OptimizeResult(x=np.asarray(best[0]), success=True, fun=best[1], nit=it)

#@accepts(Model, Sample, Conditions, Unchecked, Set(["analytical", "numerical", "cn", "implicit", "explicit"]))
#@returns(Unchecked)
def solve_all_conditions(model, sample=None, condition_combinations=None, method=None):
    """Solve the model for all relevant conditions.

    This takes the following parameters (note that there are two legal
    parameterizations for this function):

    - `model` - A Model() object
    - `sample` - A Sample() object which has conditions for each of
      the required conditions in `model`. Conditions may equivalently be
      specified via `condition_combinations`.
    - `condition_combinations` - A list of dicts, where each dict
      specifies a combination of condition names and values for which to
      solve the model (the same format as Sample.condition_combinations()
      outputs). Conditions must be the same as the required conditions in
      `model`. Conditions may equivalently be specified via `sample`.
    - `method` - A string describing the solver method.  Can be
      "analytical", "numerical", "cn", "implicit", or "explicit".

    For each combination of relevant condition-values, this will solve the
    model for that "condition combination." It returns a dictionary indexed
    by a frozenset of the condition names and values, with the Solution
    object as the value, e.g.:
    
        {frozenset({('reward', 3)}): <Solution object>,
         frozenset({('reward', 1)}): <Solution object>}

    This function will automatically parallelize if set_N_cpus() has
    been called.
    """
    if sample is not None and condition_combinations is None:
        conds = sample.condition_combinations(required_conditions=model.required_conditions)
    elif condition_combinations is not None and sample is None:
        assert all(set(cond_combo.keys()) == set(model.required_conditions) for cond_combo in condition_combinations)
        conds = condition_combinations
    elif condition_combinations is None and sample is None:
        raise ValueError("Must specify either `sample` or `condition_combinations` for solve_all_conditions().")
    else:
        raise ValueError("Cannot specify both `sample` and `condition_combinations` for solve_all_conditions().")

    if method is None:
        meth = model.solve
    elif method == "analytical":
        meth = model.solve_analytical
    elif method == "numerical":
        meth = model.solve_numerical
    elif method == "cn":
        meth = model.solve_numerical_cn
    elif method == "implicit":
        meth = model.solve_numerical_implicit
    elif method == "explicit":
        meth = model.solve_numerical_explicit
    else:
        raise ValueError("Invalid method "+method)

    cache = {}
    if _parallel_pool is None: # No parallelization
        for c in conds:
            cache[frozenset(c.items())] = meth(conditions=c)
        return cache
    else: # Parallelize across pool
        if paranoid_settings.get('enabled') is False:
            # The *2 makes sure that this runs on all subprocesses,
            # since you can't broadcast commands to all processes
            _parallel_pool.map(lambda x : paranoid_settings.set(enabled=False), [None]*_parallel_pool.n_cpus*2)
        sols = _parallel_pool.map(meth, conds, chunksize=1)
        for c,s in zip(conds, sols):
            cache[frozenset(c.items())] = s
        return cache


# TODO explicitly test this in unit tests
@accepts(Model, Maybe(Sample), Maybe(Conditions), Maybe(Set(["analytical", "numerical", "cn", "implicit", "explicit"])))
# @returns(Solution) # This doesn't actually return a solution, only a solution-like object
@requires('sample is not None --> all((c in sample.condition_names() for c in model.required_conditions))')
@requires('conditions is not None and sample is not None --> all((c in sample.condition_names() for c in conditions))')
@requires('conditions is not None and sample is None --> all((c in conditions for c in model.required_conditions))')
@requires("method == 'explicit' --> model.can_solve_explicit(conditions=conditions)")
@requires("method == 'cn' --> model.can_solve_cn()")
def solve_partial_conditions(model, sample=None, conditions=None, method=None):
    """Solve a model without specifying the value of all conditions

    This function solves `model` according to the ratio of trials in
    `sample`.  For example, suppose `sample` has 100 trials with high
    coherence and 50 with low coherence.  This will then return a
    solution with 2/3*(PDF high coherence) + 1/3*(PDF low coherence).
    This is especially useful when comparing a model to a sample which
    may have many different conditions.  

    Alternatively, if no sample is available, it will solve all
    conditions passed in `conditions` in equal ratios.

    The advantage to this function over Model.solve() is that the
    former can only handle a single value for each condition, whereas
    this function accepts can do it lists for condition values as
    well.

    The `conditions` variable limits the solution to a subset of
    `sample` which satisfy `conditions`.  The elements of the
    dictionary `conditions` should be specified either as values or as
    a list of values.

    Optionally, `method` describes the solver to use.  It can be
    "analytical", "numerical", "cn" (Crank-Nicolson), "implicit"
    (backward Euler), "explicit" (forward Euler), or None
    (auto-detect method).

    This function will automatically parallelize if set_N_cpus() has
    been called.

    """
    if conditions is None:
        conditions = {}
    T_dur = model.T_dur
    dt = model.dt
    if sample:
        # If a sample is passed, include only the parts of the sample
        # that satisfy the passed conditions.
        samp = sample.subset(**conditions)
        # Specially handle the case where one parameter value from conditions is
        # not in the sample.  This was implemented specifically for the
        # psychometric curve calculations.  In theory this could be made more
        # general so that it can still succeed if more than one parameter value
        # is missing.
        if len(samp) == 0:
            for cond in conditions.keys():
                samp = sample.subset(**{c : conditions[c] for c in conditions.keys() if c != cond})
                if len(samp) != 0:
                    break
            else:
                raise ValueError("More than one condition not found in the sample")
            df = samp.to_pandas_dataframe()
            cvals = conditions[cond] if isinstance(conditions[cond], (list, np.ndarray)) else [conditions[cond]]
            dfs = []
            for cv in cvals:
                new_df = df.copy()
                new_df[cond] = cv
                dfs.append(new_df)
            samp = Sample.from_pandas_dataframe(pandas.concat(dfs), 'RT', 'choice')
    else:
        # If no sample is passed, create a dummy sample.  For this, we
        # need all of the conditions to be specified in the
        # "conditions" variable.  We then construct a sample with
        # exactly one element correct and zero elements error for each
        # potential combinations of conditions.  For instance, if
        # there are two conditions with only one value of each passed
        # (as scalars), only one element will be created.  If each of
        # these two instead has two values passed (as a list), create
        # four elements in the sample, etc.
        assert len(set(model.required_conditions) - set(conditions.keys())) == 0, \
            "If no sample is passed, all conditions must be specified"
        # Build cond_combs as the data matrix iteratively. Initial
        # value is a correct response with an RT of 1 (as per the
        # first expected elements of Sample.from_numpy_array().
        cond_combs = [[0, 1]]
        all_conds = list(sorted(conditions.keys()))
        for c in all_conds:
            vs = conditions[c]
            if not isinstance(vs, list):
                vs = [vs]
            cond_combs = [cc + [v] for cc in cond_combs for v in vs]
        # Quick fix for bug with a tuple as a condition and only one set of
        # conditions
        if len(cond_combs) == 1:
            cond_combs = cond_combs + cond_combs
        samp = Sample.from_numpy_array(np.asarray(cond_combs, dtype=object), all_conds, choice_names=model.choice_names)
    model_choice_upper = 0*model.t_domain()
    model_choice_lower = 0*model.t_domain()
    model_undec = -1 # Set to dummy value -1 so we can detect this in our loop
    # If we have an overlay, this function should not calculate the
    # (incorrect) undecided probability
    if not isinstance(model.get_dependence("overlay"), OverlayNone):
        model_undec = None
    all_conds = solve_all_conditions(model, sample=samp, method=method)
    for conds in samp.condition_combinations(required_conditions=model.required_conditions):
        subset = samp.subset(**conds)
        sol = all_conds[frozenset(conds.items())]
        model_choice_upper += len(subset)/len(samp)*sol.pdf(model.choice_names[0])
        model_choice_lower += len(subset)/len(samp)*sol.pdf(model.choice_names[1])
        # We can't get the size of the undecided pdf until we have a
        # specific set of conditions.  Once we do, if the simulation
        # method doesn't support an undecided probability, set it to
        # None.  If it does, add it together, making sure they are
        # always the same size.  (They may not be the same size if the
        # bound depends on a parameter.)  If they are ever not the
        # same size, set it to None rather than trying to align them.
        if sol.undec is not None and isinstance(model_undec, int) and model_undec == -1:
            model_undec = len(subset)/len(samp)*sol.pdf_undec()
        if sol.undec is not None and model_undec is not None and len(model_undec) == len(sol.undec):
            model_undec += len(subset)/len(samp)*sol.pdf_undec()
        else:
            model_undec = None
    sol = Solution(model_choice_upper*model.dt, model_choice_lower*model.dt, model, conditions={}, pdf_undec=model_undec)
    sol.partial_conditions = conditions
    return sol

@accepts(Model)
@returns(Boolean)
def hit_boundary(model):
    """Returns True if any Fitted objects are close to their min/max value"""
    components_list = [model.get_dependence("drift"),
                       model.get_dependence("noise"),
                       model.get_dependence("bound"),
                       model.get_dependence("IC"),
                       model.get_dependence("overlay")]
    hit = False
    for component in components_list:
        for param_name in component.required_parameters:
            pv = getattr(component, param_name) # Parameter value in the object
            if isinstance(pv, Fitted):
                if (pv - pv.minval)/(pv.maxval-pv.minval) < .01: # No abs because pv always > pv.minval
                    _logger.warning("%s hit the lower boundary of %f with value %f" % (param_name, pv.minval, pv))
                    hit = True
                if (pv.maxval-pv)/(pv.maxval-pv.minval) < .01: # No abs because pv.maxval always > pv
                    _logger.warning("%s hit the lower boundary of %f with value %f" % (param_name, pv.maxval, pv))
                    hit = True
    return hit

@accepts(Fittable)
@returns(Boolean)
def dependence_hit_boundary(pv):
    """Returns True if a Fitted instance has hit the boundary.

    Fitted instances may have minimum or maximum values attached to
    them.  If it does, and if it has gotten close to this min/max
    while fitting, return True.  Otherwise, or if the value is not a
    Fitted object, return False.
    """
    if isinstance(pv, Fitted):
        if (pv - pv.minval)/(pv.maxval-pv.minval) < .01: # No abs because pv always > pv.minval
            return True
        if (pv.maxval-pv)/(pv.maxval-pv.minval) < .01: # No abs because pv.maxval always > pv
            return True
    return False

def display_model(model, print_output=True):
    """A readable way to display models.

    `model` should be any Model object.  Prints a description of the
    model, and does not return anything.
    """
    OUT = ""
    assert isinstance(model, Model), "Invalid model"
    # Separate the code to display a single component so we can reuse
    # it to display the components of chains (e.g. OverlayChain).
    def display_component(component, prefix=""):
        OUT = ""
        OUT += prefix+"%s" % component.name + "\n"
        fixed = []
        fitted = []
        fittable = []
        if len(component.required_parameters) == 0:
            OUT += prefix+"(No parameters)" + "\n"
        for param_name in component.required_parameters:
            pv = getattr(component, param_name) # Parameter value in the object
            if isinstance(pv, Fitted):
                if dependence_hit_boundary(pv):
                    fitted.append(prefix+"- %s: %f (WARNING: AT BOUNDARY)" % (param_name, pv))
                else:
                    fitted.append(prefix+"- %s: %f" % (param_name, pv))
            elif isinstance(pv, Fittable):
                fittable.append(prefix+"- %s: Fittable (default %f)" % (param_name, pv.default()))
            else:
                fixed.append(prefix+"- %s: %f" % (param_name, pv))
        for t,vs in [("Fixed", fixed), ("Fitted", fitted), ("Fittable", fittable)]:
            if len(vs) > 0:
                OUT += prefix+t+" parameters:" + "\n"
                for v in vs:
                    OUT += v + "\n"
        return OUT
    # Start displaying the model information
    OUT += ("Model %s information:\n" % model.name) if model.name != "" else "Model information:" + "\n"
    OUT += "Choices: '%s' (upper boundary), '%s' (lower boundary)\n" % model.choice_names
    for component in model.dependencies:
        OUT += "%s component %s:" % (component.depname, type(component).__name__) + "\n"
        if isinstance(component, OverlayChain):
            for o in component.overlays:
                OUT += "    %s component %s:" % (o.depname, type(o).__name__) + "\n"
                OUT += display_component(o, prefix="        ")
        else:
            OUT += display_component(component, prefix="    ")
    if not isinstance(model.fitresult, FitResultEmpty):
        OUT += "Fit information:\n"
        OUT += "    Loss function: %s\n" % model.fitresult.loss
        OUT += "    Loss function value: %s\n" % model.fitresult.value()
        OUT += "    Fitting method: %s\n" % model.fitresult.fitting_method
        OUT += "    Solver: %s\n" % ("forward Euler" if model.fitresult.method == "explicit" \
                                     else "backward Euler" if model.fitresult.method == "implicit" \
                                     else "Crank-Nicoloson" if model.fitresult.method == "cn" \
                                     else model.fitresult.method)
        OUT += "    Other properties:\n"
        for p,v in model.fitresult.properties.items():
            OUT += "        - %s: %s\n" % (p,repr(v))
    if not print_output:
        return OUT
    else:
        print(OUT)

def gddm(drift=0, noise=1, bound=1, nondecision=0, starting_position=0, mixture_coef=.02, name="", parameters={}, conditions=[], dx=param.dx, dt=param.dt, T_dur=param.T_dur, choice_names=param.choice_names):
    """Return a model without the use of PyDDM's object-oriented interface.

    PyDDM has two interfaces: this one (the gddm function), and the
    object-oriented interface.  This interface supports almost all of the
    features of PyDDM, and makes it much simpler to specify models.  Models
    created either way can be used interchangably.

    To create a model with the gddm function, there are three essential pieces
    of information that must be specified: the model parameters (values which
    are fit to data), the data conditions (e.g. extra properties of each trial),
    and the model form.

    Parameters are values in the model which are left free, allowing them to be
    fit to data.  Parameters are specified in the `parameters` argument and is a
    dictionary, where each key is the name of a parameter.  The value associated
    with each key should be a tuple of size two, indicating the minimum and
    maximum valid value of the parameter.  The parameter may also be a single
    value, meaning the parameter is fixed to a given value.  Parameters must be
    floating point numbers.

    Conditions are properties of the data which may be used by the model.  For
    instance, each trial may have a motion coherence or signal strength.
    Conditions must be defined when loading the data.  The `conditions` argument
    should be a list of the names of the relevant conditions in the Sample used
    for fitting.  Conditions do not need to be floats or even numbers: they can
    be strings, arrays, lists, or any other Python object.

    To specify model form, there are six model components available to this
    function: drift rate, noise level (standard deviation of noise), bound
    height, starting position, non-decision time, and uniform mixture model
    coefficient.  Each may be a single fixed constant, a parameter, a condition,
    or a Python function.  Functions can be any valid Python function, and may
    involve parameters or conditions.  The name of the function arguments should
    share the same name as the relevant parameter or condition.  For example, to
    scale the drift rate by a parameter named "drift" and a condition named
    "coherence", set

        drift = lambda drift, coherence : drift*coherence

    In more detail, the six model components that can be specified this way:

    - `drift`: The drift rate. In addition to parameters and conditions, you can
      also use "t" to specify the current point in time, and "x" to specify the
      current position in space.  (These can be used time-dependent drift rate
      and leaky/unstable integration, respectively.)  All drift rate functions
      supported by PyDDM are supported by this function.

    - `noise`: The standard deviation of the noise. In addition to parameters and
      conditions, you can also use "t" to specify the current point in time, and
      "x" to specify the current position in space.  All noise functions
      supported by PyDDM are supported by this function.

    - `bound`: The bound height as distance from the center.  Therefore, total
      bound height is twice this value. In addition to parameters and
      conditions, you can also use "t" to specify the current point in time
      (This can be used for collapsing or expanding bounds.)  Default value is a
      constant bound of height 1 (hence, total separation between the bounds is
      2).  All bound heights supported by PyDDM are supported by this function.

    - `starting_point`: Starting point bias.  Positive values are a bias towards
      the upper bound, and negative values are biased towards the lower bound.
      Can be specified by parameters and conditions.  Default value is 0 (no
      bias), and the maximum and minimum values are +1 (top bound) and -1 (lower
      bound), respectively.  In addition to accepting parameters and conditions,
      the starting point function can also accept the argument "x", the vector
      of all positions in space.  To use a distribution of starting points,
      return a vector the same size as "x".  Otherwise, if a single value is
      returned, it will be assumed to be a single point.

    - `nondecision`: The non-decision time.  Can be specified by parameters and
      conditions.  Default value is 0.  Non-decision time may be negative.  In
      addition to parameters and conditions, the function for non-decision time
      may also accept the argument "T", which is a vector of times from -T_dur
      to +T_dur, spaced by dx.  To return a distribution of non-decision times,
      return a vector of the same size as "T".  Otherwise, non-decision time
      will be assumed to be a single point.

    - `mixturecoef`: The uniform distribution mixture model, used to allow
      likelihood fitting.  Can not be given by a function, and must be either a
      constant or a parameter.  The uniform mixture model can be disabled by
      setting this to 0.

    Other parameters:

    - T_dur: the duration of the simulation, in units of seconds.  Default is 2.0 sec.
    - dx and dt: Numerical simulation parameters, in units of seconds.  Default is 0.005.
    - choice_names: The names of the upper and lower bound, as a tuple.  Default is ("correct", "error").
    - name: The name of this model.  Defaults to "".

    Note on functions which depend on x (the position in space): For performance
    reasons, this variable is always passed as a numpy array of x values.  So,
    make sure your function can accept a vector of x.  Parameters, conditions,
    and "t" (the current time) are passed as floats.

    """
    assert isinstance(parameters, dict), "Parameters must be a dictionary, with keys containing the parameter name and values containing the value of the parameter or the range of valid parameter values for fitting."
    for c in conditions:
        assert c not in parameters.keys(), f"Condition and parameter cannot have the same name.  Invalid name '{c}'."
        assert c not in ["x", "t", "T", "dx"], f"Condition cannot be named 'x', 't', 'T', or 'dx'.  Invalid name '{c}'."
        assert c.isidentifier() and not keyword.iskeyword(c), f"Condition names must be valid names for variables in Python.  Invalid name '{c}'."

    for name,p in parameters.items():
        assert name not in ["x", "t", "T", "dx"], f"Parameters cannot be named 'x', 't', 'T', or 'dx'.  Invalid name '{name}'."
        assert isinstance(p, (int,float,np.float_,np.int_)) or (isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], (float,int,np.int_,np.float_)) and isinstance(p[1], (float,int,np.int_,np.float_))), f"Parameters must be a single number or a tuple of numbers (representing a range for fitting).  Invalid parameter '{name}'."
        assert name.isidentifier() and not keyword.iskeyword(name), f"Parameter names must be valid names for variables in Python.  Invalid parameter name '{name}'."

    # Either the fittable or the constant value
    _fittables = {pn : Fittable(minval=pv[0], maxval=pv[1]) if isinstance(pv, tuple) else pv for pn,pv in parameters.items()}
    def _parse_dep(val, name, special="xt"):
        """Determine whether `val` can be turned into a PyDDM model, and if so, parse relevant information.

        `name` is the dependence name for error message outputs.
        `special` is either "", "x", "t", "T", or "xt", describing what the dependence supports.
        """
        if val in conditions:
            val = eval(f"lambda {val}: {val}")
        if val in parameters.keys():
            return "val",None,_fittables[val]
        elif isinstance(val, (int,float,np.float_,np.int_)):
            return "val",None,val
        elif hasattr(val, "__call__"):
            sig = inspect.getfullargspec(val)
            assert len(sig.kwonlyargs) == 0, f"Keyword only args not supported for {name}"
            assert sig.varkw is None, f"Keyword only args not supported for {name}"
            assert sig.varargs is None, f"Variable arguments not supported for {name}"
            assert sig.defaults is None, f"Default arguments not supported for {name}"
            _required_conditions = []
            _required_parameters = []
            _required_xt = []
            for arg in sig.args:
                assert arg != "dx", f"{name} cannot depend on dx"
                if arg in parameters.keys():
                    _required_parameters.append(arg)
                elif arg in conditions:
                    _required_conditions.append(arg)
                elif arg in ["x", "t", "T"]:
                    _descr = {"x": "the vector of particle positions",
                              "t": "the current time in the simulation",
                              "T": "the vector of all time points in the simulation"}
                    assert arg in special, f"In PyDDM, the '{arg}' argument usually indicates {_descr[arg]}, but this argument cannot be used in the {name} function."
                    _required_xt.append(arg)
                else:
                    raise ValueError(f"Invalid argument '{arg}' to the {name} function.  All arguments to {name} must be parameters, conditions, or the special value{'s' if len(special)>1 else ''} {' or '.join(list(special))}.  Did you forget to add '{arg}' to the list of parameters or conditions passed to the model?")
            return "func", (_required_parameters,_required_conditions,_required_xt), val
        else:
            raise ValueError(f"Invalid value for {name}, please provide a number, parameter (as a string), or function")

    typ, parsed, drift = _parse_dep(drift, "drift")
    # If drift is a parameter or value
    if typ == "val":
        driftobj = DriftConstant(drift=drift)
    # If it is a function
    elif typ == "func":
        _required_parameters_drift,_required_conditions_drift,_required_xt_drift = parsed
        class DriftEasy(Drift):
            name = "easy_drift"
            required_parameters = _required_parameters_drift
            required_conditions = _required_conditions_drift
            def get_drift(self, x, t, conditions, **kwargs):
                extras = {}
                if "t" in _required_xt_drift:
                    extras["t"] = t
                if "x" in _required_xt_drift:
                    extras["x"] = x
                return drift(**{v: getattr(self, v) for v in _required_parameters_drift}, **{v: conditions[v] for v in _required_conditions_drift}, **extras)
            def _uses_t(self):
                return "t" in _required_xt_drift
            def _uses_x(self):
                return "x" in _required_xt_drift
        driftobj = DriftEasy(**{fname:fval for fname,fval in _fittables.items() if fname in _required_parameters_drift})

    typ, parsed, noise = _parse_dep(noise, "noise")
    # If noise is a parameter or value
    if typ == "val":
        noiseobj = NoiseConstant(noise=noise)
    # If it is a function
    elif typ == "func":
        _required_parameters_noise,_required_conditions_noise,_required_xt_noise = parsed
        class NoiseEasy(Noise):
            name = "easy_noise"
            required_parameters = _required_parameters_noise
            required_conditions = _required_conditions_noise
            def get_noise(self, x, t, conditions, **kwargs):
                extras = {}
                if "t" in _required_xt_noise:
                    extras["t"] = t
                if "x" in _required_xt_noise:
                    extras["x"] = x
                return noise(**{v: getattr(self, v) for v in _required_parameters_noise}, **{v: conditions[v] for v in _required_conditions_noise}, **extras)
            def _uses_t(self):
                return "t" in _required_xt_noise
            def _uses_x(self):
                return "x" in _required_xt_noise
        noiseobj = NoiseEasy(**{fname:fval for fname,fval in _fittables.items() if fname in _required_parameters_noise})

    typ, parsed, bound = _parse_dep(bound, "bound", "t")
    # If it is a parameter or value
    if typ == "val":
        boundobj = BoundConstant(B=bound)
    # If it is a function
    elif typ == "func":
        _required_parameters_bound,_required_conditions_bound,_required_t_bound = parsed
        class BoundEasy(Bound):
            name = "easy_bound"
            required_parameters = _required_parameters_bound
            required_conditions = _required_conditions_bound
            def get_bound(self, t, conditions, **kwargs):
                extras = {"t": t} if "t" in _required_t_bound else {}
                return bound(**{v: getattr(self, v) for v in _required_parameters_bound}, **{v: conditions[v] for v in _required_conditions_bound}, **extras)
            def _uses_t(self):
                return "t" in _required_t_bound
        boundobj = BoundEasy(**{fname:fval for fname,fval in _fittables.items() if fname in _required_parameters_bound})

    typ, parsed, starting_position = _parse_dep(starting_position, "starting_position", "x")
    # If starting_position is a parameter or value
    if typ == "val":
        icobj = ICPointRatio(x0=starting_position)
    # If it is a function
    elif typ == "func":
        _required_parameters_x0,_required_conditions_x0,_required_x_x0 = parsed
        if "x" in _required_x_x0:
            class ICDistributionEasy(InitialCondition):
                name = "easy_distribution_initial_conditions"
                required_parameters = _required_parameters_x0
                required_conditions = _required_conditions_x0
                def get_IC(self, x, dx, conditions):
                    return starting_position(**{v: getattr(self, v) for v in _required_parameters_x0}, **{v: conditions[v] for v in _required_conditions_x0}, x=x)
            icobj = ICDistributionEasy(**{fname:fval for fname,fval in _fittables.items() if fname in _required_parameters_x0})
        else:
            class ICPointRatioEasy(ICPointRatio):
                name = "easy_starting_point"
                required_parameters = _required_parameters_x0
                required_conditions = _required_conditions_x0
                def get_starting_point(self, conditions):
                    return starting_position(**{v: getattr(self, v) for v in _required_parameters_x0}, **{v: conditions[v] for v in _required_conditions_x0})
            icobj = ICPointRatioEasy(**{fname:fval for fname,fval in _fittables.items() if fname in _required_parameters_x0})

    overlayobjs = []
    typ, parsed, nondecision = _parse_dep(nondecision, "nondecision", "T")
    # If nondecision is a parameter or value
    if typ == "val":
        overlayobjs.append(OverlayNonDecision(nondectime=nondecision))
    # If it is a function
    elif typ == "func":
        _required_parameters_nd,_required_conditions_nd,_required_T_overlay = parsed
        if "T" in _required_T_overlay: # Distribution
            class OverlayNonDecisionDistributionEasy(OverlayNonDecision):
                name = "easy_distribution_nondecision"
                required_parameters = _required_parameters_nd
                required_conditions = _required_conditions_nd
                def apply(self, solution):        # Make sure params are within range
                    # Extract components of the solution object for convenience
                    choice_upper = solution.choice_upper
                    choice_lower = solution.choice_lower
                    conditions = solution.conditions
                    dt = solution.dt
                    # Create the weights for different timepoints
                    times = np.asarray(list(range(-len(choice_upper), len(choice_upper))))*dt
                    weights = nondecision(**{v: getattr(self, v) for v in _required_parameters_nd}, **{v: conditions[v] for v in _required_conditions_nd}, T=times)
                    assert len(weights) == len(times), "Invalid distribution of starting points, must be the same length as the argument T."
                    if np.sum(weights) > 0:
                        weights /= np.sum(weights) # Ensure it integrates to 1
                    newchoice_upper = np.convolve(weights, choice_upper, mode="full")[len(choice_upper):(2*len(choice_upper))]
                    newchoice_lower = np.convolve(weights, choice_lower, mode="full")[len(choice_upper):(2*len(choice_upper))]
                    return Solution(newchoice_upper, newchoice_lower, solution.model, solution.conditions, solution.undec)
            overlayobjs.append(OverlayNonDecisionDistributionEasy(**{fname:fval for fname,fval in _fittables.items() if fname in _required_parameters_nd}))
        else: # Single point
            class OverlayNonDecisionEasy(OverlayNonDecision):
                name = "easy_nondecision"
                required_parameters = _required_parameters_nd
                required_conditions = _required_conditions_nd
                def get_nondecision_time(self, conditions):
                    return nondecision(**{v: getattr(self, v) for v in _required_parameters_nd}, **{v: conditions[v] for v in _required_conditions_nd})
            overlayobjs.append(OverlayNonDecisionEasy(**{fname:fval for fname,fval in _fittables.items() if fname in _required_parameters_nd}))

    typ, parsed, mixture_coef = _parse_dep(mixture_coef, "mixture_coef", "")
    # If mixture coefficient is a parameter or value
    if typ == "val":
        overlayobjs.append(OverlayUniformMixture(umixturecoef=mixture_coef))
    # If it is a function
    elif typ == "func":
        raise ValueError("mixture_coef cannot be a function here, please use the full object oriented version of PyDDM for this functionality.")
    return Model(drift=driftobj, noise=noiseobj, bound=boundobj, IC=icobj, overlay=OverlayChain(overlays=overlayobjs), dx=dx, dt=dt, T_dur=T_dur, choice_names=choice_names)
