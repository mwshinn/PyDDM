# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

import logging
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from .tridiag import TriDiagMatrix

from . import parameters as param
from .analytic import analytic_ddm
from .models.drift import DriftConstant, Drift
from .models.noise import NoiseConstant, Noise
from .models.ic import ICPointSourceCenter, ICPoint, ICPointRatio, InitialCondition
from .models.bound import BoundConstant, BoundCollapsingLinear, Bound
from .models.overlay import OverlayNone, Overlay
from .models.paranoid_types import Conditions
from .sample import Sample
from .solution import Solution
from .fitresult import FitResult, FitResultEmpty
from .logger import logger as _logger

from paranoid.types import Numeric, Number, Self, List, Generic, Positive, Positive0, String, Boolean, Natural1, Natural0, Dict, Set, Integer, NDArray, Maybe, Nothing
from paranoid.decorators import accepts, returns, requires, ensures, paranoidclass, paranoidconfig
import dis

try:
    from . import csolve
    HAS_CSOLVE = True
except ImportError:
    HAS_CSOLVE = False

# "Model" describes how a variable is dependent on other variables.
# Principally, we want to know how drift and noise depend on x and t.
# `name` is the type of dependence (e.g. "linear") for methods which
# implement the algorithms, and any parameters these algorithms could
# need should be passed as kwargs. To compare to legacy code, the
# `name` used to be `f_mu_setting` or `f_sigma_setting` and kwargs now
# encompassed (e.g.) `param_mu_t_temp`.



##Pre-defined list of models that can be used, and the corresponding default parameters
@paranoidclass
class Model(object):
    """A full simulation of a single DDM-style model.

    Each model simulation depends on five key components:
    
    - A description of how drift rate (drift) changes throughout the simulation.
    - A description of how variability (noise) changes throughout the simulation.
    - A description of how the boundary changes throughout the simulation.
    - Starting conditions for the model
    - Specific details of a task which cause dynamic changes in the model (e.g. a stimulus intensity change)

    This class manages these, and also provides the affiliated
    services, such as analytical or numerical simulations of the
    resulting reaction time distribution.
    """
    # TODO it would be nice to have explicit "save" and "load"
    # functions, which just provide safe wrappers around saving and
    # loading text files with "repr" and "exec".
    @staticmethod
    def _test(v):
        assert v.get_dependence("drift") in Generic(Drift)
        assert v.get_dependence("noise") in Generic(Noise)
        assert v.get_dependence("bound") in Generic(Bound)
        assert v.get_dependence("IC") in Generic(InitialCondition)
        assert v.get_dependence("overlay") in Generic(Overlay)
        assert v.dx in Positive()
        assert v.dt in Positive()
        assert v.T_dur in Positive()
        assert v.name in String()
        assert v.required_conditions in List(String)
        assert v.fitresult in Generic(FitResult)
    @staticmethod
    def _generate():
        # TODO maybe generate better models?
        # TODO test a model where the bound size changes based on a parameter
        #yield Model(dx=.01, dt=.01, T_dur=2)
        yield Model() # Default model
        yield Model(dx=.05, dt=.01, T_dur=3) # Non-default numerics
        #yield Model(dx=.005, dt=.005, T_dur=.5)
    def __init__(self, drift=DriftConstant(drift=0),
                 noise=NoiseConstant(noise=1),
                 bound=BoundConstant(B=1),
                 IC=ICPointSourceCenter(),
                 overlay=OverlayNone(), name="",
                 dx=param.dx, dt=param.dt,
                 T_dur=param.T_dur, fitresult=None,
                 choice_names=param.choice_names):
        """Construct a Model object from the 5 key components.

        The five key components of our DDM-style models describe how
        the drift rate (`drift`), noise (`noise`), and bounds (`bound`)
        change over time, and the initial conditions (`IC`).

        These five components are given by the parameters `drift`,
        `noise`, `bound`, and `IC`, respectively.  They should be
        types which inherit from the types Drift, Noise, Bound, and
        InitialCondition, respectively.  They default to constant
        unitary values.

        Additionally, simulation parameters can be set, such as time
        and spacial precision (`dt` and `dx`) and the simulation
        duration `T_dur`.  If not specified, they will be taken from
        the defaults specified in the parameters file.

        If you are creating a model, `fitresult` should always be
        None.  This is provided as an optional parameter because when
        models are output as a string (using "str" or "repr"), they
        must save fitting information.  Thus, this allows you to fit a
        model, convert it to a string, save that string to a text
        file, and then run "exec" on that file in a new script to load
        the model.

        By default, the choice associated with the upper boundary is "correct
        responses" and the lower boundary is "error responses".  To change
        these, set the `choice_names` argument to be a tuple containing two
        strings, with the names of the boundaries.  So the default is
        ("correct", "error"), but could be anything, e.g. ("left", "right"),
        ("high value" and "low value"), etc.  This is sometimes referred to as
        "accuracy coding" and "stimulus coding".  When fitting data, this must
        match the choice names of the sample.

        The `name` parameter is exclusively for convenience, and may
        be used in plotting or in debugging.

        """
        assert isinstance(name, str)
        self.name = name
        assert isinstance(drift, Drift)
        self._driftdep = drift
        assert isinstance(noise, Noise)
        self._noisedep = noise
        assert isinstance(bound, Bound)
        self._bounddep = bound
        assert isinstance(IC, InitialCondition)
        self._IC = IC
        assert isinstance(overlay, Overlay)
        self._overlay = overlay
        self.dependencies = [self._driftdep, self._noisedep, self._bounddep, self._IC, self._overlay]
        self.required_conditions = list(set([x for l in self.dependencies for x in l.required_conditions]))
        assert isinstance(choice_names, tuple) and len(choice_names) == 2, "choice_names must be a tuple of length 2"
        self.choice_names = choice_names
        self.dx = dx
        self.dt = dt
        if self.dx > .01:
            _logger.warning("dx is large.  Estimated pdfs may be imprecise.  Decrease dx to 0.01 or less.")
        if self.dt > .01:
            _logger.warning("dt is large.  Estimated pdfs may be imprecise.  Decrease dt to 0.01 or less.")
        self.T_dur = T_dur
        self.fitresult = FitResultEmpty() if fitresult is None else fitresult # If the model was fit, store the status here
    def __eq__(self, other):
        for i in range(0, len(self.dependencies)):
            if self.dependencies[i] != other.dependencies[i]:
                return False
        if self.dx != other.dx:
            return False
        if self.dt != other.dt:
            return False
        if self.fitresult != other.fitresult:
            return False
        if self.name != other.name:
            return False
        if self.choice_names != other.choice_names:
            return False
        return True

    # Get a string representation of the model
    def __repr__(self, pretty=False):
        # Use a list so they can be sorted
        allobjects = [("name", self.name), ("drift", self.get_dependence('drift')),
                      ("noise", self.get_dependence('noise')), ("bound", self.get_dependence('bound')),
                      ("IC", self.get_dependence('ic')),
                      ("overlay", self.get_dependence('overlay')), ("dx", self.dx),
                      ("dt", self.dt), ("T_dur", self.T_dur)]
        params = ""
        for n,o in allobjects:
            params += n + "=" + o.__repr__()
            if (n,o) != allobjects[-1]:
                if pretty:
                    params += ",\n" + " "*(len(type(self).__name__)+1)
                else:
                    params += ", "
        if not isinstance(self.fitresult, FitResultEmpty):
            if pretty:
                params += ",\n  fitresult=" + repr(self.fitresult)
            else:
                params += ", fitresult=" + repr(self.fitresult)
        if self.choice_names != ("correct", "error"):
            if pretty:
                params += ",\n  choice_names="+repr(self.choice_names)
            else:
                params += ", choice_names="+repr(self.choice_names)
        return type(self).__name__ + "(" + params + ")"
    def __str__(self):
        return self.__repr__(pretty=True)
    def show(self, print_output=True):
        """Show information about the model"""
        from .functions import display_model
        return display_model(self, print_output=print_output)
    def fit(self, sample, fitparams=None, fitting_method="differential_evolution",
            lossfunction=None, verify=False, method=None, verbose=True):
        """Fit a model to data.

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

        If `verify` is False (the default), checking for programming
        errors is disabled during the fit. This can decrease runtime and
        may prevent crashes.  If verification is already disabled, this
        does not re-enable it.

        `method` gives the method used to solve the model, and can be
        "analytical", "numerical", "cn", "implicit", or "explicit".

        `verbose` enables out-of-boundaries warnings and prints the model
        information at each evaluation of the fitness function.

        No return value.

        After running this function, the model will be modified to include
        a "FitResult" object, accessed as m.fitresult.  This can be used
        to get the value of the objective function, as well as to access
        diagnostic information about the fit.

        This function will automatically parallelize if set_N_cpus() has
        been called.

        """
        from .functions import fit_adjust_model
        from .models.loss import LossLikelihood
        if lossfunction is None:
            lossfunction = LossLikelihood
        fit_adjust_model(sample=sample, model=self, fitparams=fitparams,
                         fitting_method=fitting_method,
                         lossfunction=lossfunction, verify=verify,
                         method=method, verbose=verbose)
    def parameters(self):
        """Return all parameters in the model

        This will return a dictionary of dictionaries.  The keys of this
        dictionary will be "drift", "noise", "bound", "IC", and "overlay".  The
        values will be dictionaries, each containing the parameters used by
        these.  Note that this includes both fixed parameters and Fittable
        parameters.  If a parameter is fittable, it will return either a
        Fittable object or a Fitted object in place of the parameter, depending
        on whether or not the model has been fit to data yet.
        """
        ret = {}
        for depname in ["drift", "noise", "bound", "IC", "overlay"]:
            ret[depname] = {}
            dep = self.get_dependence(depname)
            for param_name in dep.required_parameters:
                param_value = getattr(dep, param_name)
                ret[depname][param_name] = param_value
        return ret

    def get_model_parameters(self):
        """Get an ordered list of all model parameters.
        
        Returns a list of each model parameter which can be varied
        during a fitting procedure.  The ordering is arbitrary but is
        guaranteed to be in the same order as
        get_model_parameter_names() and set_model_parameters().  If
        multiple parameters refer to the same "Fittable" object, then
        that object will only be listed once.
        """
        params = []
        for dep in self.dependencies:
            for param_name in dep.required_parameters:
                param_value = getattr(dep, param_name)
                # If this can be fit to data
                if isinstance(param_value, Fittable):
                    # be fit) via optimization If we have the same
                    # Fittable object in two different components
                    # inside the model, we only want to list it once.
                    if id(param_value) not in map(id, params):
                        params.append(param_value)
        return params

    def get_model_parameter_names(self):
        """Get an ordered list of the names of all parameters in the model.

        Returns the name of each model parameter.  The ordering is
        arbitrary, but is uaranteed to be in the same order as
        get_model_parameters() and set_model_parameters(). If multiple
        parameters refer to the same "Fittable" object, then that
        object will only be listed once, however the names of the
        parameters will be separated by a "/" character.
        """
        params = []
        param_names = []
        for dep in self.dependencies:
            for param_name in dep.required_parameters:
                param_value = getattr(dep, param_name)
                # If this can be fit to data
                if isinstance(param_value, Fittable):
                    # If we have the same Fittable object in two
                    # different components inside the model, we only
                    # want to list it once.
                    if id(param_value) not in map(id, params):
                        param_names.append(param_name)
                        params.append(param_value)
                    else:
                        ind = list(map(id, params)).index(id(param_value))
                        if param_name not in param_names[ind].split("/"):
                            param_names[ind] += "/" + param_name
        return param_names

    def set_model_parameters(self, params):
        """Set the parameters of the model from an ordered list.

        Takes as an argument a list of parameters in the same order as
        those from get_model_parameters().  Sets the associated
        parameters as a "Fitted" object. If multiple parameters refer
        to the same "Fittable" object, then that object will only be
        listed once.
        """
        old_params = self.get_model_parameters()
        param_object_ids = list(map(id, old_params))
        assert len(params) == len(param_object_ids), "Invalid number of parameters specified: " \
            "got %i, expected %i" % (len(params), len(param_object_ids))
        new_params = [p if isinstance(p, Fittable) else op.make_fitted(p) \
                      for p,op in zip(params, old_params)]
        for dep in self.dependencies:
            for param_name in dep.required_parameters:
                param_value = getattr(dep, param_name)
                # If this can be fit to data
                if isinstance(param_value, Fittable):
                    i = param_object_ids.index(id(param_value))
                    setattr(dep, param_name, new_params[i])


    def get_fit_result(self):
        """Returns a FitResult object describing how the model was fit.
        
        Returns the FitResult object describing the last time this
        model was fit to data, including the loss function, fitting
        method, and the loss function value.  If the model was never
        fit to data, this will return FitResultEmpty.
        """
        return self.fitresult
    
    def get_dependence(self, name):
        """Return the dependence object given by the string `name`."""
        if name.lower() in ["drift", "driftdep", "_driftdep"]:
            return self._driftdep
        elif name.lower() in ["noise", "noisedep", "_noisedep"]:
            return self._noisedep
        elif name.lower() in ["b", "bound", "bounddep", "_bounddep"]:
            return self._bounddep
        elif name.lower() in ["ic", "initialcondition", "_ic"]:
            return self._IC
        elif name.lower() in ["overlay", "_overlay"]:
            return self._overlay
        raise NameError("Invalid dependence name")

    def check_conditions_satisfied(self, conditions):
        rc = list(sorted(self.required_conditions))
        ck = list(sorted(conditions.keys()))
        assert set(rc) - set(ck) == set(), \
            "Please specify valid conditions for this simulation.\nSpecified: %s\nExpected: %s" % (str(ck), str(rc))

    def get_model_type(self):
        """Return a dictionary which fully specifies the class of the five key model components."""
        tt = lambda x : (x.depname, type(x))
        return dict(map(tt, self.dependencies))
    @accepts(Self, Conditions, Maybe(Positive0))
    def x_domain(self, conditions, t=None):
        """A list which spans from the lower boundary to the upper boundary by increments of dx."""
        # Find the maximum size of the bound across the t-domain in
        # case we have increasing bounds
        if t is None:
            B = max([self.get_dependence("bound").get_bound(t=t, conditions=conditions) for t in self.t_domain()])
        else:
            B = self.get_dependence("bound").get_bound(t=t, conditions=conditions)
        B = np.ceil(B/self.dx)*self.dx # Align the bound to dx borders
        return np.arange(-B, B+0.1*self.dx, self.dx) # +.1*dx is to ensure that the largest number in the array is B
    def t_domain(self):
        """A list of all of the timepoints over which the joint PDF will be defined (increments of dt from 0 to T_dur)."""
        return np.arange(0., self.T_dur+0.1*self.dt, self.dt)
    def flux(self, x, t, conditions):
        """The flux across the boundary at position `x` at time `t`."""
        drift_flux = self.get_dependence('drift').get_flux(x, t, dx=self.dx, dt=self.dt, conditions=conditions)
        noise_flux = self.get_dependence('noise').get_flux(x, t, dx=self.dx, dt=self.dt, conditions=conditions)
        return drift_flux + noise_flux
    def IC(self, conditions):
        """The initial distribution at t=0.

        Returns a length N ndarray (where N is the size of x_domain())
        which should sum to 1.
        """
        return self.get_dependence('IC').get_IC(self.x_domain(conditions=conditions), dx=self.dx, conditions=conditions)

    @accepts(Self, conditions=Conditions, cutoff=Boolean, seed=Natural0, rk4=Boolean)
    @returns(NDArray(t=Number, d=1))
    @ensures('0 < len(return) <= len(self.t_domain())')
    @ensures('not cutoff --> len(return) == len(self.t_domain())')
    def simulate_trial(self, conditions={}, cutoff=True, rk4=True, seed=0):
        """Simulate the decision variable for one trial.

        Given conditions `conditions`, this function will simulate the
        decision variable for a single trial.  It will cut off the
        simulation when the decision variable crosses the boundary
        unless `cutoff` is set to False.  By default, Runge-Kutta is
        used to simulate the trial, however if `rk4` is set to False,
        the less efficient Euler's method is used instead. This
        returns a trajectory of the simulated trial over time as a
        numpy array.

        Note that this will return the same trajectory on each run
        unless the random seed `seed` is varied.

        Also note that you shouldn't normally need to use this
        function.  To simulate an entire probability distributions,
        call Model.solve() and the results of the simulation will be
        in the returned Solution object.  This is only useful for
        finding individual trajectories instead of the probability
        distribution as a whole.

        """
        self.check_conditions_satisfied(conditions)
        
        h = self.dt
        T = self.t_domain()

        # Choose a starting position from the IC
        rng = np.random.RandomState(seed)
        ic = self.IC(conditions=conditions)
        x0 = rng.choice(self.x_domain(conditions=conditions), p=ic)
        pos = [x0]

        # Convenience functions
        _driftdep = self.get_dependence("drift")
        _noisedep = self.get_dependence("noise")
        fm = lambda x,t : _driftdep.get_drift(t=t, x=x, conditions=conditions)
        fs = lambda x,t : _noisedep.get_noise(t=t, x=x, conditions=conditions)
        
        for i in range(1, len(T)):
            # Stochastic Runge-Kutta order 4.  See "Introduction to
            # Stochastic Differential Equations" by Thomas C. Gard
            rn = rng.randn()
            dw = np.sqrt(h)*rn
            drift1 = fm(t=T[i-1], x=pos[i-1])
            s1  = fs(t=T[i-1], x=pos[i-1])
            if rk4: # Use Runge-Kutta order 4
                drift2 = fm(t=(T[i-1]+h/2), x=(pos[i-1] + drift1*h/2 + s1*dw/2)) # Should be dw/sqrt(2)?
                s2  = fs(t=(T[i-1]+h/2), x=(pos[i-1] + drift1*h/2 + s1*dw/2))
                drift3 = fm(t=(T[i-1]+h/2), x=(pos[i-1] + drift2*h/2 + s2*dw/2))
                s3  = fs(t=(T[i-1]+h/2), x=(pos[i-1] + drift2*h/2 + s2*dw/2))
                drift4 = fm(t=(T[i-1]+h), x=(pos[i-1] + drift3*h + s3*dw)) # Should this be 1/2*s3*dw?
                s4  = fs(t=(T[i-1]+h), x=(pos[i-1] + drift3*h + s3*dw)) # Should this be 1/2*s3*dw?
                dx = h*(drift1 + 2*drift2 + 2*drift3 + drift4)/6 + dw*(s1 + 2*s2 + 2*s3 + s4)/6
            else: # Use Euler's method
                dx = h*drift1 + dw*s1
            pos.append(pos[i-1] + dx)
            B = self.get_dependence("bound").get_bound(t=T[i], conditions=conditions)
            if cutoff and (pos[i] > B or pos[i] < -B):
                break

        traj = self.get_dependence("overlay").apply_trajectory(trajectory=np.asarray(pos), model=self, seed=seed, rk4=rk4, conditions=conditions)
        if cutoff is False and len(traj) < len(T):
            traj = np.append(traj, [traj[-1]]*(len(T)-len(traj)))
        if len(traj) > len(T):
            traj = traj[0:len(T)]
        return traj


    @accepts(Self, Conditions, Natural1, Boolean, Natural0)
    @returns(Sample)
    @paranoidconfig(max_runtime=.1)
    def simulated_solution(self, conditions={}, size=1000, rk4=True, seed=0):
        """Simulate individual trials to obtain a distribution.

        Given conditions `conditions` and the number `size` of trials
        to simulate, this will run the function "simulate_trial"
        `size` times, and use the result to find a histogram analogous
        to solve.  Returns a Sample object.

        Note that in practice you should never need to use this
        function.  This function uses an outdated method to simulate
        the model and should be used for comparison perposes only.  To
        produce a probability density function of boundary crosses,
        use Model.solve().  To sample from the probability
        distribution (e.g. for finding confidence intervals for
        limited amounts of data), call Model.solve() and then use the
        Solution.sample() function of the resulting Solution.

        """
        _logger.warning("To generate a sample from a model, please use Solution.sample().  The only practical purpose of the simulated_solution function is debugging the simulate_trial function for custom Overlays.")
        choice_upper_times = []
        choice_lower_times = []
        undec_count = 0

        T = self.t_domain()

        for s in range(0, size):
            if s % 200 == 0:
                _logger.info("Simulating trial %i" % s)
            timecourse = self.simulate_trial(conditions=conditions, seed=(hash((s, seed)) % 2**32), cutoff=True, rk4=rk4)
            T_finish = T[len(timecourse) - 1]
            B = self.get_dependence("bound").get_bound(t=T_finish, conditions=conditions)
            # Correct for the fact that the particle could have
            # crossed at any point between T_finish-dt and T_finish.
            dt_correction = self.dt/2
            # Determine whether the sim is a correct or error trial.
            if timecourse[-1] > B:
                choice_upper_times.append(T_finish - dt_correction)
            elif timecourse[-1] < -B:
                choice_lower_times.append(T_finish - dt_correction)
            elif len(timecourse) == len(T):
                undec_count += 1
            else:
                raise SystemError("Internal error: Invalid particle simulation")
            
        aa = lambda x : np.asarray(x)
        conds = {k:(aa(len(choice_upper_times)*[v]), aa(len(choice_lower_times)*[v]), aa(undec_count*[v])) for k,v in conditions.items() if k and v}
        return Sample(aa(choice_upper_times), aa(choice_lower_times), undec_count, **conds)

    @accepts(Self)
    @returns(Boolean)
    def has_analytical_solution(self):
        """Is it possible to find an analytic solution for this model?"""
        # First check to make sure drift doesn't vary with time or
        # particle location
        if self.get_dependence("drift")._uses_t() or self.get_dependence("drift")._uses_x():
            return False
        # Check noise to make sure it doesn't vary with time or particle location
        if self.get_dependence("noise")._uses_t() or self.get_dependence("noise")._uses_x():
            return False
        # Check to make sure bound is one that we can solve for
        if self.get_dependence("bound")._uses_t() and self.get_dependence("bound").__class__ != BoundCollapsingLinear:
            return False
        # Make sure initial condition is a single point
        if not issubclass(self.get_dependence("IC").__class__,(ICPointSourceCenter,ICPoint,ICPointRatio)):
            return False
        # Assuming none of these is the case, return True.
        return True

    @accepts(Self, conditions=Conditions)
    @returns(Boolean)
    def can_solve_explicit(self, conditions={}):
        """Check explicit method stability criterion"""
        self.check_conditions_satisfied(conditions)
        noise_max = max((self._noisedep.get_noise(x=0, t=t, dx=self.dx, dt=self.dt, conditions=conditions) for t in self.t_domain()))
        return noise_max**2 * self.dt/(self.dx**2) < 1

    @accepts(Self, conditions=Conditions)
    @returns(Boolean)
    def can_solve_cn(self, conditions={}):
        """Check whether this model is compatible with Crank-Nicolson solver.

        All bound functions which do not depend on time are compatible."""
        # TODO in the future, instead of looking for parameters this
        # way, we should use "t in [i.argrepr for i in
        # dis.get_instructions(get_bound)]" to see if it is used in the
        # function rather than looking to see if it is passed to the
        # function.
        if self.get_dependence("bound")._uses_t():
            return False
        return True
    
    @accepts(Self, conditions=Conditions, return_evolution=Boolean, force_python=Boolean)
    @returns(Solution)
    def solve(self, conditions={}, return_evolution=False, force_python=False):
        """Solve the model using an analytic solution if possible, and a numeric solution if not.

        First, it tries to use Crank-Nicolson as the solver, and then backward
        Euler.  See documentation of Model.solve_numerical() for more information.

        The return_evolution argument should be set to True if you need to use
        the Solution.get_evolution() function from the returned Solution.

        Return a Solution object describing the joint PDF distribution of reaction times.

        """
        # TODO solves this using the dis module as described in the
        # comment for can_solve_cn
        self.check_conditions_satisfied(conditions)
        if self.has_analytical_solution() and return_evolution is False:
            return self.solve_analytical(conditions=conditions)
        elif isinstance(self.get_dependence("bound"), BoundConstant) and return_evolution is False and (force_python or not HAS_CSOLVE):
            return self.solve_numerical_cn(conditions=conditions)
        else:
            return self.solve_numerical_implicit(conditions=conditions, return_evolution=return_evolution, force_python=force_python)

    @accepts(Self, conditions=Conditions, force_python=Boolean)
    @returns(Solution)
    def solve_analytical(self, conditions={}, force_python=False):
        """Solve the model with an analytic solution, if possible.

        Analytic solutions are only possible in a select number of
        special cases; in particular, it works for simple DDM and for
        linearly collapsing bounds and arbitrary single-point initial 
        conditions. (See Anderson (1960) for implementation details.)  
        For most reasonably complex models, the method will fail.  
        Check whether a solution is possible with has_analytic_solution().

        If successful, this returns a Solution object describing the
        joint PDF.  If unsuccessful, this will raise an exception.
        """
        assert self.has_analytical_solution(), "Cannot solve for this model analytically"
        self.check_conditions_satisfied(conditions)
        
        #calculate shift in initial conditions if present
        if isinstance(self.get_dependence('IC'),(ICPoint, ICPointRatio)):
            ic = self.IC(conditions=conditions)
            assert np.count_nonzero(ic)==1, "Cannot solve analytically for models with non-point initial conditions"
            shift = np.flatnonzero(ic) / (len(ic) - 1) #rescale to proprotion of total bound height
        else:
            shift = None
        
        # The analytic_ddm function does the heavy lifting.
        if isinstance(self.get_dependence('bound'), BoundCollapsingLinear): # Linearly Collapsing Bound
            anal_pdf_choice_upper, anal_pdf_choice_lower = analytic_ddm(self.get_dependence("drift").get_drift(t=0, x=0, conditions=conditions),
                                                       self.get_dependence("noise").get_noise(t=0, x=0, conditions=conditions),
                                                       self.get_dependence("bound").get_bound(t=0, x=0, conditions=conditions),
                                                       self.t_domain(), shift, -self.get_dependence("bound").t,
                                                       force_python=force_python) # TODO why must this be negative? -MS
        else: # Constant bound DDM
            anal_pdf_choice_upper, anal_pdf_choice_lower = analytic_ddm(self.get_dependence("drift").get_drift(t=0, x=0, conditions=conditions),
                                                        self.get_dependence("noise").get_noise(t=0, x=0, conditions=conditions),
                                                       self.get_dependence("bound").get_bound(t=0, x=0, conditions=conditions), 
                                                       self.t_domain(), shift,
                                                       force_python=force_python)

        ## Remove some abnormalities such as NaN due to trivial reasons.
        anal_pdf_choice_upper[anal_pdf_choice_upper==np.NaN] = 0. # FIXME Is this a bug? You can't use == to compare nan to nan...
        anal_pdf_choice_upper[0] = 0.
        anal_pdf_choice_lower[anal_pdf_choice_lower==np.NaN] = 0.
        anal_pdf_choice_lower[0] = 0.

        # Fix numerical errors
        anal_pdf_choice_upper *= self.dt
        anal_pdf_choice_lower *= self.dt
        pdfsum = np.sum(anal_pdf_choice_upper) + np.sum(anal_pdf_choice_lower)
        if pdfsum > 1:
            if pdfsum > 1.01 and param.renorm_warnings:
                _logger.warning(("Renormalizing probability density from " + str(pdfsum) + " to 1."
                    + "  Try decreasing dt.  If that doesn't eliminate this warning, it may be due"
                    + " to extreme parameter values and/or bugs in your model spefication."))
                _logger.debug(self.parameters())
            anal_pdf_choice_upper /= pdfsum
            anal_pdf_choice_lower /= pdfsum

        sol = Solution(anal_pdf_choice_upper, anal_pdf_choice_lower, self, conditions=conditions)
        return self.get_dependence('overlay').apply(sol)
    
    def solve_numerical_c(self, conditions={}):
        """Solve the DDM model using the implicit method with C extensions.

        This function should give near identical results to
        solve_numerical_implicit.  However, it uses compiled C code instead of
        Python code to do so, which should make it much (10-100x) faster.

        This does not current work with non-Gaussian diffusion matrices (a
        currently undocumented feature).
        """

        get_drift = self.get_dependence("drift").get_drift
        drift_uses_t = self.get_dependence("drift")._uses_t()
        drift_uses_x = self.get_dependence("drift")._uses_x()
        if not drift_uses_t and not drift_uses_x:
            drifttype = 0
            drift = np.asarray([get_drift(conditions=conditions, x=0, t=0)])
        elif drift_uses_t and not drift_uses_x:
            drifttype = 1
            drift = np.asarray([get_drift(t=t, conditions=conditions, x=0) for t in self.t_domain()])
        elif not drift_uses_t and drift_uses_x:
            drifttype = 2
            drift = np.asarray(get_drift(x=self.x_domain(conditions=conditions), t=0, conditions=conditions))
        elif drift_uses_t and drift_uses_x:
            drifttype = 3
            # TODO: Right now this calculates and passes the maximum x domain,
            # even if it is not necessary to do so.  Performance could be
            # improved by only calculating the parts of x domain that are
            # needed and then finding a way to index that within C.
            # Alternatively, it could only calculate the ones it needs, and
            # then fill the extra space with nonsense values just to form a
            # rectangular matrix, since they will never be read within the C
            # code.  maxt is a workaround so we don't have to find the maximum
            # in the t domain on each iteration.
            maxt = self.t_domain()[np.argmax([self.get_dependence("bound").get_bound(t=t, conditions=conditions) for t in self.t_domain()])]
            xdomain = self.x_domain(t=maxt, conditions=conditions)
            drift = np.concatenate([get_drift(t=t, x=xdomain, conditions=conditions) for t in self.t_domain()])
        get_noise = self.get_dependence("noise").get_noise
        noise_uses_t = self.get_dependence("noise")._uses_t()
        noise_uses_x = self.get_dependence("noise")._uses_x()
        if not noise_uses_t and not noise_uses_x:
            noisetype = 0
            noise = np.asarray([get_noise(conditions=conditions, x=0, t=0)])
        elif noise_uses_t and not noise_uses_x:
            noisetype = 1
            noise = np.asarray([get_noise(t=t, conditions=conditions, x=0) for t in self.t_domain()])
        elif not noise_uses_t and noise_uses_x:
            noisetype = 2
            noise = np.asarray(get_noise(x=self.x_domain(conditions=conditions), conditions=conditions, t=0))
        elif noise_uses_t and noise_uses_x:
            noisetype = 3
            # See comment in drifttype = 3
            maxt = self.t_domain()[np.argmax([self.get_dependence("bound").get_bound(t=t, conditions=conditions) for t in self.t_domain()])]
            xdomain = self.x_domain(t=maxt, conditions=conditions)
            noise = np.concatenate([get_noise(t=t, x=xdomain, conditions=conditions) for t in self.t_domain()])
        bound_uses_t = self.get_dependence("bound")._uses_t()
        if not bound_uses_t:
            boundtype = 0
            bound = np.asarray([self.get_dependence("Bound").get_bound(conditions=conditions, t=0)])
        elif bound_uses_t:
            boundtype = 1
            bound = np.asarray([self.get_dependence("Bound").get_bound(t=t, conditions=conditions) for t in self.t_domain()])
        res = csolve.implicit_time(drift, drifttype, noise, noisetype, bound, boundtype, self.get_dependence("IC").get_IC(self.x_domain(conditions=conditions), self.dx, conditions=conditions), self.T_dur, self.dt, self.dx, len(self.t_domain()))
        # TODO: Handle the pdf going below zero, returning pdfcurr, and fix numerical errors
        choice_upper = (res[0]*self.dt)
        choice_upper[choice_upper<0] = 0
        choice_lower = (res[1]*self.dt)
        choice_lower[choice_lower<0] = 0
        undec = res[2]
        undec[undec<0] = 0
        return self.get_dependence('overlay').apply(Solution(choice_upper, choice_lower, self, conditions=conditions, pdf_undec=undec))

    @accepts(Self, method=Set(["explicit", "implicit", "cn"]), conditions=Conditions, return_evolution=Boolean, force_python=Boolean)
    @returns(Solution)
    @requires("method == 'explicit' --> self.can_solve_explicit(conditions=conditions)")
    @requires("method == 'cn' --> self.can_solve_cn()")
    def solve_numerical(self, method="cn", conditions={}, return_evolution=False, force_python=False):
        """Solve the DDM model numerically.

        Use `method` to solve the DDM.  `method` can either be
        "explicit", "implicit", or "cn" (for Crank-Nicolson).  This is
        the core DDM solver of this library.

        Crank-Nicolson is the default and works for any model with
        constant bounds.

        Implicit is the fallback method.  It should work well in most
        cases and is generally stable.

        Normally, the explicit method should not be used. Also note
        the stability criteria for explicit method is:

          | noise^2/2 * dt/dx^2 < 1/2

        It returns a Solution object describing the joint PDF.  This
        method should not fail for any model type.

        `return_evolution` (default=False) governs whether or not the function 
        returns the full evolution of the pdf as part of the Solution object. 
        This only works with methods "explicit" or "implicit", not with "cn".

        `force_python` makes PyDDM use the solver written in Python instead of
        the optimized solver written in C.

        """
        self.check_conditions_satisfied(conditions)
        if method == "cn":
            if return_evolution == False:
                return self.solve_numerical_cn(conditions=conditions)
            else:
                _logger.warning("return_evolution is not supported with the Crank-Nicolson solver, using implicit (backward Euler) instead.")
                method = "implicit"
        if method == "implicit" and HAS_CSOLVE and not force_python and not return_evolution:
            return self.solve_numerical_c(conditions=conditions)

        # Initial condition of decision variable
        pdf_curr = self.IC(conditions=conditions)
        # Output correct and error pdfs.  If pdf_corr + pdf_err +
        # undecided probability are summed, they equal 1.  So these
        # are componets of the joint pdf.
        pdf_choice_upper = np.zeros(len(self.t_domain())) # Not a proper pdf on its own (doesn't sum to 1)
        pdf_choice_lower = np.zeros(len(self.t_domain())) # Not a proper pdf on its own (doesn't sum to 1)
        x_list = self.x_domain(conditions=conditions)

        # If evolution of pdf should be returned, preallocate np.array pdf_evolution for performance reasons
        if return_evolution:
            pdf_evolution = np.zeros((len(x_list), len(self.t_domain())))
            pdf_evolution[:,0] = pdf_curr

        # Find maximum bound for increasing bounds
        _bound_func = self.get_dependence("bound").get_bound
        bmax = max([_bound_func(t=t, conditions=conditions) for t in self.t_domain()])
        # Looping through time and updating the pdf.
        for i_t, t in enumerate(self.t_domain()[:-1]): # -1 because nothing will happen at t=0 so each step computes the value for the next timepoint
            # Alias pdf_prev to pdf_curr for clarity
            pdf_prev = pdf_curr

            # For efficiency only do diffusion if there's at least
            # some densities remaining in the channel.
            if np.sum(pdf_curr[:]) < 0.0001:
                break
            
            # Boundary at current time-step.
            bound = self.get_dependence('bound').get_bound(t=t, conditions=conditions)

            # Now figure out which x positions are still within
            # the (collapsing) bound.
            assert bmax >= bound, "Invalid change in bound" # Ensure the bound didn't expand
            bound_shift = bmax - bound
            # Note that we linearly approximate the bound by the two surrounding grids sandwiching it.
            x_index_inner = int(np.ceil(bound_shift/self.dx)) # Index for the inner bound (smaller matrix)
            x_index_outer = int(np.floor(bound_shift/self.dx)) # Index for the outer bound (larger matrix)

            # We weight the upper and lower matrices according to
            # how far away the bound is from each.  The weight of
            # each matrix is approximated linearly. The inner
            # bound is 0 when bound exactly at grids.
            weight_inner = (bound_shift - x_index_outer*self.dx)/self.dx
            weight_outer = 1. - weight_inner # The weight of the lower bound matrix, approximated linearly.
            x_list_inbounds = x_list[x_index_outer:len(x_list)-x_index_outer] # List of x-positions still within bounds.

            # Diffusion Matrix for Implicit Method. Here defined as
            # Outer Matrix, and inder matrix is either trivial or an
            # extracted submatrix.
            drift_matrix = self.get_dependence('drift').get_matrix(x=x_list_inbounds, t=t,
                                                                   dt=self.dt, dx=self.dx,
                                                                   conditions=conditions,
                                                                   implicit=(method!="explicit"))
            noise_matrix = self.get_dependence('noise').get_matrix(x=x_list_inbounds, t=t,
                                                                   dt=self.dt, dx=self.dx, conditions=conditions,
                                                                   implicit=(method!="explicit"))
            if method == "implicit":
                diffusion_matrix = TriDiagMatrix.eye(len(x_list_inbounds)) + drift_matrix + noise_matrix
            elif method == "explicit":
                # Explicit method flips sign except for the identity matrix
                diffusion_matrix_explicit = TriDiagMatrix.eye(len(x_list_inbounds)) - drift_matrix - noise_matrix

            ### Compute Probability density functions (pdf)
            # PDF for outer matrix
            if method == "implicit":
                pdf_outer = diffusion_matrix.spsolve(pdf_prev[x_index_outer:len(x_list)-x_index_outer])
            elif method == "explicit":
                pdf_outer = diffusion_matrix_explicit.dot(pdf_prev[x_index_outer:len(x_list)-x_index_outer]).squeeze()
            # If the bounds are the same the bound perfectly
            # aligns with the grid), we don't need so solve the
            # diffusion matrix again since we don't need a linear
            # approximation.
            if x_index_inner == x_index_outer:
                pdf_inner = pdf_outer
            else:
                # Need a separate matrix here to get the proper corrections
                drift_matrix = self.get_dependence('drift').get_matrix(x=x_list_inbounds[1:-1], t=t,
                                                                    dt=self.dt, dx=self.dx,
                                                                    conditions=conditions,
                                                                    implicit=(method!="explicit"))
                noise_matrix = self.get_dependence('noise').get_matrix(x=x_list_inbounds[1:-1], t=t,
                                                                    dt=self.dt, dx=self.dx, conditions=conditions,
                                                                    implicit=(method!="explicit"))
                if method == "implicit":
                    diffusion_matrix = TriDiagMatrix.eye(len(x_list_inbounds)-2) + drift_matrix + noise_matrix
                elif method == "explicit":
                    # Explicit method flips sign except for the identity matrix
                    diffusion_matrix_explicit = TriDiagMatrix.eye(len(x_list_inbounds)) - drift_matrix - noise_matrix
                if method == "implicit":
                    pdf_inner = diffusion_matrix.spsolve(pdf_prev[x_index_inner:len(x_list)-x_index_inner])
                elif method == "explicit":
                    pdf_inner = diffusion_matrix_explicit.dot(pdf_prev[x_index_inner:len(x_list)-x_index_inner])

            # Pdfs out of bound is considered decisions made.
            pdf_choice_lower[i_t+1] += weight_outer * np.sum(pdf_prev[:x_index_outer]) \
                              + weight_inner * np.sum(pdf_prev[:x_index_inner])
            pdf_choice_upper[i_t+1] += weight_outer * np.sum(pdf_prev[len(x_list)-x_index_outer:]) \
                               + weight_inner * np.sum(pdf_prev[len(x_list)-x_index_inner:])
            # Reconstruct current probability density function,
            # adding outer and inner contribution to it.  Use
            # .fill() method to avoid allocating memory with
            # np.zeros().
            pdf_curr.fill(0) # Reset
            pdf_curr[x_index_outer:len(x_list)-x_index_outer] += weight_outer*pdf_outer # For explicit, should be pdf_outer.T?
            pdf_curr[x_index_inner:len(x_list)-x_index_inner] += weight_inner*pdf_inner

            # Increase current, transient probability of crossing
            # either bounds, as flux.  Corr is a correct answer, err
            # is an incorrect answer
            _inner_B_choice_upper = x_list[len(x_list)-1-x_index_inner]
            _outer_B_choice_upper = x_list[len(x_list)-1-x_index_outer]
            _inner_B_choice_lower = x_list[x_index_inner]
            _outer_B_choice_lower = x_list[x_index_outer]
            if len(pdf_inner) == 0: # Otherwise we get an error when bounds collapse to 0
                pdf_inner = np.array([0])
            pdf_choice_upper[i_t+1] += weight_outer * pdf_outer[-1] * self.flux(_outer_B_choice_upper, t, conditions=conditions) \
                               +  weight_inner * pdf_inner[-1] * self.flux(_inner_B_choice_upper, t, conditions=conditions)
            pdf_choice_lower[i_t+1]  += weight_outer * pdf_outer[0] * self.flux(_outer_B_choice_lower, t, conditions=conditions) \
                                +  weight_inner * pdf_inner[0] * self.flux(_inner_B_choice_lower, t, conditions=conditions)

            # Renormalize when the channel size has <1 grid, although
            # all hell breaks loose in this regime.
            if bound < self.dx:
                pdf_choice_upper[i_t+1] *= (1+ (1-bound/self.dx))
                pdf_choice_lower[i_t+1] *= (1+ (1-bound/self.dx))

            # If evolution of pdf should be returned, append pdf_curr to pdf_evolution
            if return_evolution:
                pdf_evolution[:,i_t+1] = pdf_curr

        # Detect and fix below zero errors
        pdf_undec = pdf_curr
        minval = np.min((np.min(pdf_choice_upper), np.min(pdf_choice_lower), np.min(pdf_undec)))
        if minval < 0:
            sum_negative_strength = np.sum(pdf_choice_upper[pdf_choice_upper<0]) + np.sum(pdf_choice_lower[pdf_choice_lower<0])
            sum_negative_strength_undec = np.sum(pdf_undec[pdf_undec<0])
            if sum_negative_strength < -.01 and param.renorm_warnings:
                _logger.warning(("Probability density included values less than zero (minimum=%f, "
                    + "total=%f.  Please decrease dt and/or avoid extreme parameter values.")
                    % (minval, sum_negative_strength))
                _logger.debug(self.parameters())
            if sum_negative_strength_undec < -.01 and param.renorm_warnings:
                _logger.warning(("Remaining FP distribution included values less than zero "
                    + "(minimum=%f, total=%f).  Please decrease dt and/or avoid extreme parameter "
                    + "values.") % (minval, sum_negative_strength_undec))
                _logger.debug(self.parameters())
            pdf_choice_lower[pdf_choice_lower < 0] = 0
            pdf_choice_lower[pdf_choice_lower < 0] = 0
            pdf_undec[pdf_undec < 0] = 0
        # Fix numerical errors
        pdfsum = np.sum(pdf_choice_upper) + np.sum(pdf_choice_lower) + np.sum(pdf_undec)
        if pdfsum > 1:
            if pdfsum > 1.01 and param.renorm_warnings:
                _logger.warning(("Renormalizing probability density from " + str(pdfsum) + "to 1. "
                    + " Try decreasing dt or using the implicit (backward Euler) method instead.  "
                    + "If that doesn't eliminate this warning, it may be due to extreme parameter "
                    + "values and/or bugs in your model spefication."))
                _logger.debug(self.parameters())
            pdf_choice_upper /= pdfsum
            pdf_choice_lower /= pdfsum
            pdf_undec /= pdfsum

        if return_evolution:
            return self.get_dependence('overlay').apply(Solution(pdf_choice_upper, pdf_choice_lower, self, conditions=conditions, pdf_undec=pdf_undec, pdf_evolution=pdf_evolution))
        
        return self.get_dependence('overlay').apply(Solution(pdf_choice_upper, pdf_choice_lower, self, conditions=conditions, pdf_undec=pdf_undec))

    @accepts(Self, Conditions)
    @returns(Solution)
    @requires("self.can_solve_explicit(conditions=conditions)")
    def solve_numerical_explicit(self, conditions={}, **kwargs):
        """Solve the model using the explicit method (Forward Euler).

        See documentation for the solve_numerical method.
        """
        return self.solve_numerical(method="explicit", conditions=conditions, **kwargs)

    @accepts(Self, Conditions)
    @returns(Solution)
    def solve_numerical_implicit(self, conditions={}, **kwargs):
        """Solve the model using the implicit method (Backward Euler).

        See documentation for the solve_numerical method.
        """
        return self.solve_numerical(method="implicit", conditions=conditions, **kwargs)

    @accepts(Self, conditions=Conditions)
    @requires("self.can_solve_cn(conditions=conditions)")
    @returns(Solution)
    def solve_numerical_cn(self, conditions={}):
        """Solve the DDM model numerically using Crank-Nicolson.

        This uses the Crank Nicolson method to solve the DDM at each
        timepoint.  Results are then compiled together.  This is the
        core DDM solver of this library.

        It returns a Solution object describing the joint PDF.
        """
        ### Initialization: Lists
        self.check_conditions_satisfied(conditions)
        pdf_curr = self.IC(conditions=conditions) # Initial condition
        pdf_outer = self.IC(conditions=conditions)
        pdf_inner = self.IC(conditions=conditions)
        # pdf_prev = np.zeros((len(pdf_curr)))
        # If pdf_corr + pdf_err + undecided probability are summed, they
        # equal 1.  So these are componets of the joint pdf.
        pdf_choice_upper = np.zeros(len(self.t_domain())+1) # Not a proper pdf on its own (doesn't sum to 1)
        pdf_choice_lower = np.zeros(len(self.t_domain())+1) # Not a proper pdf on its own (doesn't sum to 1)
        x_list = self.x_domain(conditions=conditions)

        bound_shift = 0.
        # Note that we linearly approximate the bound by the two surrounding grids sandwiching it.
        x_index_inner = int(np.ceil(bound_shift/self.dx)) # Index for the inner bound (smaller matrix)
        x_index_outer = int(np.floor(bound_shift/self.dx)) # Index for the outer bound (larger matrix)

        # We weight the upper and lower matrices according to
        # how far away the bound is from each.  The weight of
        # each matrix is approximated linearly. The inner
        # bound is 0 when bound exactly at grids.
        weight_inner = (bound_shift - x_index_outer*self.dx)/self.dx
        weight_outer = 1. - weight_inner # The weight of the lower bound matrix, approximated linearly.
        x_list_inbounds = x_list[x_index_outer:len(x_list)-x_index_outer] # List of x-positions still within bounds.


        prev_t = 0
        prev_i_t = 0
        # Looping through time and updating the pdf.
        for i_t, t in enumerate(self.t_domain()):
            # Update Previous state.
            pdf_outer_prev = pdf_outer.copy()
            pdf_inner_prev = pdf_inner.copy()

            # For efficiency only do diffusion if there's at least
            # some densities remaining in the channel.
            if np.sum(pdf_curr[:])>0.0001:
                ## Define the boundaries at current time.
                bound = self.get_dependence('bound').get_bound(t=t, conditions=conditions) # Boundary at current time-step.

                # Now figure out which x positions are still within
                # the (collapsing) bound.
                assert self.get_dependence("bound").get_bound(t=0, conditions=conditions) >= bound, "Invalid change in bound" # Ensure the bound didn't expand
                bound_shift = self.get_dependence("bound").get_bound(t=0, conditions=conditions) - bound
                # Note that we linearly approximate the bound by the two surrounding grids sandwiching it.
                x_index_inner_prev = x_index_inner
                x_index_outer_prev = x_index_outer
                x_index_inner = int(np.ceil(bound_shift/self.dx)) # Index for the inner bound (smaller matrix)
                x_index_outer = int(np.floor(bound_shift/self.dx)) # Index for the outer bound (larger matrix)
                x_index_inner_shift = x_index_inner - x_index_inner_prev
                x_index_outer_shift = x_index_outer - x_index_outer_prev
                x_index_io_shift = x_index_inner - x_index_outer
                x_index_io_shift_prev = x_index_inner_prev - x_index_outer_prev
                # We weight the upper and lower matrices according to
                # how far away the bound is from each.  The weight of
                # each matrix is approximated linearly. The inner
                # bound is 0 when bound exactly at grids.
                weight_inner_prev = weight_inner
                weight_outer_prev = weight_outer
                weight_inner = (bound_shift - x_index_outer*self.dx)/self.dx
                weight_outer = 1. - weight_inner # The weight of the lower bound matrix, approximated linearly.
                x_list_inbounds_prev = x_list_inbounds.copy() # List of x-positions still within bounds.
                x_list_inbounds = x_list[x_index_outer:len(x_list)-x_index_outer] # List of x-positions still within bounds.

                # Diffusion Matrix for Implicit Method. Here defined as
                # Outer Matrix, and inner matrix is either trivial or an
                # extracted submatrix.
                # diffusion_matrix_prev = 2.* np.diag(np.ones(len(x_list_inbounds_prev))) - diffusion_matrix       #Diffusion Matrix for Implicit Method. Here defined as Outer Matrix, and inner matrix is either trivial or an extracted submatrix.
                local_dt = t - prev_t if t!=0 else self.t_domain()[1]
                drift_matrix = self.get_dependence('drift').get_matrix(x=x_list_inbounds, t=t,
                                                                       dt=local_dt, dx=self.dx, conditions=conditions,
                                                                       implicit=True)
                drift_matrix *= .5
                noise_matrix = self.get_dependence('noise').get_matrix(x=x_list_inbounds,
                                                                       t=t, dt=local_dt, dx=self.dx, conditions=conditions,
                                                                       implicit=True)
                noise_matrix *= .5
                diffusion_matrix = TriDiagMatrix.eye(len(x_list_inbounds))
                diffusion_matrix += drift_matrix
                diffusion_matrix += noise_matrix

                drift_matrix_prev = self.get_dependence('drift').get_matrix(x=x_list_inbounds_prev, t=prev_t,
                                                                            dt=local_dt, dx=self.dx, conditions=conditions,
                                                                            implicit=True)
                drift_matrix_prev *= .5
                noise_matrix_prev = self.get_dependence('noise').get_matrix(x=x_list_inbounds_prev, t=prev_t,
                                                                            dt=local_dt, dx=self.dx, conditions=conditions,
                                                                            implicit=True)
                noise_matrix_prev *= .5
                diffusion_matrix_prev = TriDiagMatrix.eye(len(x_list_inbounds_prev))
                diffusion_matrix_prev -= drift_matrix_prev
                diffusion_matrix_prev -= noise_matrix_prev

                ### Compute Probability density functions (pdf)
                # PDF for outer matrix
                # Probability density function for outer matrix.
                # Considers the whole space in the previous step
                # for matrix multiplication, then restrains to current
                # space when solving for matrix_diffusion.. Separating
                # outer and inner pdf_prev
                # 
                # For constant bounds pdf_inner is unnecessary.
                # For changing bounds pdf_inner is always needed,
                # even if the current inner and outer bounds
                # coincide. I currently make this generally but
                # we can constrain it to changing-bound
                # simulations only.
                so_from = x_index_outer_shift
                so_to = len(x_list_inbounds)+x_index_outer_shift
                si_from = x_index_io_shift
                si_to = len(x_list_inbounds)-x_index_io_shift
                si2_from = x_index_io_shift_prev
                si2_to = len(x_list_inbounds_prev)-x_index_io_shift_prev
                si3_from = x_index_inner_shift
                si3_to = len(x_list)-2*x_index_inner+x_index_inner_shift

                pdf_outer = diffusion_matrix.spsolve(diffusion_matrix_prev.dot(pdf_outer_prev)[so_from:so_to])
                if x_index_inner == x_index_outer: # Should always be the case, since we removed CN changing bounds support
                    pdf_inner = pdf_outer
                else:
                    # Need a separate matrix here to get the proper corrections
                    drift_matrix = self.get_dependence('drift').get_matrix(x=x_list_inbounds[si_from:si_to], t=t,
                                                                           dt=local_dt, dx=self.dx, conditions=conditions,
                                                                           implicit=True)
                    drift_matrix *= .5
                    noise_matrix = self.get_dependence('noise').get_matrix(x=x_list_inbounds[si_from:si_to],
                                                                           t=t, dt=local_dt, dx=self.dx, conditions=conditions,
                                                                           implicit=True)
                    noise_matrix *= .5
                    diffusion_matrix = TriDiagMatrix.eye(len(x_list_inbounds[si_from:si_to]))
                    diffusion_matrix += drift_matrix
                    diffusion_matrix += noise_matrix

                    drift_matrix_prev = self.get_dependence('drift').get_matrix(x=x_list_inbounds_prev[si2_from:si2_to], t=prev_t,
                                                                                dt=local_dt, dx=self.dx, conditions=conditions,
                                                                                implicit=True)
                    drift_matrix_prev *= .5
                    noise_matrix_prev = self.get_dependence('noise').get_matrix(x=x_list_inbounds_prev[si2_from:si2_to], t=prev_t,
                                                                                dt=local_dt, dx=self.dx, conditions=conditions,
                                                                                implicit=True)
                    noise_matrix_prev *= .5
                    diffusion_matrix_prev = TriDiagMatrix.eye(len(x_list_inbounds_prev[si2_from:si2_to]))
                    diffusion_matrix_prev -= drift_matrix_prev
                    diffusion_matrix_prev -= noise_matrix_prev
                    pdf_inner = diffusion_matrix.spsolve(diffusion_matrix_prev.dot(pdf_inner_prev)[si3_from:si3_to])


                # Pdfs out of bound is considered decisions made.
                pdf_choice_lower[i_t+1] += weight_outer_prev * np.sum(pdf_outer_prev[:x_index_outer_shift]) \
                                   + weight_inner_prev * np.sum(pdf_inner_prev[:x_index_inner_shift])
                pdf_choice_upper[i_t+1] += weight_outer_prev * np.sum(pdf_outer_prev[len(pdf_outer_prev)-x_index_outer_shift:]) \
                                   + weight_inner_prev * np.sum(pdf_inner_prev[len(pdf_inner_prev)-x_index_inner_shift:])
                # Reconstruct current probability density function,
                # adding outer and inner contribution to it.  Use
                # .fill() method to avoid allocating memory with
                # np.zeros().
                pdf_curr.fill(0) # Reset
                pdf_curr[x_index_outer:len(x_list)-x_index_outer] += weight_outer*pdf_outer
                pdf_curr[x_index_inner:len(x_list)-x_index_inner] += weight_inner*pdf_inner

            else:
                break #break if the remaining densities are too small....

            # Increase current, transient probability of crossing
            # either bounds, as flux.  Corr is a correct answer, err
            # is an incorrect answer
            _inner_B_choice_upper = x_list[len(x_list)-1-x_index_inner]
            _outer_B_choice_upper = x_list[len(x_list)-1-x_index_outer]
            _inner_B_choice_lower = x_list[x_index_inner]
            _outer_B_choice_lower = x_list[x_index_outer]
            flux_outer_B_choice_upper = self.flux(_outer_B_choice_upper, t, conditions=conditions)
            flux_inner_B_choice_upper = self.flux(_inner_B_choice_upper, t, conditions=conditions)
            flux_outer_B_choice_lower = self.flux(_outer_B_choice_lower, t, conditions=conditions)
            flux_inner_B_choice_lower = self.flux(_inner_B_choice_lower, t, conditions=conditions)
            if len(pdf_inner) == 0: # Otherwise we get an error when bounds collapse to 0
                pdf_inner = np.array([0])
            pdf_choice_upper[i_t+1] += 0.5*weight_outer * pdf_outer[-1] * flux_outer_B_choice_upper \
                               +  0.5*weight_inner * pdf_inner[-1] * flux_inner_B_choice_upper
            pdf_choice_lower[i_t+1] += 0.5*weight_outer * pdf_outer[0]  * flux_outer_B_choice_lower \
                               +  0.5*weight_inner * pdf_inner[0]  * flux_inner_B_choice_lower
            pdf_choice_upper[i_t]   += 0.5*weight_outer * pdf_outer[-1] * flux_outer_B_choice_upper \
                               +  0.5*weight_inner * pdf_inner[-1] * flux_inner_B_choice_upper
            pdf_choice_lower[i_t]   += 0.5*weight_outer * pdf_outer[0]  * flux_outer_B_choice_lower \
                               +  0.5*weight_inner * pdf_inner[0]  * flux_inner_B_choice_lower

            # Renormalize when the channel size has <1 grid, although
            # all hell breaks loose in this regime.
            if bound < self.dx:
                pdf_choice_upper[i_t+1] *= (1+ (1-bound/self.dx))
                pdf_choice_lower[i_t+1] *= (1+ (1-bound/self.dx))
            prev_t = t
            prev_i_t = i_t

        # Fix the time-offest error of dt/2
        pdf_choice_upper = np.concatenate([[0], np.mean([pdf_choice_upper[1:], pdf_choice_upper[:-1]], axis=0)])
        pdf_choice_lower = np.concatenate([[0], np.mean([pdf_choice_lower[1:], pdf_choice_lower[:-1]], axis=0)])
        # Fix the truncation error at the end
        pdf_choice_upper = pdf_choice_upper[:-1]
        pdf_choice_lower = pdf_choice_lower[:-1]
        # Detect and fix below zero errors.  Here, we don't worry
        # about undecided probability as we did with the implicit
        # method, because CN tends to oscillate around zero,
        # especially when noise (sigma) is large.  The user would be
        # directed to decrease dt.
        pdf_undec = pdf_curr
        minval = np.min((np.min(pdf_choice_upper), np.min(pdf_choice_lower)))
        if minval < 0:
            sum_negative_strength = np.sum(pdf_choice_upper[pdf_choice_upper<0]) + np.sum(pdf_choice_lower[pdf_choice_lower<0])
            # For small errors, don't bother alerting the user
            if sum_negative_strength < -.01 and param.renorm_warnings:
                _logger.warning(("Probability density included values less than zero (minimum=%f, "
                    + "total=%f).  Please decrease dt and/or avoid extreme parameter values.")
                    % (minval, sum_negative_strength))
                _logger.debug(self.parameters())
            pdf_choice_upper[pdf_choice_upper < 0] = 0
            pdf_choice_lower[pdf_choice_lower < 0] = 0
        # Fix numerical errors
        pdfsum = np.sum(pdf_choice_upper) + np.sum(pdf_choice_lower)
        if pdfsum > 1 and False:
            print('Renorm for', pdfsum)
            # If it is only a small renormalization, don't bother alerting the user.
            if pdfsum > 1.01 and param.renorm_warnings:
                _logger.warning(("Renormalizing probability density from " + str(pdfsum) + " to 1."
                    + "  Try decreasing dt.  If that doesn't eliminate this warning, it may be due"
                    + " to extreme parameter values and/or bugs in your model spefication."))
                _logger.debug(self.parameters())
            pdf_choice_upper /= pdfsum
            pdf_choice_lower /= pdfsum

        # TODO Crank-Nicolson still has something weird going on with pdf_curr near 0, where it seems to oscillate
        return self.get_dependence('overlay').apply(Solution(pdf_choice_upper, pdf_choice_lower, self, conditions=conditions, pdf_undec=None))
    # def fit(self, sample, fitparams=None, fitting_method="differential_evolution",
    #         lossfunction=LossLikelihood, verify=False, method=None, verbose=True):
    #     functions.fit_adjust_model.__doc__
    #     pyddm.fit_adjust_model(sample=sample, model=self, fitparams=fitparams, fitting_method=fitting_method,
    #                            lossfunction=lossfunction, verify=verify, method=method, verbose=verbose)


@paranoidclass
class Fittable(float):
    """For parameters that should be adjusted when fitting a model to data.
        
    Each Fittable object does not need any parameters, however several
    parameters may improve the ability to fit the model.  In
    particular, `maxval` and `minval` ensure we do not choose an
    invalid parameter value.  `default` is the value to start with
    when fitting; if it is not given, it will be selected at random.
    """
    @staticmethod
    def _test(v):
        assert v in Numeric()
        # This if statement technically violates the liskov
        # substitution principle.  However this is only the case
        # because Fittable inherits from float.  If it didn't, passing
        # nan as the "value" for a Fittable wouldn't be necessary and
        # everything would be fine.  Inheriting from float is a
        # convenience so that we don't have to redefine all of
        # python's floating point internal methods for the Fitted
        # class.  In theory you could get around this problem by
        # making Fitted inherit from Fittable (which would inherit
        # only from nothing) and float via multiple inheritance, but
        # this gets complicated quickly since float is a builtin.
        if type(v) is Fittable:
            assert np.isnan(v), "Fittable has already been fitted"
        elif type(v) is Fitted:
            assert v in Number(), "Fitted value is invalid"
    @staticmethod
    def _generate():
        yield Fittable()
        yield Fittable(minval=1)
        yield Fittable(maxval=3)
        yield Fittable(minval=-1, maxval=1)
        yield Fittable(default_value=.001)
        yield Fittable(minval=3, default_value=6)
        yield Fittable(minval=10, maxval=100, default_value=20)
        yield Fitted(0)
        yield Fitted(1)
        yield Fitted(4, minval=0, maxval=10)
    def __new__(cls, *args, **kwargs):
        if len(args) == 1 or 'val' in kwargs.keys():
            val = args[0] if len(args) == 1 else kwargs['val']
            raise ValueError("No positional arguments for Fittables. Received argument: %s, kwargs: %s." % (str(val), str(kwargs)))
        return float.__new__(cls, np.nan)
    def __getnewargs_ex__(self):
        return ((), {"minval": self.minval, "maxval": self.maxval, "default": self.default_value})
    def __init__(self, *args, **kwargs):
        if len(args) == 2:
            minval,maxval = args
            default_value = kwargs['default'] if 'default' in kwargs else None
        elif len(args) == 3:
            minval,maxval,default_value = args
        elif len(args) == 1 or len(args) > 3:
            raise ValueError(f"Invalid number of positional arguments to Fittable: {args}")
        else:
            minval = kwargs['minval'] if "minval" in kwargs else -np.inf
            maxval = kwargs['maxval'] if "maxval" in kwargs else np.inf
            default_value = kwargs['default'] if 'default' in kwargs else None
        object.__setattr__(self, 'minval', minval)
        object.__setattr__(self, 'maxval', maxval)
        object.__setattr__(self, 'default_value', default_value)
    def __setattr__(self, name, val):
        """No changing the state."""
        raise AttributeError
    def __delattr__(self, name):
        """No deletions of existing parameters."""
        raise AttributeError
    def __repr__(self):
        components = []
        if not np.isnan(self):
            components.append(str(float(self)))
        if self.minval != -np.inf:
            components.append("minval=" + self.minval.__repr__())
        if self.maxval != np.inf:
            components.append("maxval=" + self.maxval.__repr__())
        if self.default_value is not None:
            components.append("default=" + self.default_value.__repr__())
        return type(self).__name__ + "(" + ", ".join(components) + ")"
    @accepts(Self)
    @returns(Number)
    def default(self):
        """Choose a default value.

        This chooses a value for the Fittable object randomly abiding
        by any constraints.  Note that calling this function multiple
        times will give different results.
        """
        if self.default_value is not None:
            return self.default_value
        else:
            maxval = self.maxval # Makes equations below more readable
            minval = self.minval
            if maxval < np.inf and minval > -np.inf:
                return np.random.beta(2, 2)*(maxval-minval) + minval
            elif maxval == np.inf and minval > -np.inf:
                return np.random.pareto(1) + minval
            elif maxval < np.inf and minval == -np.inf:
                return maxval - np.random.pareto(1)
            elif maxval == np.inf and minval == -np.inf:
                return np.random.standard_cauchy()
            else:
                raise ValueError("Error with the maximum or minimum bounds")
    @accepts(Self, Number)
    @returns(Self)
    def make_fitted(self, val):
        return Fitted(float(val), maxval=self.maxval, minval=self.minval)

class Fitted(Fittable):
    """Parameters which used to be Fittable but now hold a value.

    This extends float so that the parameters for the Fittable can be
    saved in the final model, even though there are no more Fittable
    objects in the final model.
    """
    def __new__(cls, val, **kwargs):
        return float.__new__(cls, val)
    def __init__(self, val, **kwargs):
        Fittable.__init__(self, **kwargs)
    def __getnewargs_ex__(self):
        return ((float(self),), {"minval": self.minval, "maxval": self.maxval, "default": self.default_value})
    def default(self):
        return float(self)

