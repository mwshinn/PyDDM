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
        "differential_evolution" is the default, which accurately locates the
        global maximum without using a derivative.  For most purposes,
        differential evolution is the recommended method.  Another option is
        "bads" for BADS (https://arxiv.org/abs/1705.04405), which may be faster
        for some models.  Other methods are only recommended in special cases.
        "simple" uses a derivative-based method to minimize, and just uses
        randomly initialized parameters and gradient descent, and "simplex" is
        the Nelder-Mead method, and is a gradient-free local search.  These are
        only recommended if you have a very simple model, and also have a good
        estimate of the global maximum. Alternatively, a custom objective
        function may be used by setting `fitting_method` to be a function which
        accepts the "x_0" parameter (for starting position) and "constraints"
        (for min and max values).  In general, it is recommended you almost
        always use differential evolution or BADS, unless you have a model which
        is highly-constrained (e.g. only one or two parameters to estimate with
        low covariance) or you already know the approximate parameter values.
        If you change this value, make sure to check that you get reliable and
        recoverable fits.

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
        if t is None and self._bounddep._uses_t():
            B = max([self._bounddep.get_bound(t=t, conditions=conditions) for t in self.t_domain()])
        else:
            B = self._bounddep.get_bound(t=0 if t is None else t, conditions=conditions)
        # This is the storage grid, not a rounding of the physical boundary.
        # Integer multiples keep the centre exactly zero and avoid an extra
        # grid point when B/dx is an integer up to floating point roundoff.
        radius = max(0, int(np.ceil(B/self.dx - 1e-10)))
        return np.arange(-radius, radius+1)*self.dx
    def t_domain(self):
        """A list of all of the timepoints over which the joint PDF will be defined (increments of dt from 0 to T_dur)."""
        return np.arange(0., self.T_dur+0.1*self.dt, self.dt)
    def flux(self, x, t, conditions):
        """Uniform-grid outgoing coefficient at the adjacent interior point.

        Numerical solvers use the spatial operator's outgoing rates directly
        to also account for fractional boundaries and single-point domains.
        """
        drift_flux = self.get_dependence('drift').get_flux(x, t, dx=self.dx, dt=self.dt, conditions=conditions)
        noise_flux = self.get_dependence('noise').get_flux(x, t, dx=self.dx, dt=self.dt, conditions=conditions)
        return drift_flux + noise_flux
    def IC(self, conditions):
        """The initial distribution at t=0.

        Returns a length N ndarray (where N is the size of x_domain())
        which should sum to 1.
        """
        x = self.x_domain(conditions=conditions)
        # A ratio is relative to the bounds at t=0, not the largest bounds
        # used to allocate storage for an expanding-bound model.
        initial_x = (self.x_domain(conditions=conditions, t=0)
                     if isinstance(self._IC, ICPointRatio) and self._bounddep._uses_t() else x)
        initial = np.asarray(self._IC.get_IC(initial_x, dx=self.dx, conditions=conditions), dtype=float)
        if initial.ndim != 1 or len(initial) != len(initial_x):
            raise ValueError("Initial condition must be a one-dimensional array matching the spatial grid")
        mass = initial.sum()
        if not np.isfinite(mass) or np.any(initial < 0) or abs(mass-1) > 1.01e-10:
            raise ValueError("Initial condition must contain finite, nonnegative probabilities summing to one")
        if len(initial_x) != len(x):
            result = np.zeros(len(x))
            offset = (len(x)-len(initial_x))//2
            result[offset:offset+len(initial_x)] = initial
            return result
        return initial

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
        Euler's method is used instead. This
        returns a trajectory of the simulated trial over time as a
        numpy array.

        Note that this will return the same trajectory on each run
        unless the random seed `seed` is varied.

        With `rk4=True`, the drift uses Runge-Kutta stages.  Noise that
        depends on position is evaluated at the start of each step, so the
        simulation follows the same Itô convention as the numerical solver.
        This does not imply fourth-order convergence for stochastic paths.

        Also note that you shouldn't normally need to use this
        function.  To simulate an entire probability distributions,
        call Model.solve() and the results of the simulation will be
        in the returned Solution object.  This is only useful for
        finding individual trajectories instead of the probability
        distribution as a whole.

        """
        return self._simulate_trial(conditions=conditions, cutoff=cutoff, rk4=rk4, seed=seed)[0]

    def _simulate_trial(self, conditions, cutoff, rk4, seed):
        """Return the public trajectory and whether it ended in a decision."""
        self.check_conditions_satisfied(conditions)
        
        h = self.dt
        T = self.t_domain()

        # Choose a starting position from the IC
        rng = np.random.RandomState(seed)
        ic = self.IC(conditions=conditions)
        x0 = rng.choice(self.x_domain(conditions=conditions), p=ic)
        pos = [x0]
        initial_bound = self._bounddep.get_bound(t=0, conditions=conditions)
        absorbed_initially = cutoff and (x0 >= initial_bound or x0 <= -initial_bound)
        absorbed = absorbed_initially

        # Convenience functions
        _driftdep = self.get_dependence("drift")
        _noisedep = self.get_dependence("noise")
        state_dependent_noise = rk4 and _noisedep._uses_x()
        fm = lambda x,t : _driftdep.get_drift(t=t, x=x, conditions=conditions)
        fs = lambda x,t : _noisedep.get_noise(t=t, x=x, conditions=conditions)
        
        for i in range(1, len(T)):
            if absorbed_initially:
                break
            # Stochastic Runge-Kutta order 4.  See "Introduction to
            # Stochastic Differential Equations" by Thomas C. Gard
            rn = rng.randn()
            dw = np.sqrt(h)*rn
            drift1 = fm(t=T[i-1], x=pos[i-1])
            s1  = fs(t=T[i-1], x=pos[i-1])
            if rk4: # Use Runge-Kutta order 4
                drift2 = fm(t=(T[i-1]+h/2), x=(pos[i-1] + drift1*h/2 + s1*dw/2)) # Should be dw/sqrt(2)?
                s2x = pos[i-1] if state_dependent_noise else (pos[i-1] + drift1*h/2 + s1*dw/2)
                s2  = fs(t=(T[i-1]+h/2), x=s2x)
                drift3 = fm(t=(T[i-1]+h/2), x=(pos[i-1] + drift2*h/2 + s2*dw/2))
                s3x = pos[i-1] if state_dependent_noise else (pos[i-1] + drift2*h/2 + s2*dw/2)
                s3  = fs(t=(T[i-1]+h/2), x=s3x)
                drift4 = fm(t=(T[i-1]+h), x=(pos[i-1] + drift3*h + s3*dw)) # Should this be 1/2*s3*dw?
                s4x = pos[i-1] if state_dependent_noise else (pos[i-1] + drift3*h + s3*dw)
                s4  = fs(t=(T[i-1]+h), x=s4x)
                dx = h*(drift1 + 2*drift2 + 2*drift3 + drift4)/6 + dw*(s1 + 2*s2 + 2*s3 + s4)/6
            else: # Use Euler's method
                dx = h*drift1 + dw*s1
            pos.append(pos[i-1] + dx)
            B = self.get_dependence("bound").get_bound(t=T[i], conditions=conditions)
            if cutoff and (pos[i] >= B or pos[i] <= -B):
                absorbed = True
                break

        traj = self.get_dependence("overlay").apply_trajectory(trajectory=np.asarray(pos), model=self, seed=seed, rk4=rk4, conditions=conditions)
        if cutoff is False and len(traj) < len(T):
            traj = np.append(traj, [traj[-1]]*(len(T)-len(traj)))
        beyond_duration = len(traj) > len(T)
        if beyond_duration:
            traj = traj[0:len(T)]
        return traj, bool(absorbed and not beyond_duration)


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
            trial_seed = hash((s, seed)) % 2**32
            if getattr(self.simulate_trial, '__func__', None) is Model.simulate_trial:
                timecourse, absorbed = self._simulate_trial(conditions=conditions, seed=trial_seed, cutoff=True, rk4=rk4)
            else:
                # Preserve custom simulate_trial overrides, which do not
                # provide the internal absorption flag.
                timecourse = self.simulate_trial(conditions=conditions, seed=trial_seed, cutoff=True, rk4=rk4)
                absorbed = None
            T_finish = T[len(timecourse) - 1]
            B = self.get_dependence("bound").get_bound(t=T_finish, conditions=conditions)
            # Correct for the fact that the particle could have
            # crossed at any point between T_finish-dt and T_finish.
            dt_correction = self.dt/2 if T_finish > 0 else 0
            # Determine whether the sim is a correct or error trial.
            if absorbed is False:
                undec_count += 1
            elif absorbed and timecourse[-1] > 0:
                choice_upper_times.append(T_finish - dt_correction)
            elif absorbed and timecourse[-1] < 0:
                choice_lower_times.append(T_finish - dt_correction)
            elif timecourse[-1] == 0 and (absorbed or B <= 0):
                # A point exactly at two coincident bounds is split evenly.
                side_rng = np.random.default_rng(np.random.SeedSequence([seed, s, 1703]))
                choice = side_rng.integers(2)
                (choice_upper_times if choice else choice_lower_times).append(T_finish - dt_correction)
            elif timecourse[-1] >= B:
                choice_upper_times.append(T_finish - dt_correction)
            elif timecourse[-1] <= -B:
                choice_lower_times.append(T_finish - dt_correction)
            elif len(timecourse) == len(T):
                undec_count += 1
            else:
                raise SystemError("Internal error: Invalid particle simulation")
            
        aa = lambda x : np.asarray(x)
        conds = {k:(aa(len(choice_upper_times)*[v]), aa(len(choice_lower_times)*[v]), aa(undec_count*[v])) for k,v in conditions.items()}
        return Sample(aa(choice_upper_times), aa(choice_lower_times), undec_count,
                      choice_names=self.choice_names, **conds)

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
        """Check positivity and stability of the explicit spatial update.

        Check every occupied grid position, including the shorter distances
        to off-grid boundaries.  The drift restrictions matter as well as
        the usual diffusion restriction sigma**2 * dt / dx**2 <= 1.
        """
        self.check_conditions_satisfied(conditions)
        x = self.x_domain(conditions=conditions)
        times = self.t_domain()
        varying = any(dep._uses_t() for dep in (self._driftdep, self._noisedep, self._bounddep))
        for t in times[:-1] if varying else times[:1]:
            bound = self._bounddep.get_bound(t=t, conditions=conditions)
            if not self._fp_explicit_stable(self._fp_matrix(x, t, bound, conditions)):
                return False
        return True

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
        elif self.can_solve_cn(conditions=conditions) and return_evolution is False:
            if HAS_CSOLVE and not force_python:
                return self.solve_numerical_cn_c(conditions=conditions)
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
        
        # Use the physical starting position, rather than a fraction of the
        # storage grid (whose endpoints may lie outside the physical bound).
        initial_bound = self._bounddep.get_bound(t=0, conditions=conditions)
        start = 0.
        if isinstance(self.get_dependence('IC'),(ICPoint, ICPointRatio)):
            ic = self.IC(conditions=conditions)
            assert np.count_nonzero(ic)==1, "Cannot solve analytically for models with non-point initial conditions"
            start = (np.flatnonzero(ic)[0] - (len(ic)-1)//2)*self.dx
            shift = .5 + start/(2*initial_bound) if initial_bound > 0 else None
        else:
            shift = None
        if initial_bound <= 0 or abs(start) >= initial_bound:
            upper, lower = np.zeros(len(self.t_domain())), np.zeros(len(self.t_domain()))
            upper[0] = 1. if start > 0 else 0. if start < 0 else .5
            lower[0] = 1-upper[0]
            return self._overlay.apply(Solution(upper, lower, self, conditions=conditions))
        
        # The analytic_ddm function does the heavy lifting.
        # Only the standard Model.t_domain guarantees a uniform grid.  Custom
        # Model subclasses may supply other timepoints and must be checked.
        uniform_model_grid = getattr(self.t_domain, '__func__', None) is Model.t_domain
        if isinstance(self.get_dependence('bound'), BoundCollapsingLinear): # Linearly Collapsing Bound
            anal_pdf_choice_upper, anal_pdf_choice_lower = analytic_ddm(self.get_dependence("drift").get_drift(t=0, x=0, conditions=conditions),
                                                       self.get_dependence("noise").get_noise(t=0, x=0, conditions=conditions),
                                                       self.get_dependence("bound").get_bound(t=0, x=0, conditions=conditions),
                                                       self.t_domain(), shift, -self.get_dependence("bound").t,
                                                       force_python=force_python, _uniform_model_grid=uniform_model_grid) # TODO why must this be negative? -MS
        else: # Constant bound DDM
            anal_pdf_choice_upper, anal_pdf_choice_lower = analytic_ddm(self.get_dependence("drift").get_drift(t=0, x=0, conditions=conditions),
                                                        self.get_dependence("noise").get_noise(t=0, x=0, conditions=conditions),
                                                       self.get_dependence("bound").get_bound(t=0, x=0, conditions=conditions), 
                                                       self.t_domain(), shift,
                                                       force_python=force_python, _uniform_model_grid=uniform_model_grid)

        ## Remove some abnormalities such as NaN due to trivial reasons.
        anal_pdf_choice_upper[0] = 0.
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

        self.check_conditions_satisfied(conditions)
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
            # Evaluate on the maximum storage grid once, before iterating t.
            xdomain = self.x_domain(conditions=conditions)
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
            xdomain = self.x_domain(conditions=conditions)
            noise = np.concatenate([get_noise(t=t, x=xdomain, conditions=conditions) for t in self.t_domain()])
        bound_uses_t = self.get_dependence("bound")._uses_t()
        if not bound_uses_t:
            boundtype = 0
            bound = np.asarray([self.get_dependence("Bound").get_bound(conditions=conditions, t=0)])
        elif bound_uses_t:
            boundtype = 1
            bound = np.asarray([self.get_dependence("Bound").get_bound(t=t, conditions=conditions) for t in self.t_domain()])
        res = csolve.implicit_time(drift, drifttype, noise, noisetype, bound, boundtype, self.IC(conditions=conditions), self.T_dur, self.dt, self.dx, len(self.t_domain()), param.numerical_mass_tolerance)
        return self._finish_numerical(res[0]*self.dt, res[1]*self.dt, conditions, undec=res[2])

    def solve_numerical_cn_c(self, conditions={}):
        """Solve the DDM model using Crank-Nicolson with C extensions.

        This function should give near identical results to
        solve_numerical_cn.  However, it uses compiled C code instead of
        Python code to do so, which should make it faster.
        """

        self.check_conditions_satisfied(conditions)
        assert self.can_solve_cn(conditions=conditions), "Crank-Nicolson requires time-independent bounds"
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
            xdomain = self.x_domain(conditions=conditions)
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
            xdomain = self.x_domain(conditions=conditions)
            noise = np.concatenate([get_noise(t=t, x=xdomain, conditions=conditions) for t in self.t_domain()])
        bound_uses_t = self.get_dependence("bound")._uses_t()
        if not bound_uses_t:
            boundtype = 0
            bound = np.asarray([self.get_dependence("Bound").get_bound(conditions=conditions, t=0)])
        elif bound_uses_t:
            boundtype = 1
            bound = np.asarray([self.get_dependence("Bound").get_bound(t=t, conditions=conditions) for t in self.t_domain()])
        res = csolve.cn_time(drift, drifttype, noise, noisetype, bound, boundtype, self.IC(conditions=conditions), self.T_dur, self.dt, self.dx, len(self.t_domain()), param.numerical_mass_tolerance, 1)
        end_flux = None
        if 0 < res[3] <= 2:
            t = self.t_domain()[res[3]]
            x = self.x_domain(conditions=conditions)
            b = self._bounddep.get_bound(t=t, conditions=conditions)
            _, lower, upper = self._fp_matrix(x, t, b, conditions)
            end_flux = (self.dt*upper[1]*res[2][upper[0]], self.dt*lower[1]*res[2][lower[0]])
        return self._finish_numerical(res[0], res[1], conditions, cn_last_step=res[3], cn_end_flux=end_flux)

    def _fp_indices(self, x, bound):
        """Indices strictly inside the physical absorbing boundaries."""
        centre = (len(x)-1)//2
        if bound <= 0:
            return centre, centre
        radius = min(centre, max(0, int(np.ceil(bound/self.dx - 1e-10))-1))
        return centre-radius, centre+radius+1

    def _fp_matrix(self, x, t, bound, conditions):
        """Return -L and its two outgoing rates, for q' = L q.

        q contains probability masses, not density values.  Only points
        strictly inside (-bound, bound) evolve.  At a fractional boundary
        the rates are the transpose of the nonuniform central-difference
        backward operator.  They preserve the local first and second
        moments, including when there is just one interior point.
        """
        first, stop = self._fp_indices(x, bound)
        matrix = TriDiagMatrix(diag=np.zeros(len(x)),
                               up=np.zeros(len(x)-1), down=np.zeros(len(x)-1))
        if first == stop:
            return matrix, (first, 0.), (first, 0.)
        inside = x[first:stop]
        args = dict(x=inside, t=t, dx=self.dx, dt=self.dt,
                    conditions=conditions, implicit=False)
        local = (self._driftdep.get_matrix(**args)
                 + self._noisedep.get_matrix(**args)) * (1/self.dt)
        # A boundary between grid points changes the two rates leaving
        # the adjacent point, not just the rate directed at the boundary.
        edges = [0] if len(inside) == 1 else [0, len(inside)-1]
        for j in edges:
            left = bound + inside[j] if j == 0 else self.dx
            right = bound - inside[j] if j == len(inside)-1 else self.dx
            args_edge = dict(x=inside[j], t=t, dx=self.dx, dt=self.dt, conditions=conditions)
            drift = self._driftdep.get_drift(**args_edge)
            variance = self._noisedep.get_noise(**args_edge)**2
            rate_left = (variance-drift*right)/(left*(left+right))
            rate_right = (variance+drift*left)/(right*(left+right))
            local.diag[j] = rate_left + rate_right
            if j > 0:
                local.up[j-1] = -rate_left
            else:
                lower = (first, rate_left)
            if j < len(inside)-1:
                local.down[j] = -rate_right
            else:
                upper = (stop-1, rate_right)
        matrix.diag[first:stop] = local.diag
        matrix.up[first:stop-1] = local.up
        matrix.down[first:stop-1] = local.down
        return matrix, lower, upper

    def _fp_absorb(self, pdf, x, bound):
        """Remove mass on or outside a boundary, including a collapsed one."""
        first, stop = self._fp_indices(x, bound)
        if first == stop:
            lower = np.sum(pdf[:first]) + .5*pdf[first]
            upper = np.sum(pdf[first+1:]) + .5*pdf[first]
            pdf.fill(0)
        else:
            lower, upper = np.sum(pdf[:first]), np.sum(pdf[stop:])
            pdf[:first] = 0
            pdf[stop:] = 0
        return lower, upper

    def _fp_explicit_stable(self, operator):
        matrix, lower, upper = operator
        return not (np.any(matrix.diag*self.dt > 1+1e-12)
                    or np.any(matrix.up > 1e-12) or np.any(matrix.down > 1e-12)
                    or lower[1] < -1e-12 or upper[1] < -1e-12)

    def _fp_step(self, pdf, old, new, step, theta, implicit_matrix=None):
        """Advance the state and absorbed mass with the same theta rule."""
        rhs = pdf if theta == 1 else pdf - (1-theta)*step*old[0].dot(pdf)
        following = rhs
        if theta != 0:
            first, stop = new[1][0], new[2][0]+1
            if implicit_matrix is None:
                matrix, scale = new[0], theta*step
                implicit_matrix = TriDiagMatrix(diag=1+scale*matrix.diag[first:stop],
                                                up=scale*matrix.up[first:stop-1],
                                                down=scale*matrix.down[first:stop-1])
            following = rhs.copy()
            following[first:stop] = implicit_matrix.spsolve(rhs[first:stop])
        lower = step*((1-theta)*old[1][1]*pdf[old[1][0]]
                      + theta*new[1][1]*following[new[1][0]])
        upper = step*((1-theta)*old[2][1]*pdf[old[2][0]]
                      + theta*new[2][1]*following[new[2][0]])
        return following, lower, upper

    def _finish_numerical(self, upper, lower, conditions, undec=None, evolution=None,
                          cn_last_step=None, cn_end_flux=None):
        arrays = [upper, lower] + ([] if undec is None else [undec])
        negative = sum(np.sum(a[a < 0]) for a in arrays)
        if negative < -.01 and param.renorm_warnings:
            _logger.warning("Negative probability mass encountered. Decrease dt and/or dx.")
        for a in arrays:
            a[a < 0] = 0
        total = sum(np.sum(a) for a in arrays)
        scale = 1.
        if total > 1:
            if total > 1.01 and param.renorm_warnings:
                _logger.warning("Renormalizing probability density from %f to 1. Decrease dt and/or dx.", total)
            for a in arrays:
                a /= total
            scale = 1/total
        solution = Solution(upper, lower, self, conditions=conditions,
                            pdf_undec=undec, pdf_evolution=evolution)
        if cn_last_step is not None and cn_last_step > 2:
            # CN conserves probability by storing half the final flux in its
            # last integrated node.  Keep that mass for cdf/prob/sample, but
            # expose the full nodal flux when a pointwise PDF is requested.
            corrections = (np.zeros_like(upper), np.zeros_like(lower))
            corrections[0][cn_last_step] = solution.choice_upper[cn_last_step]
            corrections[1][cn_last_step] = solution.choice_lower[cn_last_step]
            solution._pdf_correction = corrections
        elif cn_last_step and cn_end_flux is not None:
            # During the two startup intervals the node stores half the
            # interval mass, rather than half the final instantaneous flux.
            corrections = (np.zeros_like(upper), np.zeros_like(lower))
            corrections[0][cn_last_step] = max(0., cn_end_flux[0])*scale*solution._choice_rescale - solution.choice_upper[cn_last_step]
            corrections[1][cn_last_step] = max(0., cn_end_flux[1])*scale*solution._choice_rescale - solution.choice_lower[cn_last_step]
            solution._pdf_correction = corrections
        return self._overlay._apply_with_pdf_correction(solution)

    def _solve_numerical_python(self, method, conditions, return_evolution=False):
        times = self.t_domain()
        x = self.x_domain(conditions=conditions)
        pdf = self.IC(conditions=conditions).copy()
        upper, lower = np.zeros(len(times)), np.zeros(len(times))
        bound = self._bounddep.get_bound(t=0, conditions=conditions)
        lower[0], upper[0] = self._fp_absorb(pdf, x, bound)
        evolution = np.zeros((len(x), len(times))) if return_evolution else None
        if evolution is not None:
            evolution[:, 0] = pdf
        varying_bound = self._bounddep._uses_t()
        varying = varying_bound or self._driftdep._uses_t() or self._noisedep._uses_t()
        old = self._fp_matrix(x, 0, bound, conditions)
        implicit_matrix = None
        cn_rhs = None
        if not varying and method != 'explicit':
            first, stop = old[1][0], old[2][0]+1
            theta = .5 if method == 'cn' else 1
            # For constant coefficients, CN and its BE half steps share
            # the same left-hand matrix.  Cache only within this solve.
            implicit_matrix = TriDiagMatrix.eye(stop-first) + theta*self.dt*old[0].splice(first, stop)
            if method == 'cn':
                cn_rhs = TriDiagMatrix.eye(stop-first) - theta*self.dt*old[0].splice(first, stop)
        if method == 'explicit' and not varying:
            assert self._fp_explicit_stable(old), "Explicit solver is unstable for these dx, dt, drift, noise, and bounds"
        last_cn_step = 0
        for i in range(1, len(times)):
            # Check the absolute mass so signed CN roundoff cannot cancel.
            # A varying explicit model must still validate every time step.
            if (i-1) % 10 == 0 and not (method == 'explicit' and varying):
                remaining = np.sum(np.abs(pdf))
                if remaining == 0 or (evolution is None and remaining < param.numerical_mass_tolerance):
                    break
            t, step = times[i], self.dt
            if varying_bound:
                bound = self._bounddep.get_bound(t=t, conditions=conditions)
            if method == 'explicit' and varying:
                # Validate the operator that will actually advance this step,
                # instead of building every operator in a separate preflight.
                assert self._fp_explicit_stable(old), "Explicit solver is unstable for these dx, dt, drift, noise, and bounds"
            if method == 'implicit' and varying_bound:
                lower[i], upper[i] = self._fp_absorb(pdf, x, bound)
                if bound <= 0:
                    break  # All mass has been absorbed, even if bounds reopen.
            new = self._fp_matrix(x, t, bound, conditions) if varying else old
            if method == 'cn':
                if i <= 2:
                    # Rannacher startup: four backward Euler half steps
                    # damp the grid-scale modes of a point initial condition.
                    # Interpolate the operator at the first half step.  Only
                    # these two startup intervals use this interpolation.
                    middle = ((old[0]+new[0])*.5,
                              (old[1][0], .5*(old[1][1]+new[1][1])),
                              (old[2][0], .5*(old[2][1]+new[2][1])))
                    pdf, half_lower, half_upper = self._fp_step(pdf, old, middle, step*.5, 1, implicit_matrix)
                    pdf, out_lower, out_upper = self._fp_step(pdf, middle, new, step*.5, 1, implicit_matrix)
                    # Both half-step masses belong to this time interval.
                    # Split its mass between the adjacent time nodes, so
                    # the output remains centred on the existing time grid.
                    lower[i] = .5*(half_lower + out_lower)
                    upper[i] = .5*(half_upper + out_upper)
                    lower[i-1] += lower[i]
                    upper[i-1] += upper[i]
                else:
                    if i == 3:
                        # Complete the last startup node with the old-state
                        # contribution from the first trapezoidal interval.
                        lower[i-1] += .5*step*old[1][1]*pdf[old[1][0]]
                        upper[i-1] += .5*step*old[2][1]*pdf[old[2][0]]
                    if cn_rhs is not None:
                        pdf[first:stop] = implicit_matrix.spsolve(cn_rhs.dot(pdf[first:stop]))
                    else:
                        pdf, _, _ = self._fp_step(pdf, old, new, step, .5, implicit_matrix)
                    # Interior nodes receive two equal trapezoidal half
                    # weights. Correct the terminal node once after looping.
                    lower[i] = step*new[1][1]*pdf[new[1][0]]
                    upper[i] = step*new[2][1]*pdf[new[2][0]]
                last_cn_step = i
            elif method == 'implicit':
                pdf, out_lower, out_upper = self._fp_step(pdf, new, new, step, 1, implicit_matrix)
                lower[i] += out_lower
                upper[i] += out_upper
            else:
                pdf, lower[i], upper[i] = self._fp_step(pdf, old, old, step, 0)
                if varying_bound:
                    out_lower, out_upper = self._fp_absorb(pdf, x, bound)
                    lower[i] += out_lower
                    upper[i] += out_upper
            old = new
            if evolution is not None:
                evolution[:, i] = pdf
        if last_cn_step > 2:
            # Also applies if the tail cutoff ended integration early.
            lower[last_cn_step] *= .5
            upper[last_cn_step] *= .5
        end_flux = None
        if method == 'cn' and 0 < last_cn_step <= 2:
            end_flux = (self.dt*old[2][1]*pdf[old[2][0]], self.dt*old[1][1]*pdf[old[1][0]])
        return self._finish_numerical(upper, lower, conditions,
                                      undec=None if method == 'cn' else pdf, evolution=evolution,
                                      cn_last_step=last_cn_step if method == 'cn' else None,
                                      cn_end_flux=end_flux)


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
        constant bounds. Its returned choice arrays contain conserved
        probability masses at the existing time nodes; see
        solve_numerical_cn for the endpoint quadrature weights.

        Implicit is the fallback method.  It should work well in most
        cases and is generally stable.

        Normally, the explicit method should not be used.  Its step must
        preserve nonnegative probability at every interior point.  On a
        uniform grid this requires noise**2 * dt/dx**2 <= 1 and
        abs(drift)*dx <= noise**2.  Fractional boundary cells can require
        smaller steps; can_solve_explicit checks the actual operator.

        It returns a Solution object describing the joint PDF.  This
        method should not fail for any model type.

        `return_evolution` (default=False) governs whether or not the function 
        returns the full evolution of the pdf as part of the Solution object. 
        This only works with methods "explicit" or "implicit", not with "cn".

        `force_python` makes PyDDM use the solver written in Python instead of
        the optimized solver written in C.

        Numerical solves may stop when the remaining absolute probability
        mass is below `pyddm.parameters.numerical_mass_tolerance` (1e-14 by
        default).  Set this to zero to integrate the complete tail.  Saved
        evolution always integrates the complete tail.

        """
        self.check_conditions_satisfied(conditions)
        if method == "cn":
            if return_evolution == False:
                if HAS_CSOLVE and not force_python:
                    return self.solve_numerical_cn_c(conditions=conditions)
                return self.solve_numerical_cn(conditions=conditions)
            else:
                _logger.warning("return_evolution is not supported with the Crank-Nicolson solver, using implicit (backward Euler) instead.")
                method = "implicit"
        if method == "implicit" and HAS_CSOLVE and not force_python and not return_evolution:
            return self.solve_numerical_c(conditions=conditions)

        return self._solve_numerical_python(method, conditions, return_evolution)

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

        This uses Crank-Nicolson after two Rannacher startup intervals
        (four backward Euler half steps). The returned choice arrays
        contain the absorbed probability, placed at t_domain() using the
        time integrator's quadrature weights. Each startup interval's
        absorbed mass is shared equally between its endpoint nodes.
        Subsequent intervals contribute dt/2 times the boundary flux at
        each endpoint. Thus ordinary interior nodes contain dt times the
        nodal density, while the final node has only a half weight. This
        preserves both probability and the interior density's time
        alignment, without counting absorption after the final time.

        Solution.pdf reports the unweighted nodal density, while cdf,
        prob, and sample use the conserved masses. The final undecided spatial
        distribution is not returned, since CN states can oscillate.

        It returns a Solution object describing the joint PDF.
        """
        self.check_conditions_satisfied(conditions)
        assert self.can_solve_cn(conditions=conditions), "Crank-Nicolson requires time-independent bounds"
        return self._solve_numerical_python("cn", conditions)


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
