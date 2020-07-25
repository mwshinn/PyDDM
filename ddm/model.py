# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from .tridiag import TriDiagMatrix

import inspect

from . import parameters as param
from .analytic import analytic_ddm
from .models.drift import DriftConstant, Drift
from .models.noise import NoiseConstant, Noise
from .models.ic import ICPointSourceCenter, InitialCondition
from .models.bound import BoundConstant, BoundCollapsingLinear, Bound
from .models.overlay import OverlayNone, Overlay
from .models.paranoid_types import Conditions
from .sample import Sample
from .solution import Solution
from .fitresult import FitResult, FitResultEmpty

from paranoid.types import Numeric, Number, Self, List, Generic, Positive, String, Boolean, Natural1, Natural0, Dict, Set, Integer, NDArray, Maybe, Nothing
from paranoid.decorators import accepts, returns, requires, ensures, paranoidclass, paranoidconfig
import dis
    
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
                 T_dur=param.T_dur, fitresult=None):
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
        self.dx = dx
        self.dt = dt
        if self.dx > .01:
            print("WARNING: dx is large.  Estimated pdfs may be imprecise.  Decrease dx to 0.01 or less.")
        if self.dt > .01:
            print("WARNING: dt is large.  Estimated pdfs may be imprecise.  Decrease dt to 0.01 or less.")
        self.T_dur = T_dur
        self.fitresult = FitResultEmpty() if fitresult is None else fitresult # If the model was fit, store the status here
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
        return type(self).__name__ + "(" + params + ")"
    def __str__(self):
        return self.__repr__(pretty=True)
    def get_model_parameters(self):
        """Get an ordered list of all model parameters.
        
        Returns a list of each model parameter which can be varied
        during a fitting procedure.  The ordering is arbitrary but is
        guaranteed to be in the same order as
        get_model_parameter_names() and set_model_parameters().
        """
        params = []
        for d in self.dependencies:
            for p in d.required_parameters:
                params.append(getattr(d, p))
        return params

    def get_model_parameter_names(self):
        """Get an ordered list of the names of all parameters in the model.

        Returns the name of each model parameter.  The ordering is
        arbitrary, but is uaranteed to be in the same order as
        get_model_parameters() and set_model_parameters().
        """
        params = []
        for d in self.dependencies:
            for p in d.required_parameters:
                params.append(p)
        return params

    def get_fit_result(self):
        """Returns a FitResult object describing how the model was fit.
        
        Returns the FitResult object describing the last time this
        model was fit to data, including the loss function, fitting
        method, and the loss function value.  If the model was never
        fit to data, this will return FitResultEmpty.
        """
        return self.fitresult
    
    def set_model_parameters(self, params):
        """Set the parameters of the model from an ordered list.

        Takes as an argument a list of parameters in the same order as
        those from get_model_parameters().  Sets the associated
        parameters.
        """
        assert len(params) == len(self.get_model_parameters()), "Invalid params"
        i = 0
        for d in self.dependencies:
            for p in d.required_parameters:
                setattr(d, p, params[i])
                i += 1

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
    @accepts(Self, Conditions, Maybe(Positive))
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
        if cutoff is False:
            if len(traj) < len(T):
                traj = np.append(traj, [traj[-1]]*(len(T)-len(traj)))
            elif len(traj) > len(T):
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
        Solution.resample() function of the resulting Solution.

        """
        corr_times = []
        err_times = []
        undec_count = 0

        T = self.t_domain()

        for s in range(0, size):
            if s % 200 == 0:
                print("Simulating trial %i" % s)
            timecourse = self.simulate_trial(conditions=conditions, seed=(hash((s, seed)) % 2**32), cutoff=True, rk4=rk4)
            T_finish = T[len(timecourse) - 1]
            B = self.get_dependence("bound").get_bound(t=T_finish, conditions=conditions)
            # Correct for the fact that the particle could have
            # crossed at any point between T_finish-dt and T_finish.
            dt_correction = self.dt/2
            # Determine whether the sim is a correct or error trial.
            if timecourse[-1] > B:
                corr_times.append(T_finish - dt_correction)
            elif timecourse[-1] < -B:
                err_times.append(T_finish - dt_correction)
            elif len(timecourse) == len(T):
                undec_count += 1
            else:
                raise SystemError("Internal error: Invalid particle simulation")
            
        aa = lambda x : np.asarray(x)
        conds = {k:(aa(len(corr_times)*[v]), aa(len(err_times)*[v]), aa(undec_count*[v])) for k,v in conditions.items() if k and v}
        return Sample(aa(corr_times), aa(err_times), undec_count, **conds)

    @accepts(Self)
    @returns(Boolean)
    def has_analytical_solution(self):
        """Is it possible to find an analytic solution for this model?"""
        mt = self.get_model_type()
        # First check to make sure drift doesn't vary with time or
        # particle location
        driftfuncsig = inspect.signature(mt["Drift"].get_drift)
        if "t" in driftfuncsig.parameters or "x" in driftfuncsig.parameters:
            return False
        # Check noise to make sure it doesn't vary with time or particle location
        noisefuncsig = inspect.signature(mt["Noise"].get_noise)
        if "t" in noisefuncsig.parameters or "x" in noisefuncsig.parameters:
            return False
        # Check to make sure bound is one that we can solve for
        if mt["Bound"] not in [BoundConstant, BoundCollapsingLinear]:
            return False
        # Make sure initial condition is at the center
        if mt["IC"] != ICPointSourceCenter:
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
        boundfuncsig = inspect.signature(self.get_dependence("bound").get_bound)
        if "t" in boundfuncsig.parameters:
            return False
        return True
    
    @accepts(Self, conditions=Conditions, return_evolution=Boolean)
    @returns(Solution)
    def solve(self, conditions={}, return_evolution=False):
        """Solve the model using an analytic solution if possible, and a numeric solution if not.

        Return a Solution object describing the joint PDF distribution of reaction times."""
        # TODO solves this using the dis module as described in the
        # comment for can_solve_cn
        self.check_conditions_satisfied(conditions)
        if self.has_analytical_solution() and return_evolution is False:
            return self.solve_analytical(conditions=conditions)
        elif isinstance(self.get_dependence("bound"), BoundConstant) and return_evolution is False:
            return self.solve_numerical_cn(conditions=conditions)
        else:
            return self.solve_numerical_implicit(conditions=conditions, return_evolution=return_evolution)

    @accepts(Self, conditions=Conditions)
    @returns(Solution)
    def solve_analytical(self, conditions={}):
        """Solve the model with an analytic solution, if possible.

        Analytic solutions are only possible in a select number of
        special cases; in particular, it works for simple DDM and for
        linearly collapsing bounds.  (See Anderson (1960) for
        implementation details.)  For most reasonably complex models,
        the method will fail.  Check whether a solution is possible
        with has_analytic_solution().

        If successful, this returns a Solution object describing the
        joint PDF.  If unsuccessful, this will raise an exception.
        """
        assert self.has_analytical_solution(), "Cannot solve for this model analytically"
        self.check_conditions_satisfied(conditions)
        # The analytic_ddm function does the heavy lifting.
        if isinstance(self.get_dependence('bound'), BoundConstant): # Simple DDM
            anal_pdf_corr, anal_pdf_err = analytic_ddm(self.get_dependence("drift").get_drift(t=0, conditions=conditions),
                                                       self.get_dependence("noise").get_noise(t=0, conditions=conditions),
                                                       self.get_dependence("bound").get_bound(t=0, conditions=conditions), self.t_domain())
        elif isinstance(self.get_dependence('bound'), BoundCollapsingLinear): # Linearly Collapsing Bound
            anal_pdf_corr, anal_pdf_err = analytic_ddm(self.get_dependence("drift").get_drift(t=0, conditions=conditions),
                                                       self.get_dependence("noise").get_noise(t=0, conditions=conditions),
                                                       self.get_dependence("bound").get_bound(t=0, conditions=conditions),
                                                       self.t_domain(), -self.get_dependence("bound").t) # TODO why must this be negative? -MS

        ## Remove some abnormalities such as NaN due to trivial reasons.
        anal_pdf_corr[anal_pdf_corr==np.NaN] = 0. # FIXME Is this a bug? You can't use == to compare nan to nan...
        anal_pdf_corr[0] = 0.
        anal_pdf_err[anal_pdf_err==np.NaN] = 0.
        anal_pdf_err[0] = 0.

        # Fix numerical errors
        pdfsum = (np.sum(anal_pdf_corr) + np.sum(anal_pdf_err))*self.dt
        if pdfsum > 1:
            if pdfsum > 1.01 and param.renorm_warnings:
                print("Warning: renormalizing probability density from", pdfsum, "to 1.  " \
                      "Try decreasing dt.  If that doesn't eliminate this warning, it may be due to " \
                      "extreme parameter values and/or bugs in your model speficiation.")
            anal_pdf_corr /= pdfsum
            anal_pdf_err /= pdfsum

        return self.get_dependence('overlay').apply(Solution(anal_pdf_corr*self.dt, anal_pdf_err*self.dt, self, conditions=conditions))

    @accepts(Self, method=Set(["explicit", "implicit", "cn"]), conditions=Conditions, return_evolution=Boolean)
    @returns(Solution)
    @requires("method == 'explicit' --> self.can_solve_explicit(conditions=conditions)")
    @requires("method == 'cn' --> self.can_solve_cn()")
    def solve_numerical(self, method="cn", conditions={}, return_evolution=False):
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
        
        return_evolution(default=False) governs whether or not the function 
        returns the full evolution of the pdf as part of the Solution object. 
        This only works with methods "explicit" or "implicit", not with "cn".
        """
        self.check_conditions_satisfied(conditions)
        if method == "cn":
            if return_evolution == False:
                return self.solve_numerical_cn(conditions=conditions)
            else:
                print("Warning: return_evolution is not supported with the Crank-Nicolson solver, using implicit (backward Euler) instead.")
                method = "implicit"

        # Initial condition of decision variable
        pdf_curr = self.IC(conditions=conditions)
        # Output correct and error pdfs.  If pdf_corr + pdf_err +
        # undecided probability are summed, they equal 1.  So these
        # are componets of the joint pdf.
        pdf_corr = np.zeros(len(self.t_domain())) # Not a proper pdf on its own (doesn't sum to 1)
        pdf_err = np.zeros(len(self.t_domain())) # Not a proper pdf on its own (doesn't sum to 1)
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
            drift_matrix = self.get_dependence('drift').get_matrix(x=x_list_inbounds, t=t, dt=self.dt, dx=self.dx, conditions=conditions)
            noise_matrix = self.get_dependence('noise').get_matrix(x=x_list_inbounds, t=t, dt=self.dt, dx=self.dx, conditions=conditions)
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
                if method == "implicit":
                    pdf_inner = diffusion_matrix.splice(1,-1).spsolve(pdf_prev[x_index_inner:len(x_list)-x_index_inner])
                elif method == "explicit":
                    pdf_inner = diffusion_matrix_explicit.splice(1,-1).dot(pdf_prev[x_index_inner:len(x_list)-x_index_inner])

            # Pdfs out of bound is considered decisions made.
            pdf_err[i_t+1] += weight_outer * np.sum(pdf_prev[:x_index_outer]) \
                              + weight_inner * np.sum(pdf_prev[:x_index_inner])
            pdf_corr[i_t+1] += weight_outer * np.sum(pdf_prev[len(x_list)-x_index_outer:]) \
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
            _inner_B_corr = x_list[len(x_list)-1-x_index_inner]
            _outer_B_corr = x_list[len(x_list)-1-x_index_outer]
            _inner_B_err = x_list[x_index_inner]
            _outer_B_err = x_list[x_index_outer]
            if len(pdf_inner) == 0: # Otherwise we get an error when bounds collapse to 0
                pdf_inner = np.array([0])
            pdf_corr[i_t+1] += weight_outer * pdf_outer[-1] * self.flux(_outer_B_corr, t, conditions=conditions) \
                            +  weight_inner * pdf_inner[-1] * self.flux(_inner_B_corr, t, conditions=conditions)
            pdf_err[i_t+1]  += weight_outer * pdf_outer[0] * self.flux(_outer_B_err, t, conditions=conditions) \
                            +  weight_inner * pdf_inner[0] * self.flux(_inner_B_err, t, conditions=conditions)

            # Renormalize when the channel size has <1 grid, although
            # all hell breaks loose in this regime.
            if bound < self.dx:
                pdf_corr[i_t+1] *= (1+ (1-bound/self.dx))
                pdf_err[i_t+1] *= (1+ (1-bound/self.dx))

            # If evolution of pdf should be returned, append pdf_curr to pdf_evolution
            if return_evolution:    
                pdf_evolution[:,i_t+1] = pdf_curr

        # Detect and fix below zero errors
        pdf_undec = pdf_curr
        minval = np.min((np.min(pdf_corr), np.min(pdf_err), np.min(pdf_undec)))
        if minval < 0:
            sum_negative_strength = np.sum(pdf_corr[pdf_corr<0]) + np.sum(pdf_err[pdf_err<0])
            sum_negative_strength_undec = np.sum(pdf_undec[pdf_undec<0])
            if sum_negative_strength < -.01 and param.renorm_warnings:
                print("Warning: probability density included values less than zero "
                      "(minimum=%f, total=%f).  "  \
                      "Please decrease dt and/or avoid extreme parameter values." % (minval, sum_negative_strength))
            if sum_negative_strength_undec < -.01 and param.renorm_warnings:
                print("Warning: remaining FP distribution included values less than zero " \
                      "(minimum=%f, total=%f).  " \
                      "Please decrease dt and/or avoid extreme parameter values." % (minval, sum_negative_strength_undec))
            pdf_corr[pdf_corr < 0] = 0
            pdf_err[pdf_err < 0] = 0
            pdf_undec[pdf_undec < 0] = 0
        # Fix numerical errors
        pdfsum = np.sum(pdf_corr) + np.sum(pdf_err) + np.sum(pdf_undec)
        if pdfsum > 1:
            if pdfsum > 1.01 and param.renorm_warnings:
                print("Warning: renormalizing probability density from", pdfsum, "to 1.  " \
                      "Try decreasing dt or using the implicit (backward Euler) method instead.  " \
                      "If that doesn't eliminate this warning, it may be due to " \
                      "extreme parameter values and/or bugs in your model speficiation.")
            pdf_corr /= pdfsum
            pdf_err /= pdfsum
            pdf_undec /= pdfsum

        if return_evolution:
            return self.get_dependence('overlay').apply(Solution(pdf_corr, pdf_err, self, conditions=conditions, pdf_undec=pdf_undec, pdf_evolution=pdf_evolution))
        
        return self.get_dependence('overlay').apply(Solution(pdf_corr, pdf_err, self, conditions=conditions, pdf_undec=pdf_undec))

    @accepts(Self, Conditions)
    @returns(Solution)
    @requires("self.can_solve_explicit(conditions=conditions)")
    def solve_numerical_explicit(self, conditions={}, **kwargs):
        """Solve the model using the explicit method (Forward Euler)."""
        return self.solve_numerical(method="explicit", conditions=conditions, **kwargs)

    @accepts(Self, Conditions)
    @returns(Solution)
    def solve_numerical_implicit(self, conditions={}, **kwargs):
        """Solve the model using the implicit method (Backward Euler)."""
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
        pdf_corr = np.zeros(len(self.t_domain())) # Not a proper pdf on its own (doesn't sum to 1)
        pdf_err = np.zeros(len(self.t_domain())) # Not a proper pdf on its own (doesn't sum to 1)
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


        # Looping through time and updating the pdf.
        for i_t, t in enumerate(self.t_domain()[:-1]): # -1 because nothing will happen at t=0 so each step computes the value for the next timepoint
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
                drift_matrix = self.get_dependence('drift').get_matrix(x=x_list_inbounds, t=t,
                                                                       dt=self.dt, dx=self.dx, conditions=conditions)
                drift_matrix *= .5
                noise_matrix = self.get_dependence('noise').get_matrix(x=x_list_inbounds,
                                                                       t=t, dt=self.dt, dx=self.dx, conditions=conditions)
                noise_matrix *= .5
                diffusion_matrix = TriDiagMatrix.eye(len(x_list_inbounds))
                diffusion_matrix += drift_matrix
                diffusion_matrix += noise_matrix

                drift_matrix_prev = self.get_dependence('drift').get_matrix(x=x_list_inbounds_prev, t=np.maximum(0,t-self.dt),
                                                                            dt=self.dt, dx=self.dx, conditions=conditions)
                drift_matrix_prev *= .5
                noise_matrix_prev = self.get_dependence('noise').get_matrix(x=x_list_inbounds_prev, t=np.maximum(0,t-self.dt),
                                                                            dt=self.dt, dx=self.dx, conditions=conditions)
                noise_matrix_prev *= .5
                diffusion_matrix_prev = TriDiagMatrix.eye(len(x_list_inbounds))
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
                    pdf_inner = diffusion_matrix.splice(si_from,si_to).spsolve(
                                                      diffusion_matrix_prev.splice(si2_from,si2_to).dot(pdf_inner_prev)[si3_from:si3_to])


                # Pdfs out of bound is considered decisions made.
                pdf_err[i_t+1] += weight_outer_prev * np.sum(pdf_outer_prev[:x_index_outer_shift]) \
                                + weight_inner_prev * np.sum(pdf_inner_prev[:x_index_inner_shift])
                pdf_corr[i_t+1] += weight_outer_prev * np.sum(pdf_outer_prev[len(pdf_outer_prev)-x_index_outer_shift:]) \
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
            _inner_B_corr = x_list[len(x_list)-1-x_index_inner]
            _outer_B_corr = x_list[len(x_list)-1-x_index_outer]
            _inner_B_err = x_list[x_index_inner]
            _outer_B_err = x_list[x_index_outer]
            flux_outer_B_corr = self.flux(_outer_B_corr, t, conditions=conditions)
            flux_inner_B_corr = self.flux(_inner_B_corr, t, conditions=conditions)
            flux_outer_B_err = self.flux(_outer_B_err, t, conditions=conditions)
            flux_inner_B_err = self.flux(_inner_B_err, t, conditions=conditions)
            if len(pdf_inner) == 0: # Otherwise we get an error when bounds collapse to 0
                pdf_inner = np.array([0])
            pdf_corr[i_t+1] += 0.5*weight_outer * pdf_outer[-1] * flux_outer_B_corr \
                            +  0.5*weight_inner * pdf_inner[-1] * flux_inner_B_corr
            pdf_err[i_t+1]  += 0.5*weight_outer * pdf_outer[0]  * flux_outer_B_err \
                            +  0.5*weight_inner * pdf_inner[0]  * flux_inner_B_err
            pdf_corr[i_t]   += 0.5*weight_outer * pdf_outer[-1] * flux_outer_B_corr \
                            +  0.5*weight_inner * pdf_inner[-1] * flux_inner_B_corr
            pdf_err[i_t]    += 0.5*weight_outer * pdf_outer[0]  * flux_outer_B_err \
                            +  0.5*weight_inner * pdf_inner[0]  * flux_inner_B_err

            # Renormalize when the channel size has <1 grid, although
            # all hell breaks loose in this regime.
            if bound < self.dx:
                pdf_corr[i_t+1] *= (1+ (1-bound/self.dx))
                pdf_err[i_t+1] *= (1+ (1-bound/self.dx))

        # Detect and fix below zero errors.  Here, we don't worry
        # about undecided probability as we did with the implicit
        # method, because CN tends to oscillate around zero,
        # especially when noise (sigma) is large.  The user would be
        # directed to decrease dt.
        pdf_undec = pdf_curr
        minval = np.min((np.min(pdf_corr), np.min(pdf_err)))
        if minval < 0:
            sum_negative_strength = np.sum(pdf_corr[pdf_corr<0]) + np.sum(pdf_err[pdf_err<0])
            # For small errors, don't bother alerting the user
            if sum_negative_strength < -.01 and param.renorm_warnings:
                print("Warning: probability density included values less than zero "
                      "(minimum=%f, total=%f).  " \
                      "Please decrease dt and/or avoid extreme parameter values." % (minval, sum_negative_strength))
            pdf_corr[pdf_corr < 0] = 0
            pdf_err[pdf_err < 0] = 0
        # Fix numerical errors
        pdfsum = np.sum(pdf_corr) + np.sum(pdf_err)
        if pdfsum > 1:
            # If it is only a small renormalization, don't bother alerting the user.
            if pdfsum > 1.01 and param.renorm_warnings:
                print("Warning: renormalizing probability density from", pdfsum, "to 1.  " \
                      "Try decreasing dt.  If that doesn't eliminate this warning, it may be due to " \
                      "extreme parameter values and/or bugs in your model speficiation.")
            pdf_corr /= pdfsum
            pdf_err /= pdfsum

        # TODO Crank-Nicolson still has something weird going on with pdf_curr near 0, where it seems to oscillate
        return self.get_dependence('overlay').apply(Solution(pdf_corr, pdf_err, self, conditions=conditions, pdf_undec=None))


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
    def __new__(cls, val=np.nan, **kwargs):
        if not np.isnan(val):
            print(val)
            raise ValueError("No positional arguments for Fittables")
        return float.__new__(cls, np.nan)
    def __init__(self, **kwargs):
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
    def default(self):
        return float(self)

