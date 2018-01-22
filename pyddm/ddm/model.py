import copy
import itertools
from functools import lru_cache

import numpy as np
from scipy import sparse, randn
import scipy.sparse.linalg

from . import parameters as param
from .analytic import analytic_ddm
from .models.mu import MuConstant, Mu
from .models.sigma import SigmaConstant, Sigma
from .models.ic import ICPointSourceCenter, InitialCondition
from .models.bound import BoundConstant, BoundCollapsingLinear, Bound
from .models.overlay import OverlayNone, Overlay


# This speeds up the code by about 10%.
sparse.csr_matrix.check_format = lambda self, full_check=True : True
    
# TODO:
# - Ensure that OverlayChain doesn't have two parameters with the same name

# This describes how a variable is dependent on other variables.
# Principally, we want to know how mu and sigma depend on x and t.
# `name` is the type of dependence (e.g. "linear") for methods which
# implement the algorithms, and any parameters these algorithms could
# need should be passed as kwargs. To compare to legacy code, the
# `name` used to be `f_mu_setting` or `f_sigma_setting` and
# kwargs now encompassed (e.g.) `param_mu_t_temp`.



##Pre-defined list of models that can be used, and the corresponding default parameters
class Model(object):
    """A full simulation of a single DDM-style model.

    Each model simulation depends on five key components:
    
    - A description of how drift rate (mu) changes throughout the simulation.
    - A description of how variability (sigma) changes throughout the simulation.
    - A description of how the boundary changes throughout the simulation.
    - Starting conditions for the model
    - Specific details of a task which cause dynamic changes in the model (e.g. a stimulus intensity change)

    This class manages these, and also provides the affiliated
    services, such as analytical or numerical simulations of the
    resulting reaction time distribution.
    """
    def __init__(self, mu=MuConstant(mu=0),
                 sigma=SigmaConstant(sigma=1),
                 bound=BoundConstant(B=1),
                 IC=ICPointSourceCenter(),
                 overlay=OverlayNone(), name="",
                 dx=param.dx, dt=param.dt, T_dur=param.T_dur):
        """Construct a Model object from the 5 key components.

        The five key components of our DDM-style models describe how
        the drift rate (`mu`), noise (`sigma`), and bounds (`bound`)
        change over time, and the initial conditions (`IC`).

        These five components are given by the parameters `mu`,
        `sigma`, `bound`, and `IC`, respectively.  They should be
        types which inherit from the types Mu, Sigma, Bound, and
        InitialCondition, respectively.  They default to constant
        unitary values.

        Additionally, simulation parameters can be set, such as time
        and spacial precision (`dt` and `dx`) and the simulation
        duration `T_dur`.  If not specified, they will be taken from
        the defaults specified in the parameters file.

        The `name` parameter is exclusively for convenience, and may
        be used in plotting or in debugging.
        """
        assert isinstance(name, str)
        self.name = name
        assert isinstance(mu, Mu)
        self._mudep = mu
        assert isinstance(sigma, Sigma)
        self._sigmadep = sigma
        assert isinstance(bound, Bound)
        self._bounddep = bound
        assert isinstance(IC, InitialCondition)
        self._IC = IC
        assert isinstance(overlay, Overlay)
        self._overlay = overlay
        self.dependencies = [self._mudep, self._sigmadep, self._bounddep, self._IC, self._overlay]
        self.required_conditions = list(set([x for l in self.dependencies for x in l.required_conditions]))
        self.dx = dx
        self.dt = dt
        self.T_dur = T_dur
    # Get a string representation of the model
    def __repr__(self, pretty=False):
        # Use a list so they can be sorted
        allobjects = [("name", self.name), ("mu", self.get_dependence('mu')),
                      ("sigma", self.get_dependence('sigma')), ("bound", self.get_dependence('bound')),
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
        return type(self).__name__ + "(" + params + ")"
    def __str__(self):
        return self.__repr__(pretty=True)
    def get_model_parameters(self):
        """Get an ordered list of all model parameters.
        
        Returns a list of each model parameter which can be varied during
        a fitting procedure.  The ordering is arbitrary but is guaranteed
        to be in the same order as set_model_parameters().
        """
        params = []
        for d in self.dependencies:
            for p in d.required_parameters:
                params.append(getattr(d, p))
        return params
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
        if name.lower() in ["mu", "mudep", "_mudep"]:
            return self._mudep
        elif name.lower() in ["sigma", "sigmadep", "_sigmadep"]:
            return self._sigmadep
        elif name.lower() in ["b", "bound", "bounddep", "_bounddep"]:
            return self._bounddep
        elif name.lower() in ["ic", "initialcondition", "_ic"]:
            return self._IC
        elif name.lower() in ["overlay", "_overlay"]:
            return self._overlay
        raise NameError("Invalid dependence name")

    def get_model_type(self):
        """Return a dictionary which fully specifies the class of the five key model components."""
        tt = lambda x : (x.depname, type(x))
        return dict(map(tt, self.dependencies))
    def x_domain(self, conditions):
        """A list which spans from the lower boundary to the upper boundary by increments of dx."""
        B = self.get_dependence("bound").B_base(conditions=conditions)
        return np.arange(-B, B+0.1*self.dx, self.dx) # +.1*dx is to ensure that the largest number in the array is B
    def t_domain(self):
        """A list of all of the timepoints over which the joint PDF will be defined (increments of dt from 0 to T_dur)."""
        return np.arange(0., self.T_dur+0.1*self.dt, self.dt)
    @staticmethod
    @lru_cache(maxsize=4)
    def _cache_eye(m, format="csr"):
        """Cache the call to sparse.eye since this decreases runtime by 5%."""
        return sparse.eye(m, format=format)
    def diffusion_matrix(self, x, t, conditions):
        """The matrix for the implicit method of solving the diffusion equation.

        - `x` - a length N ndarray representing the domain over which
          the matrix is to be defined. Usually a contiguous subset of
          x_domain().
        - `t` - The timepoint at which the matrix is valid.

        Returns a size NxN scipy sparse array.
        """
        mu_matrix = self.get_dependence('mu').get_matrix(x=x, t=t, dt=self.dt, dx=self.dx, conditions=conditions)
        sigma_matrix = self.get_dependence('sigma').get_matrix(x=x, t=t, dt=self.dt, dx=self.dx, conditions=conditions)
        return self._cache_eye(len(x), format="csr") + mu_matrix + sigma_matrix
    def flux(self, x, t, conditions):
        """The flux across the boundary at position `x` at time `t`."""
        mu_flux = self.get_dependence('mu').get_flux(x, t, dx=self.dx, dt=self.dt, conditions=conditions)
        sigma_flux = self.get_dependence('sigma').get_flux(x, t, dx=self.dx, dt=self.dt, conditions=conditions)
        return mu_flux + sigma_flux
    def IC(self, conditions):
        """The initial distribution at t=0.

        Returns a length N ndarray (where N is the size of x_domain())
        which should sum to 1.
        """
        return self.get_dependence('IC').get_IC(self.x_domain(conditions=conditions), dx=self.dx, conditions=conditions)

    def simulate_trial(self, conditions=None):
        """Simulate the decision variable for one trial.

        Given conditions `conditions`, this function will simulate the
        decision variable for a single trial.  It will *not* cut the
        simulation off when it goes beyond the boundary.  This returns
        a trajectory of the simulated trial over time as a numpy
        array.
        """

        # TODO this doesn't support OU models since it doesn't take X.
        T = self.t_domain()
        if conditions is None:
            conditions = {}
        mu = np.asarray([self.get_dependence("mu").get_mu(t=t, dx=self.dx, dt=self.dt, conditions=conditions) for t in T])
        sigma = np.asarray([self.get_dependence("sigma").get_sigma(t=t, dx=self.dx, dt=self.dt, conditions=conditions) for t in T])
        randnorm = randn(len(T))
        pos = np.cumsum(mu*self.dt + sigma*randnorm*np.sqrt(self.dt))
        return pos

    def simulated_solution(self, conditions=None, size=1000):
        """Simulate individual trials to obtain a distribution.

        Given conditions `conditions` and the number `size` of trials
        to simulate, this will run the function "simulate_trial"
        `size` times, and use the result to find a histogram analogous
        to solve.  Returns a sample object.
        """

        # TODO this doesn't support OU models.  It could also be made
        # more efficient by stopping the simulation once it has
        # crossed threshold.
        if conditions is None:
            conditions = {}
        corr_times = []
        err_times = []
        undec_count = 0
        for _ in range(0, size):
            timecourse = self.simulate_trial(conditions=conditions)
            bound = np.asarray([self.get_dependence("bound").get_bound(t, conditions=conditions) for t in self.t_domain()])
            cross_corr = [i for i in range(0, len(timecourse)) if bound[i] <= timecourse[i]]
            cross_err = [i for i in range(0, len(timecourse)) if -bound[i] >= timecourse[i]]
            if (cross_corr and cross_err and cross_corr[0] < cross_err[0]) or (cross_corr and not cross_err):
                corr_times.append(self.t_domain()[cross_corr[0]])
            elif (cross_err and cross_corr and cross_err[0] < cross_corr[0]) or (cross_err and not cross_corr):
                err_times.append(self.t_domain()[cross_err[0]])
            elif (not cross_corr) and (not cross_err):
                undec_count += 1
            else:
                raise ValueError("Internal error")
        return Sample(corr_times, err_times, undec_count)

    def has_analytical_solution(self):
        """Is it possible to find an analytic solution for this model?"""
        mt = self.get_model_type()
        return mt["Mu"] == MuConstant and mt["Sigma"] == SigmaConstant and \
            (mt["Bound"] in [BoundConstant, BoundCollapsingLinear]) and \
            mt["IC"] == ICPointSourceCenter
        
    def solve(self, conditions=None):
        """Solve the model using an analytic solution if possible, and a numeric solution if not.

        Return a Solution object describing the joint PDF distribution of reaction times."""
        # Don't use {} as a default argument since it is mutable
        if conditions is None:
            conditions = {}
        if self.has_analytical_solution():
            return self.solve_analytical(conditions=conditions)
        else:
            return self.solve_numerical(conditions=conditions)

    def solve_analytical(self, conditions=None):
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
        # Don't use {} as a default argument since it is mutable
        if conditions is None:
            conditions = {}
        # The analytic_ddm function does the heavy lifting.
        if isinstance(self.get_dependence('bound'), BoundConstant): # Simple DDM
            anal_pdf_corr, anal_pdf_err = analytic_ddm(self.get_dependence("mu").get_mu(t=0, conditions=conditions),
                                                       self.get_dependence("sigma").get_sigma(t=0, conditions=conditions),
                                                       self.get_dependence("bound").B_base(conditions=conditions), self.t_domain())
        elif isinstance(self.get_dependence('bound'), BoundCollapsingLinear): # Linearly Collapsing Bound
            anal_pdf_corr, anal_pdf_err = analytic_ddm(self.get_dependence("mu").get_mu(t=0, conditions=conditions),
                                                       self.get_dependence("sigma").get_sigma(t=0, conditions=conditions),
                                                       self.get_dependence("bound").B_base(conditions=conditions),
                                                       self.t_domain(), -self.get_dependence("bound").t) # TODO why must this be negative? -MS

        ## Remove some abnormalities such as NaN due to trivial reasons.
        anal_pdf_corr[anal_pdf_corr==np.NaN] = 0. # FIXME Is this a bug? You can't use == to compare nan to nan...
        anal_pdf_corr[0] = 0.
        anal_pdf_err[anal_pdf_err==np.NaN] = 0.
        anal_pdf_err[0] = 0.
        return self.get_dependence('overlay').apply(Solution(anal_pdf_corr*self.dt, anal_pdf_err*self.dt, self, conditions=conditions))

    def solve_numerical(self, conditions=None):
        """Solve the DDM model numerically.

        This uses the implicit method to solve the DDM at each
        timepoint.  Results are then compiled together.  This is the
        core DDM solver of this library.

        It returns a Solution object describing the joint PDF.  This
        method should not fail for any model type.
        """
        # Don't use {} as a default argument since it is mutable
        if conditions is None:
            conditions = {}
        ### Initialization: Lists
        pdf_curr = self.IC(conditions=conditions) # Initial condition
        pdf_prev = np.zeros((len(pdf_curr)))
        # If pdf_corr + pdf_err + undecided probability are summed, they
        # equal 1.  So these are componets of the joint pdf.
        pdf_corr = np.zeros(len(self.t_domain())) # Not a proper pdf on its own (doesn't sum to 1)
        pdf_err = np.zeros(len(self.t_domain())) # Not a proper pdf on its own (doesn't sum to 1)
        x_list = self.x_domain(conditions=conditions)

        # Looping through time and updating the pdf.
        for i_t, t in enumerate(self.t_domain()[:-1]): # -1 because nothing will happen at t=0 so each step computes the value for the next timepoint
            # Update Previous state. To be frank pdf_prev could be
            # removed for max efficiency. Leave it just in case.
            pdf_prev = copy.copy(pdf_curr)

            # For efficiency only do diffusion if there's at least
            # some densities remaining in the channel.
            if sum(pdf_curr[:])>0.0001:
                ## Define the boundaries at current time.
                bound = self.get_dependence('bound').get_bound(t, conditions=conditions) # Boundary at current time-step.

                # Now figure out which x positions are still within
                # the (collapsing) bound.
                assert self.get_dependence("bound").B_base(conditions=conditions) >= bound, "Invalid change in bound" # Ensure the bound didn't expand
                bound_shift = self.get_dependence("bound").B_base(conditions=conditions) - bound
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
                diffusion_matrix = self.diffusion_matrix(x_list_inbounds, t, conditions=conditions)

                ### Compute Probability density functions (pdf)
                # PDF for outer matrix
                pdf_outer = sparse.linalg.spsolve(diffusion_matrix, pdf_prev[x_index_outer:len(x_list)-x_index_outer])
                # If the bounds are the same the bound perfectly
                # aligns with the grid), we don't need so solve the
                # diffusion matrix again since we don't need a linear
                # approximation.
                if x_index_inner == x_index_outer:
                    pdf_inner = copy.copy(pdf_outer)
                else:
                    pdf_inner = sparse.linalg.spsolve(diffusion_matrix[1:-1, 1:-1], pdf_prev[x_index_inner:len(x_list)-x_index_inner])

                # Pdfs out of bound is consideered decisions made.
                pdf_err[i_t+1] += weight_outer * np.sum(pdf_prev[:x_index_outer]) \
                                  + weight_inner * np.sum(pdf_prev[:x_index_inner])
                pdf_corr[i_t+1] += weight_outer * np.sum(pdf_prev[len(x_list)-x_index_outer:]) \
                                   + weight_inner * np.sum(pdf_prev[len(x_list)-x_index_inner:])
                # Reconstruct current proability density function,
                # adding outer and inner contribution to it.  Use
                # .fill() method to avoid alocating memory with
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
            if len(pdf_inner) == 0: # Fix error when bounds collapse to 0
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

        return self.get_dependence('overlay').apply(Solution(pdf_corr, pdf_err, self, conditions=conditions))

class _Sample_Iter_Wraper(object):
    """Provide an iterator for sample objects.

    `sample_obj` is the Sample which we plan to iterate.  `correct`
    should be either True (to iterate through correct responses) or
    False (to iterate through error responses).

    Each step of the iteration returns a two-tuple, where the first
    element is the reaction time, and the second element is a
    dictionary of conditions.
    """
    def __init__(self, sample_obj, correct):
        self.sample = sample_obj
        self.i = 0
        self.correct = correct
    def __iter__(self):
        return self
    def next(self):
        if self.i == len(self.sample):
            raise StopIteration
        self.i += 1
        if self.correct:
            rt = self.sample.corr
            ind = 0
        elif not self.correct:
            rt = self.sample.err
            ind = 1
        return (rt[self.i-1], {k : self.sample.conditions[k][ind][self.i-1] for k in self.sample.conditions.keys()})
        
# TODO require passing non-decision trials
class Sample(object):
    """Describes a sample from some (empirical or simulated) distribution.

    Similarly to Solution, this is a glorified container for three
    items: a list of correct reaction times, a list of error reaction
    times, and the number of non-decision trials.  Each can have
    different properties associated with it, known as "conditions"
    elsewhere in this codebase.  This is to specifiy the experimental
    parameters of the trial, to allow fitting of stimuli by (for
    example) color or intensity.

    To specify conditions, pass a keyword argument to the constructor.
    The name should be the name of the property, and the value should
    be a tuple of length two or three.  The first element of the tuple
    should be a list of length equal to the number of correct trials,
    and the second should be equal to the number of error trials.  If
    there are any non-decision trials, the third argument should
    contain a list of length equal to `non_decision`.

    Optionally, additional data can be associated with each
    independent data point.  These should be passed as keyword
    arguments, where the keyword name is the property and the value is
    a tuple.  The tuple should have either two or three elements: the
    first two should be lists of properties for the correct and error
    reaction times, where the properties correspond to reaction times
    in the correct or error lists.  Optionally, a third list of length
    equal to the number of non-decision trials gives a list of
    conditions for these trials.  If multiple properties are passed as
    keyword arguments, the ordering of the non-decision time
    properties (in addition to those of the correct and error
    distributions) will correspond to one another.
    """
    def __init__(self, sample_corr, sample_err, non_decision=0, **kwargs):
        self.corr = sample_corr
        self.err = sample_err
        self.non_decision = non_decision
        # Make sure the kwarg parameters/conditions are in the correct
        # format
        for _,v in kwargs.items():
            assert isinstance(v, tuple)
            assert len(v) in [2, 3]
            assert len(v[0]) == len(self.corr)
            assert len(v[1]) == len(self.err)
            if len(v) == 3:
                assert len(v[2]) == non_decision
            else:
                assert non_decision == 0
        self.conditions = kwargs
    def __len__(self):
        """The number of samples"""
        return len(self.corr) + len(self.err) + self.non_decision
    def __iter__(self):
        """Iterate through each reaction time, with no regard to whether it was a correct or error trial."""
        return np.concatenate([self.corr, self.err]).__iter__()
    def __add__(self, other):
        assert sorted(self.conditions.keys()) == sorted(other.conditions.keys()), "Canot add with unlike conditions"
        corr = self.corr + other.corr
        err = self.err + other.err
        non_decision = self.non_decision + other.non_decision
        conditions = {}
        for k in self.conditions.keys():
            sc = self.conditions
            oc = other.conditions
            conditions[k] = (sc[k][0]+oc[k][0], sc[k][1]+oc[k][1],
                             (sc[k][2] if len(sc[k]) == 3 else [])
                             + (oc[k][2] if len(oc[k]) == 3 else []))
        return Sample(corr, err, non_decision, **conditions)
    @staticmethod
    def from_numpy_array(data, column_names):
        """Generate a Sample object from a numpy array.
        
        `data` should be an n x m array (n rows, m columns) where
        m>=2. The first column should be the response times, and the
        second column should be whether the trial was correct or an
        error (1 == correct, 0 == error).  Any remaining columns
        should be conditions.  `column_names` should be a list of
        strings of length m indicating the names of the conditions.
        The first two values can be anything, since these correspond
        to special columns as described above.  (However, for the
        bookkeeping, it might be wise to make them "rt" and "correct"
        or something of the sort.)  Remaining elements are the
        condition names corresponding to the columns.  This function
        does not yet work with no-decision trials.
        """
        # TODO this function doesn't do validity checks yet
        c = data[:,1].astype(bool)
        nc = (1-data[:,1]).astype(bool)
        def pt(x): # Pythonic types
            arr = np.asarray(x)
            if np.all(arr == np.round(arr)):
                arr = arr.astype(int)
            return arr.tolist()

        conditions = {k: (pt(data[c,i+2]), pt(data[nc,i+2]), []) for i,k in enumerate(column_names[2:])}
        return Sample(pt(data[c,0]), pt(data[nc,0]), 0, **conditions)
    def items(self, correct):
        """Iterate through the reaction times.

        This takes only one argument: a boolean `correct`, true if we
        want to iterate through the correct trials, and false if we
        want to iterate through the error trials.  

        For each iteration, a two-tuple is returned.  The first
        element is the reaction time, the second is a dictionary
        containing the conditions associated with that reaction time.
        """
        return _Sample_Iter_Wraper(self, correct=correct)
    def subset(self, **kwargs):
        """Subset the data by filtering based on specified properties.

        Each keyword argument should be the name of a property.  These
        keyword arguments may have one of three values:

        - A list: For each element in the returned subset, the
          specified property is in this list of values.
        - A function: For each element in the returned subset, the
          specified property causes the function to evaluate to True.
        - Anything else: Each element in the returned subset must have
          this value for the specified property.

        Return a sample object representing the filtered sample.
        """
        mask_corr = np.ones(len(self.corr)).astype(bool)
        mask_err = np.ones(len(self.err)).astype(bool)
        mask_non = np.ones(self.non_decision).astype(bool)
        for k,v in kwargs.items():
            if hasattr(v, '__call__'):
                mask_corr = np.logical_and(mask_corr, map(v, self.conditions[k][0]))
                mask_err = np.logical_and(mask_err, map(v, self.conditions[k][1]))
                mask_non = [] if self.non_decision == 0 else np.logical_and(mask_non, map(v, self.conditions[k][2]))
            if hasattr(v, '__contains__'):
                mask_corr = np.logical_and(mask_corr, [i in v for i in self.conditions[k][0]])
                mask_err = np.logical_and(mask_err, [i in v for i in self.conditions[k][1]])
                mask_non = [] if self.non_decision == 0 else np.logical_and(mask_non, [i in v for i in self.conditions[k][2]])
            else:
                mask_corr = np.logical_and(mask_corr, [i == v for i in self.conditions[k][0]])
                mask_err = np.logical_and(mask_err, [i == v for i in self.conditions[k][1]])
                mask_non = [] if self.non_decision == 0 else np.logical_and(mask_non, [i == v for i in self.conditions[k][2]])
        filtered_conditions = {k : (list(itertools.compress(v[0], mask_corr)),
                                    list(itertools.compress(v[1], mask_err)),
                                    (list(itertools.compress(v[2], mask_non)) if len(v) == 3 else []))
                               for k,v in self.conditions.items()}
        return Sample(list(itertools.compress(self.corr, list(mask_corr))),
                      list(itertools.compress(self.err, list(mask_err))),
                      sum(mask_non),
                      **filtered_conditions)
                      
    def condition_names(self):
        """The names of conditions which hold some non-zero value in this sample."""
        return list(self.conditions.keys())
    def condition_values(self, cond):
        """The values of a condition that have at least one element in the sample.

        `cond` is the name of the condition from which to get the
        observed values.  Returns a list of these values.
        """
        cs = self.conditions
        return sorted(list(set(cs[cond][0]).union(set(cs[cond][1]))))
    def condition_combinations(self, required_conditions=None):
        """Get all values for set conditions and return every combination of them.

        Since PDFs of solved models in general depend on all of the
        conditions, this returns a list of dictionaries.  The keys of
        each dictionary are the names of conditions, and the value is
        a particular value held by at least one element in the sample.
        Each list contains all possible combinations of condition values.

        If `required_conditions` is iterable, only the conditions with
        names found within `required_conditions` will be included.
        """
        cs = self.conditions
        conditions = []
        names = self.condition_names()
        if required_conditions is not None:
            names = [n for n in names if n in required_conditions]
        for c in names:
            conditions.append(list(set(cs[c][0]).union(set(cs[c][1]))))
        combs = []
        for p in itertools.product(*conditions):
            if len(self.subset(**dict(zip(names, p)))) != 0:
                combs.append(dict(zip(names, p)))
        if len(combs) == 0:
            return [{}]
        return combs

    @staticmethod
    def t_domain(dt=.01, T_dur=2):
        """The times that corresponds with pdf/cdf_corr/err parameters (their support)."""
        return np.linspace(0, T_dur, T_dur/dt+1)

    def pdf_corr(self, dt=.01, T_dur=2):
        """The correct component of the joint PDF."""
        return np.histogram(self.corr, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self)/dt # dt/2 terms are for continuity correction

    def pdf_err(self, dt=.01, T_dur=2):
        """The error (incorrect) component of the joint PDF."""
        return np.histogram(self.err, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self)/dt # dt/2 terms are for continuity correction

    def cdf_corr(self, dt=.01, T_dur=2):
        """The correct component of the joint CDF."""
        return np.cumsum(self.pdf_corr(dt=dt, T_dur=T_dur))*dt

    def cdf_err(self, dt=.01, T_dur=2):
        """The error (incorrect) component of the joint CDF."""
        return np.cumsum(self.pdf_err(dt=dt, T_dur=T_dur))*dt

    def prob_correct(self):
        """The probability of selecting the right response."""
        return len(self.corr)/len(self)

    def prob_error(self):
        """The probability of selecting the incorrect (error) response."""
        return len(self.err)/len(self)

    def prob_undecided(self):
        """The probability of selecting neither response (undecided)."""
        return self.non_decision/len(self)

    def prob_correct_forced(self):
        """The probability of selecting the correct response if a response is forced."""
        return self.prob_correct() + self.prob_undecided()/2.

    def prob_error_forced(self):
        """The probability of selecting the incorrect response if a response is forced."""
        return self.prob_error() + self.prob_undecided()/2.

    def mean_decision_time(self):
        """The mean decision time in the correct trials (excluding undecided trials)."""
        return np.mean(self.corr)


    

class Solution(object):
    """Describes the result of an analytic or numerical DDM run.

    This is a glorified container for a joint pdf, between the
    response options (correct, error, and no decision) and the
    response time distribution associated with each.  It stores a copy
    of the response time distribution for both the correct case and
    the incorrect case, and the rest of the properties can be
    calculated from there.  

    It also stores a full deep copy of the model used to simulate it.
    This is most important for storing, e.g. the dt and the name
    associated with the simulation, but it is also good to keep the
    whole object as a full record describing the simulation, so that
    the full parametrization of every run is recorded.  Note that this
    may increase memory requirements when many simulations are run.

    """
    def __init__(self, pdf_corr, pdf_err, model, conditions):
        """Create a Solution object from the results of a model
        simulation.

        Constructor takes three arguments.

            - `model` - the Model object used to generate `pdf_corr` and `pdf_err`
            - `pdf_corr` - a size N numpy ndarray describing the correct portion of the joint pdf
            - `pdf_err` - a size N numpy ndarray describing the error portion of the joint pdf
        """
        self.model = copy.deepcopy(model) # TODO this could cause a memory leak if I forget it is there...
        self.corr = pdf_corr
        self._pdf_corr = pdf_corr # for backward compatibility
        self.err = pdf_err
        self._pdf_err = pdf_err # for backward compatibility
        self.conditions = conditions

    def pdf_corr(self):
        """The correct component of the joint PDF."""
        return self.corr/self.model.dt

    def pdf_err(self):
        """The error (incorrect) component of the joint PDF."""
        return self.err/self.model.dt

    def cdf_corr(self):
        """The correct component of the joint CDF."""
        return np.cumsum(self.corr)

    def cdf_err(self):
        """The error (incorrect) component of the joint CDF."""
        return np.cumsum(self.err)

    def prob_correct(self):
        """The probability of selecting the right response."""
        return np.sum(self.corr)

    def prob_error(self):
        """The probability of selecting the incorrect (error) response."""
        return np.sum(self.err)

    def prob_undecided(self):
        """The probability of selecting neither response (undecided)."""
        return 1 - self.prob_correct() - self.prob_error()

    def prob_correct_forced(self):
        """The probability of selecting the correct response if a response is forced."""
        return self.prob_correct() + self.prob_undecided()/2.

    def prob_error_forced(self):
        """The probability of selecting the incorrect response if a response is forced."""
        return self.prob_error() + self.prob_undecided()/2.

    def mean_decision_time(self):
        """The mean decision time in the correct trials (excluding undecided trials)."""
        return np.sum((self.corr)*self.model.t_domain()) / self.prob_correct()

    @staticmethod
    def _sample_from_histogram(hist, hist_bins, k, seed=0):
        """Generate a sample from a histogram.

        Given a histogram, imply the distribution and generate a
        sample. `hist` should be either a normalized histogram
        (i.e. summing to 1 or less for non-decision) or else a listing
        of bin membership counts, of size n.  (I.e. there are n bins.)
        The bins should be labeled by `hist_bins`, a list of size
        n+1. This will sample from the distribution and return a list
        of size `k`.  This uses the naive method of selecting
        uniformly from the area of each histogram bar according to the
        size of the bar, rather than non-parametrically estimating the
        distribution and then sampling from the distribution (which
        would arguably be better).
        """
        assert len(hist_bins) == len(hist) + 1, "Incorrect bin specification"
        rng = np.random.RandomState(seed)
        sample = []
        h = hist
        norm = np.round(np.sum(h), 2)
        assert norm <= 1 or int(norm) == norm, "Invalid histogram of size %f" % norm
        if norm >= 1:
            h = h/norm
            norm = 1
        hcum = np.cumsum(h)
        rns = rng.rand(k)
        for rn in rns:
            ind = next((i for i in range(0, len(hcum)) if rn < hcum[i]), np.nan)
            if np.isnan(ind):
                sample.append(np.nan)
            else:
                sample.append(rng.uniform(low=hist_bins[ind], high=hist_bins[ind+1]))
        return sample

    def resample(self, k=1, seed=0):
        """Generate a list of reaction times sampled from the PDF.

        `k` is the number of TRIALS, not the number of samples.  Since
        we are only showing the distribution from the correct trials,
        we guarantee that, for an identical seed, the sum of the two
        return values will be less than `k`.  If no non-decision
        trials exist, the sum of return values will be equal to `k`.

        This relies on the assumption that reaction time cannot be
        less than 0.

        Returns a tuple, where the first element is a list of correct
        reaction times, and the second element is a list of error
        reaction times.
        """
        # To sample from both correct and error distributions, we make
        # the correct answers be positive and the incorrect answers be
        # negative.
        pdf_domain = self.model.t_domain()
        assert pdf_domain[0] == 0, "Invalid PDF domain"
        combined_domain = list(reversed(np.asarray(pdf_domain)*-1)) + list(pdf_domain[1:])
        # Continuity correction doesn't work here.  Instead we say
        # that the histogram value at time dt belongs in the [0, dt]
        # bin, the value at time 2*dt belongs in the [dt, 2*dt] bin,
        # etc.
        assert self.pdf_err()[0] == 0 and self.pdf_corr()[0] == 0, "Invalid pdfs"
        combined_pdf = list(reversed(self.pdf_err()[1:]))+list(self.pdf_corr()[1:])
        sample = self._sample_from_histogram(np.asarray(combined_pdf)*self.model.dt, combined_domain, k, seed=seed)
        corr_sample = [x for x in sample if x >= 0]
        err_sample = [-x for x in sample if x < 0]
        non_decision = k - (len(corr_sample) + len(err_sample))
        conditions = {k : ([v]*len(corr_sample), [v]*len(err_sample), [v]*non_decision) for k,v in self.conditions.items()}
        return Sample(corr_sample, err_sample, non_decision, **conditions)

class Fittable(float):
    """For parameters that should be adjusted when fitting a model to data.
        
    Each Fittable object does not need any parameters, however several
    parameters may improve the ability to fit the model.  In
    particular, `maxval` and `minval` ensure we do not choose an
    invalid parameter value.  `default` is the value to start with
    when fitting; if it is not given, it will be selected at random.
    """
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

