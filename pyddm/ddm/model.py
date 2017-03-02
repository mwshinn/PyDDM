import numpy as np
import copy
from . import parameters as param
from .analytic import analytic_ddm, analytic_ddm_linbound
from scipy import sparse
import scipy.sparse.linalg
import itertools

# TODO:
# - Clean up mu_base to return the value of mu at time t
# - Use mu_base as a default mu diffusion matrix implementation
# - Fix documentation

# This describes how a variable is dependent on other variables.
# Principally, we want to know how mu and sigma depend on x and t.
# `name` is the type of dependence (e.g. "linear") for methods which
# implement the algorithms, and any parameters these algorithms could
# need should be passed as kwargs. To compare to legacy code, the
# `name` used to be `f_mu_setting` or `f_sigma_setting` and
# kwargs now encompassed (e.g.) `param_mu_t_temp`.

class Dependence(object):
    """An abstract class describing how one variable depends on other variables.

    This is an abstract class which is inherrited by other abstract
    classes only, and has the highest level machinery for describing
    how one variable depends on others.  For example, an abstract
    class that inherits from Dependence might describe how the drift
    rate may change throughout the simulation depending on the value
    of x and t, and then this class would be inherited by a concrete
    class describing an implementation.  For example, the relationship
    between drift rate and x could be linear, exponential, etc., and
    each of these would be a subsubclass of Dependence.

    In order to subclass Dependence, you must set the (static) class
    variable `depname`, which gives an alpha-numeric string describing
    which variable could potentially depend on other variables.

    Each subsubclass of dependence must also define two (static) class
    variables.  First, it must define `name`, which is an
    alpha-numeric plus underscores name of what the algorithm is, and
    also `required_parameters`, a python list of names (strings) for
    the parameters that must be passed to this algorithm.  (This does
    not include globally-relevant variables like dt, it only includes
    variables relevant to a particular instance of the algorithm.)  An
    optional (static) class variable is `default_parameters`, which is
    a dictionary indexed by the parameter names from
    `required_parameters`.  Any parameters referenced here will be
    given a default value.

    Dependence will check to make sure all of the required parameters
    have been supplied, with the exception of those which have default
    versions.  It also provides other convenience and safety features,
    such as allowing tests for equality of derived algorithms and for
    ensuring extra parameters were not assigned.
    """
    def __init__(self, **kwargs):
        """Create a new Dependence object with parameters specified in **kwargs."""
        assert hasattr(self, "depname"), "Dependence needs a parameter name"
        assert hasattr(self, "name"), "Dependence classes need a name"
        assert hasattr(self, "required_parameters"), "Dependence needs a list of required params"
        if hasattr(self, "default_parameters"):
            args = self.default_parameters
            args.update(kwargs)
        else:
            args = kwargs
        if not hasattr(self, "required_conditions"):
            object.__setattr__(self, 'required_conditions', [])
        passed_args = sorted(args.keys())
        expected_args = sorted(self.required_parameters)
        assert passed_args == expected_args, "Provided %s arguments, expected %s" % (str(passed_args), str(expected_args))
        for key, value in args.items():
            setattr(self, key, value)

    def __eq__(self, other):
        """Equality is defined as having the same algorithm type and the same parameters."""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __setattr__(self, name, val):
        """Only allow the required parameters to be assigned."""
        if name in self.required_parameters:
            return object.__setattr__(self, name, val) # No super() for python2 compatibility
        raise LookupError
    def __delattr__(self, name):
        """No not allow a required parameter to be deleted."""
        raise LookupError
    def __repr__(self):
        params = ""
        # If it is a sub-sub-class, then print the parameters it was
        # instantiated with
        if self.name:
            for p in self.required_parameters:
                params += str(p) + "=" + getattr(self, p).__repr__()
                if p != self.required_parameters[-1]:
                    params += ", "
        return type(self).__name__ + "(" + params + ")"
    def __str__(self):
        return self.__repr__()
    def __hash__(self):
        return hash(repr(self))

class InitialCondition(Dependence):
    """Subclass this to compute the initial conditions of the simulation.

    This abstract class describes initial PDF at the beginning of a
    simulation.  To subclass it, implement get_IC(x).

    Also, since it inherits from Dependence, subclasses must also
    assign a `name` and `required_parameters` (see documentation for
    Dependence.)
    """
    depname = "IC"
    def get_IC(self, x, dx, **kwargs):
        """Get the initial conditions (a PDF) withsupport `x`.

        `x` is a length N ndarray representing the support of the
        initial condition PDF, i.e. the x-domain.  This returns a
        length N ndarray describing the distribution.
        """
        raise NotImplementedError

class ICPointSourceCenter(InitialCondition):
    """Initial condition: a dirac delta function in the center of the domain."""
    name = "point_source_center"
    required_parameters = []
    def get_IC(self, x, dx, **kwargs):
        pdf = np.zeros(len(x))
        pdf[int((len(x)-1)/2)] = 1. # Initial condition at x=0, center of the channel.
        return pdf

# Dependence for testing.
class ICUniform(InitialCondition):
    """Initial condition: a uniform distribution."""
    name = "uniform"
    required_parameters = []
    def get_IC(self, x, dx, **kwargs):
        pdf = np.zeros(len(x))
        pdf = 1/(len(x))*np.ones((len(x)))
        return pdf

class Mu(Dependence):
    """Subclass this to specify how drift rate varies with position and time.

    This abstract class provides the methods which define a dependence
    of mu on x and t.  To subclass it, implement get_matrix and
    get_flux.  All subclasses must include a parameter mu in
    required_parameters, which is the drift rate at the start of the
    simulation.

    Also, since it inherits from Dependence, subclasses must also
    assign a `name` and `required_parameters` (see documentation for
    Dependence.)
    """
    depname = "Mu"
    def get_matrix(self, x, t, dx, dt, adj=0, conditions={}, **kwargs):
        """The drift component of the implicit method diffusion matrix across the domain `x` at time `t`.

        `x` should be a length N ndarray.  
        `adj` is the optional amount by which to adjust `mu`.  This is
        most relevant for tasks which modify `mu` over time.

        Returns a sparse numpy matrix.
        """
        mu = self.get_mu(x=x, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)
        return sparse.diags([ 0.5*dt/dx * (mu + adj),
                             -0.5*dt/dx * (mu + adj)],
                            [1, -1], shape=(len(x), len(x)), format="csr")
    # Amount of flux from bound/end points to correct and erred
    # response probabilities, due to different parameters.
    def get_flux(self, x_bound, t, dx, dt, adj=0, conditions={}, **kwargs):
        """The drift component of flux across the boundary at position `x_bound` at time `t`.

        Flux here is essentially the amount of the mass of the PDF
        that is past the boundary point `x_bound`.

        `adj` is the optional amount by which to adjust `mu`.  This is
        most relevant for tasks which modify `mu` over time.
        """
        return 0.5*dt/dx * np.sign(x_bound) * (self.get_mu(x=x_bound, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs) + adj)
    def mu_base(self, conditions={}):
        """Return the value of mu at the beginning of the simulation."""
        assert "mu" in self.required_parameters, "Mu must be a required parameter"
        return self.mu
    def get_mu(self, conditions={}, **kwargs):
        raise NotImplementedError

class MuConstant(Mu):
    """Mu dependence: drift rate is constant throughout the simulation.

    Only take one parameter: mu, the constant drift rate.

    Note that this is a special case of MuLinear."""
    name = "constant"
    required_parameters = ["mu"]
    def get_mu(self, **kwargs):
        return self.mu

class MuLinear(Mu):
    """Mu dependence: drift rate varies linearly with position and time.

    Take three parameters:

    - `mu` - The starting drift rate
    - `x` - The coefficient by which mu varies with x
    - `t` - The coefficient by which mu varies with t
    """
    name = "linear_xt"
    required_parameters = ["mu", "x", "t"]
    # Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) -
    # (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the
    # same x,t with p(x,t) (probability distribution function). Hence
    # we use x[1:]/[:-1] respectively for the +/-1 off-diagonal.
    def get_matrix(self, x, t, dx, dt, adj=0, conditions={}, **kwargs):
        return sparse.diags( 0.5*dt/dx * (self.mu + adj + self.x*x[1:]  + self.t*t), 1) \
             + sparse.diags(-0.5*dt/dx * (self.mu + adj + self.x*x[:-1] + self.t*t),-1)
    def get_flux(self, x_bound, t, dx, dt, adj=0, conditions={}, **kwargs):
        return 0.5*dt/dx * np.sign(x_bound) * (self.mu + adj + self.x*x_bound + self.t*t)
    def get_mu(self, x, t, **kwargs):
        return self.mu + self.x*x_bound + self.t*t

class MuSinCos(Mu):
    """Mu dependence: drift rate varies linearly with sin(x) and cos(t).

    This was intended for testing purposes and is not intended to be
    used seriously.

    Take three parameters:

    - `mu` - The starting drift rate
    - `x` - The coefficient by which mu varies with sin(x)
    - `t` - The coefficient by which mu varies with cos(t)
    """
    name = "sinx_cost"
    required_parameters = ["mu", "x", "t"]
    def get_matrix(self, x, t, dx, dt, adj=0, conditions={}, **kwargs):
        return sparse.diags( 0.5*dt/dx * (self.mu + adj + self.x*np.sin(x[1:])  + self.t*np.cos(t)), 1) \
             + sparse.diags(-0.5*dt/dx * (self.mu + adj + self.x*np.sin(x[:-1]) + self.t*np.cos(t)),-1)
    def get_flux(x_bound, t, dx, dt, adj=0, conditions={}, **kwargs):
        return 0.5*dt/dx * np.sign(x_bound) * (self.mu + adj + self.x*np.sin(x_bound) + self.t*np.cos(t))
    def get_mu(self, x, t, **kwargs):
        return self.mu + self.x*np.sin(x_bound) + self.t*np.cos(t)

class Sigma(Dependence):
    """Subclass this to specify how diffusion rate/noise varies with position and time.

    This abstract class provides the methods which define a dependence
    of sigma on x and t.  To subclass it, implement get_matrix and
    get_flux.  All subclasses must include a parameter sigma in
    required_parameters, which is the diffusion rate/noise at the
    start of the simulation.

    Also, since it inherits from Dependence, subclasses must also
    assign a `name` and `required_parameters` (see documentation for
    Dependence.)
    """
    depname = "Sigma"
    def get_matrix(self, x, t, dx, dt, adj=0, conditions={}, **kwargs):
        """The diffusion component of the implicit method diffusion matrix across the domain `x` at time `t`.

        `x` should be a length N ndarray.
        `t` should be a float for the time.
        `adj` is the optional amount by which to adjust `sigma`.  This
        is most relevant for tasks which modify `sigma` over time.
        """
        sigma = self.get_sigma(x=x, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs)
        return sparse.diags([1.0*(sigma + adj)**2 * dt/dx**2,
                             -0.5*(sigma + adj)**2 * dt/dx**2,
                             -0.5*(sigma + adj)**2 * dt/dx**2],
                            [0, 1, -1], shape=(len(x), len(x)), format="csr")
    def get_flux(self, x_bound, t, dx, dt, adj=0, conditions={}, **kwargs):
        """The diffusion component of flux across the boundary at position `x_bound` at time `t`.

        Flux here is essentially the amount of the mass of the PDF
        that is past the boundary point `x_bound` at time `t` (a float).

        `adj` is the optional amount by which to adjust `sigma`.  This
        is most relevant for tasks which modify `sigma` over time.
        """
        return 0.5*dt/dx**2 * (self.get_sigma(x=x_bound, t=t, dx=dx, dt=dt, conditions=conditions, **kwargs) + adj)**2
    def sigma_base(self, conditions={}):
        """Return the value of sigma at the beginning of the simulation."""
        assert "sigma" in self.required_parameters, "Sigma must be a required parameter"
        return self.sigma
    def get_sigma(self, conditions={}, **kwargs):
        raise NotImplementedError

class SigmaConstant(Sigma):
    """Simga dependence: diffusion rate/noise is constant throughout the simulation.

    Only take one parameter: sigma, the diffusion rate.

    Note that this is a special case of SigmaLinear."""
    name = "constant"
    required_parameters = ["sigma"]
    def get_sigma(self, **kwargs):
        return self.sigma

class SigmaLinear(Sigma):
    """Sigma dependence: diffusion rate varies linearly with position and time.

    Take three parameters:

    - `sigma` - The starting diffusion rate/noise
    - `x` - The coefficient by which sigma varies with x
    - `t` - The coefficient by which sigma varies with t
    """
    name = "linear_xt"
    required_parameters = ["sigma", "x", "t"]
    def get_matrix(self, x, t, dx, dt, adj=0, conditions={}, **kwargs):
        diagadj = self.sigma + adj + self.x*x + self.t*t
        diagadj[diagadj<0] = 0
        return sparse.diags(1.0*(diagadj)**2 * dt/dx**2, 0) \
             - sparse.diags(0.5*(diagadj[1:])**2 * dt/dx**2, 1) \
             - sparse.diags(0.5*(diagadj[:-1])**2 * dt/dx**2,-1)
    def get_flux(self, x_bound, t, dx, dt, adj=0, conditions={}, **kwargs):
        fluxadj = (self.sigma + adj + self.x*x_bound + self.t*t)
        if fluxadj < 0:
            return 0
        return 0.5*dt/dx**2 * fluxadj**2
    def get_sigma(self, x, t, **kwargs):
        return self.sigma + self.x*x + self.t*t

class SigmaSinCos(Sigma):
    """Sigma dependence: diffusion rate varies linearly with sin(x) and cos(t).

    This was intended for testing purposes and is not intended to be
    used seriously.

    Take three parameters:

    - `sigma` - The starting diffusion rate/noise
    - `x` - The coefficient by which sigma varies with sin(x)
    - `t` - The coefficient by which sigma varies with cos(t)
    """
    name = "sinx_cost"
    required_parameters = ["sigma", "x", "t"]
    def get_matrix(self, x, t, dx, dt, adj=0, conditions={}, **kwargs):
        return sparse.diags(1.0*(self.sigma + adj + self.x*np.sin(x)      + self.t*np.cos(t))**2 * dt/dx**2, 0) \
             - sparse.diags(0.5*(self.sigma + adj + self.x*np.sin(x[1:])  + self.t*np.cos(t))**2 * dt/dx**2, 1) \
             - sparse.diags(0.5*(self.sigma + adj + self.x*np.sin(x[:-1]) + self.t*np.cos(t))**2 * dt/dx**2,-1)
    def get_flux(self, x_bound, t, dx, dt, adj=0, conditions={}, **kwargs):
        return 0.5*dt/dx**2 * (self.sigma + self.x*np.sin(x_bound) + self.t*np.cos(t))**2
    def get_sigma(self, x, t, **kwargs):
        return self.sigma + self.x*np.sin(x) + self.t*np.cos(t)

class Bound(Dependence):
    """Subclass this to specify how bounds vary with time.

    This abstract class provides the methods which define a dependence
    of the bounds on t.  To subclass it, implement get_bound.  All
    subclasses must include a parameter `B` in required_parameters,
    which is the upper bound at the start of the simulation.  (The
    lower bound is symmetrically -B.)

    Also, since it inherits from Dependence, subclasses must also
    assign a `name` and `required_parameters` (see documentation for
    Dependence.)
    """
    depname = "Bound"
    ## Second effect of Collapsing Bounds: Collapsing Center: Positive
    ## and Negative states are closer to each other over time.
    def get_bound(self, t, conditions={}, **kwargs):
        """Return the bound at time `t`."""
        raise NotImplementedError
    def B_base(self, conditions={}):
        assert "B" in self.required_parameters, "B must be a required parameter"
        return self.B

class BoundConstant(Bound):
    """Bound dependence: bound is constant throuhgout the simulation.

    Takes only one parameter: `B`, the constant bound."""
    name = "constant"
    required_parameters = ["B"]
    def get_bound(self, t, adj=0, conditions={}, **kwargs):
        return self.B

class BoundCollapsingLinear(Bound):
    """Bound dependence: bound collapses linearly over time.

    Takes two parameters: 

    `B` - the bound at time t = 0.
    `t` - the slope, i.e. the coefficient of time, should be greater
    than zero.
    """
    name = "collapsing_linear"
    required_parameters = ["B", "t"]
    def get_bound(self, t, adj=0, conditions={}, **kwargs):
        return max(self.B + adj - self.t*t, 0.)

class BoundCollapsingExponential(Bound):
    """Bound dependence: bound collapses exponentially over time.

    Takes two parameters: 

    `B` - the bound at time t = 0.
    `tau` - the time constant for the collapse, should be greater than
    zero.
    """
    name = "collapsing_exponential"
    required_parameters = ["B", "tau"]
    def get_bound(self, t, adj=0, conditions={}, **kwargs):
        return (self.B + adj) * np.exp(-self.tau*t)
    
class Task(Dependence):
    depname = "Task"
    def adjust_mu(self, mu, t, conditions={}):
        return 0
    def adjust_sigma(self, sigma, t, conditions={}):
        return 0

class TaskFixedDuration(Task):
    name = "Fixed_Duration"
    required_parameters = []

class TaskDurationParadigm(Task):
    name = "Duration_Paradigm"
    required_parameters = ["duration"]
    def adjust_mu(self, mu, t, conditions={}):
        if t < self.duration:
            return 0
        else:
            return -mu

class TaskPulseParadigm(Task):
    name = "Pulse_Paradigm"
    required_parameters = ["onset", "duration", "adjustment"]
    default_parameters = {"duration" : .1, "adjustment" : .15}
    def adjust_mu(self, mu, t, conditions={}):
        if (t > self.onset) and (t < (self.onset + self.duration)):
            return mu * self.adjustment
        else:
            return 0

class TaskDelay(Task):
    name = "Delay"
    required_parameters = ["delay"]
    def adjust_mu(self, mu, t, conditions={}):
        if t < self.delay:
            return -mu
        else:
            return 0
    def adjust_sigma(self, sigma, t, conditions={}):
        if t < self.delay:
            return -sigma
        else:
            return 0
    
class TaskIndependentDelays(Task):
    name = "Independent_Delays"
    required_parameters = ["delay_mu", "delay_sigma"]
    def adjust_mu(self, mu, t, conditions={}):
        if t < self.delay_mu:
            return -mu
        else:
            return 0
    def adjust_sigma(self, sigma, t, conditions={}):
        if t < self.delay_sigma:
            return -sigma
        else:
            return 0
    

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
                 task=TaskFixedDuration(), name="",
                 dx=param.dx, dt=param.dt, T_dur=param.T_dur):
        """Construct a Model object from the 5 key components.

        The five key components of our DDM-style models describe how
        the drift rate (`mu`), noise (`sigma`), and bounds (`bound`)
        change over time, the initial conditions (`IC`), and a
        description of a potential task which could modify the model
        during the simulation (`task`).

        These five components are given by the parameters `mu`, `sigma`,
        `bound`, `IC`, and `task`, respectively.  They should be types
        which inherit from the types Mu, Sigma, Bound, Task, and
        InitialCondition, respectively.  They default to constant
        unitary values.

        Additionally, simulation parameters can be set, such as time
        and spacial precision (`dt` and `dx`) and the simulation
        duration `T_dur`.  If not specified, they will be taken from
        the defaults specified in the parameters file.

        The `name` parameter is exclusively for convenience, and may
        be used in plotting or in debugging.
        """
        assert name.__str__() == name # TODO crappy way to test type(name) == str for Python2 and Python3
        self.name = name
        assert isinstance(mu, Mu)
        self._mudep = mu
        assert isinstance(sigma, Sigma)
        self._sigmadep = sigma
        assert isinstance(bound, Bound)
        self._bounddep = bound
        assert isinstance(task, Task)
        self._task = task
        assert isinstance(IC, InitialCondition)
        self._IC = IC
        self.dependencies = [self._mudep, self._sigmadep, self._bounddep, self._task, self._IC]
        self.required_conditions = list(set([x for l in self.dependencies for x in l.required_conditions]))
        self.dx = dx
        self.dt = dt
        self.T_dur = T_dur
    # Get a string representation of the model
    def __repr__(self, pretty=False):
        # Use a list so they can be sorted
        allobjects = [("name", self.name), ("mu", self._mudep),
                      ("sigma", self._sigmadep), ("bound", self._bounddep),
                      ("task", self._task), ("IC", self._IC),
                      ("dx", self.dx), ("dt", self.dt), ("T_dur", self.T_dur)]
        params = ""
        for n,o in allobjects:
            params += n + "=" + o.__repr__()
            if (n,o) != allobjects[-1]:
                if pretty == True:
                    params += ",\n" + " "*(len(type(self).__name__)+1)
                else:
                    params += ", "
        return type(self).__name__ + "(" + params + ")"
    def __str__(self):
        return self.__repr__(pretty=True)
    # Get an ordered list of all model parameters.  Guaranteed to be
    # in the same order as set_model_parameters().
    def get_model_parameters(self):
        params = []
        for d in self.dependencies:
            for p in d.required_parameters:
                params.append(getattr(d, p))
        return params
    # Accepts a list of parameters in the same order as
    # get_model_parameters.
    def set_model_parameters(self, params):
        assert len(params) == len(self.get_model_parameters()), "Invalid params"
        i = 0
        for d in [self._mudep, self._sigmadep, self._bounddep, self._task, self._IC]:
            for p in d.required_parameters:
                setattr(d, p, params[i])
                i += 1

    def get_dependence(self, name):
        if name.lower() in ["mu", "mudep", "_mudep"]:
            return self._mudep
        elif name.lower() in ["sigma", "sigmadep", "_sigmadep"]:
            return self._sigmadep
        elif name.lower() in ["b", "bound", "bounddep", "_bounddep"]:
            return self._bounddep
        elif name.lower() in ["ic", "initialcondition", "_ic"]:
            return self._IC
        elif name.lower() in ["task", "_task"]:
            return self._task
        raise NameError("Invalid dependence name")

    def get_model_type(self):
        """Return a dictionary which fully specifies the class of the five key model components."""
        tt = lambda x : (x.depname, type(x))
        return dict(map(tt, [self._mudep, self._sigmadep, self._bounddep, self._task, self._IC]))
    def bound_base(self, conditions={}):
        """The boundary at the beginning of the simulation."""
        return self._bounddep.B_base(conditions=conditions)
    def mu_base(self, conditions={}):
        """The drift rate at the beginning of the simulation."""
        return self._mudep.mu_base(conditions=conditions)
    def sigma_base(self, conditions={}):
        """The noise at the beginning of the simulation."""
        return self._sigmadep.sigma_base(conditions=conditions)
    def x_domain(self):
        """A list which spans from the lower boundary to the upper boundary by increments of dx."""
        B = self.bound_base()
        return np.arange(-B, B+0.1*self.dx, self.dx) # +.1*dx is to ensure that the largest number in the array is B
    def t_domain(self):
        """A list of all of the timepoints over which the joint PDF will be defined (increments of dt from 0 to T_dur)."""
        return np.arange(0., self.T_dur+0.1*self.dt, self.dt)
    def mu_task_adj(self, t, conditions={}):
        """The amount by which we should adjust the drift rate at time `t` for the current task."""
        return self._task.adjust_mu(self.mu_base(conditions=conditions), t, conditions=conditions)
    def sigma_task_adj(self, t, conditions={}):
        """The amount by which we should adjust the drift rate at time `t` for the current task."""
        return self._task.adjust_sigma(self.sigma_base(conditions=conditions), t, conditions=conditions)
    # TODO: This, as well as mu_matrix and sigma_matrix, are
    # bottlenecks.  Maybe we could cache or pre-allocate memory?
    def diffusion_matrix(self, x, t, conditions={}):
        """The matrix for the implicit method of solving the diffusion equation.

        - `x` - a length N ndarray representing the domain over which
          the matrix is to be defined. Usually a contiguous subset of
          x_domain().
        - `t` - The timepoint at which the matrix is valid.

        Returns a size NxN scipy sparse array.
        """
        mu_matrix = self._mudep.get_matrix(x=x, t=t, dt=self.dt, dx=self.dx, adj=self.mu_task_adj(t, conditions=conditions), conditions=conditions)
        sigma_matrix = self._sigmadep.get_matrix(x=x, t=t, dt=self.dt, dx=self.dx, adj=self.sigma_task_adj(t, conditions=conditions), conditions=conditions)
        return sparse.eye(len(x), format="csr") + mu_matrix + sigma_matrix
    def flux(self, x, t, conditions={}):
        """The flux across the boundary at position `x` at time `t`."""
        mu_flux = self._mudep.get_flux(x, t, adj=self.mu_task_adj(t, conditions=conditions), dx=self.dx, dt=self.dt, conditions=conditions)
        sigma_flux = self._sigmadep.get_flux(x, t, adj=self.sigma_task_adj(t, conditions=conditions), dx=self.dx, dt=self.dt, conditions=conditions)
        return mu_flux + sigma_flux
    def bound(self, t, conditions={}):
        """The upper boundary of the simulation at time `t`."""
        return self._bounddep.get_bound(t, conditions=conditions)
    def IC(self, conditions={}):
        """The initial distribution at t=0.

        Returns a length N ndarray (where N is the size of x_domain())
        which should sum to 1.
        """
        return self._IC.get_IC(self.x_domain(), dx=self.dx, conditions=conditions)

    def has_analytical_solution(self):
        """Is it possible to find an analytic solution for this model?"""
        mt = self.get_model_type()
        return mt["Mu"] == MuConstant and mt["Sigma"] == SigmaConstant and \
            (mt["Bound"] in [BoundConstant, BoundCollapsingLinear]) and \
            mt["Task"] == TaskFixedDuration and mt["IC"] == ICPointSourceCenter
        
    def solve(self, conditions={}):
        """Solve the model using an analytic solution if possible, and a numeric solution if not.

        Return a Solution object describing the joint PDF distribution of reaction times."""
        if self.has_analytical_solution():
            return self.solve_analytical(conditions=conditions)
        else:
            return self.solve_numerical(conditions=conditions)

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
        # The analytic_ddm function does the heavy lifting.
        if type(self._bounddep) == BoundConstant: # Simple DDM
            anal_pdf_corr, anal_pdf_err = analytic_ddm(self.mu_base(conditions=conditions),
                                                       self.sigma_base(conditions=conditions),
                                                       self.bound_base(conditions=conditions), self.t_domain())
        elif type(self._bounddep) == BoundCollapsingLinear: # Linearly Collapsing Bound
            anal_pdf_corr, anal_pdf_err = analytic_ddm(self.mu_base(conditions=conditions),
                                                       self.sigma_base(conditions=conditions),
                                                       self.bound_base(conditions=conditions),
                                                       self.t_domain(), -self._bounddep.t) # TODO why must this be negative? -MS

        ## Remove some abnormalities such as NaN due to trivial reasons.
        anal_pdf_corr[anal_pdf_corr==np.NaN] = 0.
        anal_pdf_corr[0] = 0.
        anal_pdf_err[anal_pdf_err==np.NaN] = 0.
        anal_pdf_err[0] = 0.
        return Solution(anal_pdf_corr*self.dt, anal_pdf_err*self.dt, self)

    def solve_numerical(self, conditions={}):
        """Solve the DDM model numerically.

        This uses the implicit method to solve the DDM at each
        timepoint.  Results are then compiled together.  This is the
        core DDM solver of this library.

        It returns a Solution object describing the joint PDF.  This
        method should not fail for any model type.
        """

        ### Initialization: Lists
        pdf_curr = self.IC(conditions=conditions) # Initial condition
        pdf_prev = np.zeros((len(pdf_curr)))
        # If pdf_corr + pdf_err + undecided probability are summed, they
        # equal 1.  So these are componets of the joint pdf.
        pdf_corr = np.zeros(len(self.t_domain())) # Not a proper pdf on its own (doesn't sum to 1)
        pdf_err = np.zeros(len(self.t_domain())) # Not a proper pdf on its own (doesn't sum to 1)
        x_list = self.x_domain()

        # Looping through time and updating the pdf.
        for i_t, t in enumerate(self.t_domain()[:-1]): # -1 because nothing will happen at t=0 so each step computes the value for the next timepoint
            # Update Previous state. To be frank pdf_prev could be
            # removed for max efficiency. Leave it just in case.
            pdf_prev = copy.copy(pdf_curr)

            # For efficiency only do diffusion if there's at least
            # some densities remaining in the channel.
            if sum(pdf_curr[:])>0.0001:
                ## Define the boundaries at current time.
                bound = self.bound(t) # Boundary at current time-step.

                # Now figure out which x positions are still within
                # the (collapsing) bound.
                assert self.bound_base() >= bound, "Invalid change in bound" # Ensure the bound didn't expand
                bound_shift = self.bound_base(conditions=conditions) - bound
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
                    # TODO this solve call is a bottleneck... can it be optimized?
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
            if len(pdf_inner) == 0: pdf_inner = np.array([0]) # Fix error when bounds collapse to 0
            pdf_corr[i_t+1] += weight_outer * pdf_outer[-1] * self.flux(_outer_B_corr, t, conditions=conditions) \
                            +  weight_inner * pdf_inner[-1] * self.flux(_inner_B_corr, t, conditions=conditions)
            pdf_err[i_t+1]  += weight_outer * pdf_outer[0] * self.flux(_outer_B_err, t, conditions=conditions) \
                            +  weight_inner * pdf_inner[0] * self.flux(_inner_B_err, t, conditions=conditions)

            # Renormalize when the channel size has <1 grid, although
            # all hell breaks loose in this regime.
            if bound < self.dx:
                pdf_corr[i_t+1] *= (1+ (1-bound/self.dx))
                pdf_err[i_t+1] *= (1+ (1-bound/self.dx))

        return Solution(pdf_corr, pdf_err, self, conditions=conditions)

class _Sample_Iter_Wraper(object):
    def __init__(self, sample_obj, correct=None):
        self.sample = sample_obj
        self.i = 0
        self.correct = correct
    def __iter__(self):
        return self
    def next(self):
        if self.i == len(sample):
            raise StopIteration
        self.i += 1
        if self.correct == True:
            rt = self.sample.corr
            ind = 0
        elif self.correct == False:
            rt = self.sample.err
            ind = 1
        return (rt[self.i-1], {k : self.sample.conditions[k][0][self.i-1] for k in self.sample.conditions.keys()})
        

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
    arguments, where the keyword name is the propertyp and the
    """
    def __init__(self, sample_corr, sample_err, non_decision=0, **kwargs):
        self.corr = sample_corr
        self.err = sample_err
        self.non_decision = non_decision
        # Make sure the kwarg parameters/conditions are in the correct
        # format
        for k,v in kwargs.items():
            assert type(v) == tuple
            assert len(v) in [2, 3]
            assert len(v[0]) == len(self.corr)
            assert len(v[1]) == len(self.err)
            if len(v) == 3:
                assert len(v[2]) == non_decision
            else:
                assert non_decision == 0
        self.conditions = kwargs
    def __len__(self):
        return len(self.corr) + len(self.err) + self.non_decision
    def __iter__(self):
        return np.concatenate([self.corr, self.err]).__iter__()
    def items(self, correct):
        return _Sample_Iter_Wraper(self, correct=correct)
    def subset(self, **kwargs):
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
        return list(self.conditions.keys())
    def condition_combinations(self, required_conditions=None):
        cs = self.conditions
        conditions = []
        names = self.condition_names()
        if required_conditions is not None:
            names = [n for n in names if n in required_conditions]
        for c in names:
            conditions.append(list(set(cs[c][0]).union(set(cs[c][1]))))
        combs = []
        for p in itertools.product(*conditions):
            combs.append(dict(zip(names, p)))
        if len(combs) == 0:
            return [{}]
        return combs
    

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
    def __init__(self, pdf_corr, pdf_err, model, conditions={}):
        """Create a Solution object from the results of a model
        simulation.

        Constructor takes three arguments.

            - `model` - the Model object used to generate `pdf_corr` and `pdf_err`
            - `pdf_corr` - a size N numpy ndarray describing the correct portion of the joint pdf
            - `pdf_err` - a size N numpy ndarray describing the error portion of the joint pdf
        """
        self.model = copy.deepcopy(model) # TODO this could cause a memory leak if I forget it is there...
        self._pdf_corr = pdf_corr
        self._pdf_err = pdf_err
        self.conditions = conditions

    def pdf_corr(self):
        """The correct component of the joint PDF."""
        return self._pdf_corr/self.model.dt

    def pdf_err(self):
        """The error (incorrect) component of the joint PDF."""
        return self._pdf_err/self.model.dt

    def cdf_corr(self):
        """The correct component of the joint CDF."""
        return np.cumsum(self._pdf_corr)

    def cdf_err(self):
        """The error (incorrect) component of the joint CDF."""
        return np.cumsum(self._pdf_err)

    def prob_correct(self):
        """The probability of selecting the right response."""
        return np.sum(self._pdf_corr)

    def prob_error(self):
        """The probability of selecting the incorrect (error) response."""
        return np.sum(self._pdf_err)

    def prob_undecided(self):
        """The probability of selecting neither response (undecided)."""
        return 1 - self.prob_correct() - self.prob_error()

    def prob_correct_forced(self):
        """The probability of selecting the correct response if a response is forced."""
        return self.prob_correct() + prob_undecided()/2.

    def prob_error_forced(self):
        """The probability of selecting the incorrect response if a response is forced."""
        return self.prob_error() + prob_undecided()/2.

    def mean_decision_time(self):
        """The mean decision time in the correct trials (excluding undecided trials)."""
        return np.sum((self._pdf_corr)*self.model.t_domain()) / self.prob_correct()

    def _sample_from_histogram(self, hist, hist_bins, k, seed=0):
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
        norm = np.round(np.sum(h), 5)
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
        corr_sample = list(filter(lambda x : x >= 0, sample))
        err_sample = list(np.asarray(list(filter(lambda x : x < 0, sample)))*-1)
        non_decision = k - (len(corr_sample) + len(err_sample))
        conditions = {k : ([v]*len(corr_sample), [v]*len(err_sample), [v]*non_decision) for k,v in self.conditions.items()}
        return Sample(corr_sample, err_sample, non_decision, **conditions)

class Fittable:
    """For parameters that should be adjusted when fitting a model to data.
        
    Each Fittable object does not need any parameters, however several
    parameters may improve the ability to fit the model.  In
    particular, `maxval` and `minval` ensure we do not choose an
    invalid parameter value.  `default` is the value to start with
    when fitting; if it is not given, it will be selected at random.
    """
    def __init__(self, minval=-np.inf, maxval=np.inf, default=None):
        self.minval = minval
        self.maxval = maxval
        self.default_value = default
    def __repr__(self):
        reprstr = "Fittable("
        if self.minval != -np.inf:
            reprstr += "minval=" + self.minval.__repr__() + ", "
        if self.maxval != np.inf:
            reprstr += "maxval=" + self.maxval.__repr__() + ", "
        reprstr += "default=" + self.default_value.__repr__()
        reprstr += ")"
        return reprstr
    def default(self):
        """Choose a default value.

        This chooses a value for the Fittable object randomly abiding
        by any constraints.  Note that calling this function multiple
        times will give different results.
        """
        if self.default_value is not None:
            return self.default_value
        else:
            maxval = self.maxval; minval = self.minval # Makes equations below more readable
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

class LossFunction(object):
    def __init__(self, sample, required_conditions=None, **kwargs):
        assert hasattr(self, "name"), "Solver needs a name"
        self.sample = sample
        self.required_conditions = required_conditions
        self.setup(**kwargs)
    def setup(self, **kwargs):
        pass
    def loss(self, model):
        raise NotImplementedError
    def cache_by_conditions(self, model):
        cache = {}
        for c in self.sample.condition_combinations(required_conditions=self.required_conditions):
            cache[frozenset(c.items())] = model.solve(conditions=c)
        return cache

class LossSquaredError(LossFunction):
    name = "Squared Difference"
    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur
        self.hists_corr = {}
        self.hists_err = {}
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            self.hists_corr[frozenset(comb.items())] = np.histogram(self.sample.subset(**comb).corr, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self.sample.subset(**comb))/dt # dt/2 (and +1) is continuity correction
            self.hists_err[frozenset(comb.items())] = np.histogram(self.sample.subset(**comb).err, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self.sample.subset(**comb))/dt
        self.target = np.concatenate([s for i in sorted(self.hists_corr.keys()) for s in [self.hists_corr[i], self.hists_err[i]]])
    def loss(self, model):
        assert model.dt == self.dt and model.T_dur == self.T_dur
        sols = self.cache_by_conditions(model)
        this = np.concatenate([s for i in sorted(self.hists_corr.keys()) for s in [sols[i].pdf_corr(), sols[i].pdf_err()]])
        return np.sum((this-self.target)**2)

SquaredErrorLoss = LossSquaredError # Named this incorrectly on the first go... lecacy
    
class LossMLE(LossFunction):
    name = "MLE"
    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur
        # Each element in the dict is indexed by the conditions of the
        # model (e.g. coherence, trial conditions) as a frozenset.
        # Each contains a tuple of lists, which are to contain the
        # position for each within a histogram.  For instance, if a
        # reaction time corresponds to position i, then we can index a
        # list representing a normalized histogram/"pdf" (given by dt
        # and T_dur) for immediate access to the probability of
        # obtaining that value.
        self.hist_indexes = {}
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            s = self.sample.subset(**comb)
            corr = [int(round(e/dt)) for e in s.corr]
            err = [int(round(e/dt)) for e in s.err]
            nondec = self.sample.non_decision
            self.hist_indexes[frozenset(comb.items())] = (corr, err, nondec)
    def loss(self, model):
        assert model.dt == self.dt and model.T_dur == self.T_dur
        sols = self.cache_by_conditions(model)
        loglikelihood = 0
        for k in sols.keys():
            loglikelihood += np.sum(np.log(sols[k].pdf_corr()[self.hist_indexes[k][0]]))
            loglikelihood += np.sum(np.log(sols[k].pdf_err()[self.hist_indexes[k][1]]))
            if sols[k].prob_undecided() > 0:
                loglikelihood += np.log(sols[k].prob_undecided())*self.hist_indexes[k][2]
        return -loglikelihood

class LossMLEMixture(LossMLE):
    name = "MLE with 2% uniform noise"
    def loss(self, model):
        assert model.dt == self.dt and model.T_dur == self.T_dur
        sols = self.cache_by_conditions(model)
        loglikelihood = 0
        for k in sols.keys():
            solpdfcorr = sols[k].pdf_corr()
            solpdfcorr[solpdfcorr<0] = 0 # Numerical errors cause this to be negative sometimes
            solpdferr = sols[k].pdf_err()
            solpdferr[solpdferr<0] = 0 # Numerical errors cause this to be negative sometimes
            pdfcorr = solpdfcorr*.98 + .01*np.ones(1+self.T_dur/self.dt)/self.T_dur*self.dt # .98 and .01, not .98 and .02, because we have both correct and error
            pdferr = solpdferr*.98 + .01*np.ones(1+self.T_dur/self.dt)/self.T_dur*self.dt
            loglikelihood += np.sum(np.log(pdfcorr[self.hist_indexes[k][0]]))
            loglikelihood += np.sum(np.log(pdferr[self.hist_indexes[k][1]]))
            if sols[k].prob_undecided() > 0:
                loglikelihood += np.log(sols[k].prob_undecided())*self.hist_indexes[k][2]
        return -loglikelihood
