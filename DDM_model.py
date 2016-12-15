from DDM_parameters import *

# This describes how a variable is dependent on other variables.
# Principally, we want to know how mu and sigma depend on x and t.
# `name` is the type of dependence (e.g. "linear") for methods which
# implement the algorithms, and any parameters these algorithms could
# need should be passed as kwargs. To compare to legacy code, the
# `name` used to be `f_mu_setting` or `f_sigma_setting` and
# kwargs now encompassed (e.g.) `param_mu_t_temp`.
class Dependence(object):
    def __init__(self, **kwargs):
        assert hasattr(self, "name"), "Dependence classes need a name"
        assert hasattr(self, "required_parameters"), "Dependence needs a list of required params"
        if hasattr(self, "default_parameters"):
            args = self.default_parameters
            args.update(kwargs)
        else:
            args = kwargs
        passed_args = sorted(args.keys())
        expected_args = sorted(self.required_parameters)
        assert passed_args == expected_args, "Provided %s arguments, expected %s" % (str(passed_args), str(expected_args))
        self.add_parameter(**args)
    def add_parameter(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    # Allow tests for equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False
    # Only allow the required parameters to be assigned
    def __setattr__(self, name, val):
        if name in self.required_parameters:
            return object.__setattr__(self, name, val) # No super() for python2 compatibility
        raise LookupError
    def __delattr__(self, name):
        raise LookupError

class InitialCondition(Dependence):
    depname = "IC"
    def get_IC(self, x):
        raise NotImplementedError

class ICPointSourceCenter(InitialCondition):
    name = "point_source_center"
    required_parameters = []
    ## Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) -
    ## (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the
    ## same x,t with p(x,t) (probability distribution function). Hence
    ## we use x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
    def get_IC(self, x):
        pdf = np.zeros(len(x))
        pdf[int((len(x)-1)/2)] = 1. # Initial condition at x=0, center of the channel.
        return pdf

# Dependence for testing.
class ICUniform(InitialCondition):
    name = "uniform"
    required_parameters = []
    def get_IC(self, x):
        pdf = np.zeros(len(x))
        pdf = 1/(len(x))*np.ones((len(x)))
        return pdf

class Mu(Dependence):
    depname = "Mu"
    def get_matrix(self, x, t, adj=0, **kwargs):
        raise NotImplementedError
    def get_flux(self, x_bound, t, adj=0, **kwargs):
        raise NotImplementedError
    def mu_base(self):
        assert "mu" in self.required_parameters, "Mu must be a required parameter"
        return self.mu

class MuLinear(Mu):
    name = "linear_xt"
    required_parameters = ["mu", "x", "t"]
    def get_matrix(self, x, t, dx, dt, adj=0, **kwargs):
        return np.diag( 0.5*dt/dx * (self.mu + adj + self.x*x[1:]  + self.t*t), 1) \
             + np.diag(-0.5*dt/dx * (self.mu + adj + self.x*x[:-1] + self.t*t),-1)
    ### Amount of flux from bound/end points to correct and erred
    ### response probabilities, due to different parameters (mu,
    ### sigma, bound)
    ## Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) -
    ## (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the
    ## same x,t with p(x,t) (probability distribution function). Hence
    ## we use x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
    def get_flux(self, x_bound, t, dx, dt, adj=0, **kwargs):
        return 0.5*dt/dx * np.sign(x_bound) * (self.mu + adj + self.x*x_bound + self.t*t)

class MuSinCos(Mu):
    name = "sinx_cost"
    required_parameters = ["mu", "x", "t"]
    def get_matrix(self, x, t, **kwargs):
        return np.diag( 0.5*dt/dx * (self.mu + adj + self.x*np.sin(x[1:])  + self.t*np.cos(t)), 1) \
             + np.diag(-0.5*dt/dx * (self.mu + adj + self.x*np.sin(x[:-1]) + self.t*np.cos(t)),-1)
    def get_flux(x_bound, t):
        return 0.5*dt/dx * np.sign(x_bound) * (self.mu + adj + self.x*np.sin(x_bound) + self.t*np.cos(t))

class Sigma(Dependence):
    depname = "Sigma"
    def get_matrix(self, x, t, adj=0, **kwargs):
        raise NotImplementedError
    def get_flux(self, x_bound, t, adj=0, **kwargs):
        raise NotImplementedError
    def sigma_base(self):
        assert "sigma" in self.required_parameters, "Sigma must be a required parameter"
        return self.sigma

class SigmaLinear(Sigma):
    name = "linear_xt"
    required_parameters = ["sigma", "x", "t"]
    def get_matrix(self, x, t, dx, dt, adj=0, **kwargs):
        return np.diag(1.0*(self.sigma + adj + self.x*x      + self.t*t)**2 * dt/dx**2, 0) \
             - np.diag(0.5*(self.sigma + adj + self.x*x[1:]  + self.t*t)**2 * dt/dx**2, 1) \
             - np.diag(0.5*(self.sigma + adj + self.x*x[:-1] + self.t*t)**2 * dt/dx**2,-1)
    def get_flux(self, x_bound, t, dx, dt, **kwargs):
        return 0.5*dt/dx**2 * (self.sigma + self.x*x_bound + self.t*t)**2

class SigmaSinCos(Sigma):
    name = "sinx_cost"
    required_parameters = ["sigma", "x", "t"]
    def get_matrix(self, x, t, dx, dt, adj=0, **kwargs):
        return np.diag(1.0*(self.sigma + adj + self.x*np.sin(x)      + self.t*np.cos(t))**2 * dt/dx**2, 0) \
             - np.diag(0.5*(self.sigma + adj + self.x*np.sin(x[1:])  + self.t*np.cos(t))**2 * dt/dx**2, 1) \
             - np.diag(0.5*(self.sigma + adj + self.x*np.sin(x[:-1]) + self.t*np.cos(t))**2 * dt/dx**2,-1)
    def get_flux(self, x_bound, t, dx, dt, adj=0, **kwargs):
        return 0.5*dt/dx**2 * (self.sigma + self.x*np.sin(x_bound) + self.t*np.cos(t))**2

class Bound(Dependence):
    depname = "Bound"
    ## Second effect of Collapsing Bounds: Collapsing Center: Positive
    ## and Negative states are closer to each other over time.
    def get_bound(self, t, **kwargs):
        raise NotImplementedError
    def B_base(self):
        assert "B" in self.required_parameters, "B must be a required parameter"
        return self.B

class BoundConstant(Bound):
    name = "constant"
    required_parameters = ["B"]
    def get_bound(self, t, adj=0, **kwargs):
        return self.B

class BoundCollapsingLinear(Bound):
    name = "collapsing_linear"
    required_parameters = ["B", "t"]
    def get_bound(self, t, adj=0, **kwargs):
        return max(self.B + adj - self.t*t, 0.)

class BoundCollapsingExponential(Bound):
    name = "collapsing_exponential"
    required_parameters = ["B", "tau"]
    def get_bound(self, t, adj=0, **kwargs):
        return (self.B + adj) * np.exp(-self.tau*t)
    
class Task(Dependence):
    depname = "Task"
    def adjust_mu(self, mu, t):
        raise NotImplementedError

class TaskFixedDuration(Task):
    name = "Fixed_Duration"
    required_parameters = []
    def adjust_mu(self, mu, t):
        return 0

class TaskPsychoPhysicalKernel(Task):
    name = "PsychoPhysical_Kernel"
    required_parameters = ["kernel"]
    def adjust_mu(self, mu, t):
        return self.kernel[int(t/dt_mu_PK)] ## Fix/Implement later

class TaskDurationParadigm(Task):
    name = "Duration_Paradigm"
    required_parameters = ["duration"]
    def adjust_mu(self, mu, t):
        if t < self.duration:
            return 0
        else:
            return -mu

class TaskPulseParadigm(Task):
    name = "Pulse_Paradigm"
    required_parameters = ["onset", "duration", "adjustment"]
    default_parameters = {"duration" : .1, "adjustment" : .15}
    def adjust_mu(self, mu, t):
        if (t > self.onset) and (t < (self.onset + self.duration)):
            return mu * self.adjustment
        else:
            return 0


##Pre-defined list of models that can be used, and the corresponding default parameters
class Model(object):
    def __init__(self, mudep, sigmadep, bounddep, task=TaskFixedDuration(),
                 IC=ICPointSourceCenter(), name="",
                 dx=dx, dt=dt, T_dur=T_dur):
        assert name.__str__() == name # TODO crappy way to test type(name) == str for Python2 and Python3
        self.name = name
        assert isinstance(mudep, Mu)
        self._mudep = mudep
        assert isinstance(sigmadep, Sigma)
        self._sigmadep = sigmadep
        assert isinstance(bounddep, Bound)
        self._bounddep = bounddep
        assert isinstance(task, Task)
        self._task = task
        assert isinstance(IC, InitialCondition)
        self._IC = IC
        self.dx = dx
        self.dt = dt
        self.T_dur = T_dur
    # Get an ordered list of all model parameters.  Guaranteed to be
    # in the same order as set_model_parameters().
    def get_model_parameters(self):
        params = []
        for d in [self._mudep, self._sigmadep, self._bounddep, self._task, self._IC]:
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
    def get_model_type(self):
        tt = lambda x : (x.depname, type(x))
        return dict(map(tt, [self._mudep, self._sigmadep, self._bounddep, self._task, self._IC]))

    def bound_base(self):
        return self._bounddep.B_base()
    def mu_base(self):
        return self._mudep.mu_base()
    def sigma_base(self):
        return self._sigmadep.sigma_base()
    def x_domain(self):
        B = self.bound_base()
        return np.arange(-B, B+0.1*dx, dx) # +.1*dx is to ensure that the largest number in the array is B
    def t_domain(self):
        return np.arange(0., self.T_dur, self.dt)
    def mu_task_adj(self, t):
        return self._task.adjust_mu(self.mu_base(), t)
    def diffusion_matrix(self, x, t):
        mu_matrix = self._mudep.get_matrix(x=x, t=t, dt=self.dt, dx=self.dx, adj=self.mu_task_adj(t))
        sigma_matrix = self._sigmadep.get_matrix(x=x, t=t, dt=self.dt, dx=self.dx)
        return np.eye(len(x)) + mu_matrix + sigma_matrix
    def flux(self, x, t):
        mu_flux = self._mudep.get_flux(x, t, adj=self.mu_task_adj(t), dx=self.dx, dt=self.dt)
        sigma_flux = self._sigmadep.get_flux(x, t, dx=self.dx, dt=self.dt)
        return mu_flux + sigma_flux
    def bound(self, t):
        return self._bounddep.get_bound(t)
    def IC(self):
        return self._IC.get_IC(self.x_domain())
    def set_mu(self, mu):
        self._mudep.mu = mu

