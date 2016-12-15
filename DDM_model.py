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
    def get_matrix(self, mu, x, t):
        raise NotImplementedError
    def get_flux(self, x_bound, mu, t):
        raise NotImplementedError

class MuLinear(Mu):
    name = "linear_xt"
    required_parameters = ["x", "t"]
    def get_matrix(self, mu, x, t):
        return np.diag( 0.5*dt/dx * (mu + self.x*x[1:]  + self.t*t), 1) \
             + np.diag(-0.5*dt/dx * (mu + self.x*x[:-1] + self.t*t),-1)
    ### Amount of flux from bound/end points to correct and erred
    ### response probabilities, due to different parameters (mu,
    ### sigma, bound)
    ## Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) -
    ## (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the
    ## same x,t with p(x,t) (probability distribution function). Hence
    ## we use x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
    def get_flux(self, x_bound, mu, t):
        return 0.5*dt/dx * np.sign(x_bound) * (mu + self.x*x_bound + self.t*t)

class MuSinCos(Mu):
    name = "sinx_cost"
    required_parameters = ["x", "t"]
    def get_matrix(self, mu, x, t):
        return np.diag( 0.5*dt/dx * (mu + self.x*np.sin(x[1:])  + self.t*np.cos(t)), 1) \
             + np.diag(-0.5*dt/dx * (mu + self.x*np.sin(x[:-1]) + self.t*np.cos(t)),-1)
    def get_flux(x_bound, x, t):
        return 0.5*dt/dx * np.sign(x_bound) * (mu + self.x*np.sin(x_bound) + self.t*np.cos(t))

class Sigma(Dependence):
    def get_matrix(self, sigma, x, t):
        raise NotImplementedError
    def get_flux(self, x_bound, sigma, t):
        raise NotImplementedError

class SigmaLinear(Sigma):
    name = "linear_xt"
    required_parameters = ["x", "t"]
    def get_matrix(self, sigma, x, t):
        return np.diag(1.0*(sigma + self.x*x      + self.t*t)**2 * dt/dx**2, 0) \
             - np.diag(0.5*(sigma + self.x*x[1:]  + self.t*t)**2 * dt/dx**2, 1) \
             - np.diag(0.5*(sigma + self.x*x[:-1] + self.t*t)**2 * dt/dx**2,-1)
    def get_flux(self, x_bound, sigma, t):
        return 0.5*dt/dx**2 * (sigma + self.x*x_bound + self.t*t)**2

class SigmaSinCos(Sigma):
    name = "sinx_cost"
    required_parameters = ["x", "t"]
    def get_matrix(self, sigma, x, t):
        return np.diag(1.0*(sigma + self.x*np.sin(x)      + self.t*np.cos(t))**2 * dt/dx**2, 0) \
             - np.diag(0.5*(sigma + self.x*np.sin(x[1:])  + self.t*np.cos(t))**2 * dt/dx**2, 1) \
             - np.diag(0.5*(sigma + self.x*np.sin(x[:-1]) + self.t*np.cos(t))**2 * dt/dx**2,-1)
    def get_flux(self, x_bound, sigma, t):
        return 0.5*dt/dx**2 * (sigma + self.x*np.sin(x_bound) + self.t*np.cos(t))**2

class Bound(Dependence):
    ## Second effect of Collapsing Bounds: Collapsing Center: Positive
    ## and Negative states are closer to each other over time.
    def get_bound(self, bound, t):
        raise NotImplementedError

class BoundConstant(Bound):
    name = "constant"
    required_parameters = []
    def get_bound(self, bound, t):
        return bound

class BoundCollapsingLinear(Bound):
    name = "collapsing_linear"
    required_parameters = ["t"]
    def get_bound(self, bound, t):
        return max(bound - self.t*t, 0.)

class BoundCollapsingExponential(Bound):
    name = "collapsing_exponential"
    required_parameters = ["tau"]
    def get_bound(self, bound, t):
        return bound * np.exp(-self.tau*t)
    
class Task(Dependence):
    def adjust_mu(self, mu, t):
        raise NotImplementedError

class TaskFixedDuration(Task):
    name = "Fixed_Duration"
    required_parameters = []
    def adjust_mu(self, mu, t):
        return mu

class TaskPsychoPhysicalKernel(Task):
    name = "PsychoPhysical_Kernel"
    required_parameters = ["kernel"]
    def adjust_mu(self, mu, t):
        return mu + self.kernel[int(t/dt_mu_PK)] ## Fix/Implement later

class TaskDurationParadigm(Task):
    name = "Duration_Paradigm"
    required_parameters = ["duration"]
    def adjust_mu(self, mu, t):
        if t < self.duration:
            return mu
        else:
            return 0

class TaskPulseParadigm(Task):
    name = "Pulse_Paradigm"
    required_parameters = ["onset", "duration", "adjustment"]
    default_parameters = {"duration" : .1, "adjustment" : 1.15}
    def adjust_mu(self, mu, t):
        if (t > self.onset) and (t < (self.onset + self.duration)):
            return mu * self.adjustment
        else:
            return mu


##Pre-defined list of models that can be used, and the corresponding default parameters
class Model(object):
    def __init__(self, mu, mudep, sigma, sigmadep, B, bounddep, task=TaskFixedDuration(), IC=ICPointSourceCenter(), name=""):
        assert name.__str__() == name # TODO crappy way to test type(name) == str for Python2 and Python3
        self.name = name
        assert isinstance(mudep, Mu)
        self.mudep = mudep
        assert isinstance(sigmadep, Sigma)
        self.sigmadep = sigmadep
        assert isinstance(bounddep, Bound)
        self.bounddep = bounddep
        assert isinstance(task, Task)
        self.task = task
        assert isinstance(IC, InitialCondition)
        self.IC = IC
        assert np.isreal(mu) and np.isreal(sigma) and np.isreal(B)
        self.parameters = {"mu" : mu, "sigma" : sigma, "B" : B}
    # Get an ordered list of all model parameters.  Guaranteed to be
    # in the same order as set_model_parameters().
    def get_model_parameters(self):
        params = []
        for k in sorted(self.parameters.keys()):
            params.append(self.parameters[k])
        for d in [self.mudep, self.sigmadep, self.bounddep, self.task, self.IC]:
            for p in d.required_parameters:
                params.append(getattr(d, p))
        return params
    # Accepts a list of parameters in the same order as
    # get_model_parameters.
    def set_model_parameters(self, params):
        assert len(params) == len(self.get_model_parameters()), "Invalid params"
        i = 0
        for d in [self.mudep, self.sigmadep, self.bounddep, self.task, self.IC]:
            for p in d.required_parameters:
                setattr(d, p, params[i])
                i += 1
    def get_model_type(self):
        return list(map(type, [self.mudep, self.sigmadep, self.bounddep, self.task, self.IC]))
