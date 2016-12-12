'''
Simulation code for Drift Diffusion Model
Author: Norman Lam (norman.lam@yale.edu)
'''
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

########################################################################################################################
### Initialization
## Flags to run various parts or not

# Parameters.
dx = .08#0.008 # grid size
T_dur = 2. # [s] Duration of simulation
dt = .05#0.005 # [s] Time-step.

#mu= Drift Rate
mu_0 = 0.*0.5 # Constant component of drift rate mu.
#mu_0_list = np.logspace(-2,1, 20) # List of mu_0, to be looped through for tasks.
mu_0_list = [-10, -5, -2, -1, -0.5, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10] # List of mu_0, to be looped through for tasks.
param_mu_x = 1.*0.5 # Parameter for x_dependence of mu. Add more if 1 param is not sufficient...
param_mu_t = 0.*0.3 # Parameter for t_dependence of mu. Add more if 1 param is not sufficient...
sigma_0 = 1.*0.5 # Constant component of sigma=noise.
param_sigma_x = 0. # Parameter for x_dependence of sigma. Add more if 1 param is not sufficient...
param_sigma_t = 0. # Parameter for t_dependence of sigma. Add more if 1 param is not sufficient...
# B = Bound
B = 1. # Boundary. Assumed to be 1
param_B_t = 0.5 # Parameter for t_dependence of B (no x-dep I sps?). Add more if 1 param is not sufficient...


# Parameters consistent with spiking circuit data.                                                                      # Arranged in params in ddm_pdf_genreal as [mu_0,param_mu_x,param_mu_t, sigma_0,param_sigma_x,param_sigma_t, B_0, param_B_t], all param_a_b are assumed to only have 1... add more if needed.
# Mu = Drift rate
mu_0 = 1.*13.97531121 # Constant component of drift rate mu.
coh_list = np.array([0.0,3.2,6.4,12.8,25.6,51.2]) # [%] For Duration Paradigm and in general
# coh_list = np.array([-51.2, -25.6, -12.8, -6.4, -3.2, 0.0, 3.2, 6.4, 12.8, 25.6, 51.2]) # [%] For Pulse Paradigm
mu_0_list = [mu_0*0.01*coh_temp for coh_temp in coh_list] # List of mu_0, to be looped through for tasks.
param_mu_x_OUpos = 6.99053975 # Note that this value largely depends on the model used...
param_mu_x_OUneg = -7.73123206 # Note that this value largely depends on the model used.... NOTE that this is the regime where control is optimal over OU+-, and OU+- are significantly different.
param_mu_t = 0. # Parameter for t_dependence of mu. Add more if 1 param is not sufficient...
# Sigma = Noise
sigma_0 = 1.*1.29705615 # Constant component of noise sigma.
param_sigma_x = 0.5 # Parameter for x_dependence of sigma. Add more if 1 param is not sufficient...
param_sigma_t = 0.5 # Parameter for t_dependence of sigma. Add more if 1 param is not sufficient...
# B = Bound
B = 1. # Boundary. Assumed to be 1
param_B_t = 1. # Parameter for t_dependence of B (no x-dep I sps?). Add more if 1 param is not sufficient...

# Declare arrays for usage and storage.
x_list = np.arange(-B, B+0.1*dx, dx) # List of x-grids (Staggered-mesh)
center_matrix_ind = (len(x_list)-1)/2 # index of the center of the matrix. Should be integer by design of x_list
t_list = np.arange(0., T_dur, dt) # t-grids


# This describes how a variable is dependent on other variables.
# Principally, we want to know how mu and sigma depend on x and t.
# `name` is the type of dependence (e.g. "linear") for methods which
# implement the algorithms, and any parameters these algorithms could
# need should be passed as kwargs. To compare to legacy code, the
# `name` used to be `f_mu_setting` or `f_sigma_setting` and
# kwargs now encompassed (e.g.) `param_mu_t_temp`.
class Dependence(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self.add_parameter(**kwargs)
        self.all_parameters = sorted(kwargs.keys())
    def add_parameter(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    # Allow tests for equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

class InitialCondition(Dependence):
    def get_IC(self, x):
        raise NotImplementedError

class ICPointSourceCenter(InitialCondition):
    def __init__(self):
        super(self.__class__, self).__init__("point_source_center")
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
    def __init__(self):
        super(self.__class__, self).__init__(name="uniform")
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
    def __init__(self, x, t):
        super(self.__class__, self).__init__(name="linear_xt", x=x, t=t)
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
    def __init__(self, x, t):
        super(self.__class__, self).__init__(name="sinx_cost", x=x, t=t)
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
    def __init__(self, x, t):
        super(self.__class__, self).__init__(name="linear_xt", x=x, t=t)
    def get_matrix(self, sigma, x, t):
        return np.diag(1.0*(sigma + self.x*x      + self.t*t)**2 * dt/dx**2, 0) \
             - np.diag(0.5*(sigma + self.x*x[1:]  + self.t*t)**2 * dt/dx**2, 1) \
             - np.diag(0.5*(sigma + self.x*x[:-1] + self.t*t)**2 * dt/dx**2,-1)
    def get_flux(self, x_bound, sigma, t):
        return 0.5*dt/dx**2 * (sigma + self.x*x_bound + self.t*t)**2

class SigmaSinCos(Sigma):
    def __init__(self, x, t):
        super(self.__class__, self).__init__(name="sinx_cost", x=x, t=t)
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
    def __init__(self):
        super(self.__class__, self).__init__(name="constant")
    def get_bound(self, bound, t):
        return bound

class BoundCollapsingLinear(Bound):
    def __init__(self, t):
        super(self.__class__, self).__init__(name="collapsing_linear", t=t)
    def get_bound(self, bound, t):
        return max(bound - self.t*t, 0.)

class BoundCollapsingExponential(Bound):
    def __init__(self, tau):
        super(self.__class__, self).__init__(name="collapsing_exponential", tau=tau)
    def get_bound(self, bound, t):
        return bound * np.exp(-self.tau*t)
    
class Task(Dependence):
    def adjust_mu(self, mu, t):
        raise NotImplementedError

class TaskFixedDuration(Task):
    def __init__(self):
        super(self.__class__, self).__init__(name="Fixed_Duration")
    def adjust_mu(self, mu, t):
        return mu

class TaskPsychoPhysicalKernel(Task):
    def __init__(self, kernel):
        super(self.__class__, self).__init__(name="PsychoPhysical_Kernel", kernel=kernel)
    def adjust_mu(self, mu, t):
        return mu + self.kernel[int(t/dt_mu_PK)] ## Fix/Implement later

class TaskDurationParadigm(Task):
    def __init__(self, duration):
        super(self.__class__, self).__init__(name="Duration_Paradigm", duration=duration)
    def adjust_mu(self, mu, t):
        if t < self.duration:
            return mu
        else:
            return 0

class TaskPulseParadigm(Task):
    def __init__(self, onset, duration=.1):
        super(self.__class__, self).__init__(name="Pulse_Paradigm", onset=onset, duration=duration)
    def adjust_mu(self, mu, t):
        if (t > self.onset) and (t < (self.onset + self.duration)):
            return mu * 1.15 # 0.15 based on spiking circuit simulations.
        else:
            return mu

    
##Pre-defined list of models that can be used, and the corresponding default parameters
class Model(object):
    def __init__(self, mu, mudep, sigma, sigmadep, B, bounddep, task=TaskFixedDuration(), IC=ICPointSourceCenter(), name=""):
        self.name = name
        self.mudep = mudep
        self.sigmadep = sigmadep
        self.bounddep = bounddep
        self.task = task
        self.IC = IC
        self.parameters = {"mu" : mu, "sigma" : sigma, "B" : B}
    # Get an ordered list of all model parameters.  Guaranteed to be
    # in the same order as set_model_parameters().
    def get_model_parameters(self):
        params = []
        for k in sorted(self.parameters.keys()):
            params.append(self.parameters[k])
        for d in [self.mudep, self.sigmadep, self.bounddep, self.task, self.IC]:
            for p in d.all_parameters:
                params.append(getattr(d, p))
        return params
    # Accepts a list of parameters in the same order as
    # get_model_parameters.
    def set_model_parameters(self, params):
        assert len(params) == len(self.get_model_parameters()), "Invalid params"
        i = 0
        for d in [self.mudep, self.sigmadep, self.bounddep, self.task, self.IC]:
            for p in d.all_parameters:
                setattr(d, p, params[i])
                i += 1

s1 = Model(name="DDM", mu=mu_0, sigma=sigma_0, B=B,
           mudep=MuLinear(x=0, t=0),
           sigmadep=SigmaLinear(x=0, t=0),
           bounddep=BoundConstant())
s2 = Model(name="CB_Lin", mu=mu_0, sigma=sigma_0, B=B,
           mudep=MuLinear(x=0, t=0),
           sigmadep=SigmaLinear(x=0, t=0),
           bounddep=BoundCollapsingLinear(t=param_B_t))
s3 = Model(name="CB_Expo", mu=mu_0, sigma=sigma_0, B=B,
           mudep=MuLinear(x=0, t=0),
           sigmadep=SigmaLinear(x=0, t=0),
           bounddep=BoundCollapsingExponential(tau=param_B_t))
s4 = Model(name="OU+", mu=mu_0, sigma=sigma_0, B=B,
           mudep=MuLinear(x=param_mu_x_OUpos, t=0),
           sigmadep=SigmaLinear(x=0, t=0),
           bounddep=BoundConstant())
s5 = Model(name="OU-", mu=mu_0, sigma=sigma_0, B=B,
           mudep=MuLinear(x=param_mu_x_OUneg, t=0),
           sigmadep=SigmaLinear(x=0, t=0),
           bounddep=BoundConstant())
models = [s1, s2, s3, s4, s5]

setting_list = [['linear_xt', 'linear_xt', 'constant', 'point_source_center'],
                ['linear_xt', 'linear_xt', 'collapsing_linear', 'point_source_center'],
                ['linear_xt', 'linear_xt', 'collapsing_exponential', 'point_source_center'],
                ['linear_xt', 'linear_xt', 'constant', 'point_source_center'],
                ['linear_xt', 'linear_xt', 'constant', 'point_source_center'],
                ['linear_xt', 'linear_xt', 'constant', 'point_source_center']]

task_list = ['Fixed_Duration', 'PsychoPhysical_Kernel', 'Duration_Paradigm', 'Pulse_Paradigm'] # Define various setting specs for each tasks...
task_params_list = [[], [], [0.1*mu_0, T_dur/2.], [0.1*mu_0, T_dur/2.]] # Temporary parameters to test the function. Later want to vary through them. See f_mu1_task for details.
models_list_all = [0,1,2,3,4] # List of models to use. See Setting_list
#param_mu_0_list = [0., 0., 0., 1.*param_mu_0, -1.5*param_mu_0, param_mu_t] # List of param_mu_0 input in DDM_pdf_general. Can do the same for sigma_0 etc if needed.
param_mu_x_list = [0., 0., 0., param_mu_x_OUpos, param_mu_x_OUneg, 0.] # List of param_mu_0 input in DDM_pdf_general. Can do the same for sigma_0 etc if needed.
param_mu_t_list = [0., 0., 0., 0., 0., param_mu_t] # List of param_mu_0 input in DDM_pdf_general. Can do the same for sigma_0 etc if needed.
param_sigma_x_list = [0., 0., 0., 0., 0., 0.] # List of param_mu_0 input in DDM_pdf_general. Can do the same for sigma_0 etc if needed.
param_sigma_t_list = [0., 0., 0., 0., 0., 0.] # List of param_mu_0 input in DDM_pdf_general. Can do the same for sigma_0 etc if needed.
param_B_t_list = [0., param_B_t, param_B_t, 0., 0., param_mu_t] # List of param_mu_0 input in DDM_pdf_general. Can do the same for sigma_0 etc if needed.
labels_list = ['DDM', 'CB_Lin', 'CB_Expo', 'OU+', 'OU-', 'DDM_t'] # Labels for figures
#color_list  = ['g', 'r', 'orange', 'b', 'c', 'k']                                                                      # Colors for figures
color_list = ['r', 'm', 'orange', 'g', 'b', 'k'] #Colors for figures. TEMP: Want r/g/b for DDM/OU+/OU-

########################################################################################################################

