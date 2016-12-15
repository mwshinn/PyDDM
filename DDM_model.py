import numpy as np
import copy
from DDM_parameters import *
from DDM_analytic import analytic_ddm, analytic_ddm_linbound

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

class MuConstant(Mu):
    name = "constant"
    required_parameters = ["mu"]
    def get_matrix(self, x, t, dx, dt, adj=0, **kwargs):
        return np.diag( 0.5*dt/dx * (self.mu + adj + 0*x[1:] ), 1) \
             + np.diag(-0.5*dt/dx * (self.mu + adj + 0*x[:-1]),-1)
    ### Amount of flux from bound/end points to correct and erred
    ### response probabilities, due to different parameters (mu,
    ### sigma, bound)
    ## Reminder: The general definition is (mu*p)_(x_{n+1},t_{m+1}) -
    ## (mu*p)_(x_{n-1},t_{m+1})... So choose mu(x,t) that is at the
    ## same x,t with p(x,t) (probability distribution function). Hence
    ## we use x_list[1:]/[:-1] respectively for the +/-1 off-diagonal.
    def get_flux(self, x_bound, t, dx, dt, adj=0, **kwargs):
        return 0.5*dt/dx * np.sign(x_bound) * (self.mu + adj)

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
    def get_matrix(self, x, t, dx, dt, adj=0, **kwargs):
        return np.diag( 0.5*dt/dx * (self.mu + adj + self.x*np.sin(x[1:])  + self.t*np.cos(t)), 1) \
             + np.diag(-0.5*dt/dx * (self.mu + adj + self.x*np.sin(x[:-1]) + self.t*np.cos(t)),-1)
    def get_flux(x_bound, t, dx, dt, adj=0, **kwargs):
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

class SigmaConstant(Sigma):
    name = "constant"
    required_parameters = ["sigma"]
    def get_matrix(self, x, t, dx, dt, adj=0, **kwargs):
        return np.diag(1.0*(self.sigma + adj + 0*x)**2     * dt/dx**2, 0) \
             - np.diag(0.5*(self.sigma + adj + 0*x[1:])**2 * dt/dx**2, 1) \
             - np.diag(0.5*(self.sigma + adj+ 0*x[:-1])**2 * dt/dx**2,-1)
    def get_flux(self, x_bound, t, dx, dt, adj=0, **kwargs):
        return 0.5*dt/dx**2 * (self.sigma + adj)**2

class SigmaLinear(Sigma):
    name = "linear_xt"
    required_parameters = ["sigma", "x", "t"]
    def get_matrix(self, x, t, dx, dt, adj=0, **kwargs):
        return np.diag(1.0*(self.sigma + adj + self.x*x      + self.t*t)**2 * dt/dx**2, 0) \
             - np.diag(0.5*(self.sigma + adj + self.x*x[1:]  + self.t*t)**2 * dt/dx**2, 1) \
             - np.diag(0.5*(self.sigma + adj + self.x*x[:-1] + self.t*t)**2 * dt/dx**2,-1)
    def get_flux(self, x_bound, t, dx, dt, adj=0, **kwargs):
        return 0.5*dt/dx**2 * (self.sigma + adj + self.x*x_bound + self.t*t)**2

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
        return np.arange(-B, B+0.1*self.dx, self.dx) # +.1*dx is to ensure that the largest number in the array is B
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


    def has_analytical_solution(self):
        mt = self.get_model_type()
        return mt["Mu"] == MuConstant and mt["Sigma"] == SigmaConstant and \
            (mt["Bound"] in [BoundConstant, BoundCollapsingLinear]) and \
            mt["Task"] == TaskFixedDuration and mt["IC"] == ICPointSourceCenter
        
    def solve(self):
        if self.has_analytical_solution():
            return self.solve_analytical()
        else:
            return self.solve_numerical()

    ########################################################################################################################
    ## Functions for Analytical Solutions.
    ### Analytical form of DDM. Can only solve for simple DDM or linearly collapsing bound... From Anderson1960
    # Note that the solutions are automatically normalized.
    def solve_analytical(self):
        '''
        Now assume f_mu, f_sigma, f_Bound are callable functions
        See DDM_pdf_general for nomenclature
        '''

        ### Initialization
        assert self.has_analytical_solution(), "Cannot solve for this model analytically"

        if type(self._bounddep) == BoundConstant: # Simple DDM
            anal_pdf_corr, anal_pdf_err = analytic_ddm(self.mu_base(),
                                                       self.sigma_base(),
                                                       self.bound_base(), self.t_domain())
        elif type(self._bounddep) == BoundCollapsingLinear: # Linearly Collapsing Bound
            anal_pdf_corr, anal_pdf_err = analytic_ddm(self.mu_base(),
                                                       self.sigma_base(),
                                                       self.bound_base(),
                                                       self.t_domain(), -self._bounddep.t) # TODO why must this be negative? -MS

        ## Remove some abnormalities such as NaN due to trivial reasons.
        anal_pdf_corr[anal_pdf_corr==np.NaN] = 0.
        anal_pdf_corr[0] = 0.
        anal_pdf_err[anal_pdf_err==np.NaN] = 0.
        anal_pdf_err[0] = 0.
        return Solution(anal_pdf_corr*self.dt, anal_pdf_err*self.dt, self)

    # Function that simulates one trial of the time-varying drift/bound DDM
    def solve_numerical(self):
        '''
        Now assume f_mu, f_sigma, f_Bound are callable functions
        '''
        ### Initialization: Lists
        pdf_curr = self.IC() # Initial condition
        pdf_prev = np.zeros((len(pdf_curr)))
        # If pdf_corr + pdf_err + undecided probability are summed, they
        # equal 1.  So these are componets of the joint pdf.
        pdf_corr = np.zeros(len(self.t_domain())) # Probability flux to correct choice.  Not a proper pdf (doesn't sum to 1)
        pdf_err = np.zeros(len(self.t_domain())) # Probability flux to erred choice. Not a proper pdf (doesn't sum to 1)
        x_list = self.x_domain()

        ##### Looping through time and updating the pdf.
        for i_t, t in enumerate(self.t_domain()[:-1]): # NOTE I translated this directly from "for i_t in range(len(t_list_temp)-1):" but I think the -1 was a bug -MS
            # Update Previous state. To be frank pdf_prev could be
            # removed for max efficiency. Leave it just in case.
            pdf_prev = copy.copy(pdf_curr)

            # If we are in a task, adjust mu according to the task

            if sum(pdf_curr[:])>0.0001: # For efficiency only do diffusion if there's at least some densities remaining in the channel.
                ## Define the boundaries at current time.
                bound = self.bound(t) # Boundary at current time-step. Can generalize to assymetric bounds

                ## Now figure out which x positions are still within the
                # (collapsing) bound.
                assert self.bound_base() >= bound, "Invalid change in bound" # Ensure the bound didn't expand
                bound_shift = self.bound_base() - bound
                # Note that we linearly approximate the bound by the two surrounding grids sandwiching it.
                x_index_inner = int(np.ceil(bound_shift/self.dx)) # Index for the inner bound (smaller matrix)
                x_index_outer = int(np.floor(bound_shift/self.dx)) # Index for the outer bound (larger matrix)
                weight_inner = (bound_shift - x_index_outer*self.dx)/self.dx # The weight of the upper bound matrix, approximated linearly. 0 when bound exactly at grids.
                weight_outer = 1. - weight_inner # The weight of the lower bound matrix, approximated linearly.
                x_list_inbounds = x_list[x_index_outer:len(x_list)-x_index_outer] # List of x-positions still within bounds.

                ## Define the diffusion matrix for implicit method
                # Diffusion Matrix for Implicit Method. Here defined as
                # Outer Matrix, and inder matrix is either trivial or an
                # extracted submatrix.
                diffusion_matrix = self.diffusion_matrix(x_list_inbounds, t)

                ### Compute Probability density functions (pdf)
                # PDF for outer matrix
                pdf_outer = np.linalg.solve(diffusion_matrix, pdf_prev[x_index_outer:len(x_list)-x_index_outer])
                # PDF for inner matrix (with optimization)
                if x_index_inner == x_index_outer: # Optimization: When bound is exactly at a grid, use the most efficient method.
                    pdf_inner = copy.copy(pdf_outer)
                else:
                    pdf_inner = np.linalg.solve(diffusion_matrix[1:-1, 1:-1], pdf_prev[x_index_inner:len(x_list)-x_index_inner])

                # Pdfs out of bound is consideered decisions made.
                pdf_err[i_t+1] += weight_outer * np.sum(pdf_prev[:x_index_outer]) \
                                  + weight_inner * np.sum(pdf_prev[:x_index_inner])
                pdf_corr[i_t+1] += weight_outer * np.sum(pdf_prev[len(x_list)-x_index_outer:]) \
                                   + weight_inner * np.sum(pdf_prev[len(x_list)-x_index_inner:])
                pdf_curr = np.zeros((len(x_list))) # Reconstruct current proability density function, adding outer and inner contribution to it.
                pdf_curr[x_index_outer:len(x_list)-x_index_outer] += weight_outer*pdf_outer
                pdf_curr[x_index_inner:len(x_list)-x_index_inner] += weight_inner*pdf_inner

            else:
                break #break if the remaining densities are too small....

            ### Increase current, transient probability of crossing either
            ### bounds, as flux.  Corr is a correct answer, err is an
            ### incorrect answer
            _inner_B_corr = x_list[len(x_list)-1-x_index_inner]
            _outer_B_corr = x_list[len(x_list)-1-x_index_outer]
            _inner_B_err = x_list[x_index_inner]
            _outer_B_err = x_list[x_index_outer]
            if len(pdf_inner) == 0: pdf_inner = np.array([0]) # Fix error when bounds collapse to 0
            pdf_corr[i_t+1] += weight_outer * pdf_outer[-1] * self.flux(_outer_B_corr, t) \
                            +  weight_inner * pdf_inner[-1] * self.flux(_inner_B_corr, t)
            pdf_err[i_t+1]  += weight_outer * pdf_outer[0] * self.flux(_outer_B_err, t) \
                            +  weight_inner * pdf_inner[0] * self.flux(_inner_B_err, t)

            if bound < self.dx: # Renormalize when the channel size has <1 grid, although all hell breaks loose in this regime.
                pdf_corr[i_t+1] *= (1+ (1-bound/self.dx))
                pdf_err[i_t+1] *= (1+ (1-bound/self.dx))

        return Solution(pdf_corr, pdf_err, self) # Only return jpdf components for correct and erred choices. Add more ouputs if needed



class Solution(object):
    def __init__(self, pdf_corr, pdf_err, model):
        self.model = copy.deepcopy(model) # TODO this could cause a memory leak if I forget it is there...
        self._pdf_corr = pdf_corr
        self._pdf_err = pdf_err

    def pdf_corr(self):
        return self._pdf_corr/self.model.dt

    def pdf_err(self):
        return self._pdf_err/self.model.dt

    def cdf_corr(self):
        return np.cumsum(self._pdf_corr)

    def cdf_err(self):
        return np.cumsum(self._pdf_err)

    def prob_correct(self):
        return np.sum(self._pdf_corr)

    def prob_error(self):
        return np.sum(self._pdf_err)

    def prob_undecided(self):
        return 1 - self.prob_correct() - self.prob_error()

    def prob_correct_forced(self):
        return self.prob_correct() + prob_undecided()/2.

    def prob_error_forced(self):
        return self.prob_error() + prob_undecided()/2.

    # Only consider correct choices. Note that Mean_Dec_Time does not
    # includes choices supposedly undecided and made at the last
    # moment.
    def mean_decision_time(self):
        return np.sum((self._pdf_corr)*self.model.t_domain()) / self.prob_correct()
