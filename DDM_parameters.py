'''
Simulation code for Drift Diffusion Model
Author: Norman Lam (norman.lam@yale.edu)
'''
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


##Pre-defined list of models that can be used, and the corresponding default parameters
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

