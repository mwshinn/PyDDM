'''
Simulation code for Thalamaus, based on 1993 Destexhe, McCormick, Sejnowski
Author: Norman Lam (john.murray@yale.edu)
'''
#import brian_no_units           #Note: speeds up the code
from numpy.fft import rfft,irfft
import time
import numpy as np
from scipy.special import erf
#from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cProfile
import re



########################################################################################################################
### Initialization: Parameters
dt = 0.001                  # [s]
mu = 0.03
sigma = 0.1
B = 1.
tau_bound = 1./1.             # [s]
N_sim = 10000
if N_sim%2 ==0:
    N_sim -=1      #Want odd for median that corresponds to a trial

## Spiking Circuit Parameters
mu = 1.*13.97531121 *0.128                  # mean = drift rate constant
# coh_list = np.array([0.0,3.2,6.4,12.8,25.6,51.2])                                                                      # For Duration Paradigm
# # coh_list = np.array([-51.2, -25.6, -12.8, -6.4, -3.2, 0.0, 3.2, 6.4, 12.8, 25.6, 51.2])                                 # For Pulse Paradigm
# mu_0_list = [mu_0*0.01*coh_temp for coh_temp in coh_list]
sigma = 1.29705615                    # SD constant, want to use D = sigma**2/2
param_mu_0_OUpos = 6.99053975           #Note that this value largely depends on the model used...
param_mu_0_OUneg = -7.73123206          #Note that this value largely depends on the model used.... NOTE that this is the regime where control is optimal over OU+-, and OU+- are significantly different.

x_sim_1_list = np.zeros(N_sim)
rt_sim_1_list = np.zeros(N_sim)
# correct_sim_1_list = np.zeros(N_sim)
# x_list_sim_1_list = [[] for _ in range(N_sim)]
# t_list_sim_1_list = [[] for _ in range(N_sim)]



########################################################################################################################
### Functions
# Function that simulates one trial of the time-varying drift/bound DDM
def sim_DDM_general(mu, f_mu_setting, sigma, f_sigma_setting, B, f_bound_setting, seed=1):
    '''
    Now assume f_mu, f_sigma, f_Bound are callable functions
    '''
    # Set random seed
    # np.random.seed(seed)

    # Initialization
    x = 0
    t = 0

    # Storage
    x_plot = [x]
    t_plot = [t]

    # Looping through time
    while abs(x)<=f_bound(B, x, t, tau_bound, f_bound_setting):                                                         # Fixed time task. Reaction time task with last if removed.
        x += f_mu(mu, x, t, f_mu_setting)*dt + np.random.randn()*f_sigma(sigma, x, t, f_sigma_setting)*np.sqrt(dt)
        t += dt

        x_plot.append(x)
        t_plot.append(t)

        if x > f_bound(B, x, t, tau_bound, f_bound_setting):
            rt = t
            choice = 1 # choose left
            break
        if x < -f_bound(B, x, t, tau_bound, f_bound_setting):
            rt = t
            choice = -1 # choose right
            break
        if t>2.:                                                                                                        # Block this if loop to become Reaction time task.
            # print('ERROR: Not hitting bound..')
            choice = 0 # undecided, choice undefined originally
            rt = 0
            break

    # If no boundary is hit before maximum time,
    # choose according to decision variable value
    # if t == t_simu[-1]:
    #     rt = t
    #     choice = 1 if x > 0 else 0

    correct = choice                                                                                                    # Trivial. Left for historical reasons or in case it is of use.
    # correct = choice==1                                                                                                 # suppose left choice is correct, but does not distinguish between incorrect and undecided.
    return choice, correct, rt, x_plot, t_plot





## Effect of drift=mu to particle
def f_mu(mu, x, t, f_mu_setting):
    if f_mu_setting == 'constant':
        return mu
    if f_mu_setting == 'OU':
        a = param_mu_0_OUpos             #To be set
        return mu + a*x
    if f_mu_setting == 'OU_negative':
        a = param_mu_0_OUneg             #To be set
        return mu + a*x
    if f_mu_setting == 't_term':
        b = 2.*mu             #To be set
        return mu + b*t
    if f_mu_setting == 'test':
        return mu*(1+np.sin(x)+np.cos(t))
    ###And so on...


## Effect of noise=sigma to particle
def f_sigma(sigma, x, t, f_sigma_setting):
    if f_sigma_setting == 'constant':
        return sigma
    ## NOTE: Have to play with it and see what types of sigmas we want to play with.
    if f_sigma_setting == 'OU':
        a = 0.5             #To be set
        return sigma + a*x
    if f_sigma_setting == 't_term':
        b = 0.5             #To be set
        return sigma + b*t
    if f_sigma_setting == 'test':
        return sigma*(1+np.sin(x)+np.cos(t))
    ###And so on...

## Bound at time t.
def f_bound(B, x, t, tau, f_bound_setting):
    if f_bound_setting == 'constant':
        return B
    if f_bound_setting == 'collapsing_linear':
        return B - t/tau
    if f_bound_setting == 'collapsing_exponential':
        return B*np.exp(-t/tau)
    ###And so on...
    #Can we find a collapsing bound signal comparing to autonomous ones (from the neuron itself).





## For smoothing data.
def sliding_win_on_lin_data(data_mat,window_width,axis=0):
    smaller_half = np.floor((window_width)/2)
    bigger_half = np.ceil((window_width)/2)
    data_mat_result = np.zeros(len(data_mat))
    for k_lin in range(len(data_mat)):
        lower_bound = int(np.maximum(k_lin-smaller_half, 0))
        upper_bound = int(np.minimum(k_lin+bigger_half, len(data_mat)))
        data_mat_result[k_lin] = np.mean(data_mat[lower_bound:upper_bound])
    return data_mat_result


















########################################################################################################################
### Compute the final position (i.e. choice), reaction time, correctness (see above, currently same as choice), and x,t trajectories.

#suffix    = suffix_joint.join(("test", parameter_1_name+"="+str(parameter_1_list[index_1])))   #If we have 1 parameters
x_sim_1_list = np.zeros(N_sim)
rt_sim_1_list = np.zeros(N_sim)
correct_sim_1_list = np.zeros(N_sim)
x_list_sim_1_list = [[] for _ in range(N_sim)]
t_list_sim_1_list = [[] for _ in range(N_sim)]
for i_sim in range(N_sim):
    # (x_sim_1, correct_sim_1, rt_sim_1, x_list_sim_1, t_list_sim_1) = sim_DDM_general(mu, 'constant', sigma, 'constant', B, 'constant')
    (x_sim_1, correct_sim_1, rt_sim_1, x_list_sim_1, t_list_sim_1) = sim_DDM_general(mu, 'constant', sigma, 'test', B, 'collapsing_exponential')
    x_sim_1_list[i_sim] = x_sim_1
    rt_sim_1_list[i_sim] = rt_sim_1
    correct_sim_1_list[i_sim] = correct_sim_1
    x_list_sim_1_list[i_sim] = x_list_sim_1
    t_list_sim_1_list[i_sim] = t_list_sim_1



########################################################################################################################
### Figure of probability density functions, for correct and erred trials separately.

n_bins = np.arange(0., 2., 0.02)                                                                                        # Time bins for the pdf distributions.
rt_sim_1_list_correct   = rt_sim_1_list[correct_sim_1_list== 1]                                                         # Reaction time distribution for correct choices.
rt_sim_1_list_incorrect = rt_sim_1_list[correct_sim_1_list==-1]                                                         # Reaction time distribution for incorrect choices.

# Plot RTs for both correct and incorrect choices.
fig1 = plt.figure(figsize=(8,10.5))
ax11 = fig1.add_subplot(211)
pdf_t_correct_sim_1, bins_t_correct_sim_1, patches_t_correct_sim_1 = ax11.hist(rt_sim_1_list_correct, n_bins, color = 'b', alpha=1.) #, label='Flash')
ax12 = fig1.add_subplot(212)
pdf_t_incorrect_sim_1, bins_t_incorrect_sim_1, patches_t_incorrect_sim_1 = ax12.hist(rt_sim_1_list_incorrect, n_bins, color = 'r', alpha=1.) #, label='Flash')
fig1.savefig('rt_dist_sim1.pdf')

print(len(pdf_t_correct_sim_1))
print len(bins_t_correct_sim_1)

np.save( "DDM_sim_t_pdf.npy", [bins_t_correct_sim_1, pdf_t_correct_sim_1/N_sim, bins_t_incorrect_sim_1, pdf_t_incorrect_sim_1/N_sim])   #Resave everytime, just to make sure I don't mess anything up..

















## Blocked now but working perfectly. Find the sample trajectory whose reaction is the median of the population.
## Blocked now but working perfectly. Find the sample trajectory whose reaction is the median of the population.
## Blocked now but working perfectly. Find the sample trajectory whose reaction is the median of the population.
# ########################################################################################################################
# ### Trajectory of median reaction time
#
# RT_1_median  = np.median(rt_sim_1_list)                                                                                 # Compute median reaction time of model 1
# RT_3_median  = np.median(rt_sim_3_list)                                                                                 # Compute median reaction time of model 3
#
# # In the case that most decisions are undecided, consider only the decided trials.
# RT_31_decided = rt_sim_31_list[rt_sim_31_list!=0]                                                                       # Compute median reaction time of model 31
# if np.sum(rt_sim_31_list!=0) %2 ==0:                                                                                    # Only consider odd number of trials, so that the median value is of a trial and not an average of 2 trials.
#     RT_31_decided = RT_31_decided[1:]
# RT_31_median = np.median(RT_31_decided)
#
#
# ## Find the set whose RT is the same as the computed median value. Inefficient but fine for now.
# RT_1_median_index_TF = rt_sim_1_list== RT_1_median
# RT_3_median_index_TF = rt_sim_3_list== RT_3_median
# RT_31_median_index_TF = rt_sim_31_list== RT_31_median
# for i_TF in range(N_sim):
#     if RT_1_median_index_TF[i_TF]==True:
#         RT_1_median_set_n = range(N_sim)[i_TF]
#     if RT_3_median_index_TF[i_TF]==True:
#         RT_3_median_set_n = range(N_sim)[i_TF]
#     if RT_31_median_index_TF[i_TF]==True:
#         RT_31_median_set_n= range(N_sim)[i_TF]
#
#
# ## Trajectories of median RTs
# x_list_sim_1_median  = x_list_sim_1_list[  RT_1_median_set_n]
# x_list_sim_1_median  = [x_sim_1_list[RT_1_median_set_n] * x_temp for x_temp in x_list_sim_1_median]
# t_list_sim_1_median  = t_list_sim_1_list[  RT_1_median_set_n]
# x_list_sim_3_median  = x_list_sim_3_list[  RT_3_median_set_n] #* x_sim_3_list[RT_3_median_set_n]
# x_list_sim_3_median  = [x_sim_3_list[RT_3_median_set_n] * x_temp for x_temp in x_list_sim_3_median]
# t_list_sim_3_median  = t_list_sim_3_list[  RT_3_median_set_n]
# x_list_sim_31_median = x_list_sim_31_list[RT_31_median_set_n] #* x_sim_31_list[RT_31_median_set_n]
# x_list_sim_31_median  = [x_sim_31_list[RT_31_median_set_n] * x_temp for x_temp in x_list_sim_31_median]
# t_list_sim_31_median = t_list_sim_31_list[RT_31_median_set_n]
#
#
#
# ## Smoothen data before plotting
#
#
#
# plt.plot(t_list_sim_1_median, sliding_win_on_lin_data(x_list_sim_1_median, 10), 'r', alpha=1., label='a=0')
# #    plt.plot(t_list_sim_2, x_list_sim_2, alpha=1.)
# plt.plot(t_list_sim_3_median, sliding_win_on_lin_data(x_list_sim_3_median, 10), 'g', alpha=1., label='a>0')
# plt.plot(t_list_sim_31_median, sliding_win_on_lin_data(x_list_sim_31_median, 10), 'b', alpha=1., label='a<0')
# plt.plot(t_list_sim_31_list[1], sliding_win_on_lin_data(x_list_sim_31_list[1], 10), 'k', alpha=1., label='a<0, undec')                                                # Undecided
# # plt.plot(t_list_sim_4, x_list_sim_4, alpha=0.1)
# plt.xlim([0.,1.2])
# # plt.xlim([0.,np.max(t_list_sim_31_median[-1], t_list_sim_1_median[-1])])
# plt.ylim([-1.,1.])
# plt.xlabel('time (s)')
# plt.ylabel('x (a.u.)')
# #    plt.xscale('log')
# plt.legend(loc=4)
#
#
# #plt.show()
# plt.savefig('Trajectories_Median.pdf')
# np.save
# np.save( "fig3_b_xy2.npy", [t_list_sim_3_median, sliding_win_on_lin_data(x_list_sim_3_median, 10)])                     #Resave everytime, just to make sure I don't mess anything up..
# np.save( "fig3_b_xy3.npy", [t_list_sim_31_median, sliding_win_on_lin_data(x_list_sim_31_median, 10)])                   #Resave everytime, just to make sure I don't mess anything up..
# np.save( "fig3_b_xy4.npy", [t_list_sim_31_list[1], sliding_win_on_lin_data(x_list_sim_31_list[1], 10)])                 #Resave everytime, just to make sure I don't mess anything up..
#
#
#

