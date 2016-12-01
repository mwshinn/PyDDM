'''
Simulation code for Drift Diffusion Model
Author: Norman Lam (norman.lam@yale.edu)
'''
#import brian_no_units           #Note: speeds up the code
from numpy.fft import rfft,irfft
import time
import numpy as np
from scipy.special import erf
#from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
import cProfile
import re
import matplotlib.cm as matplotlib_cm
import math
#import sys
#sys.path.append('C:\Users\nhl8\Desktop\CCNSS\Drift_Diffusion_Model')
from ddm import analytic_ddm
import multiprocessing
import copy


########################################################################################################################
### Initialization
## Flags to run various parts or not
Flag_random_traces = 1     # Plot some traces that have somewhat arbitrary trajectories
#Flag_coherence_tasks  = 1  # Do various tasks (fixed time, Psychophysical Kernel, Duration Paradigm, Pulse Paradigm). Nah... not a good idea to do this here...
Flag_Pulse = 0
Flag_Duration = 1
Flag_PK = 0

#Load parameters and functions
execfile('DDM_parameters.py')
execfile('DDM_functions.py')


########################################################################################################################
### Load Data to be fitted.

## Define Paths
path_joint = '/'                                                                                                        # Joint between various folder sub-name strings
path_cwd = '/net/murray/nhl8/0906-Decision_Making_EI/'       #Current Directory
path_shared = '/net/murray/nhl8/Data_Temp/1109/Decision_Making_EI/Retune_Stability'     # Shared folder
suffix_joint = "_"                                                                                                      # Joint between strings of suffix sub-parts
empty_joint = ""
equal_joint = "="
foldername_joint = "_"
set_number = 1000                                                                                                        # Number of sets of repeated simulations

### Information for the indices and savepath of the simulation.
variable_list = ['Control', 'gEI-3p', 'gEE-2p']         #Manually used variable.
#variable_list = ['Control']         #Manually used variable.
color_list    = ['r', 'g', 'b']
variable_suffix = "1.84_2.07_38"

## Parameters scanned through.
parameter_1_name = "coherence"
parameter_1_list = np.array([0.0,3.2,6.4,12.8,25.6,51.2])                                                               # For Psychometric Function and Duration
# parameter_1_list = np.array([-51.2,-25.6,-12.8,-6.4,-3.2,0.0,3.2,6.4,12.8,25.6,51.2])                                 # For Pulse
#parameter_1_list = sorted(set(np.concatenate((parameter_list_rest_1,parameter_list_rest_2),axis=0)))		#To join results from multiple simulations

#Initialization: Create Store arrays
# win_12_list     = np.zeros((len(parameter_1_list), len(variable_list), set_number))                                         # Which group (output 1 or 2) wins the race
# RT_win_list     = np.zeros((len(parameter_1_list), len(variable_list), set_number))                                         # Reaction time of the winner
win_lose_undec_list_count = np.zeros((3, len(parameter_1_list), len(variable_list)))                                                 # Probability of group E1 winning the race



### Import files, over the span of parameter_list = coherence levels.
for i_set in range(set_number):
    folder_name_shared_set = foldername_joint.join(('Set', str(i_set+1)))
    for i_variable_manual in range(len(variable_list)):
    #   folder_name_shared_temp = empty_joint.join((eval("foldername_shared"+suffix_temp), str(variable_list_temp[i_variable_manual])))
        variable_manual = suffix_joint.join(variable_list[i_variable_manual], variable_suffix)
    #    folder_name_shared_variable = foldername_joint.join((folder_name_shared_variable, variable_name+"="+str(variable_manual)))   # Folder name specified by parameter 2, that contains all simulations with varying parameter 1
#        folder_name_shared_variable = foldername_joint.join((folder_name_shared_variable, variable_manual))   # Folder name specified by parameter 2, that contains all simulations with varying parameter 1
        folder_name_shared_variable = variable_manual   # Folder name specified by parameter 2, that contains all simulations with varying parameter 1
        #Now loop over all parameters simulated.
        # for i_parameter_2_list in range(len(parameter_2_list)):
        #     p2 =  parameter_2_list[i_parameter_2_list]
        #     folder_name_shared_p2  = foldername_joint.join(("" , parameter_2_name+"="+str(p2)))   # Folder name specified by parameter 2, that contains all simulations with varying parameter 1
        if True:        ## Temp, in absence of p2
            for i_parameter_1_list in range(len(parameter_1_list)):
                # Initialization
                p1 = parameter_1_list[i_parameter_1_list]
                folder_name_p1  = equal_joint.join((parameter_1_name, str(p1)))   # Set file name for soma spikes.
#                path_temp  = path_joint.join((path_shared, folder_name_shared_variable, folder_name_shared_set, folder_name_shared_p2, folder_name_p1, ''))   # Set file name for soma spikes.
                path_temp  = path_joint.join((path_shared, folder_name_shared_variable, folder_name_shared_set, folder_name_p1, ''))   # Set file name for soma spikes.
                print(path_temp)                                                                                        # To check which simulation set does not work.
                ## IF I were to use r_smooth data
                # filename_r_smooth_Pe1  = ("r_smooth_E1.txt")   # psth-smoothened curves of population spike rates of N1.
                # filename_r_smooth_Pe2  = ("r_smooth_E2.txt")   # psth-smoothened curves of population spike rates of N2.
                ## If I were to use win12/RTwin data directly
                filename_win12_RTwin = ("win12_RTwin.txt")   # Set file name for soma spikes.
                win12_RTwin_temp = np.loadtxt(path_temp+filename_win12_RTwin)
                win_12_temp = win12_RTwin_temp[0]
                if win_12_temp==1:
                    win_lose_undec_list_count[0, i_parameter_1_list, i_variable_manual] +=1     # Win
                elif win_12_temp==2:
                    win_lose_undec_list_count[1, i_parameter_1_list, i_variable_manual] +=1     # Lose
                elif win_12_temp==0:
                    win_lose_undec_list_count[2, i_parameter_1_list, i_variable_manual] +=1     # Undecided
                # Note that we do not add undecided trials to win/lose cases, as we only want to extract the parameters and compare to DDM...
                # RT_win_list[i_parameter_1_list, i_variable_manual, i_set] = win12_RTwin_temp[1]

## Use count such that more simulation runs will have more weight... or use prob for normalization/ criteria of needing a probability is an issue
win_lose_undec_list_Prob = win_lose_undec_list_count/set_number                                                         # Used for plotting if nothing else.






########################################################################################################################
### Fit unstable/leaky integrator (+/- OU) to increased/decreased E/I balance (gEI-3p / gEE-2p)
## Use count such that more simulation runs will have more weight... or use prob for normalization/ criteria of needing a probability is an issue

#res = minimize(MLE_model_fit_over_coh, [mu_0, sigma_0, param_mu_0_list[i_2fit], param_B_0_list[i_2fit]], tol=10.**-6, args = (i_2fit, win_lose_undec_list_count))
params_initial_guess = [13.86678439, 1.19844365, 0., 0.]                                                                # From MSE fit
params_OU_0_pos =  1.
params_OU_0_neg = -1.

#res = minimize(MLE_model_fit_over_coh, [mu_0, sigma_0, param_mu_0_list[i_2fit], param_B_0_list[i_2fit]], tol=10.**-6, args = (i_2fit, win_lose_undec_list_Prob))
res_OU_pos = minimize(MLE_model_fit_OU_over_coh, params_OU_0_pos, tol=10.**-6, args = (params_initial_guess, 3, win_lose_undec_list_Prob[:,:,1]))
res_OU_neg = minimize(MLE_model_fit_OU_over_coh, params_OU_0_neg, tol=10.**-6, args = (params_initial_guess, 4, win_lose_undec_list_Prob[:,:,2]))
params_fitted_OU_pos = copy.copy(params_initial_guess)
params_fitted_OU_pos[2] = res_OU_pos.x
params_fitted_OU_neg = copy.copy(params_initial_guess)
params_fitted_OU_neg[2] = res_OU_neg.x
print params_fitted_OU_pos
print params_fitted_OU_neg

## Temporary, should make the code more versatile/ general
for i_coh in range(len(parameter_1_list)):
    i_2fit_OU_pos = 3
    params_fitted_coh_OU_pos                     = copy.copy(params_fitted_OU_pos)
    params_fitted_coh_OU_pos[0]                 *= (0.+ 0.01*parameter_1_list[i_coh])                                                             #Modify  mu based on coherence level
    (Prob_list_corr_fitted, Prob_list_err_fitted)= DDM_pdf_general(params_fitted_coh_OU_pos, i_2fit_OU_pos, 0)
    Prob_list_sum_corr_fitted                    = np.sum(Prob_list_corr_fitted)
    Prob_list_sum_err_fitted                     = np.sum(Prob_list_err_fitted)
    Prob_list_sum_undec_fitted                   = 1. - Prob_list_sum_corr_fitted - Prob_list_sum_err_fitted
    Prob_corr_err_undec_fitted[:,i_coh, 1]          = [Prob_list_sum_corr_fitted, Prob_list_sum_err_fitted, Prob_list_sum_undec_fitted]
for i_coh in range(len(parameter_1_list)):
    i_2fit_OU_neg = 4
    params_fitted_coh_OU_neg                     = copy.copy(params_fitted_OU_neg)
    params_fitted_coh_OU_neg[0]                 *= (0.+ 0.01*parameter_1_list[i_coh])                                                             #Modify  mu based on coherence level
    (Prob_list_corr_fitted, Prob_list_err_fitted)= DDM_pdf_general(params_fitted_coh_OU_neg, i_2fit_OU_neg, 0)
    Prob_list_sum_corr_fitted                    = np.sum(Prob_list_corr_fitted)
    Prob_list_sum_err_fitted                     = np.sum(Prob_list_err_fitted)
    Prob_list_sum_undec_fitted                   = 1. - Prob_list_sum_corr_fitted - Prob_list_sum_err_fitted
    Prob_corr_err_undec_fitted[:,i_coh, 2]          = [Prob_list_sum_corr_fitted, Prob_list_sum_err_fitted, Prob_list_sum_undec_fitted]


########################################################################################################################
## Plot spiking circuit and fitted DDM psychometric functions.
fig2 = plt.figure(figsize=(8.,10.5))
n_skip=8 # sparsen scatter plots
### E1 win rate vs Coherence ###
ax21 = fig2.add_subplot(411)
for i_var in range(len(variable_list)):
    ax21.plot( parameter_1_list, win_lose_undec_list_Prob[  0,:,i_var], label=variable_list[i_var], c=color_list[i_var])
    ax21.plot( parameter_1_list, Prob_corr_err_undec_fitted[0,:,i_var], label=variable_list[i_var]+"DDM", c=color_list[i_var], marker ="x" )
#    ax21.plot( parameter_1_list, win_0_list_Prob[:,i_var], label=variable_list[i_var], c=color_list[i_var])
# ax21.set_ylim(0,360)
# ax21.set_xlim(0,t_sim)
#ax21.set_xscale('log')
ax21.set_ylabel('E1 Win Probability')
#ax21.set_xlabel('Coherence (%)')
ax21.set_title('E1 Win Probability')
ax21.legend(frameon=False    )

### E2 win rate vs Coherence ###
ax22 = fig2.add_subplot(412)
for i_var in range(len(variable_list)):
    ax22.plot( parameter_1_list, win_lose_undec_list_Prob[  1,:,i_var], label=variable_list[i_var], c=color_list[i_var])
    ax22.plot( parameter_1_list, Prob_corr_err_undec_fitted[1,:,i_var], label=variable_list[i_var]+"DDM", c=color_list[i_var], marker ="x")
               #, marker='.',c=color_list[i_var],edgecolor='none',alpha=0.5)
# ax22.set_ylim(0,360)
# ax22.set_xlim(0,t_sim)
#ax22.set_xscale('log')
ax22.set_ylabel('E2 Win Probability')
#ax22.set_xlabel('Coherence (%)')
# ax22.set_title('E2 Win Probability')
ax22.legend(frameon=False    )

### Undecided Probability vs Coherence###
ax23 = fig2.add_subplot(413)
for i_var in range(len(variable_list)):
    ax23.plot( parameter_1_list, win_lose_undec_list_Prob[  2,:,i_var], label=variable_list[i_var], c=color_list[i_var])
    ax23.plot( parameter_1_list, Prob_corr_err_undec_fitted[2,:,i_var], label=variable_list[i_var]+"DDM", c=color_list[i_var], marker ="x")
# Note: [::n] means jump to the nth element, skipping the n-1 ones...
# ax24.set_ylim(0,360)
# ax24.set_xlim(0,t_sim)
ax23.set_ylabel('Undecision Probability')
ax23.set_xlabel('Coherence (%)')
# ax23.set_title('Undecision Probability')
ax23.legend(frameon=False    )

### Winning Population Reaction Time vs Coherence###
ax24 = fig2.add_subplot(414)
for i_var in range(len(variable_list)):
    ax24.plot( parameter_1_list, win_lose_undec_list_Prob[  2,:,i_var], label=variable_list[i_var], c=color_list[i_var])
    ax24.plot( parameter_1_list, Prob_corr_err_undec_fitted[2,:,i_var], label=variable_list[i_var]+"DDM", c=color_list[i_var], marker ="x")
# Note: [::n] means jump to the nth element, skipping the n-1 ones...
# ax24.set_ylim(0,360)
# ax24.set_xlim(0,t_sim)
ax24.set_ylabel('Undecision Probability')
ax24.set_xlabel('Coherence (%)')
# ax24.set_title('Undecision Probability')
ax24.legend(frameon=False    )

# Maybe can plot steady state diring rate of winner? later...# ax24 = fig2.add_subplot(313)

fig2.savefig(path_cwd+'Win_Loss_Undec_Prob_DDM_fit.pdf')











########################################################################################################################
### Actual results...
########################################################################################################################
## Vary coherence and do each tasks (fixed-time, psychophysical kernel, Duration Paradigm, Pulse Paradigm)
models_list = [0,1,2,3,4]                                #List of models to use. See Setting_list
#models_list = models_list_all                                #List of models to use. See Setting_list

#mu_0_list = np.arange(0.2,2.01,0.2)                         #List of coherence-equivalence to be looped through
# mu_0_list = np.logspace(-2,1, 50)                         #List of coherence-equivalence to be looped through
# mu_0_list = [mu_0]
Prob_final_corr  = np.zeros((len(mu_0_list), len(models_list)))                                                         # Array to store the total correct probability for each mu & model.
Prob_final_err   = np.zeros((len(mu_0_list), len(models_list)))                                                         # Array to store the total erred probability for each mu & model.
Prob_final_undec = np.zeros((len(mu_0_list), len(models_list)))                                                         # Array to store the total undecided probability for each mu & model.
Mean_Dec_Time    = np.zeros((len(mu_0_list), len(models_list)))                                                         # Array to store the total correct probability for each mu & model.

Prob_final_corr_Analy  = np.zeros((len(mu_0_list), 2))
Prob_final_err_Analy   = np.zeros((len(mu_0_list), 2))
Prob_final_undec_Analy = np.zeros((len(mu_0_list), 2))
Mean_Dec_Time_Analy    = np.zeros((len(mu_0_list), 2))


## Define an array to hold the data of simulations over all models, using the original parameters (mu_0)
Prob_final_corr_0  = np.zeros((len(models_list)))
Prob_final_err_0   = np.zeros((len(models_list)))
Prob_final_undec_0 = np.zeros((len(models_list)))
Mean_Dec_Time_0    = np.zeros((len(models_list)))

traj_mean_pos_all = np.zeros((len(t_list), len(models_list)))

### Compute the probability distribution functions for the correct and erred choices                                    # NOTE: First set T_dur to be the the duration of the fixed duration task.
for i_models in range(len(models_list)):
    index_model_2use = models_list[i_models]
    for i_mu0 in range(len(mu_0_list)):
        mu_temp = mu_0_list[i_mu0]
        (Prob_list_corr_temp, Prob_list_err_temp) = DDM_pdf_general([mu_temp, param_mu_x_list[index_model_2use], param_mu_t_list[index_model_2use], sigma_0, param_sigma_x_list[index_model_2use], param_sigma_t_list[index_model_2use], B, param_B_t_list[index_model_2use]], index_model_2use, 0)                               # Simple DDM
        Prob_list_sum_corr_temp  = np.sum(Prob_list_corr_temp)
        Prob_list_sum_err_temp  = np.sum(Prob_list_err_temp)
        Prob_list_sum_undec_temp  = 1 - Prob_list_sum_corr_temp - Prob_list_sum_err_temp


        #Outputs...
        # Forced Choices: The monkey will always make a decision: Split the undecided probability half-half for corr/err choices.
        Prob_final_undec[i_mu0, i_models] = Prob_list_sum_undec_temp
        Prob_final_corr[i_mu0, i_models]  = Prob_list_sum_corr_temp + Prob_final_undec[i_mu0, i_models]/2.
        Prob_final_err[i_mu0, i_models]   = Prob_list_sum_err_temp  + Prob_final_undec[i_mu0, i_models]/2.
        # Mean_Dec_Time[i_mu0, i_models]    = np.sum((Prob_list_corr_temp+Prob_list_err_temp) *t_list) / np.sum((Prob_list_corr_temp+Prob_list_err_temp))   # Regardless of choice made. Note that Mean_Dec_Time does not includes choices supposedly undecided and made at the last moment.
        Mean_Dec_Time[i_mu0, i_models]    = np.sum((Prob_list_corr_temp)*t_list) / np.sum((Prob_list_corr_temp))   # Regardless of choice made. Note that Mean_Dec_Time does not includes choices supposedly undecided and made at the last moment.

        ##Normalize to fit to the analytical solution. (Anderson 1960)
        if index_model_2use ==1 or index_model_2use ==2:
            Prob_final_corr[i_mu0, i_models] = Prob_list_sum_corr_temp / (Prob_list_sum_corr_temp + Prob_list_sum_err_temp)
            Prob_final_err[i_mu0, i_models]  = Prob_list_sum_err_temp / (Prob_list_sum_corr_temp + Prob_list_sum_err_temp)

        ## Analytical solutions (normalized) for simple DDM and CB_Linear, computed if they are in model_list
        if index_model_2use ==0 or index_model_2use==1:
            #note the temporary -ve sign for param_B_0...not sure if I still need it in exponential decay case etc...
            (Prob_list_corr_Analy_temp, Prob_list_err_Analy_temp) = DDM_pdf_analytical([mu_temp, sigma_0, param_mu_x_list[index_model_2use], B, -param_B_t_list[index_model_2use]], index_model_2use, 0)                               # Simple DDM
            Prob_list_sum_corr_Analy_temp  = np.sum(Prob_list_corr_Analy_temp)
            Prob_list_sum_err_Analy_temp   = np.sum(Prob_list_err_Analy_temp)
            Prob_list_sum_undec_Analy_temp = 1 - Prob_list_sum_corr_Analy_temp - Prob_list_sum_err_Analy_temp
            #Outputs...
            # Forced Choices: The monkey will always make a decision: Split the undecided probability half-half for corr/err choices. Actually don't think the analytical solution has undecided trials...
            Prob_final_undec_Analy[i_mu0, i_models] = Prob_list_sum_undec_Analy_temp
            Prob_final_corr_Analy[i_mu0, i_models]  = Prob_list_sum_corr_Analy_temp + Prob_final_undec_Analy[i_mu0, i_models]/2.
            Prob_final_err_Analy[i_mu0, i_models]   = Prob_list_sum_err_Analy_temp  + Prob_final_undec_Analy[i_mu0, i_models]/2.
            # Mean_Dec_Time_Analy[i_mu0, i_models]    = np.sum((Prob_list_corr_Analy_temp+Prob_list_err_Analy_temp) *t_list) / np.sum((Prob_list_corr_Analy_temp+Prob_list_err_Analy_temp))      # Regardless of choices. Note that Mean_Dec_Time does not includes choices supposedly undecided and made at the last moment.
            Mean_Dec_Time_Analy[i_mu0, i_models]    = np.sum((Prob_list_corr_Analy_temp)*t_list) / np.sum((Prob_list_corr_Analy_temp))                                                           # Only consider correct choices. Note that Mean_Dec_Time does not includes choices supposedly undecided and made at the last moment.

    ## Compute the default models (based on spiking circuit) for the various models.
    (Prob_list_corr_0_temp, Prob_list_err_0_temp) = DDM_pdf_general([mu_0_list[0], param_mu_x_list[index_model_2use], param_mu_t_list[index_model_2use], sigma_0, param_sigma_x_list[index_model_2use], param_sigma_t_list[index_model_2use], B, param_B_t_list[index_model_2use]], index_model_2use, 0)                               # Simple DDM
    Prob_list_sum_corr_0_temp  = np.sum(Prob_list_corr_0_temp)
    Prob_list_sum_err_0_temp   = np.sum(Prob_list_err_0_temp)
    Prob_list_sum_undec_0_temp = 1. - Prob_list_sum_corr_0_temp - Prob_list_sum_err_0_temp
    #Outputs...
    Prob_final_corr_0[i_models]  = Prob_list_sum_corr_0_temp
    Prob_final_err_0[i_models]   = Prob_list_sum_err_0_temp
    Prob_final_undec_0[i_models] = Prob_list_sum_undec_0_temp
    Mean_Dec_Time_0[i_models]    = np.sum((Prob_list_corr_0_temp+Prob_list_err_0_temp) *t_list) / np.sum((Prob_list_corr_0_temp+Prob_list_err_0_temp))


### Plot correct probability, erred probability, indecision probability, and mean decision time.
fig1 = plt.figure(figsize=(8,10.5))
ax11 = fig1.add_subplot(411)
for i_models in range(len(models_list)):
    index_model_2use = models_list[i_models]
    ax11.plot(coh_list, Prob_final_corr[:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
    if index_model_2use ==0 or index_model_2use==1:
        ax11.plot(coh_list, Prob_final_corr_Analy[:,i_models], color=color_list[index_model_2use], linestyle=':')      #, label=labels_list[index_model_2use]+"_A" )
#fig1.ylim([-1.,1.])
#ax11.set_xlabel('mu_0 (~coherence)')
ax11.set_ylabel('Probability')
ax11.set_title('Correct Probability')
# ax11.set_xscale('log')
ax11.legend(loc=4)

ax12 = fig1.add_subplot(412)
for i_models in range(len(models_list)):
    index_model_2use = models_list[i_models]
    ax12.plot(coh_list, Prob_final_err[:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
    if index_model_2use ==0 or index_model_2use==1:
        ax12.plot(coh_list, Prob_final_err_Analy[:,i_models], color=color_list[index_model_2use], linestyle=':')       #, label=labels_list[index_model_2use]+"_A" )

#fig1.ylim([-1.,1.])
#ax12.set_xlabel('mu_0 (~coherence)')
ax12.set_ylabel('Probability')
ax12.set_title('Erred Probability')
# ax12.set_xscale('log')
ax12.legend(loc=1)


ax13 = fig1.add_subplot(413)
for i_models in range(len(models_list)):
    index_model_2use = models_list[i_models]
    ax13.plot(coh_list, Prob_final_undec[:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
    if index_model_2use ==0 or index_model_2use==1:
        ax13.plot(coh_list, Prob_final_undec_Analy[:,i_models], color=color_list[index_model_2use], linestyle=':')     #, label=labels_list[index_model_2use]+"_A" )
#fig1.ylim([-1.,1.])
#ax13.set_xlabel('mu_0 (~coherence)')
ax13.set_ylabel('Probability')
ax13.set_title('Undecision Probability')
# ax13.set_xscale('log')
ax13.legend(loc=1)

ax14 = fig1.add_subplot(414)
for i_models in range(len(models_list)):
    index_model_2use = models_list[i_models]
    ax14.plot(coh_list, Mean_Dec_Time[:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
    if index_model_2use ==0 or index_model_2use==1:
        ax14.plot(coh_list, Mean_Dec_Time_Analy[:,i_models], color=color_list[index_model_2use], linestyle=':')        #, label=labels_list[index_model_2use]+"_A" )
#fig1.ylim([-1.,1.])
ax14.set_xlabel('Coherence (%)')
ax14.set_ylabel('Time (s)')
ax14.set_title('Mean Decision Time')
# ax14.set_xscale('log')
ax14.legend(loc=3)

fig1.savefig('Fixed_Task_Performance.pdf')
np.save( "fig3_c_x.npy", coh_list)   #Resave everytime, just to make sure I don't mess anything up..
np.save( "fig3_c_y.npy", Prob_final_corr)   #Resave everytime, just to make sure I don't mess anything up..
# mean time & indecision probabilitiies as SI?





########################################################################################################################
## Fig 2: compare analytical vs implicit method for simple DDM and Collapsing bound (linear). Others have no analytical forms.
if Flag_Compare_num_analy_sim:
    ###models_list_fig2 = [0,1]                                #List of models to use. See Setting_list (DDM and CB_Lin only here)
    mu_0_F2 = mu_0_list[-3]                                                                                                 # Set a particular mu and play with variour settings...
    (Prob_list_corr_1_fig2     , Prob_list_err_1_fig2     ) = DDM_pdf_general(   [mu_0_F2, 0., 0., sigma_0, 0., 0., B, 0.]                   , 0)
    (Prob_list_corr_1_Anal_fig2, Prob_list_err_1_Anal_fig2) = DDM_pdf_analytical([mu_0_F2        , sigma_0, 0.    , B, 0.]                   , 0)
    (Prob_list_corr_2_fig2     , Prob_list_err_2_fig2     ) = DDM_pdf_general(   [mu_0_F2, 0., 0., sigma_0, 0., 0., B,  param_B_t ]          , 1)
    (Prob_list_corr_2_Anal_fig2, Prob_list_err_2_Anal_fig2) = DDM_pdf_analytical([mu_0_F2        , sigma_0, 0.    , B, -param_B_t]           , 1)
    (Prob_list_corr_3_fig2     , Prob_list_err_3_fig2     ) = DDM_pdf_general(   [mu_0_F2, 0., 0., sigma_0, 0., 0., B,  param_B_t, T_dur/4. ], 0, 3)
    (Prob_list_corr_4_fig2     , Prob_list_err_4_fig2     ) = DDM_pdf_general(   [mu_0_F2, 0., 0., sigma_0, 0., 0., B,  param_B_t, T_dur/4. ], 1, 3)
    # (Prob_list_corr_2_fig2     , Prob_list_err_2_fig2     ) = DDM_pdf_general(   [mu_0_F2, 0., 0., sigma_0, sigma_0, sigma_0, B,  param_B_t ]          , 1)           # Testing various x&t varying params (mu, sigma etc), and comparing to simulations


    # Cumulative Sums
    Prob_list_cumsum_corr_1_fig2  = np.cumsum(Prob_list_corr_1_fig2)
    Prob_list_cumsum_err_1_fig2  = np.cumsum(Prob_list_err_1_fig2)
    Prob_list_cumsum_corr_1_Anal_fig2  = np.cumsum(Prob_list_corr_1_Anal_fig2)
    Prob_list_cumsum_err_1_Anal_fig2  = np.cumsum(Prob_list_err_1_Anal_fig2)
    Prob_list_cumsum_corr_2_fig2  = np.cumsum(Prob_list_corr_2_fig2)
    Prob_list_cumsum_err_2_fig2  = np.cumsum(Prob_list_err_2_fig2)
    Prob_list_cumsum_corr_2_Anal_fig2  = np.cumsum(Prob_list_corr_2_Anal_fig2)
    Prob_list_cumsum_err_2_Anal_fig2  = np.cumsum(Prob_list_err_2_Anal_fig2)
    Prob_list_cumsum_corr_3_fig2  = np.cumsum(Prob_list_corr_3_fig2)
    Prob_list_cumsum_err_3_fig2  = np.cumsum(Prob_list_err_3_fig2)
    Prob_list_cumsum_corr_4_fig2  = np.cumsum(Prob_list_corr_4_fig2)
    Prob_list_cumsum_err_4_fig2  = np.cumsum(Prob_list_err_4_fig2)


    # # Note that the analtical forms are normalized, so we
    # Prob_list_corr_1_fig2        = Prob_list_corr_1_fig2       /(Prob_list_cumsum_corr_1_fig2[-1]+Prob_list_cumsum_err_1_fig2[-1])
    # Prob_list_err_1_fig2         = Prob_list_err_1_fig2        /(Prob_list_cumsum_corr_1_fig2[-1]+Prob_list_cumsum_err_1_fig2[-1])
    # Prob_list_cumsum_corr_1_fig2 = Prob_list_cumsum_corr_1_fig2/(Prob_list_cumsum_corr_1_fig2[-1]+Prob_list_cumsum_err_1_fig2[-1])
    # Prob_list_cumsum_err_1_fig2  = Prob_list_cumsum_err_1_fig2 /(Prob_list_cumsum_corr_1_fig2[-1]+Prob_list_cumsum_err_1_fig2[-1])
    # Prob_list_corr_2_fig2        = Prob_list_corr_2_fig2       /(Prob_list_cumsum_corr_2_fig2[-1]+Prob_list_cumsum_err_2_fig2[-1])
    # Prob_list_err_2_fig2         = Prob_list_err_2_fig2        /(Prob_list_cumsum_corr_2_fig2[-1]+Prob_list_cumsum_err_2_fig2[-1])
    # Prob_list_cumsum_corr_2_fig2 = Prob_list_cumsum_corr_2_fig2/(Prob_list_cumsum_corr_2_fig2[-1]+Prob_list_cumsum_err_2_fig2[-1])
    # Prob_list_cumsum_err_2_fig2  = Prob_list_cumsum_err_2_fig2 /(Prob_list_cumsum_corr_2_fig2[-1]+Prob_list_cumsum_err_2_fig2[-1])
    # Prob_list_corr_4_fig2        = Prob_list_corr_4_fig2       /(Prob_list_cumsum_corr_4_fig2[-1]+Prob_list_cumsum_err_4_fig2[-1])
    # Prob_list_err_4_fig2         = Prob_list_err_4_fig2        /(Prob_list_cumsum_corr_4_fig2[-1]+Prob_list_cumsum_err_4_fig2[-1])
    # Prob_list_cumsum_corr_4_fig2 = Prob_list_cumsum_corr_4_fig2/(Prob_list_cumsum_corr_4_fig2[-1]+Prob_list_cumsum_err_4_fig2[-1])
    # Prob_list_cumsum_err_4_fig2  = Prob_list_cumsum_err_4_fig2 /(Prob_list_cumsum_corr_4_fig2[-1]+Prob_list_cumsum_err_4_fig2[-1])

    # In case the trial model has no analytical solution, use simulation (see DDM_sim_compare_pdf.py) instead.
    [bins_edge_t_correct_sim_temp, pdf_t_correct_sim_temp, bins_edge_t_incorrect_sim_temp, pdf_t_incorrect_sim_temp] = np.load('DDM_sim_t_pdf.npy')
    bins_t_correct_sim_temp = 0.5*(bins_edge_t_correct_sim_temp[1:] + bins_edge_t_correct_sim_temp[:-1])
    bins_t_incorrect_sim_temp = 0.5*(bins_edge_t_incorrect_sim_temp[1:] + bins_edge_t_incorrect_sim_temp[:-1])
    norm_sim_temp = np.sum(pdf_t_correct_sim_temp + pdf_t_incorrect_sim_temp)
    dt_ratio = 10.                                                                                                          # Ratio in time step, and thus 1/ number of datapoints, between simulation and numerical solutions.


    ### Plot correct probability, erred probability, indecision probability, and mean decision time.
    fig2 = plt.figure(figsize=(8,10.5))
    ax21 = fig2.add_subplot(411)                                                                                            # PDF, Correct
    ax21.plot(t_list, Prob_list_corr_1_fig2, 'r', label='DDM' )
    ax21.plot(t_list, Prob_list_corr_1_Anal_fig2, 'r:', label='DDM_A' )
    ax21.plot(t_list, Prob_list_corr_2_fig2, 'b', label='test' )
    ax21.plot(t_list, Prob_list_corr_2_fig2, 'b', label='CB_Lin' )
    ax21.plot(t_list, Prob_list_corr_2_Anal_fig2, 'b:', label='CB_Lin_A' )
    ax21.plot(t_list, Prob_list_corr_3_fig2, 'r--', label='DDM_P' )
    ax21.plot(t_list, Prob_list_corr_4_fig2, 'b--', label='CB_Lin_P' )
    ## ax21.plot(bins_t_correct_sim_temp, pdf_t_correct_sim_temp/dt_ratio, 'k-.', label='sim' )
    #fig1.ylim([-1.,1.])
    ax21.set_xlabel('time (s)')
    ax21.set_ylabel('PDF (normalized)')
    ax21.set_title('Correct PDF, Analytical vs Numerical')
    # ax21.set_xscale('log')
    ax21.legend(loc=1)

    ax22 = fig2.add_subplot(412)                                                                                            # PDF, erred
    ax22.plot(t_list, Prob_list_err_1_fig2, 'r', label='DDM' )
    ax22.plot(t_list, Prob_list_err_1_Anal_fig2, 'r:', label='DDM_A' )
    ax22.plot(t_list, Prob_list_err_2_fig2, 'b', label='test' )
    ax22.plot(t_list, Prob_list_err_2_fig2, 'b', label='CB_Lin' )
    ax22.plot(t_list, Prob_list_err_2_Anal_fig2, 'b:', label='CB_Lin_A' )
    ax22.plot(t_list, Prob_list_err_3_fig2, 'r--', label='DDM_P' )
    ax22.plot(t_list, Prob_list_err_4_fig2, 'b--', label='CB_Lin_P' )
    ## ax22.plot(bins_t_incorrect_sim_temp, pdf_t_incorrect_sim_temp/dt_ratio, 'k-.', label='sim' )
    #fig1.set_ylim([-1.,1.])
    ax22.set_xlabel('time (s)')
    ax22.set_ylabel('PDF (normalized)')
    ax22.set_title('Erred PDF, Analytical vs Numerical')
    # fig1.set_xscale('log')
    ax22.legend(loc=1)

    ax23 = fig2.add_subplot(413)                                                                                            # CDF, Correct
    ax23.plot(t_list, Prob_list_cumsum_corr_1_fig2, 'r', label='DDM' )
    ax23.plot(t_list, Prob_list_cumsum_corr_1_Anal_fig2, 'r:', label='DDM_A' )
    ax23.plot(t_list, Prob_list_cumsum_corr_2_fig2, 'b', label='test' )
    ax23.plot(t_list, Prob_list_cumsum_corr_2_fig2, 'b', label='CB_Lin' )
    ax23.plot(t_list, Prob_list_cumsum_corr_2_Anal_fig2, 'b:', label='CB_Lin_A' )
    ax23.plot(t_list, Prob_list_cumsum_corr_3_fig2, 'r--', label='DDM_P' )
    ax23.plot(t_list, Prob_list_cumsum_corr_4_fig2, 'b--', label='CB_Lin_P' )
    ## ax23.plot(bins_t_correct_sim_temp, np.cumsum(pdf_t_correct_sim_temp), 'k-.', label='sim' )

    #fig1.ylim([-1.,1.])
    ax23.set_xlabel('time (s)')
    ax23.set_ylabel('CDF (normalized)')
    ax23.set_title('Correct CDF, Analytical vs Numerical')
    # ax23.set_xscale('log')
    ax23.legend(loc=4)

    ax24 = fig2.add_subplot(414)                                                                                            # CDF, Erred
    ax24.plot(t_list, Prob_list_cumsum_err_1_fig2, 'r', label='DDM' )
    ax24.plot(t_list, Prob_list_cumsum_err_1_Anal_fig2, 'r:', label='DDM_A' )
    ax24.plot(t_list, Prob_list_cumsum_err_2_fig2, 'b', label='test' )
    ax24.plot(t_list, Prob_list_cumsum_err_2_fig2, 'b', label='CB_Lin' )
    ax24.plot(t_list, Prob_list_cumsum_err_2_Anal_fig2, 'b:', label='CB_Lin_A' )
    ax24.plot(t_list, Prob_list_cumsum_err_3_fig2, 'r--', label='DDM_P' )
    ax24.plot(t_list, Prob_list_cumsum_err_4_fig2, 'b--', label='CB_Lin_P' )
    ## ax24.plot(bins_t_incorrect_sim_temp, np.cumsum(pdf_t_incorrect_sim_temp), 'k-.', label='sim' )
    #fig1.set_ylim([-1.,1.])
    ax24.set_xlabel('time (s)')
    ax24.set_ylabel('CDF (normalized)')
    ax24.set_title('Erred CDF, Analytical vs Numerical')
    # fig1.set_xscale('log')
    ax24.legend(loc=4)

    fig2.savefig('Numerical_vs_Analytical_DDM_CBLin.pdf')




















