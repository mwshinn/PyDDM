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
B_max = 1. # Maximum boundary. Assumed to be 1

x_list = np.arange(-B_max, B_max+0.1*dx, dx) # List of x-grids (Staggered-mesh)
t_list = np.arange(0., T_dur, dt) # t-grids


########################################################################################################################

