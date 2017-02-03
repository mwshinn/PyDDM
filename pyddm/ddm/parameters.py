'''
Simulation code for Drift Diffusion Model
Author: Norman Lam (norman.lam@yale.edu)
'''
from __future__ import print_function, unicode_literals, absolute_import, division

# Default simulaiton parameters.  These can be overridden by user
# code on a per-model basis.

# Parameters.
dx = .05#0.008 # grid size
T_dur = 2. # [s] Duration of simulation
dt = .003#0.005 # [s] Time-step.
