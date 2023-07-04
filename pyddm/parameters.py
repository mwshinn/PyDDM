# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

# Default simulaiton parameters.  These can be overridden by user
# code on a per-model basis.

# Parameters.
dx = .005 #0.008 # grid size
T_dur = 2. # [s] Duration of simulation
dt = .005 #0.005 # [s] Time-step.
choice_names = ("correct", "error") # Default upper and lower boundary

# Display warnings
renorm_warnings = True
