# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
#           2018 Gangyu Robert Yang
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

import numpy as np
try:
    from . import csolve
    HAS_CSOLVE = True
except ImportError:
    HAS_CSOLVE = False

def analytic_ddm_linbound(a1, b1, a2, b2, teval):
    '''
    Calculate the reaction time distribution of a Drift Diffusion model
    with linear boundaries, zero drift, and noise = 1.

    The upper boundary is y(t) = a1 + b1*t
    The lower boundary is y(t) = a2 + b2*t
    The starting point is 0
    teval is the array of time where the reaction time distribution is evaluated

    Return the reaction time distribution of crossing the upper boundary

    Reference:
    Anderson, Theodore W. "A modification of the sequential probability ratio test
    to reduce the sample size." The Annals of Mathematical Statistics (1960): 165-197.
    '''
    # Avoid dividing by zero
    teval[teval==0] = 1e-30
    
    # Change of variables
    tmp = -2.*((a1-a2)/teval+b1-b2)

    # Initialization
    nMax     = 100  # Maximum looping number
    errbnd   = 1e-10 # Error bound for looping
    suminc   = 0
    checkerr = 0

    for n in range(nMax):
        # increment
        inc = np.exp(tmp*n*((n+1)*a1-n*a2))*((2*n+1)*a1-2*n*a2)-\
              np.exp(tmp*(n+1)*(n*a1-(n+1)*a2))*((2*n+1)*a1-2*(n+1)*a2)
        suminc += inc
        # Break when the relative increment is low for three consecutive updates
        # np.where statement avoids dividing by 0
        if np.max(np.abs(inc/np.where(suminc==0,1e-30,suminc))) < errbnd:
            checkerr += 1
            if checkerr == 3:
                break
        else:
            checkerr = 0

    # Probability Distribution of reaction time
    dist = np.exp(-(a1+b1*teval)**2./teval/2)/np.sqrt(2*np.pi)/teval**1.5*suminc
    dist = dist*(dist>0) # make sure non-negative
    return dist

def analytic_ddm(drift, noise, b, teval, shift=None, b_slope=0, force_python=False):
    '''
    Calculate the reaction time distribution of a Drift Diffusion model
    Parameters
    -------------------------------------------------------------------
    drift : Drift rate
    noise : Noise intensity
    b     : Constant boundary (half of total bound height)
    teval : The array of time points where the reaction time distribution is evaluated
    shift : (Optional) A shift in the starting point on the interval [0,1], expressed as a proportion
              of total bound height 2*b, where 0.5 is the center.
    b_slope : (Optional) If provided, then the upper boundary is B(t) = b + b_slope*t,
              and the lower boundary is B(t) = -b - b_slope*t
    force_python : Force PyDDM to use the pure Python solver instead of the C solver.  Usually, the C
                     solver is about 30% faster.

    Return:
    dist_cor : Reaction time distribution at teval for correct trials
    dist_err : Reaction time distribution at teval for error trials
    '''
    
    #find bounds based on initial condition
    if shift is None:
        b_lower = b
        b_upper = b
    else:
        b_lower = 2. * b * shift
        b_upper = 2. * b - b_lower
    
    # Scale b, drift, and (implicitly) noise so new noise is 1
    b_lower /= noise
    b_upper /= noise
    drift   /= noise
    b_slope /= noise

    # Get valid time points (before two bounds collapsed)
    teval_valid = teval[b+b_slope*teval>0]

    if force_python or not HAS_CSOLVE:
        dist_choice_upper = analytic_ddm_linbound(b_upper, -drift+b_slope, -b_lower, -drift-b_slope, teval_valid)
        dist_choice_lower = analytic_ddm_linbound(b_lower,  drift+b_slope, -b_upper,  drift-b_slope, teval_valid)
    else:
        dt = teval_valid[1]-teval_valid[0]
        dist_choice_upper = csolve.analytic_ddm_linbound(b_upper, -drift+b_slope, -b_lower, -drift-b_slope, len(teval_valid), dt)
        dist_choice_lower = csolve.analytic_ddm_linbound(b_lower,  drift+b_slope, -b_upper,  drift-b_slope, len(teval_valid), dt)

    # For invalid time points, set the probability to be a very small number
    if len(teval_valid) < len(teval):
        eps = np.ones(len(teval)-len(teval_valid)) * 1e-100
        dist_choice_upper = np.concatenate((dist_choice_upper,eps))
        dist_choice_lower = np.concatenate((dist_choice_lower,eps))

    return dist_choice_upper, dist_choice_lower
