# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
#           2018 Gangyu Robert Yang
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

import numpy as np

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
    # errbnd   = 1e-7 # Error bound for looping
    suminc   = 0
    # checkerr = 0

    for n in range(nMax):
        # increment
        inc = np.exp(tmp*n*((n+1)*a1-n*a2))*((2*n+1)*a1-2*n*a2)-\
              np.exp(tmp*(n+1)*(n*a1-(n+1)*a2))*((2*n+1)*a1-2*(n+1)*a2)
        suminc += inc
        # Break when the relative increment is low for two consecutive updates
        # if(max(abs(inc/suminc)) < errbnd):
        #     checkerr += 1
        #     if(checkerr == 2):
        #         break
        # else:
        #     checkerr = 0

    # Probability Distribution of reaction time
    dist = np.exp(-(a1+b1*teval)**2./teval/2)/np.sqrt(2*np.pi)/teval**1.5*suminc
    dist = dist*(dist>0) # make sure non-negative
    return dist

def analytic_ddm(drift, noise, b, teval, b_slope=0):
    '''
    Calculate the reaction time distribution of a Drift Diffusion model
    Parameters
    -------------------------------------------------------------------
    drift : Drift rate
    noise : Noise intensity
    B     : Constant boundary
    teval : The array of time points where the reaction time distribution is evaluated
    b_slope : (Optional) If provided, then the upper boundary is B(t) = b + b_slope*t,
              and the lower boundary is B(t) = -b - b_slope*t

    Return:
    dist_cor : Reaction time distribution at teval for correct trials
    dist_err : Reaction time distribution at teval for error trials
    '''
    # Scale B, drift, and (implicitly) noise so new noise is 1
    b       /= noise
    drift   /= noise
    b_slope /= noise

    # Get valid time points (before two bounds collapsed)
    teval_valid = teval[b+b_slope*teval>0]

    dist_cor = analytic_ddm_linbound(b, -drift+b_slope, -b, -drift-b_slope, teval_valid)
    dist_err = analytic_ddm_linbound(b,  drift+b_slope, -b,  drift-b_slope, teval_valid)

    # For invalid time points, set the probability to be a very small number
    if len(teval_valid) < len(teval):
        eps = np.ones(len(teval)-len(teval_valid)) * 1e-100
        dist_cor = np.concatenate((dist_cor,eps))
        dist_err = np.concatenate((dist_err,eps))

    return dist_cor, dist_err
