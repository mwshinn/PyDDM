from pyddm.models import InitialCondition
import numpy as np
import math
from paranoid import *
from pyddm.models.paranoid_types import Conditions
import scipy.stats
import pyddm as ddm


# Start ICPointSideBias
import numpy as np
from pyddm import InitialCondition
class ICPointSideBias(InitialCondition):
    name = "A starting point with a left or right bias."
    required_parameters = ["x0"]
    required_conditions = ["left_is_correct"]
    def get_IC(self, x, dx, conditions):
        start = np.round(self.x0/dx)
        # Positive bias for left choices, negative for right choices
        if not conditions['left_is_correct']:
            start = -start
        shift_i = int(start + (len(x)-1)/2)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=self.x0.
        return pdf
# End ICPointSideBias

# Start ICPointSideBiasInterp
import numpy as np
import scipy.stats
from pyddm import InitialCondition
class ICPointSideBiasInterp(InitialCondition):
    name = "A dirac delta function at a position dictated by the left or right side."
    required_parameters = ["x0"]
    required_conditions = ["left_is_correct"]
    def get_IC(self, x, dx, conditions):
        start_in = np.floor(self.x0/dx)
        start_out = np.sign(start_in)*(np.abs(start_in)+1)
        w_in = np.abs(start_out - self.x0/dx)
        w_out = np.abs(self.x0/dx - start_in)
        if not conditions['left_is_correct']:
            start_in = -start_in
            start_out = -start_out
        shift_in_i = int(start_in + (len(x)-1)/2)
        shift_out_i = int(start_out + (len(x)-1)/2)
        if w_in>0:
            assert shift_in_i>= 0 and shift_in_i < len(x), "Invalid initial conditions"
        if w_out>0:
            assert shift_out_i>= 0 and shift_out_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_in_i] = w_in # Initial condition at the inner grid next to x=self.x0.
        pdf[shift_out_i] = w_out # Initial condition at the outer grid next to x=self.x0.
        return pdf
# End ICPointSideBiasInterp

# Start ICPointSideBiasRatio
import numpy as np
from pyddm import InitialCondition
class ICPointSideBiasRatio(InitialCondition):
    name = "A side-biased starting point expressed as a proportion of the distance between the bounds."
    required_parameters = ["x0"]
    required_conditions = ["left_is_correct"]
    def get_IC(self, x, dx, conditions):
        x0 = self.x0/2 + .5 #rescale to between 0 and 1
        # Bias > .5 for left side correct, bias < .5 for right side correct.
        # On original scale, positive bias for left, negative for right
        if not conditions['left_is_correct']:
            x0 = 1-x0
        shift_i = int((len(x)-1)*x0)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=x0*2*B.
        return pdf
# End ICPointSideBiasRatio

# Start ICPointRange
import numpy as np
import scipy.stats
from pyddm import InitialCondition
class ICPointRange(InitialCondition):
    name = "A shifted side-biased uniform distribution"
    required_parameters = ["x0", "sz"]
    required_conditions = ["left_is_correct"]
    def get_IC(self, x, dx, conditions, *args, **kwargs):
        # Check for valid initial conditions
        assert abs(self.x0) + abs(self.sz) < np.max(x), \
            "Invalid x0 and sz: distribution goes past simulation boundaries"
        # Positive bias for left correct, negative for right
        x0 = self.x0 if conditions["left_is_correct"] else -self.x0
        # Use "+dx/2" because numpy ranges are not inclusive on the upper end
        pdf = scipy.stats.uniform(x0 - self.sz, 2*self.sz+dx/10).pdf(x)
        return pdf/np.sum(pdf)
# End ICPointRange

# Start ICCauchy
import numpy as np
import scipy.stats
from pyddm import InitialCondition
class ICCauchy(InitialCondition):
    name = "Cauchy distribution"
    required_parameters = ["scale"]
    def get_IC(self, x, dx, *args, **kwargs):
        pdf = scipy.stats.cauchy(0, self.scale).pdf(x)
        return pdf/np.sum(pdf)
# End ICCauchy


# Start OverlayNonDecisionGaussian
import numpy as np
import scipy
from pyddm import Overlay, Solution
class OverlayNonDecisionGaussian(Overlay):
    name = "Add a Gaussian-distributed non-decision time"
    required_parameters = ["nondectime", "ndsigma"]
    def apply(self, solution):
        # Make sure params are within range
        assert self.ndsigma > 0, "Invalid st parameter"
        # Extract components of the solution object for convenience
        choice_upper = solution.choice_upper
        choice_lower = solution.choice_lower
        dt = solution.model.dt
        # Create the weights for different timepoints
        times = np.asarray(list(range(-len(choice_upper), len(choice_upper))))*dt
        weights = scipy.stats.norm(scale=self.ndsigma, loc=self.nondectime).pdf(times)
        if np.sum(weights) > 0:
            weights /= np.sum(weights) # Ensure it integrates to 1
        newchoice_upper = np.convolve(weights, choice_upper, mode="full")[len(choice_upper):(2*len(choice_upper))]
        newchoice_lower = np.convolve(weights, choice_lower, mode="full")[len(choice_upper):(2*len(choice_upper))]
        return Solution(newchoice_upper, newchoice_lower, solution.model,
                        solution.conditions, solution.undec)
# End OverlayNonDecisionGaussian

# Start OverlayNonDecisionLR
import numpy as np
from pyddm import OverlayNonDecision, Solution
class OverlayNonDecisionLR(OverlayNonDecision):
    name = "Separate non-decision time for left and right sides"
    required_parameters = ["nondectimeL", "nondectimeR"]
    required_conditions = ["side"] # Side coded as 0=L or 1=R
    def get_nondecision_time(self, conditions):
        assert conditions['side'] in [0, 1], "Invalid side"
        return self.nondectimeL if conditions['side'] == 0 else self.nondectimeR
# End OverlayNonDecisionLR

# Start DriftCoherence
from pyddm import Drift
class DriftCoherence(Drift):
    name = "Drift depends linearly on coherence"
    required_parameters = ["driftcoh"] # <-- Parameters we want to include in the model
    required_conditions = ["coh"] # <-- Task parameters ("conditions"). Should be the same name as in the sample.
    
    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        return self.driftcoh * conditions['coh']
# End DriftCoherence

# Start DriftCoherenceRewBias
from pyddm import Drift
class DriftCoherenceRewBias(Drift):
    name = "Drift depends linearly on coherence, with a reward bias"
    required_parameters = ["driftcoh", "rewbias"] # <-- Parameters we want to include in the model
    required_conditions = ["coh", "highreward"] # <-- Task parameters ("conditions"). Should be the same name as in the sample.
    
    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        rew_bias = self.rewbias * (1 if conditions['highreward'] == 1 else -1)
        return self.driftcoh * conditions['coh'] + rew_bias
# End DriftCoherenceRewBias

# Start DriftCoherenceLeak
from pyddm import Drift
class DriftCoherenceLeak(Drift):
    name = "Leaky drift depends linearly on coherence"
    required_parameters = ["driftcoh", "leak"] # <-- Parameters we want to include in the model
    required_conditions = ["coh"] # <-- Task parameters ("conditions"). Should be the same name as in the sample.
    
    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, x, conditions, **kwargs):
        return self.driftcoh * conditions['coh'] + self.leak * x
# End DriftCoherenceLeak

# Start DriftSine
import numpy as np
from pyddm.models import Drift
class DriftSine(Drift):
    name = "Sine-wave drifts"
    required_conditions = ["frequency"]
    required_parameters = ["scale"]
    def get_drift(self, t, conditions, **kwargs):
        return np.sin(t*conditions["frequency"]*2*np.pi)*self.scale
# End DriftSine

# Start DriftPulse
from pyddm.models import Drift
class DriftPulse(Drift):
    name = "Drift for a pulse paradigm"
    required_parameters = ["start", "duration", "drift"]
    required_conditions = []
    def get_drift(self, t, conditions, **kwargs):
        if self.start <= t <= self.start + self.duration:
            return self.drift
        return 0
# End DriftPulse

# Start DriftPulseCoh
from pyddm.models import Drift
class DriftPulseCoh(Drift):
    name = "Drift for a coherence-dependent pulse paradigm"
    required_parameters = ["start", "duration", "drift"]
    required_conditions = ["coherence"]
    def get_drift(self, t, conditions, **kwargs):
        if self.start <= t <= self.start + self.duration:
            return self.drift * conditions["coherence"]
        return 0
# End DriftPulseCoh

# Start DriftPulse2
from pyddm.models import Drift
class DriftPulse2(Drift):
    name = "Drift for a pulse paradigm, with baseline drift"
    required_parameters = ["start", "duration", "drift", "drift0"]
    required_conditions = []
    def get_drift(self, t, conditions, **kwargs):
        if self.start <= t <= self.start + self.duration:
            return self.drift
        return self.drift0
# End DriftPulse2


# Start BoundCollapsingStep
from pyddm.models import Bound
class BoundCollapsingStep(Bound):
    name = "Step collapsing bounds"
    required_conditions = []
    required_parameters = ["B0", "stepheight", "steplength"]
    def get_bound(self, t, **kwargs):
        stepnum = t//self.steplength
        step = self.B0 - stepnum * self.stepheight
        return max(step, 0)
# End BoundCollapsingStep

from pyddm.models import Bound
import numpy as np
# Start BoundCollapsingWeibull
import numpy as np
from pyddm import Bound
class BoundCollapsingWeibull(Bound):
    name = "Weibull CDF collapsing bounds"
    required_parameters = ["a", "aprime", "lam", "k"]
    def get_bound(self, t, **kwargs):
        l = self.lam
        a = self.a
        aprime = self.aprime
        k = self.k
        return a - (1 - np.exp(-(t/l)**k)) * (a - aprime)
# End BoundCollapsingWeibull

# Start BoundCollapsingExponentialDelay
class BoundCollapsingExponentialDelay(Bound):
    """Bound collapses exponentially over time.

    Takes three parameters: 

    `B` - the bound at time t = 0.
    `tau` - the time constant for the collapse, should be greater than
    zero.
    `t1` - the time at which the collapse begins, in seconds
    """
    name = "Delayed exponential collapsing bound"
    required_parameters = ["B", "tau", "t1"]
    def get_bound(self, t, conditions, **kwargs):
        if t <= self.t1:
            return self.B
        if t > self.t1:
            return self.B * np.exp(-self.tau*(t-self.t1))
# End BoundCollapsingExponentialDelay

# Start LossByMeans
import numpy as np
from pyddm import LossFunction
class LossByMeans(LossFunction):
    name = "Mean RT and accuracy"
    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur
    def loss(self, model):
        sols = self.cache_by_conditions(model)
        MSE = 0
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            c = frozenset(comb.items())
            s = self.sample.subset(**comb)
            MSE += (sols[c].prob("correct") - s.prob("correct"))**2
            if sols[c].prob("correct") > 0:
                MSE += (sols[c].mean_decision_time() - np.mean(list(s)))**2
        return MSE
# End LossByMeans

# Start BoundSpeedAcc
from pyddm import Bound
class BoundSpeedAcc(Bound):
    name = "constant"
    required_parameters = ["Bacc", "Bspeed"]
    required_conditions = ['speed_trial']
    def get_bound(self, conditions, *args, **kwargs):
        assert self.Bacc > 0
        assert self.Bspeed > 0
        if conditions['speed_trial'] == 1:
            return self.Bspeed
        else:
            return self.Bacc
# End BoundSpeedAcc

# Start urgency_gain
def urgency_gain(t, gain_start, gain_slope):
    return gain_start + t*gain_slope
# End urgency_gain

# Start DriftUrgencyGain
from pyddm import Drift
class DriftUrgencyGain(Drift):
    name = "drift rate with an urgency function"
    required_parameters = ["snr", "gain_start", "gain_slope"]
    def get_drift(self, t, **kwargs):
        return self.snr * urgency_gain(t, self.gain_start, self.gain_slope)
# End DriftUrgencyGain

# Start NoiseUrgencyGain
from pyddm import Noise
class NoiseUrgencyGain(Noise):
    name = "noise level with an urgency function"
    required_parameters = ["gain_start", "gain_slope"]
    def get_noise(self, t, **kwargs):
        return urgency_gain(t, self.gain_start, self.gain_slope)
# End NoiseUrgencyGain

# Start prepare_sample_for_variable_drift
import pyddm as ddm
import numpy as np
RESOLUTION = 11
def prepare_sample_for_variable_drift(sample, resolution=RESOLUTION):
    new_samples = []
    for i in range(0, resolution):
        choice_upper = sample.choice_upper.copy()
        choice_lower = sample.choice_lower.copy()
        undecided = sample.undecided
        conditions = copy.deepcopy(sample.conditions)
        conditions['driftnum'] = (np.asarray([i]*len(choice_upper)),
                                  np.asarray([i]*len(choice_lower)),
                                  np.asarray([i]*undecided))
        new_samples.append(ddm.Sample(choice_upper, choice_lower, undecided, choice_names=samplue.choice_names, **conditions))
    new_sample = new_samples.pop()
    for s in new_samples:
        new_sample += s
    return new_sample
# End prepare_sample_for_variable_drift

# Start DriftUniform
class DriftUniform(ddm.Drift):
    """Drift with trial-to-trial variability.
    
    Note that this is a numerical approximation to trial-to-trial
    drift rate variability and is inefficient.  It also requires
    running the "prepare_sample_for_variable_drift" function above on
    the data.
    """
    name = "Uniformly-distributed drift"
    resolution = RESOLUTION # Number of bins, should be an odd number
    required_parameters = ['drift', 'width'] # Mean drift and the width of the uniform distribution
    required_conditions = ['driftnum']
    def get_drift(self, conditions, **kwargs):
        stepsize = self.width/(self.resolution-1)
        mindrift = self.drift - self.width/2
        return mindrift + stepsize*conditions['driftnum']
# End DriftUniform

# Start DriftMomentToMoment
class DriftMomentToMoment(ddm.Drift):
    """Drift rate which depends on trial-wise observations over time"""
    name = "Moment-to-moment drift"
    BINSIZE = .1 # 100 ms per bin
    required_parameters = ['drift_multiplier'] # How much to scale moment-to-moment drift
    required_conditions = ['signal'] # should be a list of values which determine the moment-to-moment drift
    def get_drift(self, t, conditions, **kwargs):
        bin_number = int(t//self.BINSIZE) # Which bin are we currently in?
        n_bins = len(conditions['signal']) # Total number of bins for this condition
        # If we are currently in a bin which exceeds the total bins, fix to the last bin
        if bin_number >= n_bins:
            bin_number = n_bins-1
        # Compute the moment-to-moment drift
        return conditions['signal'][bin_number] * self.drift_multiplier
# End DriftMomentToMoment

# Start UrgencyMomentToMoment
def signal_to_urgency(t, signal, binsize=.1):
    bin_number = int(t//binsize) # Which bin are we currently in?
    n_bins = len(signal) # Total number of bins for this condition
    # If we are currently in a bin which exceeds the total bins, fix to the last bin
    if bin_number >= n_bins:
        bin_number = n_bins-1
    return 1 + signal[bin_number]

class DriftUrgencyMomentToMoment(ddm.Drift):
    """Drift rate which varies over time, differently for each trial"""
    name = "Moment-to-moment urgency drift"
    required_parameters = ['snr'] # How much to scale moment-to-moment drift
    required_conditions = ['signal'] # should be a list of values which determine the moment-to-moment drift
    def get_drift(self, t, conditions, **kwargs):
        return signal_to_urgency(t, conditions['signal']) * self.snr

class NoiseUrgencyMomentToMoment(ddm.Noise):
    """Noise rate which varies over time, differently for each trial"""
    name = "Moment-to-moment urgency noise"
    required_parameters = [] # How much to scale moment-to-moment drift
    required_conditions = ['signal'] # should be a list of values which determine the moment-to-moment drift
    def get_noise(self, t, conditions, **kwargs):
        return signal_to_urgency(t, conditions['signal'])
# End UrgencyMomentToMoment
