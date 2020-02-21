from ddm.models import InitialCondition
import numpy as np
import math
from paranoid import *
from ddm.models.paranoid_types import Conditions
import scipy.stats
import ddm


# Start ICPointRew
import numpy as np
from ddm import InitialCondition
class ICPointRew(InitialCondition):
    name = "A reward-biased starting point."
    required_parameters = ["x0"]
    required_conditions = ["highreward"]
    def get_IC(self, x, dx, conditions):
        start = np.round(self.x0/dx)
        # Positive bias for high reward conditions, negative for low reward
        if not conditions['highreward']:
            start = -start
        shift_i = int(start + (len(x)-1)/2)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=self.x0.
        return pdf
# End ICPointRew

# Start ICPointRewInterp
import numpy as np
import scipy.stats
from ddm import InitialCondition
class ICPointRewInterp(InitialCondition):
    name = "A dirac delta function at a position dictated by reward."
    required_parameters = ["x0"]
    required_conditions = ["highreward"]
    def get_IC(self, x, dx, conditions):
        start_in = np.floor(self.x0/dx)
        start_out = np.sign(start_in)*(np.abs(start_in)+1)
        w_in = np.abs(start_out - self.x0/dx)
        w_out = np.abs(self.x0/dx - start_in)
        if not conditions['highreward']:
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
# End ICPointRewInterp

# Start ICPointRange
import numpy as np
import scipy.stats
from ddm import InitialCondition
class ICPointRange(InitialCondition):
    name = "A shifted reward-biased uniform distribution"
    required_parameters = ["x0", "sz"]
    required_conditions = ["highreward"]
    def get_IC(self, x, dx, conditions, *args, **kwargs):
        # Check for valid initial conditions
        assert abs(self.x0) + abs(self.sz) < np.max(x), \
            "Invalid x0 and sz: distribution goes past simulation boundaries"
        # Positive bias for high reward conditions, negative for low reward
        x0 = self.x0 if conditions["highreward"] else -self.x0
        # Use "+dx/2" because numpy ranges are not inclusive on the upper end
        pdf = scipy.stats.uniform(x0 - self.sz, 2*self.sz+dx/10).pdf(x)
        return pdf/np.sum(pdf)
# End ICPointRange

# Start ICCauchy
import numpy as np
import scipy.stats
from ddm import InitialCondition
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
from ddm import Overlay, Solution
class OverlayNonDecisionGaussian(Overlay):
    name = "Add a Gaussian-distributed non-decision time"
    required_parameters = ["nondectime", "ndsigma"]
    def apply(self, solution):
        # Make sure params are within range
        assert self.ndsigma > 0, "Invalid st parameter"
        # Extract components of the solution object for convenience
        corr = solution.corr
        err = solution.err
        dt = solution.model.dt
        # Create the weights for different timepoints
        times = np.asarray(list(range(-len(corr), len(corr))))*dt
        weights = scipy.stats.norm(scale=self.ndsigma, loc=self.nondectime).pdf(times)
        if np.sum(weights) > 0:
            weights /= np.sum(weights) # Ensure it integrates to 1
        newcorr = np.convolve(weights, corr, mode="full")[len(corr):(2*len(corr))]
        newerr = np.convolve(weights, err, mode="full")[len(corr):(2*len(corr))]
        return Solution(newcorr, newerr, solution.model,
                        solution.conditions, solution.undec)
# End OverlayNonDecisionGaussian

# Start OverlayNonDecisionLR
import numpy as np
from ddm import Overlay, Solution
class OverlayNonDecisionLR(Overlay):
    name = "Separate non-decision time for left and right sides"
    required_parameters = ["nondectimeL", "nondectimeR"]
    required_conditions = ["side"] # Side coded as 0=L or 1=R
    def apply(self, solution):
        # Check parameters and conditions
        assert solution.conditions['side'] in [0, 1], "Invalid side"
        # Unpack solution object
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        # Compute non-decision time
        ndtime = self.nondectimeL if cond['side'] == 0 else self.nondectimeR
        shifts = int(ndtime/m.dt) # truncate
        # Shift the distribution
        newcorr = np.zeros(corr.shape, dtype=corr.dtype)
        newerr = np.zeros(err.shape, dtype=err.dtype)
        if shifts > 0:
            newcorr[shifts:] = corr[:-shifts]
            newerr[shifts:] = err[:-shifts]
        elif shifts < 0:
            newcorr[:shifts] = corr[-shifts:]
            newerr[:shifts] = err[-shifts:]
        else:
            newcorr = corr
            newerr = err
        return Solution(newcorr, newerr, m, cond, undec)
# End OverlayNonDecisionLR

# Start DriftCoherence
from ddm import Drift
class DriftCoherence(Drift):
    name = "Drift depends linearly on coherence"
    required_parameters = ["driftcoh"] # <-- Parameters we want to include in the model
    required_conditions = ["coh"] # <-- Task parameters ("conditions"). Should be the same name as in the sample.
    
    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        return self.driftcoh * conditions['coh']
# End DriftCoherence

# Start DriftCoherenceRewBias
from ddm import Drift
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
from ddm import Drift
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
from ddm.models import Drift
class DriftSine(Drift):
    name = "Sine-wave drifts"
    required_conditions = ["frequency"]
    required_parameters = ["scale"]
    def get_drift(self, t, conditions, **kwargs):
        return np.sin(t*conditions["frequency"]*2*np.pi)*self.scale
# End DriftSine

# Start DriftPulse
from ddm.models import Drift
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
from ddm.models import Drift
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
from ddm.models import Drift
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
from ddm.models import Bound
class BoundCollapsingStep(Bound):
    name = "Step collapsing bounds"
    required_conditions = []
    required_parameters = ["B0", "stepheight", "steplength"]
    def get_bound(self, t, **kwargs):
        stepnum = t//self.steplength
        step = self.B0 - stepnum * self.stepheight
        return max(step, 0)
# End BoundCollapsingStep

from ddm.models import Bound
import numpy as np
# Start BoundCollapsingWeibull
import numpy as np
from ddm import Bound
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

# Start LossByMeans
import numpy as np
from ddm import LossFunction
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
            MSE += (sols[c].prob_correct() - s.prob_correct())**2
            if sols[c].prob_correct() > 0:
                MSE += (sols[c].mean_decision_time() - np.mean(list(s)))**2
        return MSE
# End LossByMeans

# Start BoundSpeedAcc
from ddm import Bound
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
from ddm import Drift
class DriftUrgencyGain(Drift):
    name = "drift rate with an urgency function"
    required_parameters = ["snr", "gain_start", "gain_slope"]
    def get_drift(self, t, **kwargs):
        return self.snr * urgency_gain(t, self.gain_start, self.gain_slope)
# End DriftUrgencyGain

# Start NoiseUrgencyGain
from ddm import Noise
class NoiseUrgencyGain(Noise):
    name = "noise level with an urgency function"
    required_parameters = ["gain_start", "gain_slope"]
    def get_noise(self, t, **kwargs):
        return urgency_gain(t, self.gain_start, self.gain_slope)
# End NoiseUrgencyGain



import ddm

class FakePointModel(ddm.Model):
    def solve(self, conditions={}, *args, **kwargs):
        corr = self.t_domain()*0
        corr[1] = .8
        err = self.t_domain()*0
        err[1] = .2
        return self._overlay.apply(ddm.Solution(corr, err, self, conditions))
FakePointModel.solve_analytical = FakePointModel.solve
FakePointModel.solve_numerical = FakePointModel.solve
FakePointModel.solve_numerical_cn = FakePointModel.solve
FakePointModel.solve_numerical_implicit = FakePointModel.solve
FakePointModel.solve_numerical_explicit = FakePointModel.solve

m = FakePointModel()
s = m.solve()

so = OverlayNonDecisionUniform(nondectime=-.05,st=.1).apply(s)
so2 = OverlayNonDecisionUniform2(nondectime=-.05,st=.1).apply(s)
assert np.all(so.corr == so2.corr)

so = OverlayNonDecisionGamma(shape=4, scale=.01, nondectime=.1).apply(s)
print(list(so.corr))
print(np.sum(so.corr))

#m = FakePointModel(overlay=OverlayNonDecisionGamma(shape=ddm.Fittable(minval=1, maxval=3), scale=ddm.Fittable(minval=0, maxval=.2), nondectime=ddm.Fittable(minval=0, maxval=1)))
m = FakePointModel(overlay=OverlayNonDecisionGaussian(ndsigma=ddm.Fittable(minval=0, maxval=1), nondectime=ddm.Fittable(minval=0, maxval=1)))
#m = FakePointModel(overlay=OverlayNonDecisionUniform(st=ddm.Fittable(minval=0, maxval=1), nondectime=ddm.Fittable(minval=0, maxval=1)))
import ddm.plot
ddm.plot.model_gui(m, ddm.Sample(np.asarray([1, 1.1, 1.2, 1.3, 1.4, 1.5]), np.asarray([1])))

from ddm import ICGaussian
r = ICPointRew(x0=.99)
#print(r.get_IC(np.linspace(-1, 1, 201), dx=.01, conditions={"highreward": 1}))

