# Model code for Shinn et al. (2020) - Confluence of timing and reward biases in perceptual decision-making dynamics
# Biorxiv link: https://www.biorxiv.org/content/10.1101/865501v1
# Copyright 2020 Max Shinn <maxwell.shinn@yale.edu>
# Available under the GPLv3
#
# These components are expected to be most useful on their own, but
# can be tested with the interactive demo at the end of this file.
# This file can either be directly imported to use the models within
# your own code, or else it can be run as a script from the command
# line for the interactive demo.

from paranoid.decorators import accepts, returns, requires, ensures, paranoidclass
from paranoid.types import RangeOpenClosed, RangeOpen, Range, Positive0, NDArray, ParametersDict, Natural0, Set, Self, Number, Positive
import ddm
import numpy as np
import scipy

# Note that BEGIN and END statements appear throughout this file, and
# are markers for keeping the file in sync with the documentation in
# the PyDDM Cookbook.

# BEGIN utility_functions
# Paranoid annotations for correctness
@accepts(Range(50, 100), RangeOpen(0, 10), RangeOpenClosed(50, 100))
@requires("coh <= max_coh")
@returns(Range(0, 1))
@ensures("return == 0 <--> coh == 50")
@ensures("return == 1 <--> coh == max_coh")
# Monotonic increasing in coh, decreasing in exponent
@ensures("coh >= coh` and exponent <= exponent` and max_coh == max_coh` --> return >= return`")
def coh_transform(coh, exponent, max_coh):
    """Transform coherence to range 0-1.

    `coh` should be in range 50-`max_coh`, and exponent greater than
    0.  Returns a number 0-1 via nonlinearity `exponent`.
    """
    coh_coef = (coh-50)/(max_coh-50)
    return coh_coef**exponent

@accepts(Positive0, Positive0, Positive0, Positive0)
@returns(Positive0)
@ensures("return >= 0")
# Monotonic in all variables, decreasing with t1, others increasing
@ensures("t >= t` and base >= base` and t1 <= t1` and slope >= slope` --> return >= return`")
def urgency(t, base, t1, slope):
    """Ramping urgency function.

    Evaluate the ramping urgency function at point `t`.  Returns
    ReLu(t-t1)*slope + base.
    """
    return base + ((t-t1)*slope if t>=t1 else 0)
# END utility_functions

# Paranoid Scientist annotations for task conditions
@paranoidclass
class Conditions(ParametersDict):
    """Paranoid Scientist definition of valid conditions for our task."""
    def __init__(self):
        super().__init__({"coherence": Range(50, 100),
                          "presample": Natural0,
                          "highreward": Set([0, 1]),
                          "blocktype": Set([1, 2])})
    def test(self, v):
        super().test(v)
        assert isinstance(v, dict), "Non-dict passed"
        assert not set(v.keys()) - {'coherence', 'presample', 'highreward', 'blocktype'}, \
            "Invalid reward keys"
    def generate(self):
        print("Generating")
        for ps,coh,hr,bt in zip([0, 400, 800, 1000], [50, 53, 57, 60, 70], [0, 1], [1, 2]):
            yield {"presample": ps, "coherence": coh, "highreward": hr, "blocktype": bt}
            

#################### Color match task with reward bias ####################
            
# BEGIN driftnoise
@paranoidclass
class DriftShinn2020(ddm.models.Drift):
    """General drift component which captures the models in Shinn et al (2020).

    Parameters are:

    - `snr` - signal noise ratio
    - `noise` - baseline
    - `t1` - ramp onset time of gain function for urgency increase
    - `t1slope` - slope of gain function for urgency increase
    - `cohexp` - exponent for nonlinear transform of coherence (fixed to 1 in the paper)
    - `maxcoh` - maximum coherence level for all trials
    - `leak` - leak constant
    - `leaktarget` - the position to which the leak leaks (e.g. 0 for leaky integration).  
          In the paper, this is set to the starting position of integration
    - `leaktargramp` - slope of linear increase over time of leaktarget parameter
    
    Requires data which has three conditions:

    - coherence - coherence level (range: from 50 to `maxcoh`)
    - presample - presample duration in ms
    - highreward - 1 if correct answer is large reward target, else 0
    """
    name = "Drift for Shinn et al (2020)."
    required_parameters = ["snr", "noise", "t1", "t1slope", "cohexp", "maxcoh", "leak",
                           "leaktarget", "leaktargramp"]
    required_conditions = ["coherence", "presample", "highreward"]
    default_parameters = {"leaktargramp": 0}
    def get_drift(self, t, x, conditions, **kwargs):
        # Coherence coefficient == coherence with a non-linear transform
        coh_coef = coh_transform(conditions["coherence"], self.cohexp, self.maxcoh)
        # Are we in sample or presample?
        is_past_delay = 1 if t > conditions["presample"]/1000 else 0
        # SNR times the gain
        cur_urgency = self.snr * urgency(t, self.noise, self.t1, self.t1slope)
        # The point to which we leak
        leaktarg = self.leaktarget if conditions["highreward"] else -self.leaktarget
        # Slope of leak target should depend on which side of the x axis we are on
        leaktargramp = self.leaktargramp if conditions["highreward"] else -self.leaktargramp
        return coh_coef * (cur_urgency * is_past_delay) - self.leak*(x-(leaktarg+leaktargramp*t))
    # The _test function is a paranoid scientist annotation to ensure
    # correctness.  It can be deleted if you don't want to use use
    # this library to ensure correctness.
    @staticmethod
    def _test(v):
        assert v.snr in Positive(), "Invalid SNR"
        assert v.noise in Positive0(), "Invalid noise"
        assert v.t1 in Positive0(), "Invalid t1"
        assert v.t1slope in Positive0(), "Invalid t1slope"
        assert v.cohexp in Positive0(), "Invalid cohexp"
        assert v.maxcoh in [63, 70], "Invalid maxcoh" # Max for monkey P was 63, for Q was 70
        assert v.leak in Positive0(), "Invalid leak"
        assert v.leaktarget in Number(), "Invalid leak"
        assert v.leaktargramp in Number(), "Invalid leak"

@paranoidclass
class NoiseShinn2020(ddm.models.Noise):
    """General drift component which captures the models in Shinn et al (2020).

    Takes parameters `noise`, `t1`, and `t1slope`.  See DriftShinn2020
    for parameter definitions.  This should not be used with any other
    drift other than DriftShinn2020.
    """
    name = "Noise for Shinn et al (2020)."
    # Same parameter meanings as in DriftShinn2020
    required_parameters = ["noise", "t1", "t1slope"]
    @accepts(Self, t=Positive0, conditions=Conditions)
    @returns(Positive)
    def get_noise(self, t, conditions, **kwargs):
        """Scale standard deviation by current urgency"""
        return urgency(t, self.noise, self.t1, self.t1slope) + .001
    # More Paranoid Scientist annotations
    @staticmethod
    def _test(v):
        assert v.noise in Positive0(), "Invalid noise"
        assert v.t1 in Positive0(), "Invalid t1"
        assert v.t1slope in Positive0(), "Invalid t1slope"
# END driftnoise

# BEGIN rewardtiming
@paranoidclass
class OverlayMappingError(ddm.models.Overlay):
    """Mapping error for rewards

    If crossing the small-reward boundary, with a certain probability,
    choose the large reward boundary instead.  `mappingcoef` is the
    probability of choosing large reward given you crossed the small
    reward boundary.
    """
    name = "Mapping error"
    required_parameters = ["mappingcoef"]
    required_conditions = ["highreward"]
    @accepts(Self, ddm.Solution)
    @returns(ddm.Solution)
    @ensures("return.conditions['highreward'] == 1 --> all(return.corr >= solution.corr)")
    @ensures("return.conditions['highreward'] == 0 --> all(return.corr <= solution.corr)")
    def apply(self, solution):
        # If large reward is correct, take some of the error
        # distribution and add it to the correct distribution.
        if solution.conditions['highreward'] == 1:
            mismapped = self.mappingcoef * solution.err
            err = solution.err - mismapped
            corr = solution.corr + mismapped
            return ddm.Solution(corr, err, solution.model, solution.conditions)
        # If small reward is correct, do the opposite
        else:
            mismapped = self.mappingcoef * solution.corr
            corr = solution.corr - mismapped
            err = solution.err + mismapped
            return ddm.Solution(corr, err, solution.model, solution.conditions)
    @staticmethod
    def _test(v):
        assert v.mappingcoef in Range(0, 1), "Invalid mapping coefficient"


@paranoidclass
class BoundCollapsingExponentialDelay(ddm.models.Bound):
    """Bound dependence: bound collapses exponentially over time.

    Takes three parameters: 

    `B` - the bound at time t = 0.
    `tau` - the time constant for the collapse, should be greater than
    zero.
    `t1` - the time at which the collapse begins in seconds
    """
    name = "collapsing_exponential"
    required_parameters = ["B", "tau", "t1"]

    @accepts(Self, Positive0, Conditions)
    @returns(Positive)
    @ensures("t > self.t1 --> return < self.B")
    @ensures("t <= self.t1 --> return == self.B")
    @ensures("self == self` and t > self.t1 and t > t` --> return < return`")
    def get_bound(self, t, conditions, **kwargs):
        if t <= self.t1:
            return self.B
        if t > self.t1:
            return self.B * np.exp(-self.tau*(t-self.t1))

    @staticmethod
    def _test(v):
        assert v.B in Positive(), "Invalid bound"
        assert v.tau in Positive(), "Invalid collapsing time constant"
        assert v.t1 in Positive0(), "Invalid t1"

# Was not used in the paper, only used in a reviewer response
@paranoidclass
class BoundCollapsingLinearDelay(ddm.models.Bound):
    """Bound dependence: bound collapses exponentially over time.

    Takes three parameters: 

    `B` - the bound at time t = 0.
    `tau` - the time constant for the collapse, should be greater than
    zero.
    `t1` - the time at which the collapse begins in seconds
    """
    name = "collapsing_exponential"
    required_parameters = ["B", "tau", "t1"]
    def get_bound(self, t, conditions, **kwargs):
        if t <= self.t1:
            return self.B
        if t > self.t1:
            return np.minimum(self.B, np.maximum(.01, self.B - self.tau*(t-self.t1)))

@paranoidclass
class ICPoint(ddm.models.InitialCondition):
    """Initial condition: a point mass at the point `x0`."""
    name = "point_source"
    required_parameters = ["x0"]
    required_conditions = ["highreward"]
    @accepts(Self, NDArray(d=1, t=Number), RangeOpen(0, 1), Conditions)
    @returns(NDArray(d=1))
    @requires("x.size > 1")
    @requires("all(x[1:]-x[0:-1] - dx < 1e-8)")
    @requires("x[0] < self.x0 < x[-1]")
    @ensures("sum(return) == max(return)")
    @ensures("all((e in [0, 1] for e in return))")
    @ensures("self.x0 == 0 --> list(reversed(return)) == list(return)")
    @ensures("x.shape == return.shape")
    def get_IC(self, x, dx, conditions={}):
        start = np.round(self.x0/dx)
        # We want x0 positive -> bias towards the large reward target.
        # Is the large reward target is the incorrect choice, flip the
        # sign.
        if not conditions['highreward']:
            start = -start
        shift_i = int(start + (len(x)-1)/2)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=self.x0.
        return pdf
    
    @staticmethod
    def _test(v):
        assert v.x0 in Number(), "Invalid starting position"


@paranoidclass
class OverlayExponentialRewardMixture(ddm.Overlay):
    """An exponential mixture distribution with rate which depends on reward.

    The output distribution should be pmixturecoef*100 percent exponential
    distribution and (1-umixturecoef)*100 percent of the distribution
    to which this overlay is applied.

    A mixture with the exponential distribution can be used to confer
    robustness when fitting using likelihood.  This results from a
    Poisson process, i.e. modeling a uniform lapse rate and hence has
    a flat hazard function.

    `ratehr` is the reate for the high reward choice, and `ratelr` is
    for the low reward choice.

    Example usage:

      | overlay = OverlayExponentialRewardMixture(pmixturecoef=.02, ratelr=1, ratehr=2)
    """
    name = "Exponential distribution mixture model (lapse rate)"
    required_parameters = ["pmixturecoef", "ratehr", "ratelr"]
    required_conditions = ["highreward"]
    @accepts(Self, ddm.Solution)
    @returns(ddm.Solution)
    def apply(self, solution):
        assert self.pmixturecoef >= 0 and self.pmixturecoef <= 1
        corr, err, m, cond = solution.corr, solution.err, solution.model, solution.conditions
        # These aren't real pdfs since they don't sum to 1, they sum
        # to 1/self.model.dt.  We can't just sum the correct and error
        # distributions to find this number because that would exclude
        # the undecided trials.
        pdfsum = 1/m.dt
        # This can be derived by hand pretty easily.
        lapses_hr = lambda t : self.ratehr*np.exp(-(self.ratehr+self.ratelr)*t) if t != 0 else 0
        lapses_lr = lambda t : self.ratelr*np.exp(-(self.ratehr+self.ratelr)*t) if t != 0 else 0
        X = [i*m.dt for i in range(0, len(corr))]
        # Bias should be based on reward, not correct vs error
        if cond['highreward'] == 1:
            Y_corr = np.asarray(list(map(lapses_hr, X)))*m.dt
            Y_err = np.asarray(list(map(lapses_lr, X)))*m.dt
        else:
            Y_corr = np.asarray(list(map(lapses_lr, X)))*m.dt
            Y_err = np.asarray(list(map(lapses_hr, X)))*m.dt
        corr = corr*(1-self.pmixturecoef) + self.pmixturecoef*Y_corr # Assume ndarrays, not lists
        err = err*(1-self.pmixturecoef) + self.pmixturecoef*Y_err
        return ddm.Solution(corr, err, m, cond)
    @staticmethod
    def _test(v):
        assert v.pmixturecoef in Range(0, 1), "Invalid pmixture coef"
        assert v.ratehr in Positive(), "Invalid rate"
        assert v.ratelr in Positive(), "Invalid rate"
# END rewardtiming


#################### Example usage ####################

import ddm.plot
from ddm import Fittable, OverlayNonDecision, OverlayChain, Model, BoundConstant


# Try both the delayed gain function version and the delayed
# collapsing bounds versions of the model.
if __name__ == "__main__":
    for URGENCY in ["collapse", "gain"]:
        # BEGIN demo
        # Define params separately from the model mechanisms, since some params will be shared.
        maxcoh = 70
        snr = Fittable(minval=0, maxval=40)
        noise = Fittable(minval=.01, maxval=4)
        leak = Fittable(minval=0, maxval=40)
        cohexp = 1
        if URGENCY == "collapse":
            t1 = 0
            t1slope = 0
            t1bound = Fittable(minval=0, maxval=1)
            tau = Fittable(minval=.1, maxval=10)
        elif URGENCY == "gain":
            t1 = Fittable(minval=0, maxval=1)
            t1slope = Fittable(minval=0, maxval=10)
        
        x0 = Fittable(minval=0, maxval=.9)
        leaktargramp = Fittable(minval=0, maxval=.9, default=0)
        
        pmixturecoef = Fittable(minval=0.001, maxval=0.1)
        ratelr = Fittable(minval=0.1, maxval=2)
        ratehr = Fittable(minval=0.1, maxval=2)
        
        mappingcoef = Fittable(minval=0, maxval=.4)
        
        nondectime = Fittable(minval=.1, maxval=.3)
    
        # Create the model components
        model_drift = DriftShinn2020(snr=snr, noise=noise,
                                     leak=leak, leaktarget=x0, leaktargramp=leaktargramp,
                                     t1=t1, t1slope=t1slope,
                                     cohexp=cohexp, maxcoh=maxcoh)
        
        model_noise = NoiseShinn2020(t1=t1, t1slope=t1slope, noise=noise)
        
        if URGENCY == "collapse":
            model_bounds = BoundCollapsingExponentialDelay(B=1, tau=tau, t1=t1bound)
        elif URGENCY == "gain":
            model_bounds = BoundConstant(B=1)
        
        model_ic = ICPoint(x0=x0)
        
        _overlay_map = OverlayMappingError(mappingcoef=mappingcoef)
        _overlay_nondecision = OverlayNonDecision(nondectime=nondectime)
        
        _overlay_mix = OverlayExponentialRewardMixture(pmixturecoef=pmixturecoef,
                                                       ratehr=ratehr, ratelr=ratelr)
        
        model_overlay = OverlayChain(overlays=[_overlay_nondecision, _overlay_map, _overlay_mix])
        
        modelparams = {"dx" : 0.005, "dt": 0.002, "T_dur" : 3.0}
        
        modelname = "Shinn et al (2020) model (%s)" % ("gain function" if URGENCY == "gain"
                                                       else "collapsing bounds")
        model = Model(name=modelname, **modelparams,
                      drift=model_drift, noise=model_noise,
                      bound=model_bounds, IC=model_ic,
                      overlay=model_overlay)
        
        ddm.plot.model_gui(model, conditions={"coherence": [50, 53, 60, 70],
                                              "presample": [0, 400, 800],
                                              "highreward": [0, 1]})
        # END demo
