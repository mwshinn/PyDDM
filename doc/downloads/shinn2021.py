# Model code for Shinn et al. (2021) - Transient neuronal suppression for exploitation of new sensory evidence
# Biorxiv link: https://www.biorxiv.org/content/10.1101/2020.11.29.403089v1
# Copyright 2021 Max Shinn <maxwell.shinn@yale.edu>
# Available under the GPLv3
#
# These components are expected to be most useful on their own, but
# can be tested with the interactive demo at the end of this file.
# This file can either be directly imported to use the models within
# your own code, or else it can be run as a script from the command
# line for the interactive demo.

# BEGIN imports
import pyddm as ddm
import pyddm.plot
import numpy as np
import scipy.stats
# END imports

# BEGIN functions
def coh_transform(coh, max_coh):
    """Convert coherence from 0-100 to -1-1"""
    return (coh-50)/(max_coh-50)

def urgency(t, base, t1, slope):
    """Urgency signal"""
    return base + ((t-t1)*slope if t>=t1 else 0)

def get_detect_prob(coh, param):
    """Probability of detection given coherence"""
    return 2/(1+np.exp(-param*(coh-50)/50))-1
# END functions

# BEGIN drift
class DriftDip(ddm.models.Drift):
    name = "Piecewise urgency signal, reward bias, and coherence change transient"
    required_parameters = ["snr", "noise", "t1", "t1slope", "maxcoh", "leak", "leaktarget",
                           "leaktargramp", "dipstart", "dipstop", "diptype", "dipparam"]
    required_conditions = ["coherence", "presample", "highreward"]
    default_parameters = {"leaktargramp": 0, "dipparam": 0, "diptype": -1}

    def get_drift(self, t, x, conditions, **kwargs):
        dipstart = min(self.dipstart, self.dipstop) + conditions["presample"]/1000
        dipstop = max(self.dipstart, self.dipstop) + conditions["presample"]/1000
        if self.diptype == 1 and dipstart < t and t < dipstop:
            return 0
        # Coherence coefficient == coherence with a non-linear transform
        coh_coef = coh_transform(conditions["coherence"], self.maxcoh)
        is_past_delay = 1 if t > conditions["presample"]/1000 else 0
        cur_urgency = self.snr * urgency(t, self.noise, self.t1, self.t1slope)
        leaktarg = self.leaktarget if conditions["highreward"] else -self.leaktarget
        leak = self.leak
        leaktargramp = self.leaktargramp if conditions["highreward"] else -self.leaktargramp
        if self.diptype == 2 and dipstart < t and t < dipstop:
            leak += self.dipparam
            leaktarg = 0
            leaktargramp = 0
        return coh_coef * (cur_urgency * is_past_delay) - leak*(x-(leaktarg+leaktargramp*(t)))
# END drift

# BEGIN noise
class NoiseDip(ddm.models.Noise):
    name = "Noise with piecewise linear urgency signal"
    required_parameters = ["noise", "t1", "t1slope", "dipstart", "dipstop", "diptype"]

    def get_noise(self, t, conditions, **kwargs):
        dipstart = min(self.dipstart, self.dipstop) + conditions["presample"]/1000
        dipstop = max(self.dipstart, self.dipstop) + conditions["presample"]/1000
        if self.diptype == 1 and dipstart < t and t < dipstop:
            return 0.001 # Not 0 to avoid numerical problems
        return urgency(t, self.noise, self.t1, self.t1slope) + .001
# END noise

# BEGIN ic
class ICPoint(ddm.models.InitialCondition):
    """Initial condition: a dirac delta function in the center of the domain."""
    name = "point_source"
    required_parameters = ["x0"]
    required_conditions = ["highreward"]
    def get_IC(self, x, dx, conditions={}):
        start = np.round(self.x0/dx)
        if not conditions['highreward']:
            start = -start
        shift_i = int(start + (len(x)-1)/2)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=self.x0.
        return pdf
# END ic


# BEGIN bound
class BoundDip(ddm.Bound):
    name = "Increasing bound for motor suppression"
    required_parameters = ["B", "dipstart", "dipstop", "diptype"]
    required_conditions = ["presample"]
    def get_bound(self, t, conditions, *args, **kwargs):
        dipstart = min(self.dipstart, self.dipstop) + conditions["presample"]/1000
        dipstop = max(self.dipstart, self.dipstop) + conditions["presample"]/1000
        if self.diptype == 3 and dipstart < t and t < dipstop:
            return self.B + 4*scipy.stats.beta.pdf(t, a=3, b=3, loc=dipstart, scale=dipstop-dipstart)
        return self.B
# END bound


# BEGIN overlay
class OverlayDipRatio(ddm.Overlay):
    name = "Probability of detecting the change"
    required_parameters = ["detect", "diptype"]
    required_conditions = ["coherence"]
    def apply(self, solution):
        if self.diptype not in [1, 2, 3]:
            return solution
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution
        diptype = m.get_dependence("drift").diptype
        def set_dip_type(m, diptype):
            m.get_dependence("drift").diptype = diptype
            m.get_dependence("noise").diptype = diptype
            m.get_dependence("bound").diptype = diptype
            m.get_dependence("overlay").diptype = diptype
        set_dip_type(m, -1)
        ratio = get_detect_prob(cond['coherence'], self.detect)
        s = m.solve_numerical_implicit(conditions=cond, return_evolution=True)
        newcorr = corr * ratio + s.corr * (1-ratio)
        newerr = err * ratio + s.err * (1-ratio)
        newevo = evolution
        #newevo = evolution * ratio + s.evolution * (1-ratio)
        set_dip_type(m, diptype)
        return ddm.Solution(newcorr, newerr, m, cond, undec, newevo)
    def apply_trajectory(self, trajectory, model, rk4, seed, conditions={}):
        if self.diptype not in [1, 2, 3]:
            return trajectory
        prob = get_detect_prob(conditions['coherence'], self.detect)
        # We have a `prob` probability of detecting the dip.  If we
        # detected the dip, just use the given trajectory.  Otherwise,
        # simulate a new trajectory without the dip.
        if prob > np.random.rand():
            return trajectory
        diptype = model.get_dependence("drift").diptype
        def set_dip_type(m, diptype):
            m.get_dependence("drift").diptype = diptype
            m.get_dependence("noise").diptype = diptype
            m.get_dependence("bound").diptype = diptype
            m.get_dependence("overlay").diptype = diptype
        set_dip_type(model, -1)
        traj = model.simulate_trial(conditions=conditions, rk4=rk4, seed=seed, cutoff=True)
        set_dip_type(model, diptype)
        return traj
# END overlay

#################### Example usage ####################

if __name__ == "__main__":
    # BEGIN demo
    DIPTYPE = 1 # Change to 1, 2, or 3 depending on which model you want
    snr = ddm.Fittable(minval=0.5, maxval=20, default=9.243318909157688)
    leak = ddm.Fittable(minval=-10, maxval=30, default=9.46411355874963)
    x0 = ddm.Fittable(minval=-.5, maxval=.5, default=0.1294632585920082)
    leaktargramp = ddm.Fittable(minval=0, maxval=3, default=0)
    noise = ddm.Fittable(minval=.2, maxval=2, default=1.1520906498077081)
    t1 = ddm.Fittable(minval=0, maxval=1, default=0.34905555600815663)
    t1slope = ddm.Fittable(minval=0, maxval=3, default=1.9643425020687162)

    dipstart = ddm.Fittable(minval=-.4, maxval=0, default=-.2)
    dipstop = ddm.Fittable(minval=0, maxval=.5, default=.05)
    nondectime = ddm.Fittable(minval=0, maxval=.3, default=.1)
    detect = ddm.Fittable(minval=2, maxval=50, default=10)
    diptype = DIPTYPE
    dipparam = ddm.Fittable(minval=0, maxval=50) if diptype == 2 else 0
    pmixturecoef = ddm.Fittable(minval=0, maxval=.2, default=.03)
    rate = ddm.Fittable(minval=.1, maxval=10, default=1)
    m = ddm.Model(drift = DriftDip(snr=snr,
                                   noise=noise,
                                   t1=t1,
                                   t1slope=t1slope,
                                   leak=leak,
                                   maxcoh=70,
                                   leaktarget=x0,
                                   leaktargramp=leaktargramp,
                                   dipstart=dipstart,
                                   dipstop=dipstop,
                                   diptype=diptype,
                                   dipparam=dipparam,
                                   ),
                  noise = NoiseDip(noise=noise,
                                   t1=t1,
                                   t1slope=t1slope,
                                   dipstart=dipstart,
                                   dipstop=dipstop,
                                   diptype=diptype,
                                   ),
                  IC =     ICPoint(x0=x0),
                  bound = BoundDip(B=1,
                                   dipstart=dipstart,
                                   dipstop=dipstop,
                                   diptype=diptype
                                   ),
                  overlay=ddm.OverlayChain(overlays=[
                                    ddm.OverlayNonDecision(nondectime=nondectime),
                                           OverlayDipRatio(detect=detect,
                                                           diptype=diptype),
                                 ddm.OverlayPoissonMixture(pmixturecoef=pmixturecoef,
                                                           rate=rate)
                                            ]),
                  dx=0.002, dt=0.002, T_dur=3.0)
    # END demo
    pyddm.plot.model_gui(model=m, conditions={"coherence": [50, 53, 60, 70], "presample": [0, 400, 800], "highreward": [0, 1]})
