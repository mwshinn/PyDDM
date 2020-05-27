# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

__all__ = ["Overlay", "OverlayNone", "OverlayChain", "OverlayUniformMixture", "OverlayPoissonMixture", "OverlayExponentialMixture", "OverlayNonDecision", "OverlayNonDecisionGamma", "OverlayNonDecisionUniform", "OverlaySimplePause", "OverlayBlurredPause"]

import numpy as np
from scipy.special import gamma as sp_gamma
import scipy.stats

from paranoid import accepts, returns, requires, ensures, Self, paranoidclass, paranoidconfig, Range, Positive, Number, List, Positive0, NDArray, Unchecked
from .paranoid_types import Conditions

from .base import Dependence
from ..solution import Solution

# TODO in unit test, ensure all overlays with reasonable parameters
# applied to a Solution add up to 1

# TODO unit tests for apply_trajectory

class Overlay(Dependence):
    """Subclasses can modify distributions after they have been generated.

    This abstract class provides the methods which define how a
    distribution should be modified after solving the model, for
    example for a mixture model.  To subclass it, implement apply.

    Also, since it inherits from Dependence, subclasses must also
    assign a `name` and `required_parameters` (see documentation for
    Dependence.)
    """
    depname = "Overlay"
    def apply(self, solution):
        """Apply the overlay to a Solution object.

        This function must be redefined in subclasses.

        This function takes a Solution object as its argument and
        returns a Solution object which was modified in some way.
        Often times, this will be by modifying `solution.corr` and
        `solution.err`.  See the documentation for Solution for more
        information about this object.

        Note that while this does not take `conditions` as an
        argument, conditions may still be accessed via
        `solution.conditions`.

        Conceptually, this function performs some transformation on
        the simulated response time (first passage time)
        distributions.  It is especially useful for non-decision times
        and mixture models, potentially in a parameter-dependent or
        condition-dependent manner.
        """
        raise NotImplementedError("Overlay model %s invalid: must define the apply(self, solution) function" % self.__class__.__name__)
    def apply_trajectory(self, trajectory, model, rk4, seed, conditions={}):
        """Apply the overlay to a simulated decision variable trajectory.

        This function is optional and may be redefined in subclasses.
        It is expected to implement the same mechanism as the method
        "apply", but to do so on simulated trajectories (i.e. from
        Model.simulate_trial) instead of on a Solution object.

        This function takes the t domain, the trajectory itself, and
        task conditions.  It returns the modified trajectory.
        """
        raise NotImplementedError("Overlay model %s not compatible with trajectory simulations" % self.__class__.__name__)

@paranoidclass
class OverlayNone(Overlay):
    """No overlay.  An identity function for Solutions.

    Example usage:

      | overlay = OverlayNone()
    """
    name = "No overlay"
    required_parameters = []
    @staticmethod
    def _test(v):
        pass
    @staticmethod
    def _generate():
        yield OverlayNone()
    @accepts(Self, Solution)
    @returns(Solution)
    def apply(self, solution):
        return solution
    @accepts(Self, NDArray(d=1, t=Number))
    @returns(NDArray(d=1, t=Number))
    def apply_trajectory(self, trajectory, **kwargs):
        return trajectory

# NOTE: This class is likely to break if any changes are made to the
# Dependence constructor.  In theory, no changes should be made to the
# Dependence constructor, but just in case...
@paranoidclass
class OverlayChain(Overlay):
    """Join together multiple overlays.

    Unlike other model components, Overlays are not mutually
    exclusive.  It is possible to transform the output solution many
    times.  Thus, this allows joining together multiple Overlay
    objects into a single object.

    It accepts one parameter: `overlays`.  This should be a list of
    Overlay objects, in the order which they should be applied to the
    Solution object.
    
    One key technical caveat is that the overlays which are chained
    together may not have the same parameter names.  Parameter names
    must be given different names in order to be a part of the same
    overlay.  This allows those parameters to be accessed by their
    name inside of an OverlayChain object.

    Example usage:

      | overlay = OverlayChain(overlays=[OverlayNone(), OverlayNone(), OverlayNone()]) # Still equivalent to OverlayNone
      | overlay = OverlayChain(overlays=[OverlayPoissonMixture(pmixturecoef=.01, rate=1), 
      |                                  OverlayUniformMixture(umixturecoef=.01)]) # Apply a Poission mixture and then a Uniform mixture
    """
    name = "Chain overlay"
    required_parameters = ["overlays"]
    @staticmethod
    def _test(v):
        assert v.overlays in List(Overlay), "overlays must be a list of Overlay objects"
    @staticmethod
    def _generate():
        yield OverlayChain(overlays=[OverlayNone()])
        yield OverlayChain(overlays=[OverlayUniformMixture(umixturecoef=.3), OverlayPoissonMixture(pmixturecoef=.2, rate=.7)])
        yield OverlayChain(overlays=[OverlayNonDecision(nondectime=.1), OverlayPoissonMixture(pmixturecoef=.1, rate=1), OverlayUniformMixture(umixturecoef=.1)])
    def __init__(self, **kwargs):
        Overlay.__init__(self, **kwargs)
        object.__setattr__(self, "required_parameters", [])
        object.__setattr__(self, "required_conditions", [])
        for o in self.overlays:
            self.required_parameters.extend(o.required_parameters)
            self.required_conditions.extend(o.required_conditions)
        assert len(self.required_parameters) == len(set(self.required_parameters)), "Two overlays in chain cannot have the same parameter names"
        object.__setattr__(self, "required_conditions", list(set(self.required_conditions))) # Avoid duplicates
    def __setattr__(self, name, value):
        if "required_parameters" in self.__dict__:
            if name in self.required_parameters:
                for o in self.overlays:
                    if name in o.required_parameters:
                        return setattr(o, name, value)
        return Overlay.__setattr__(self, name, value)
    def __getattr__(self, name):
        if name in self.required_parameters:
            for o in self.overlays:
                if name in o.required_parameters:
                    return getattr(o, name)
        else:
            return Overlay.__getattribute__(self, name)
    def __repr__(self):
        overlayreprs = list(map(repr, self.overlays))
        return "OverlayChain(overlays=[" + ", ".join(overlayreprs) + "])"
    @accepts(Self, Solution)
    @returns(Solution)
    def apply(self, solution):
        assert isinstance(solution, Solution)
        newsol = solution
        for o in self.overlays:
            newsol = o.apply(newsol)
        return newsol
    @accepts(Self, NDArray(d=1, t=Number))
    @returns(NDArray(d=1, t=Number))
    @paranoidconfig(unit_test=False)
    def apply_trajectory(self, trajectory, **kwargs):
        for o in self.overlays:
            trajectory = o.apply_trajectory(trajectory=trajectory, **kwargs)
        return trajectory

@paranoidclass
class OverlayUniformMixture(Overlay):
    """A uniform mixture distribution.

    The output distribution should be umixturecoef*100 percent uniform
    distribution and (1-umixturecoef)*100 percent of the distribution
    to which this overlay is applied.

    A mixture with the uniform distribution can be used to confer
    robustness when fitting using likelihood.

    Example usage:

      | overlay = OverlayUniformMixture(umixturecoef=.01)
    """
    name = "Uniform distribution mixture model"
    required_parameters = ["umixturecoef"]
    @staticmethod
    def _test(v):
        assert v.umixturecoef in Range(0, 1), "Invalid mixture coef"
    @staticmethod
    def _generate():
        yield OverlayUniformMixture(umixturecoef=0)
        yield OverlayUniformMixture(umixturecoef=1)
        yield OverlayUniformMixture(umixturecoef=.02)
        yield OverlayUniformMixture(umixturecoef=.5)
    @accepts(Self, Solution)
    @returns(Solution)
    def apply(self, solution):
        assert self.umixturecoef >= 0 and self.umixturecoef <= 1
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution
        # To make this work with undecided probability, we need to
        # normalize by the sum of the decided density.  That way, this
        # function will never touch the undecided pieces.
        norm = np.sum(corr)+np.sum(err)
        corr = corr*(1-self.umixturecoef) + .5*self.umixturecoef/len(m.t_domain())*norm
        err = err*(1-self.umixturecoef) + .5*self.umixturecoef/len(m.t_domain())*norm
        return Solution(corr, err, m, cond, undec, evolution)

@paranoidclass
class OverlayExponentialMixture(Overlay):
    """An exponential mixture distribution.

    The output distribution should be pmixturecoef*100 percent exponential
    distribution and (1-umixturecoef)*100 percent of the distribution
    to which this overlay is applied.

    A mixture with the exponential distribution can be used to confer
    robustness when fitting using likelihood.

    Note that this is called OverlayPoissonMixture and not
    OverlayExponentialMixture because the exponential distribution is
    formed from a Poisson process, i.e. modeling a uniform lapse rate.

    Example usage:

      | overlay = OverlayPoissonMixture(pmixturecoef=.02, rate=1)
    """
    name = "Poisson distribution mixture model (lapse rate)"
    required_parameters = ["pmixturecoef", "rate"]
    @staticmethod
    def _test(v):
        assert v.pmixturecoef in Range(0, 1), "Invalid mixture coef"
        assert v.rate in Positive(), "Invalid rate"
    @staticmethod
    def _generate():
        yield OverlayPoissonMixture(pmixturecoef=0, rate=1)
        yield OverlayPoissonMixture(pmixturecoef=.5, rate=.1)
        yield OverlayPoissonMixture(pmixturecoef=.02, rate=10)
        yield OverlayPoissonMixture(pmixturecoef=1, rate=1)
    @accepts(Self, Solution)
    @returns(Solution)
    def apply(self, solution):
        assert self.pmixturecoef >= 0 and self.pmixturecoef <= 1
        assert isinstance(solution, Solution)
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution
        # To make this work with undecided probability, we need to
        # normalize by the sum of the decided density.  That way, this
        # function will never touch the undecided pieces.
        norm = np.sum(corr)+np.sum(err)
        lapses = lambda t : 2*self.rate*np.exp(-1*self.rate*t)
        X = m.dt * np.arange(0, len(corr))
        Y = lapses(X)
        Y /= np.sum(Y)
        corr = corr*(1-self.pmixturecoef) + .5*self.pmixturecoef*Y*norm # Assume numpy ndarrays, not lists
        err = err*(1-self.pmixturecoef) + .5*self.pmixturecoef*Y*norm
        #print(corr)
        #print(err)
        return Solution(corr, err, m, cond, undec, evolution)

# Backward compatibility
class OverlayPoissonMixture(OverlayExponentialMixture):
    pass

@paranoidclass
class OverlayNonDecision(Overlay):
    """Add a non-decision time

    This shifts the reaction time distribution by `nondectime` seconds
    in order to create a non-decision time.

    Example usage:

      | overlay = OverlayNonDecision(nondectime=.2)
    """
    name = "Add a non-decision by shifting the histogram"
    required_parameters = ["nondectime"]
    @staticmethod
    def _test(v):
        assert v.nondectime in Number(), "Invalid non-decision time"
    @staticmethod
    def _generate():
        yield OverlayNonDecision(nondectime=0)
        yield OverlayNonDecision(nondectime=.5)
        yield OverlayNonDecision(nondectime=-.5)
    @accepts(Self, Solution)
    @returns(Solution)
    @ensures("set(return.corr.tolist()) - set(solution.corr.tolist()).union({0.0}) == set()")
    @ensures("set(return.err.tolist()) - set(solution.err.tolist()).union({0.0}) == set()")
    @ensures("solution.prob_undecided() <= return.prob_undecided()")
    def apply(self, solution):
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution
        shifts = int(self.nondectime/m.dt) # truncate
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
        return Solution(newcorr, newerr, m, cond, undec, evolution)
    @accepts(Self, NDArray(d=1, t=Number), Unchecked)
    @returns(NDArray(d=1, t=Number))
    def apply_trajectory(self, trajectory, model, **kwargs):
        shift = int(self.nondectime/model.dt)
        if shift > 0:
            trajectory = np.append([trajectory[0]]*shift, trajectory)
        elif shift < 0:
            if len(trajectory) > abs(shift):
                trajectory = trajectory[abs(shift):]
            else:
                trajectory = np.asarray([trajectory[-1]])
        return trajectory

@paranoidclass
class OverlayNonDecisionUniform(Overlay):
    """Add a uniformly-distributed non-decision time.

    The center of the distribution of non-decision times is at
    `nondectime`, and it extends `halfwidth` on each side.

    Example usage:

      | overlay = OverlayNonDecisionUniform(nondectime=.2, halfwidth=.02)

    """
    name = "Uniformly-distributed non-decision time"
    required_parameters = ["nondectime", "halfwidth"]
    @staticmethod
    def _test(v):
        assert v.nondectime in Number(), "Invalid non-decision time"
        assert v.halfwidth in Positive0(), "Invalid halfwidth parameter"
    @staticmethod
    def _generate():
        yield OverlayNonDecisionUniform(nondectime=.3, halfwidth=.01)
        yield OverlayNonDecisionUniform(nondectime=0, halfwidth=.1)
    @accepts(Self, Solution)
    @returns(Solution)
    @ensures("np.sum(return.corr) <= np.sum(solution.corr)")
    @ensures("np.sum(return.err) <= np.sum(solution.err)")
    def apply(self, solution):
        # Make sure params are within range
        assert self.halfwidth >= 0, "Invalid st parameter"
        # Extract components of the solution object for convenience
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution
        # Describe the width and shift of the uniform distribution in
        # terms of list indices
        shift = int(self.nondectime/m.dt) # Discretized non-decision time
        width = int(self.halfwidth/m.dt) # Discretized uniform distribution half-width
        offsets = list(range(shift-width, shift+width+1))
        # Create new correct and error distributions and iteratively
        # add shifts of each distribution to them.  Use this over the
        # np.convolution because it handles negative non-decision
        # times.
        newcorr = np.zeros(corr.shape, dtype=corr.dtype)
        newerr = np.zeros(err.shape, dtype=err.dtype)
        for offset in offsets:
            if offset > 0:
                newcorr[offset:] += corr[:-offset]/len(offsets)
                newerr[offset:] += err[:-offset]/len(offsets)
            elif offset < 0:
                newcorr[:offset] += corr[-offset:]/len(offsets)
                newerr[:offset] += err[-offset:]/len(offsets)
            else:
                newcorr += corr/len(offsets)
                newerr += err/len(offsets)
        return Solution(newcorr, newerr, m, cond, undec, evolution)
    @accepts(Self, NDArray(d=1, t=Number), Unchecked)
    @returns(NDArray(d=1, t=Number))
    def apply_trajectory(self, trajectory, model, **kwargs):
        ndtime = np.random.rand()*2*self.halfwidth + (self.nondectime-self.halfwidth)
        shift = int(ndtime/model.dt)
        if shift > 0:
            np.append([trajectory[0]]*shift, trajectory)
        elif shift < 0:
            if len(trajectory) > abs(shift):
                trajectory = trajectory[abs(shift):]
            else:
                trajectory = np.asarray([trajectory[-1]])
        return trajectory


@paranoidclass
class OverlayNonDecisionGamma(Overlay):
    """Add a gamma-distributed non-decision time

    This shifts the reaction time distribution by an amount of time
    specified by the gamma distribution with shape parameter `shape`
    (sometimes called "k") and scale parameter `scale` (sometimes
    called "theta").  The distribution is then further shifted by
    `nondectime` seconds.

    Example usage:

      | overlay = OverlayNonDecisionGamma(nondectime=.2, shape=1.5, scale=.05)

    """
    name = "Add a gamma-distributed non-decision time"
    required_parameters = ["nondectime", "shape", "scale"]
    @staticmethod
    def _test(v):
        assert v.nondectime in Number(), "Invalid non-decision time"
        assert v.shape in Positive0(), "Invalid shape parameter"
        assert v.shape >= 1, "Shape parameter must be >= 1"
        assert v.scale in Positive(), "Invalid scale parameter"
    @staticmethod
    def _generate():
        yield OverlayNonDecisionGamma(nondectime=.3, shape=2, scale=.01)
        yield OverlayNonDecisionGamma(nondectime=0, shape=1.1, scale=.1)
    @accepts(Self, Solution)
    @returns(Solution)
    @ensures("np.sum(return.corr) <= np.sum(solution.corr)")
    @ensures("np.sum(return.err) <= np.sum(solution.err)")
    @ensures("np.all(return.corr[0:int(self.nondectime//return.model.dt)] == 0)")
    def apply(self, solution):
        # Make sure params are within range
        assert self.shape >= 1, "Invalid shape parameter"
        assert self.scale > 0, "Invalid scale parameter"
        # Extract components of the solution object for convenience
        corr = solution.corr
        err = solution.err
        dt = solution.model.dt
        # Create the weights for different timepoints
        times = np.asarray(list(range(-len(corr), len(corr))))*dt
        weights = scipy.stats.gamma(a=self.shape, scale=self.scale, loc=self.nondectime).pdf(times)
        if np.sum(weights) > 0:
            weights /= np.sum(weights) # Ensure it integrates to 1
        newcorr = np.convolve(corr, weights, mode="full")[len(corr):(2*len(corr))]
        newerr = np.convolve(err, weights, mode="full")[len(corr):(2*len(corr))]
        return Solution(newcorr, newerr, solution.model,
                        solution.conditions, solution.undec, solution.evolution)
    @accepts(Self, NDArray(d=1, t=Number), Unchecked)
    @returns(NDArray(d=1, t=Number))
    def apply_trajectory(self, trajectory, model, **kwargs):
        ndtime = scipy.stats.gamma(a=self.shape, scale=self.scale, loc=self.nondectime).rvs()
        shift = int(ndtime/model.dt)
        if shift > 0:
            np.append([trajectory[0]]*shift, trajectory)
        elif shift < 0:
            if len(trajectory) > abs(shift):
                trajectory = trajectory[abs(shift):]
            else:
                trajectory = np.asarray([trajectory[-1]])
        return trajectory

@paranoidclass
class OverlaySimplePause(Overlay):
    name = "Brief pause in integration by shifting part of the histogram"
    required_parameters = ['pausestart', 'pausestop']
    @staticmethod
    def _test(v):
        assert v.pausestart in Positive0(), "Invalid start time"
        assert v.pausestop in Positive0(), "Invalid non-decision time"
        assert v.pausestart <= v.pausestop, "Pause start time must be before stop time"
    @staticmethod
    def _generate():
        yield OverlaySimplePause(pausestart=0, pausestop=0)
        yield OverlaySimplePause(pausestart=.1, pausestop=.2)
    @accepts(Self, Solution)
    @returns(Solution)
    @ensures("set(return.corr.tolist()) - set(solution.corr.tolist()).union({0.0}) == set()")
    @ensures("set(return.err.tolist()) - set(solution.err.tolist()).union({0.0}) == set()")
    @ensures("solution.prob_undecided() <= return.prob_undecided()")
    @ensures('self.pausestart == self.pausestop --> solution == return')
    def apply(self, solution):
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution
        start = int(self.pausestart/m.dt) # truncate
        stop = int((self.pausestop)/m.dt) # truncate
        if stop <= start:
            return solution
        newcorr = np.zeros(corr.shape, dtype=corr.dtype)
        newerr = np.zeros(err.shape, dtype=err.dtype)
        newcorr[0:start] = corr[0:start]
        newerr[0:start] = err[0:start]
        newcorr[stop:] = corr[start:-(stop-start)]
        newerr[stop:] = err[start:-(stop-start)]
        return Solution(newcorr, newerr, m, cond, undec, evolution)

@paranoidclass
class OverlayBlurredPause(Overlay):
    name = "Brief pause in integration, pause length by gamma distribution"
    required_parameters = ['pausestart', 'pausestop', 'pauseblurwidth']
    @staticmethod
    def _test(v):
        assert v.pausestart in Positive0(), "Invalid start time"
        assert v.pausestop in Positive0(), "Invalid stop time"
        assert v.pauseblurwidth in Positive(), "Invalid width"
        assert v.pausestart <= v.pausestop, "Pause start time must be before stop time"
        assert v.pausestart + v.pauseblurwidth/2 <= v.pausestop, "Blur must be shorter than pause"
    @staticmethod
    def _generate():
        yield OverlayBlurredPause(pausestart=0, pausestop=.1, pauseblurwidth=.1)
        yield OverlayBlurredPause(pausestart=.1, pausestop=.2, pauseblurwidth=.01)
    @accepts(Self, Solution)
    @returns(Solution)
    @ensures("solution.prob_undecided() <= return.prob_undecided()")
    @ensures('self.pausestart == self.pausestop --> solution == return')
    def apply(self, solution):
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution
        # Make gamma distribution
        gamma_mean = self.pausestop - self.pausestart
        gamma_var = pow(self.pauseblurwidth, 2)
        shape = gamma_mean**2/gamma_var
        scale = gamma_var/gamma_mean
        gamma_pdf = lambda t : 1/(sp_gamma(shape)*(scale**shape)) * t**(shape-1) * np.exp(-t/scale)
        gamma_vals = np.asarray([gamma_pdf(t) for t in m.t_domain() - self.pausestart if t >= 0])
        sumgamma = np.sum(gamma_vals)
        gamma_start = next(i for i,t in enumerate(m.t_domain() - self.pausestart) if t >= 0)

        # Generate first part of pdf (before the pause)
        newcorr = np.zeros(m.t_domain().shape, dtype=corr.dtype)
        newerr = np.zeros(m.t_domain().shape, dtype=err.dtype)
        # Generate pdf after the pause
        for i,t in enumerate(m.t_domain()):
            #print(np.sum(newcorr)+np.sum(newerr))
            if 0 <= t < self.pausestart:
                newcorr[i] = corr[i]
                newerr[i] = err[i]
            elif self.pausestart <= t:
                newcorr[i:] += corr[gamma_start:len(corr)-(i-gamma_start)]*gamma_vals[int(i-gamma_start)]/sumgamma
                newerr[i:] += err[gamma_start:len(corr)-(i-gamma_start)]*gamma_vals[int(i-gamma_start)]/sumgamma
            else:
                raise ValueError("Invalid domain")
        return Solution(newcorr, newerr, m, cond, undec, evolution)

