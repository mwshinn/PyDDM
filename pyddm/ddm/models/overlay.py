__ALL__ = ["Overlay", "OverlayNone", "OverlayChain", "OverlayUniformMixture", "OverlayPoissonMixture", "OverlayDelay"]

import numpy as np

from paranoid import accepts, returns, requires, ensures, Self, paranoidclass, Range, Positive, Number, List

from .base import Dependence
from ..solution import Solution

# TODO in unit test, ensure all overlays with reasonable parameters
# applied to a Solution add up to 1

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
        """Apply the overlay to a solution.

        `solution` should be a Solution object.  Return a new solution
        object.
        """
        raise NotImplementedError

@paranoidclass
class OverlayNone(Overlay):
    name = "None"
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

# NOTE: This class is likely to break if any changes are made to the
# Dependence constructor.  In theory, no changes should be made to the
# Dependence constructor, but just in case...
@paranoidclass
class OverlayChain(Overlay):
    name = "Chain overlay"
    required_parameters = ["overlays"]
    @staticmethod
    def _test(v):
        assert v.overlays in List(Overlay), "overlays must be a list of Overlay objects"
    @staticmethod
    def _generate():
        yield OverlayChain(overlays=[OverlayNone()])
        yield OverlayChain(overlays=[OverlayUniformMixture(umixturecoef=.3), OverlayPoissonMixture(pmixturecoef=.2, rate=.7)])
        yield OverlayChain(overlays=[OverlayDelay(delaytime=.1), OverlayPoissonMixture(pmixturecoef=.1, rate=1), OverlayUniformMixture(umixturecoef=.1)])
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

@paranoidclass
class OverlayUniformMixture(Overlay):
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
        # These aren't real pdfs since they don't sum to 1, they sum
        # to 1/self.model.dt.  We can't just sum the correct and error
        # distributions to find this number because that would exclude
        # the non-decision trials.
        pdfsum = 1/m.dt
        corr = corr*(1-self.umixturecoef) + .5*self.umixturecoef/pdfsum/m.T_dur # Assume numpy ndarrays, not lists
        err = err*(1-self.umixturecoef) + .5*self.umixturecoef/pdfsum/m.T_dur
        corr[0] = 0
        err[0] = 0
        return Solution(corr, err, m, cond)

@paranoidclass
class OverlayPoissonMixture(Overlay):
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
        # These aren't real pdfs since they don't sum to 1, they sum
        # to 1/self.model.dt.  We can't just sum the correct and error
        # distributions to find this number because that would exclude
        # the non-decision trials.
        pdfsum = 1/m.dt
        # Pr = lambda ru, rr, P, t : (rr*P)/((rr+ru))*(1-numpy.exp(-1*(rr+ru)*t))
        # P0 = lambda ru, rr, P, t : P*numpy.exp(-(rr+ru)*t) # Nondecision
        # Pr' = lambda ru, rr, P, t : (rr*P)*numpy.exp(-1*(rr+ru)*t)
        # lapses_cdf = lambda t : 1-np.exp(-(2*self.rate)*t)
        lapses = lambda t : 2*self.rate*np.exp(-2*self.rate*t) if t != 0 else 0
        X = [i*m.dt for i in range(0, len(corr))]
        Y = np.asarray(list(map(lapses, X)))/pdfsum
        corr = corr*(1-self.pmixturecoef) + .5*self.pmixturecoef*Y # Assume numpy ndarrays, not lists
        err = err*(1-self.pmixturecoef) + .5*self.pmixturecoef*Y
        #print(corr)
        #print(err)
        return Solution(corr, err, m, cond)

@paranoidclass
class OverlayDelay(Overlay):
    name = "Add a delay by shifting the histogram"
    required_parameters = ["delaytime"]
    @staticmethod
    def _test(v):
        assert v.delaytime in Number(), "Invalid delay time"
    @staticmethod
    def _generate():
        yield OverlayDelay(delaytime=0)
        yield OverlayDelay(delaytime=.5)
        yield OverlayDelay(delaytime=-.5)
    @accepts(Self, Solution)
    @returns(Solution)
    @ensures("set(return.corr) - set(solution.corr).union({0.0}) == set()")
    @ensures("set(return.err) - set(solution.err).union({0.0}) == set()")
    @ensures("solution.prob_undecided() <= return.prob_undecided()")
    def apply(self, solution):
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        shifts = int(self.delaytime/m.dt) # round
        newcorr = np.zeros(corr.shape)
        newerr = np.zeros(err.shape)
        if shifts > 0:
            newcorr[shifts:] = corr[:-shifts]
            newerr[shifts:] = err[:-shifts]
        elif shifts < 0:
            newcorr[:shifts] = corr[-shifts:]
            newerr[:shifts] = err[-shifts:]
        else:
            newcorr = corr
            newerr = err
        return Solution(newcorr, newerr, m, cond)
