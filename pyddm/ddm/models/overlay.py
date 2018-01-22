import numpy as np

from .base import Dependence

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

    def get_solution_components(self, solution): # DEPRECIATED TODO delete this
        return (solution.corr, solution.err, solution.model, solution.conditions)

class OverlayNone(Overlay):
    name = "None"
    required_parameters = []
    def apply(self, solution):
        return solution

# NOTE: This class is likely to break if any changes are made to the
# Dependence constructor.  In theory, no changes should be made to the
# Dependence constructor, but just in case...
class OverlayChain(Overlay):
    name = "Chain overlay"
    required_parameters = ["overlays"]
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
    def apply(self, solution):
        assert isinstance(solution, Solution)
        newsol = solution
        for o in self.overlays:
            newsol = o.apply(newsol)
        return newsol

class OverlayUniformMixture(Overlay):
    name = "Uniform distribution mixture model"
    required_parameters = ["umixturecoef"]
    def apply(self, solution):
        assert self.umixturecoef >= 0 and self.umixturecoef <= 1
        corr, err, m, cond = self.get_solution_components(solution)
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

class OverlayPoissonMixture(Overlay):
    name = "Poisson distribution mixture model (lapse rate)"
    required_parameters = ["mixturecoef", "rate"]
    def apply(self, solution):
        assert self.mixturecoef >= 0 and self.mixturecoef <= 1
        assert isinstance(solution, Solution)
        corr, err, m, cond = self.get_solution_components(solution)
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
        corr = corr*(1-self.mixturecoef) + .5*self.mixturecoef*Y # Assume numpy ndarrays, not lists
        err = err*(1-self.mixturecoef) + .5*self.mixturecoef*Y
        #print(corr)
        #print(err)
        return Solution(corr, err, m, cond)

class OverlayDelay(Overlay):
    name = "Add a delay by shifting the histogram"
    required_parameters = ["delaytime"]
    def apply(self, solution):
        corr, err, m, cond = self.get_solution_components(solution)
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
