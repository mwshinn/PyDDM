# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

import copy
import logging
import numpy as np
from paranoid.types import NDArray, Generic, Number, Self, Positive0, Range, Natural1, Natural0, Maybe, Boolean
from paranoid.decorators import accepts, returns, requires, ensures, paranoidclass
from .models.paranoid_types import Conditions
from .sample import Sample
from .logger import logger as _logger

@paranoidclass
class Solution(object):
    """Describes the result of an analytic or numerical DDM run.

    This is a glorified container for a joint pdf, between the
    response options (correct, error, and undecided) and the response
    time distribution associated with each.  It stores a copy of the
    response time distribution for both the correct case and the
    incorrect case, and the rest of the properties can be calculated
    from there.

    It also stores a full deep copy of the model used to simulate it.
    This is most important for storing, e.g. the dt and the name
    associated with the simulation, but it is also good to keep the
    whole object as a full record describing the simulation, so that
    the full parametrization of every run is recorded.  Note that this
    may increase memory requirements when many simulations are run.
    """
    @staticmethod
    def _test(v):
        # TODO should these be Positive0 instead of Number?
        assert v.corr in NDArray(d=1, t=Number), "Invalid corr histogram"
        assert v.err in NDArray(d=1, t=Number), "Invalid err histogram"
        if v.undec is not None:
            assert v.undec in NDArray(d=1, t=Number), "Invalid err histogram"
            assert len(v.undec) == len(v.model.x_domain(conditions=v.conditions))
        #assert v.model is Generic(Model), "Invalid model" # TODO could cause inf recursion issue
        assert len(v.corr) == len(v.err) == len(v.model.t_domain()), "Histogram lengths must match"
        assert 0 <= np.sum(v.corr) + np.sum(v.err) <= 1, "Histogram should integrate " \
            " to 1, not to " + str(np.sum(v.corr)+np.sum(v.err))
        assert v.conditions in Conditions()
    @staticmethod
    def _generate():
        from .model import Model # Importing here avoids a recursion issue
        m = Model()
        T = m.t_domain()
        lT = len(T)
        X = m.x_domain(conditions={})
        lX = len(X)
        # All undecided
        yield Solution(np.zeros(lT), np.zeros(lT), m, next(Conditions().generate()))
        # Uniform
        yield Solution(np.ones(lT)/(2*lT), np.ones(lT)/(2*lT), m, next(Conditions().generate()))
        # With uniform undecided probability
        yield Solution(np.ones(lT)/(3*lT), np.ones(lT)/(3*lT), m, next(Conditions().generate()), pdf_undec=np.ones(lX)/(3*lX))
        # With uniform undecided probability with collapsing bounds
        from .models.bound import BoundCollapsingExponential
        m2 = Model(bound=BoundCollapsingExponential(B=1, tau=1))
        T2 = m2.t_domain()
        lT2 = len(T2)
        X2 = m2.x_domain(conditions={})
        lX2 = len(X2)
        yield Solution(np.ones(lT2)/(3*lT2), np.ones(lT2)/(3*lT2), m2, next(Conditions().generate()), pdf_undec=np.ones(lX2)/(3*lX2))
        
    def __init__(self, pdf_corr, pdf_err, model, conditions, pdf_undec=None, pdf_evolution=None):
        """Create a Solution object from the results of a model
        simulation.

        Constructor takes four arguments.

            - `pdf_corr` - a size N numpy ndarray describing the correct portion of the joint pdf
            - `pdf_err` - a size N numpy ndarray describing the error portion of the joint pdf
            - `model` - the Model object used to generate `pdf_corr` and `pdf_err`
            - `conditions` - a dictionary of condition names/values used to generate the solution
            - `pdf_undec` - a size M numpy ndarray describing the final state of the simulation.  None if unavailable.
            - `pdf_evolution` - a size M-by-N numpy ndarray describing the state of the simulation at each time step. None if unavailable.
        """
        self.model = copy.deepcopy(model) # TODO this could cause a memory leak if I forget it is there...
        self.corr = pdf_corr
        self.err = pdf_err
        self.undec = pdf_undec
        self.evolution = pdf_evolution
        # Correct floating point errors to always get prob <= 1
        if np.sum(self.corr + self.err) > 1:
            self.corr /= 1.00000000001
            self.err /= 1.00000000001
        self.conditions = conditions

    def __eq__(self, other):
        if not np.allclose(self.corr, other.corr) or \
           not np.allclose(self.err, other.err):
            return False
        for k in self.conditions:
            if k not in other.conditions:
                return False
            if np.issubdtype(type(self.conditions[k]), np.floating) and \
               np.issubdtype(type(other.conditions[k]), np.floating):
                compare_func = np.allclose
            else:
                compare_func = lambda x,y: np.all(np.equal(x,y))
            if not compare_func(self.conditions[k], other.conditions[k]):
                return False
        if self.undec is not None:
            if not np.allclose(self.undec, other.undec):
                return False
        return True

    @accepts(Self)
    @returns(NDArray(d=1, t=Positive0))
    def pdf_corr(self):
        """The correct component of the joint PDF."""
        return self.corr/self.model.dt

    @accepts(Self)
    @returns(NDArray(d=1, t=Positive0))
    def pdf_err(self):
        """The error (incorrect) component of the joint PDF."""
        return self.err/self.model.dt

    @accepts(Self)
    @returns(NDArray(d=1, t=Positive0))
    @requires("self.undec is not None")
    def pdf_undec(self):
        """The final state of the simulation, same size as `x_domain()`.

        If the model contains overlays, this represents the final
        state of the simulation *before* the overlays are applied.
        This is because overlays do not specify what to do with the
        diffusion locations corresponding to undercided probabilities.
        Additionally, all of the necessary information may not be
        stored, such as the case with a non-decision time overlay.

        This means that in the case of models with a non-decision time
        t_nd, this gives the undecided probability at time T_dur +
        t_nd.

        If no overlays are in the model, then pdf_corr() + pdf_err() +
        pdf_undec() should always equal 1 (plus or minus floating
        point errors).
        """
        # Do this here to avoid import recursion
        from .models.overlay import OverlayNone
        # Common mistake so we want to warn the user of any possible
        # misunderstanding.
        if not isinstance(self.model.get_dependence("overlay"), OverlayNone):
            _logger.warning(("Undecided probability accessed for model with overlays.  Undecided "
                + "probability applies *before* overlays.  Please see the pdf_undec docs for more "
                + "information and to prevent misunderstanding."))
        if self.undec is not None:
            return self.undec/self.model.dx
        else:
            raise ValueError("Final state unavailable")


    @accepts(Self)
    @returns(NDArray(d=2, t=Positive0))
    @requires("self.evolution is not None")
    def pdf_evolution(self):
        """The evolving state of the simulation: An array of size `x_domain() x t_domain()`
        whose columns contain the cross-sectional pdf for every time step.
        
        If the model contains overlays, this represents the evolving
        state of the simulation *before* the overlays are applied.
        This is because overlays do not specify what to do with the
        diffusion locations corresponding to undercided probabilities.
        Additionally, all of the necessary information may not be
        stored, such as the case with a non-decision time overlay.

        This means that in the case of models with a non-decision time
        t_nd, this gives the evolving probability at time T_dur +
        t_nd.

        If no overlays are in the model, then 
        sum(pdf_corr()[0:t]*dt) + sum(pdf_err()[0:t]*dt) + sum(pdf_evolution()[:,t]*dx)
        should always equal 1 (plus or minus floating point errors).

        Note that this function will fail if the solution was not generated to
        contain information about the evolution of the pdf.  This is not
        enabled by default, as it causes substantial memory overhead.  To
        enable this, see the documentation for the Model.solve() argument
        "return_pdf", which should be set to True.
        """
        # Do this here to avoid import recursion
        from .models.overlay import OverlayNone
        # Common mistake so we want to warn the user of any possible
        # misunderstanding.
        if not isinstance(self.model.get_dependence("overlay"), OverlayNone):
            _logger.warning(("Probability evolution accessed for model with overlays.  Probability"
                + "evolution applies *before* overlays.  Please see the evolution docs for more "
                + "information and to prevent misunderstanding."))
        if self.evolution is not None:
            return self.evolution/self.model.dx
        else:
            raise ValueError("Probability evolution unavailable")


    @accepts(Self)
    @returns(NDArray(d=1, t=Positive0))
    def cdf_corr(self):
        """The correct component of the joint CDF."""
        return np.cumsum(self.corr)

    @accepts(Self)
    @returns(NDArray(d=1, t=Positive0))
    def cdf_err(self):
        """The error (incorrect) component of the joint CDF."""
        return np.cumsum(self.err)

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_correct(self):
        """Probability of correct response within the time limit."""
        return np.sum(self.corr)

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_error(self):
        """Probability of incorrect (error) response within the time limit."""
        return np.sum(self.err)

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_undecided(self):
        """The probability of not responding during the time limit."""
        udprob = 1 - np.sum(self.corr) - np.sum(self.err)
        if udprob < 0:
            _logger.warning("Setting undecided probability from %f to 0" % udprob)
            _logger.debug(self.model.parameters())
            udprob = 0
        return udprob

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_correct_forced(self):
        """Probability of correct response if a response is forced.

        Forced responses are selected randomly."""
        return self.prob_correct() + self.prob_undecided()/2.

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_error_forced(self):
        """Probability of incorrect response if a response is forced.

        Forced responses are selected randomly."""
        return self.prob_error() + self.prob_undecided()/2.

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("self.undec is not None")
    def prob_correct_sign(self):
        """Probability of correct response if a response is forced.

        Forced responses are selected by the position of the decision
        variable at the end of the time limit T_dur.

        This is only available for the implicit method.
        """
        return self.prob_correct() + np.sum(self.undec[len(self.undec)//2+1:])

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("self.undec is not None")
    def prob_error_sign(self):
        """Probability of incorrect response if a response is forced.

        Forced responses are selected by the position of the decision
        variable at the end of the time limit T_dur.

        This is only available for the implicit method.
        """
        return self.prob_error() + np.sum(self.undec[:len(self.undec)//2])

    @accepts(Self)
    @requires('self.prob_correct() > 0')
    @returns(Positive0)
    def mean_decision_time(self):
        """The mean decision time in the correct trials (excluding undecided trials)."""
        return np.sum(self.corr*self.model.t_domain()) / self.prob_correct()

    @accepts(Self, Natural1, seed=Maybe(Natural0))
    @returns(Sample)
    @ensures("len(return) == k")
    def resample(self, k=1, seed=None):
        """Generate a list of reaction times sampled from the PDF.

        `k` is the number of TRIALS, not the number of samples.  Since
        we are only showing the distribution from the correct trials,
        we guarantee that, for an identical seed, the sum of the two
        return values will be less than `k`.  If no undecided
        trials exist, the sum of return values will be equal to `k`.

        This relies on the assumption that reaction time cannot be
        less than 0.

        `seed` specifies the random seed to use in sampling.  If unspecified,
        it does not set a random seed.

        Returns a Sample object representing the distribution.

        """
        if seed is None:
            rng = np.random
        else:
            rng = np.random.RandomState(seed)
        # Exclude the last point in the t domain because we will add
        # uniform noise to the sample and this would put us over the
        # model's T_dur.
        shorter_t_domain = self.model.t_domain()[:-1]
        shorter_pdf_corr = self.pdf_corr()[:-1]
        shorter_pdf_corr[-1] += self.pdf_corr()[-1]
        shorter_pdf_err = self.pdf_err()[:-1]
        shorter_pdf_err[-1] += self.pdf_err()[-1]
        # Concatenate the correct and error distributions as well as
        # their probabilities, and add an undecided component on the
        # end.  Shift the error t domain by the maximum plus one.
        shift = np.max(shorter_t_domain)+1
        combined_domain = list(shorter_t_domain) + list(shorter_t_domain+shift) + [-1]
        combined_probs = list(shorter_pdf_corr*self.model.dt) + list(shorter_pdf_err*self.model.dt) + [self.prob_undecided()]
        if np.abs(np.sum(combined_probs)-1) >= .0001:
            _logger.warning("Distribution sums to %f rather than 1" % np.sum(combined_probs))
            _logger.debug(self.model.parameters())
        samp = rng.choice(combined_domain, p=combined_probs, replace=True, size=k)
        undecided = np.sum(samp==-1)
        samp = samp[samp != -1] # Remove undecided trials
        # Each point x on the pdf represents the space from x to x+dt.
        # So sample and then add uniform noise to each element.
        samp += rng.uniform(0, self.model.dt, len(samp))
        # Find correct and error trials
        corr_sample = samp[samp<shift]
        err_sample = samp[samp>=shift]-shift
        # Build Sample object
        aa = np.asarray
        conditions = {k : (aa([v]*len(corr_sample)), aa([v]*len(err_sample)), aa([v]*int(undecided))) for k,v in self.conditions.items()}
        return Sample(corr_sample, err_sample, undecided, **conditions)

    @accepts(Self, Positive0, Boolean)
    @returns(Positive0)
    def evaluate(self, rt, correct):
        """Evaluate the pdf at a given response time.

        `rt` is a time, greater than zero, at which to evaluate the pdf.
        `correct` is whether this response should be evaluated in the correct
        or error distribution, with True for correct or False for error.

        Returns the value of the pdf at the given RT.  Note that, despite being
        from a probability distribution, this may be positive, since it is a
        continuous probability distribution.
        """
        i = round(rt/self.model.dt)
        if i >= len(self.pdf_corr()):
            _logger.warning("T_dur="+str(self.model.T_dur)+" is too short for rt="+str(rt))
            return 0
        return (self.pdf_corr() if correct else self.pdf_err())[i]
