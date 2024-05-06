# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

import copy
import logging
import numpy as np
from paranoid.types import NDArray, Generic, Number, Self, Positive0, Range, Natural1, Natural0, Maybe, Boolean, Or, String, Set, Constant, Numeric
from paranoid.decorators import accepts, returns, requires, ensures, paranoidclass
from .models.paranoid_types import Conditions, Choice
from .sample import Sample
from .logger import logger as _logger, deprecation_warning

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
        assert v.choice_upper in NDArray(d=1, t=Number), "Invalid corr histogram"
        assert v.choice_lower in NDArray(d=1, t=Number), "Invalid err histogram"
        if v.undec is not None:
            assert v.undec in NDArray(d=1, t=Number), "Invalid err histogram"
            assert len(v.undec) == len(v.model.x_domain(conditions=v.conditions))
        #assert v.model is Generic(Model), "Invalid model" # TODO could cause inf recursion issue
        assert len(v.choice_upper) == len(v.choice_lower) == len(v.t_domain), "Histogram lengths must match"
        assert 0 <= np.sum(v.choice_upper) + np.sum(v.choice_lower) <= 1, "Histogram should integrate " \
            " to 1, not to " + str(np.sum(v.choice_upper)+np.sum(v.choice_lower))
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
        
    def __init__(self, pdf_choice_upper, pdf_choice_lower, model, conditions, pdf_undec=None, pdf_evolution=None):
        """Create a Solution object from the results of a model
        simulation.

        Constructor takes four arguments.

            - `pdf_choice_upper` - a size N numpy ndarray describing the upper boundary crossing portion of the joint pdf
            - `pdf_choice_lower` - a size N numpy ndarray describing the lower boundary crossing portion of the joint pdf
            - `model` - the Model object used to generate `pdf_corr` and `pdf_err`
            - `conditions` - a dictionary of condition names/values used to generate the solution
            - `pdf_undec` - a size M numpy ndarray describing the final state of the simulation.  None if unavailable.
            - `pdf_evolution` - a size M-by-N numpy ndarray describing the state of the simulation at each time step. None if unavailable.
        """
        self.model = model # If the model changes, this var will change too.  So create local versions of important variables below.
        self.choice_upper = pdf_choice_upper
        self.choice_lower = pdf_choice_lower
        self.undec = pdf_undec
        self.evolution = pdf_evolution
        self.choice_names = model.choice_names
        self.dx = model.dx
        self.dt = model.dt
        self.T_dur = model.T_dur
        self.t_domain = model.t_domain()
        self.model_name = model.name
        self.model_parameters = model.parameters()
        # Import here to avoid recursion
        from .models.overlay import OverlayNone
        self.is_overlay_none = isinstance(model.get_dependence("overlay"), OverlayNone)
        # Correct floating point errors to always get prob <= 1
        if np.sum(self.choice_upper + self.choice_lower) > 1:
            self.choice_upper /= 1.00000000001
            self.choice_lower /= 1.00000000001
        self.conditions = conditions

    def __eq__(self, other):
        if not np.allclose(self.choice_upper, other.choice_upper) or \
           not np.allclose(self.choice_lower, other.choice_lower):
            return False
        if self.choice_names != other.choice_names:
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
    @property
    def corr(self):
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("For backward compatibility with .corr only")
        deprecation_warning(instead="Solution.choice_upper", isfunction=False)
        return self.choice_upper
    @property
    def err(self):
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("For backward compatibility with .corr only")
        deprecation_warning(instead="Solution.choice_lower", isfunction=False)
        return self.choice_lower

    @accepts(Self, Choice)
    @returns(Set([1, 2]))
    def _choice_name_to_id(self, choice):
        """Get an ID from the choice name.

        If the choice corresponds to the upper boundary, return 1.  If it
        corresponds to the lower boundary, return 2.  Otherwise, print an
        error.
        """
        # Do is this way in case someone names their choices "_bottom" and
        # "_top" in reverse.
        if choice in [1, self.choice_names[0]]:
            return 1
        if choice in [0, 2, self.choice_names[1]]:
            return 2
        if choice in ["_top", "top", "top_bound", "upper_bound", "upper"]:
            return 1
        if choice in ["_bottom", "bottom_bound", "lower_bound", "lower", "bottom"]:
            return 2
        raise NotImplementedError("\"choice\" needs to be '"+self.choice_names[0]+"' or '"+self.choice_names[1]+"' to use this function, not '"+choice+"'")

    @accepts(Self, Choice)
    @returns(NDArray(d=1, t=Positive0))
    def pdf(self, choice):
        """The probability density function of model RTs for a given choice.

        `choice` should be the name of the choice for which to obtain the pdf,
        corresponding to the upper or lower boundary crossings.  E.g.,
        "correct", "error", or the choice names specified in the model's
        choice_names parameter.

        Note that the return value will not sum to one, but both choices plus
        the undecided distribution will collectively sum to one.

        """
        v = self.choice_upper if self._choice_name_to_id(choice) == 1 else self.choice_lower
        return v/self.dt

    @accepts(Self)
    @returns(NDArray(d=1, t=Positive0))
    def pdf_corr(self):
        """The correct component of the joint PDF.

        This method is deprecated, use Solution.pdf() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"pdf\" instead.")
        deprecation_warning(instead="Solution.pdf('correct')")
        return self.choice_upper/self.dt

    @accepts(Self)
    @returns(NDArray(d=1, t=Positive0))
    def pdf_err(self):
        """The error (incorrect) component of the joint PDF.

        This method is deprecated, use Solution.pdf() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"pdf\" instead.")
        deprecation_warning(instead="Solution.pdf('error')")
        return self.choice_lower/self.dt

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
        # Common mistake so we want to warn the user of any possible
        # misunderstanding.
        if not self.is_overlay_none:
            _logger.warning(("Undecided probability accessed for model with overlays.  Undecided "
                + "probability applies *before* overlays.  Please see the pdf_undec docs for more "
                + "information and to prevent misunderstanding."))
        if self.undec is not None:
            return self.undec/self.dx
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
        "return_evolution", which should be set to True.
        """
        # Common mistake so we want to warn the user of any possible
        # misunderstanding.
        if not self.is_overlay_none:
            _logger.warning(("Probability evolution accessed for model with overlays.  Probability"
                + "evolution applies *before* overlays.  Please see the evolution docs for more "
                + "information and to prevent misunderstanding."))
        if self.evolution is not None:
            return self.evolution/self.dx
        else:
            raise ValueError("Probability evolution unavailable")

    @accepts(Self, Choice)
    @returns(NDArray(d=1, t=Positive0))
    def cdf(self, choice):
        """The cumulative density function of model RTs for a given choice.

        `choice` should be the name of the choice for which to obtain the cdf,
        corresponding to the upper or lower boundary crossings.  E.g.,
        "correct", "error", or the choice names specified in the model's
        choice_names parameter.

        Note that the return value will not converge to one, but both choices plus
        the undecided distribution will collectively converge to one.

        """
        v = self.choice_upper if self._choice_name_to_id(choice) == 1 else self.choice_lower
        return np.cumsum(v)

    @accepts(Self)
    @returns(NDArray(d=1, t=Positive0))
    def cdf_corr(self):
        """The correct component of the joint CDF.

        This method is deprecated, use Solution.cdf() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"cdf\" instead.")
        deprecation_warning(instead="Solution.cdf('correct')")
        return np.cumsum(self.choice_upper)

    @accepts(Self)
    @returns(NDArray(d=1, t=Positive0))
    def cdf_err(self):
        """The error (incorrect) component of the joint CDF.

        This method is deprecated, use Solution.cdf() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"cdf\" instead.")
        deprecation_warning(instead="Solution.cdf('error')")
        return np.cumsum(self.choice_lower)

    @accepts(Self, Choice)
    @returns(Range(0, 1))
    def prob(self, choice):
        """Probability of a given choice response within the time limit.

        `choice` should be the name of the choice for which to obtain the
        probability, corresponding to the upper or lower boundary crossings.
        E.g., "correct", "error", or the choice names specified in the model's
        """
        v = self.choice_upper if self._choice_name_to_id(choice) == 1 else self.choice_lower
        return np.sum(v)

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_correct(self):
        """Probability of correct response within the time limit.

        This method is deprecated, use Solution.prob() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"prob\" instead.")
        deprecation_warning(instead="Solution.prob('correct')")
        return np.sum(self.choice_upper)

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_error(self):
        """Probability of incorrect (error) response within the time limit.

        This method is deprecated, use Solution.prob() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"prob\" instead.")
        deprecation_warning(instead="Solution.prob('error')")
        return np.sum(self.choice_lower)

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_undecided(self):
        """The probability of not responding during the time limit."""
        udprob = 1 - np.sum(self.choice_upper) - np.sum(self.choice_lower)
        if udprob < 0:
            _logger.warning("Setting undecided probability from %f to 0" % udprob)
            _logger.debug(self.model_parameters)
            udprob = 0
        return udprob

    @accepts(Self, Choice)
    @returns(Range(0, 1))
    def prob_forced(self, choice):
        """Probability of a given response if a response is forced.

        `choice` should be the name of the choice for which to obtain the
        probability, corresponding to the upper or lower boundary crossings.
        E.g., "correct", "error", or the choice names specified in the model's

        If a trajectory is undecided at the time limit (T_dur), then a response
        is selected randomly.
        """
        return self.prob(choice) + self.prob_undecided()/2.

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_correct_forced(self):
        """Probability of correct response if a response is forced.

        Forced responses are selected randomly.

        This method is deprecated, use Solution.prob_forced() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"prob_forced\" instead.")
        deprecation_warning(instead="Solution.prob_forced('correct')")
        return self.prob_correct() + self.prob_undecided()/2.

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_error_forced(self):
        """Probability of incorrect response if a response is forced.

        Forced responses are selected randomly.

        This method is deprecated, use Solution.prob_forced() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"prob_forced\" instead.")
        deprecation_warning(instead="Solution.prob_forced('error')")
        return self.prob_error() + self.prob_undecided()/2.

    @accepts(Self, Choice)
    @returns(Range(0, 1))
    @requires("self.undec is not None")
    def prob_sign(self, choice):
        """Probability of a given response if a response is forced.

        `choice` should be the name of the choice for which to obtain the
        probability, corresponding to the upper or lower boundary crossings.
        E.g., "correct", "error", or the choice names specified in the model's
        choice_names parameter.

        If a trajectory is undecided at the time limit (T_dur), then a response
        is sampled from the distribution of decision variables at the final
        timepoint.

        """
        if self._choice_name_to_id(choice) == 1:
            undec = np.sum(self.undec[len(self.undec)//2+1:])
        elif self._choice_name_to_id(choice) == 2:
            undec = np.sum(self.undec[:len(self.undec)//2])
        else:
            raise NotImplementedError("\"choice\" needs to be '"+self.choice_names[0]+"' or '"+self.choice_names[1]+"' to use this function, not '"+choice+"'")
        return self.prob(choice) + undec

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("self.undec is not None")
    def prob_correct_sign(self):
        """Probability of correct response if a response is forced.

        Forced responses are selected by the position of the decision
        variable at the end of the time limit T_dur.

        This is only available for the implicit method.

        This method is deprecated, use Solution.prob_sign() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"prob_sign\" instead.")
        deprecation_warning(instead="Solution.prob_sign('correct')")
        return self.prob_correct() + np.sum(self.undec[len(self.undec)//2+1:])

    @accepts(Self)
    @returns(Range(0, 1))
    @requires("self.undec is not None")
    def prob_error_sign(self):
        """Probability of incorrect response if a response is forced.

        Forced responses are selected by the position of the decision
        variable at the end of the time limit T_dur.

        This is only available for the implicit method.

        This method is deprecated, use Solution.prob_sign() instead.
        """
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  Use \"prob_sign\" instead.")
        deprecation_warning(instead="Solution.prob_sign('error')")
        return self.prob_error() + np.sum(self.undec[:len(self.undec)//2])

    @accepts(Self)
    @requires('self.prob_correct() > 0')
    @returns(Positive0)
    def mean_decision_time(self):
        """The mean decision time in the correct trials (excluding undecided trials)."""
        if self.choice_names != ("correct", "error"):
            raise NotImplementedError("Choice names need to be set to \"correct\" and \"error\" to use this function.  See mean_rt function.")
        return np.sum(self.choice_upper*self.t_domain) / self.prob_correct()

    @accepts(Self)
    @returns(Numeric)
    def mean_rt(self):
        """The mean decision time (excluding undecided trials)."""
        if self.prob("upper")+self.prob("lower") == 0:
            return np.nan
        return np.sum((self.choice_upper+self.choice_lower)*self.t_domain) / (self.prob("upper")+self.prob("lower"))

    def resample(self, k=1, seed=None):
        """Use the "sample()" function instead."""
        return self.sample(k=k, seed=seed)

    @accepts(Self, Natural1, seed=Maybe(Natural0))
    @returns(Sample)
    @ensures("len(return) == k")
    def sample(self, k=1, seed=None):
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
        shorter_t_domain = self.t_domain[:-1]
        shorter_pdf_choice_upper = self.pdf("_top")[:-1]
        shorter_pdf_choice_upper[-1] += self.pdf("_top")[-1]
        shorter_pdf_choice_lower = self.pdf("_bottom")[:-1]
        shorter_pdf_choice_lower[-1] += self.pdf("_bottom")[-1]
        # Concatenate the correct and error distributions as well as
        # their probabilities, and add an undecided component on the
        # end.  Shift the error t domain by the maximum plus one.
        shift = np.max(shorter_t_domain)+1
        combined_domain = list(shorter_t_domain) + list(shorter_t_domain+shift) + [-1]
        combined_probs = list(shorter_pdf_choice_upper*self.dt) + list(shorter_pdf_choice_lower*self.dt) + [self.prob_undecided()]
        if np.abs(np.sum(combined_probs)-1) >= .0001:
            _logger.warning("Distribution sums to %f rather than 1" % np.sum(combined_probs))
            _logger.debug(self.model_parameters)
        samp = rng.choice(combined_domain, p=combined_probs, replace=True, size=k)
        undecided = np.sum(samp==-1)
        samp = samp[samp != -1] # Remove undecided trials
        # Each point x on the pdf represents the space from x to x+dt.
        # So sample and then add uniform noise to each element.
        samp += rng.uniform(0, self.dt, len(samp))
        # Find correct and error trials
        choice_upper_sample = samp[samp<shift]
        choice_lower_sample = samp[samp>=shift]-shift
        # Build Sample object
        aa = np.asarray
        conditions = {k : (aa([v]*len(choice_upper_sample)), aa([v]*len(choice_lower_sample)), aa([v]*int(undecided))) for k,v in self.conditions.items()}
        return Sample(choice_upper_sample, choice_lower_sample, undecided, choice_names=self.choice_names, **conditions)

    @accepts(Self, Positive0, Maybe(Choice), Maybe(Boolean))
    @requires("(choice is None) != (correct is None)")
    @returns(Positive0)
    def evaluate(self, rt, choice=None, correct=None):
        """Evaluate the pdf at a given response time.

        `rt` is a time, greater than zero, at which to evaluate the pdf.

        `choice` is whether to evaluate on the upper or lower boundary, given
        as the name of the choice, e.g. "correct", "error", or the choice names
        specified in the model's choice_names parameter.

        `correct` is a deprecated parameter for backward compatibility, please
        use `choice` instead.

        Returns the value of the pdf at the given RT.  Note that, despite being
        from a probability distribution, this may be greater than 0, since it
        is a continuous probability distribution.

        """
        if correct is not None:
            assert choice is None, "Either choice or correct argument must be None"
            assert self.choice_names == ("correct", "error")
            choice = self.choice_names[0] if correct else self.choice_names[1]
            deprecation_warning(instead="Solution.evaluate(rt, 'correct') or Solution.evaluate(rt, 'error')")
        else:
            assert choice is not None, "Choice and correct arguments cannot both be None"
        i = round(rt/self.dt)
        if i >= len(self.pdf(choice)):
            _logger.warning("T_dur="+str(self.T_dur)+" is too short for rt="+str(rt))
            return 0
        return self.pdf(choice)[i]
