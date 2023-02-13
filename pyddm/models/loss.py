# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

__all__ = ['LossFunction', 'LossSquaredError', 'LossLikelihood', 'LossBIC', 'LossRobustLikelihood', 'LossRobustBIC']

import logging
import numpy as np

from paranoid.decorators import accepts, returns, requires, ensures, paranoidclass
from paranoid.types import Self, Number, Positive0, Natural1
from ..sample import Sample
from ..model import Model
from ..logger import logger as _logger

class LossFunction(object):
    """An abstract class for a function to assess goodness of fit.

    This is an abstract class for describing how well data fits a model.

    When subclasses are initialized, they will be initialized with the
    Sample object to which the model should be fit.  Because the data
    will not change but the model will change, this is specified with
    initialization.  

    The optional `required_conditions` argument limits the
    stratification of `sample` by conditions to only the conditions
    mentioned in `required_conditions`.  This decreases computation
    time by only solving the model for the condition names listed in
    `required_conditions`.  For example, a simple DDM with no drift
    and constant variaince would mean `required_conditions` is an
    empty list.

    The optional `method` argument can be "analytical", "numerical",
    "cn", "implicit", or "explicit".

    This will automatically parallelize if set_N_cpus() has been
    called.
    """
    @classmethod
    def _generate(cls):
        # Return an instance of each subclass which doesn't have a
        # "setup" method, i.e. it takes no arguments.
        subs = cls.__subclasses__()
        for s in subs:
            # Check if setup is the same as its parent.
            if s.setup is LossFunction.setup:
                samp = Sample.from_numpy_array(np.asarray([[.3, 1], [.4, 0], [.1, 0], [.2, 1]]), [])
                yield s(sample=samp, dt=.01, T_dur=2)
    def __init__(self, sample, required_conditions=None, method=None, **kwargs):
        assert hasattr(self, "name"), "Solver needs a name"
        self.sample = sample
        self.required_conditions = required_conditions
        self.method = method
        self.setup(**kwargs)
    def setup(self, **kwargs):
        """Initialize the loss function.

        The optional `setup` function is executed at the end of the
        initializaiton.  It is executed only once at the beginning of
        the fitting procedure.

        This function may optionally be redefined in subclasses.
        """
        pass
    def loss(self, model):
        """Compute the value of the loss function for the given model.

        This function must be redefined in subclasses.

        `model` should be a Model object.  This should return a
        floating point value, where smaller values mean a better fit
        of the model to the data.
        """
        raise NotImplementedError("Loss function %s invalid: must define the loss(self, model) function" % self.__class__.__name__)
    def cache_by_conditions(self, model):
        """Solve the model for all relevant conditions.

        Solve `model` for each combination of conditions found within the
        dataset.  For example, if `required_conditions` is ["hand", "color"],
        and hand can be left or right and color can be blue or green, solves
        the model for: hand=left and color=blue; hand=right and color=blue;
        hand=left and color=green, hand=right and color=green.

        This is a convenience function for defining new loss
        functions.  There is generally no need to redefine this
        function in subclasses.

        """
        from ..functions import solve_all_conditions
        return solve_all_conditions(model, sample=self.sample, method=self.method)
@paranoidclass
class LossSquaredError(LossFunction):
    """Squared-error loss function"""
    name = "Squared Error"
    @staticmethod
    def _test(v):
        assert v.dt in Positive0()
        assert v.T_dur in Positive0()
        assert v.hists_choice_upper != {}
        assert v.hists_choice_lower != {}
        assert v.target.size == 2*len(v.hists_choice_upper.keys())*(v.T_dur/v.dt+1)
    @staticmethod
    def _generate():
        yield LossSquaredError(sample=next(Sample._generate()), dt=.01, T_dur=3)
    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur
        self.hists_choice_upper = {}
        self.hists_choice_lower = {}
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            self.hists_choice_upper[frozenset(comb.items())] = np.histogram(self.sample.subset(**comb).choice_upper, bins=int(T_dur/dt)+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self.sample.subset(**comb))/dt # dt/2 (and +1) is continuity correction
            self.hists_choice_lower[frozenset(comb.items())] = np.histogram(self.sample.subset(**comb).choice_lower, bins=int(T_dur/dt)+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self.sample.subset(**comb))/dt
        self.target = np.concatenate([s for i in sorted(self.hists_choice_upper.keys()) for s in [self.hists_choice_upper[i], self.hists_choice_lower[i]]])
    @accepts(Self, Model)
    @returns(Number)
    @requires("model.dt == self.dt and model.T_dur == self.T_dur")
    def loss(self, model):
        assert model.dt == self.dt and model.T_dur == self.T_dur
        sols = self.cache_by_conditions(model)
        this = np.concatenate([s for i in sorted(self.hists_choice_upper.keys()) for s in [sols[i].pdf("_top"), sols[i].pdf("_bottom")]])
        return np.sum((this-self.target)**2)*self.dt**2

@paranoidclass
class LossLikelihood(LossFunction):
    """Likelihood loss function"""
    name = "Negative log likelihood"
    _robustness_param = 0
    @staticmethod
    def _test(v):
        assert v.dt in Positive0()
        assert v.T_dur in Positive0()
    @staticmethod
    def _generate():
        yield LossLikelihood(sample=next(Sample._generate()), dt=.01, T_dur=3)
    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur
        # Each element in the dict is indexed by the conditions of the
        # model (e.g. coherence, trial conditions) as a frozenset.
        # Each contains a tuple of lists, which are to contain the
        # position for each within a histogram.  For instance, if a
        # reaction time corresponds to position i, then we can index a
        # list representing a normalized histogram/"pdf" (given by dt
        # and T_dur) for immediate access to the probability of
        # obtaining that value.
        self.hist_indexes = {}
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            s = self.sample.subset(**comb)
            maxt = max(max(s.choice_upper) if s.choice_upper.size != 0 else -1, max(s.choice_lower) if s.choice_lower.size != 0 else -1)
            assert maxt <= self.T_dur, "Simulation time T_dur=%f not long enough for these data. (max sample RT=%f)" % (self.T_dur, maxt)
            # Find the integers which correspond to the timepoints in
            # the pdfs.  Also don't group them into the first bin
            # because this creates bias.
            choice_upper = [int(round(e/dt)) for e in s.choice_upper]
            choice_lower = [int(round(e/dt)) for e in s.choice_lower]
            undec = self.sample.undecided
            self.hist_indexes[frozenset(comb.items())] = (choice_upper, choice_lower, undec)
    @accepts(Self, Model)
    @returns(Number)
    @requires("model.dt == self.dt and model.T_dur == self.T_dur")
    def loss(self, model):
        assert model.dt == self.dt and model.T_dur == self.T_dur
        sols = self.cache_by_conditions(model)
        loglikelihood = 0
        for k in sols.keys():
            # nans come from negative values in the pdfs, which in
            # turn come from the dx parameter being set too low.  This
            # comes up when fitting, because sometimes the algorithm
            # will "explore" and look at extreme parameter values.
            # For example, this arises when standard deviation is very
            # close to 0.  We will issue a warning now, but throwing
            # an exception may be the better way to handle this to
            # make sure it doesn't go unnoticed.
            with np.errstate(all='raise', under='ignore'):
                try:
                    loglikelihood += np.sum(np.log(sols[k].pdf("_top")[self.hist_indexes[k][0]] + self._robustness_param))
                    loglikelihood += np.sum(np.log(sols[k].pdf("_bottom")[self.hist_indexes[k][1]] + self._robustness_param))
                except FloatingPointError:
                    minlike = min(np.min(sols[k].pdf("_top")), np.min(sols[k].pdf("_bottom")))
                    if minlike == 0:
                        _logger.warning("Infinite likelihood encountered. Please either use a Robust likelihood method (e.g. LossRobustLikelihood or LossRobustBIC) or even better use a mixture model (via an Overlay) which covers the full range of simulated times to avoid infinite negative log likelihood.  See the FAQs in the documentation for more information.")
                    elif minlike < 0:
                        _logger.warning("Infinite likelihood encountered. Simulated histogram is less than zero in likelihood calculation.  Try decreasing dt.")
                    _logger.debug(model.parameters())
                    return np.inf
            # This is not a valid way to incorporate undecided trials into a likelihood
            #if sols[k].prob_undecided() > 0:
            #    loglikelihood += np.log(sols[k].prob_undecided())*self.hist_indexes[k][2]
        return -loglikelihood


@paranoidclass
class LossBIC(LossLikelihood):
    """BIC loss function, functionally equivalent to LossLikelihood"""
    name = "BIC"
    @staticmethod
    def _test(v):
        assert v.nparams in Natural1()
        assert v.samplesize in Natural1()
    @staticmethod
    def _generate():
        samp = Sample.from_numpy_array(np.asarray([[.3, 1], [.4, 0], [.1, 0], [.2, 1]]), [])
        yield LossBIC(sample=samp, nparams=4, samplesize=100, dt=.01, T_dur=3)
    def setup(self, nparams, samplesize, **kwargs):
        self.nparams = nparams
        self.samplesize = samplesize
        LossLikelihood.setup(self, **kwargs)
    @accepts(Self, Model)
    @returns(Number)
    @requires("model.dt == self.dt and model.T_dur == self.T_dur")
    def loss(self, model):
        loglikelihood = -LossLikelihood.loss(self, model)
        return np.log(self.samplesize)*self.nparams - 2*loglikelihood

class LossRobustLikelihood(LossLikelihood):
    """Likelihood loss function which will not fail for infinite likelihoods.

    Usually you will want to use LossLikelihood instead.  See the FAQs
    in the documentation for more information on how this differs from
    LossLikelihood.
    """
    _robustness_param = 1e-20

class LossRobustBIC(LossBIC):
    """BIC loss function which will not fail for infinite likelihoods.

    Usually you will want to use LossBIC instead.  See the FAQs in the
    documentation for more information on how this differs from
    LossBIC.
    """
    _robustness_param = 1e-20
