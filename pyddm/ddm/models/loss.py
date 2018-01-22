import numpy as np

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

    If `pool` is None, then solve normally.  If `pool` is a Pool
    object from pathos.multiprocessing, then parallelize the loop.
    Note that `pool` must be pathos.multiprocessing.Pool, not
    multiprocessing.Pool, since the latter does not support pickling
    functions, whereas the former does.
    """
    def __init__(self, sample, required_conditions=None, pool=None, **kwargs):
        assert hasattr(self, "name"), "Solver needs a name"
        self.sample = sample
        self.required_conditions = required_conditions
        self.pool = pool
        self.setup(**kwargs)
    def setup(self, **kwargs):
        """Initialize the loss function.

        The optional `setup` function is executed at the end of the
        initializaiton.  It is executed only once at the beginning of
        the fitting procedure.
        """
        pass
    def loss(self, model):
        """Compute the value of the loss function for the given model.

        `model` should be a Model object.  This should return a
        floating point value, where smaller values mean a better fit
        of the model to the data.

        """
        raise NotImplementedError
    def cache_by_conditions(self, model):
        """Solve the model for all relevant conditions.

        If `required_conditions` isn't None, solve `model` for each
        combination of conditions found within the dataset.  For
        example, if `required_conditions` is ["hand", "color"], and
        hand can be left or right and color can be blue or green,
        solves the model for: hand=left and color=blue; hand=right and
        color=blue; hand=left and color=green, hand=right and
        color=green.

        If `required_conditions` is None, use all of the conditions
        found within the sample.
        """
        cache = {}
        conditions = self.sample.condition_combinations(required_conditions=self.required_conditions)
        if self.pool is None: # No parallelization
            for c in conditions:
                cache[frozenset(c.items())] = model.solve(conditions=c)
            return cache
        else: # Parallelize across Pool
            sols = self.pool.map(lambda x : model.solve(conditions=x), conditions)
            for c,s in zip(conditions,sols):
                cache[frozenset(c.items())] = s
            return cache
                
class LossSquaredError(LossFunction):
    name = "Squared Difference"
    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur
        self.hists_corr = {}
        self.hists_err = {}
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            self.hists_corr[frozenset(comb.items())] = np.histogram(self.sample.subset(**comb).corr, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self.sample.subset(**comb))/dt # dt/2 (and +1) is continuity correction
            self.hists_err[frozenset(comb.items())] = np.histogram(self.sample.subset(**comb).err, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self.sample.subset(**comb))/dt
        self.target = np.concatenate([s for i in sorted(self.hists_corr.keys()) for s in [self.hists_corr[i], self.hists_err[i]]])
    def loss(self, model):
        assert model.dt == self.dt and model.T_dur == self.T_dur
        sols = self.cache_by_conditions(model)
        this = np.concatenate([s for i in sorted(self.hists_corr.keys()) for s in [sols[i].pdf_corr(), sols[i].pdf_err()]])
        return np.sum((this-self.target)**2)

SquaredErrorLoss = LossSquaredError # Named this incorrectly on the first go... lecacy

class LossLikelihood(LossFunction):
    name = "Likelihood"
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
            maxt = max(max(s.corr) if s.corr else -1, max(s.err) if s.err else -1)
            assert maxt < self.T_dur, "Simulation time T_dur not long enough for these data"
            # Find the integers which correspond to the timepoints in
            # the pdfs.  Exclude all data points where this index
            # rounds to 0 because this is always 0 in the pdf (no
            # diffusion has happened yet) and you can't take the log
            # of 0.  Also don't group them into the first bin because
            # this creates bias.
            corr = [int(round(e/dt)) for e in s.corr if int(round(e/dt)) > 0]
            err = [int(round(e/dt)) for e in s.err if int(round(e/dt)) > 0]
            nondec = self.sample.non_decision
            self.hist_indexes[frozenset(comb.items())] = (corr, err, nondec)
    def loss(self, model):
        assert model.dt == self.dt and model.T_dur == self.T_dur
        sols = self.cache_by_conditions(model)
        loglikelihood = 0
        for k in sols.keys():
            loglikelihood += np.sum(np.log(sols[k].pdf_corr()[self.hist_indexes[k][0]]))
            loglikelihood += np.sum(np.log(sols[k].pdf_err()[self.hist_indexes[k][1]]))
            if sols[k].prob_undecided() > 0:
                loglikelihood += np.log(sols[k].prob_undecided())*self.hist_indexes[k][2]
            # nans come from negative values in the pdfs, which in
            # turn come from the dx parameter being set too low.  This
            # comes up when fitting, because sometimes the algorithm
            # will "explore" and look at extreme parameter values.
            # For example, this arrises when variance is very close to
            # 0.  We will issue a warning now, but throwing an
            # exception may be the better way to handle this to make
            # sure it doesn't go unnoticed.
            if np.isnan(loglikelihood):
                print("Warning: parameter values too extreme for dx.")
                return np.inf
        return -loglikelihood

class LossBIC(LossLikelihood):
    name = "Use BIC loss function, functionally equivalent to LossLikelihood"
    def setup(self, nparams, samplesize, **kwargs):
        self.nparams = nparams
        self.samplesize = samplesize
        LossLikelihood.setup(self, **kwargs)
    def loss(self, model):
        loglikelihood = -LossLikelihood.loss(self, model)
        return np.log(self.samplesize)*self.nparams - 2*loglikelihood

class LossLikelihoodMixture(LossLikelihood):
    name = "Likelihood with 2% uniform noise"
    def loss(self, model):
        assert model.dt == self.dt and model.T_dur == self.T_dur
        sols = self.cache_by_conditions(model)
        loglikelihood = 0
        for k in sols.keys():
            solpdfcorr = sols[k].pdf_corr()
            solpdfcorr[solpdfcorr<0] = 0 # Numerical errors cause this to be negative sometimes
            solpdferr = sols[k].pdf_err()
            solpdferr[solpdferr<0] = 0 # Numerical errors cause this to be negative sometimes
            pdfcorr = solpdfcorr*.98 + .01*np.ones(1+self.T_dur/self.dt)/self.T_dur*self.dt # .98 and .01, not .98 and .02, because we have both correct and error
            pdferr = solpdferr*.98 + .01*np.ones(1+self.T_dur/self.dt)/self.T_dur*self.dt
            loglikelihood += np.sum(np.log(pdfcorr[self.hist_indexes[k][0]]))
            loglikelihood += np.sum(np.log(pdferr[self.hist_indexes[k][1]]))
            if sols[k].prob_undecided() > 0:
                loglikelihood += np.log(sols[k].prob_undecided())*self.hist_indexes[k][2]
        return -loglikelihood
