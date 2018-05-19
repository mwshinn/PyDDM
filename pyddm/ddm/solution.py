import copy
import numpy as np
from math import fsum
from paranoid.types import NDArray, Generic, Number, Self, Positive0, Range, Natural1, Natural0
from paranoid.decorators import accepts, returns, requires, ensures, paranoidclass
from .models.paranoid_types import Conditions
from .sample import Sample

@paranoidclass
class Solution(object):
    """Describes the result of an analytic or numerical DDM run.

    This is a glorified container for a joint pdf, between the
    response options (correct, error, and no decision) and the
    response time distribution associated with each.  It stores a copy
    of the response time distribution for both the correct case and
    the incorrect case, and the rest of the properties can be
    calculated from there.  

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
        assert len(v.corr) == len(v.err), "Histogram lengths must match"
        assert 0 <= fsum(v.corr.tolist() + v.err.tolist()) <= 1, "Histogram does not integrate " \
            " to 1, not to " + str(fsum(v.corr.tolist() + v.err.tolist()))
        assert v.conditions in Conditions()
    @staticmethod
    def _generate():
        from .model import Model # TODO fix hack
        m = Model()
        X = m.t_domain()
        l = len(X)
        # All undecided
        yield Solution(np.zeros(l), np.zeros(l), m, next(Conditions().generate()))
        # Uniform
        yield Solution(np.ones(l)/(2*l), np.ones(l)/(2*l), m, next(Conditions().generate()))
    def __init__(self, pdf_corr, pdf_err, model, conditions, pdf_undec=None):
        """Create a Solution object from the results of a model
        simulation.

        Constructor takes four arguments.

            - `pdf_corr` - a size N numpy ndarray describing the correct portion of the joint pdf
            - `pdf_err` - a size N numpy ndarray describing the error portion of the joint pdf
            - `model` - the Model object used to generate `pdf_corr` and `pdf_err`
            - `conditions` - a dictionary of condition names/values used to generate the solution
            - `pdf_undec` - a size M numpy ndarray describing the final state of the simulation.  None if unavailable.
        """
        self.model = copy.deepcopy(model) # TODO this could cause a memory leak if I forget it is there...
        self.corr = pdf_corr 
        self.err = pdf_err
        self.undec = pdf_undec
        # Correct floating point errors to always get prob <= 1
        if fsum(self.corr.tolist() + self.err.tolist()) > 1:
            self.corr /= 1.00000000001
            self.err /= 1.00000000001
        self.conditions = conditions

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

    # TODO This doesn't take non-decision time into consideration
    @accepts(Self)
    @returns(NDArray(d=1, t=Positive0))
    def pdf_undec(self):
        """The final state of the simulation, same size as `x_domain()`."""
        if self.undec is not None:
            return self.undec/self.model.dx
        else:
            raise ValueError("Final state unavailable")

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
        """The probability of selecting the right response."""
        return fsum(self.corr)

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_error(self):
        """The probability of selecting the incorrect (error) response."""
        return fsum(self.err)

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_undecided(self):
        """The probability of selecting neither response (undecided)."""
        udprob = 1 - fsum(self.corr.tolist() + self.err.tolist())
        if udprob < 0:
            print("Warning, setting undecided probability from %f to 0" % udprob)
            edprob = 0
        return udprob

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_correct_forced(self):
        """The probability of selecting the correct response if a response is forced."""
        return self.prob_correct() + self.prob_undecided()/2.

    @accepts(Self)
    @returns(Range(0, 1))
    def prob_error_forced(self):
        """The probability of selecting the incorrect response if a response is forced."""
        return self.prob_error() + self.prob_undecided()/2.

    @accepts(Self)
    @requires('self.prob_correct() > 0')
    @returns(Positive0)
    def mean_decision_time(self):
        """The mean decision time in the correct trials (excluding undecided trials)."""
        return fsum((self.corr)*self.model.t_domain()) / self.prob_correct()

    @staticmethod
    def _sample_from_histogram(hist, hist_bins, k, seed=0):
        """Generate a sample from a histogram.

        Given a histogram, imply the distribution and generate a
        sample. `hist` should be either a normalized histogram
        (i.e. summing to 1 or less for non-decision) or else a listing
        of bin membership counts, of size n.  (I.e. there are n bins.)
        The bins should be labeled by `hist_bins`, a list of size
        n+1. This will sample from the distribution and return a list
        of size `k`.  This uses the naive method of selecting
        uniformly from the area of each histogram bar according to the
        size of the bar, rather than non-parametrically estimating the
        distribution and then sampling from the distribution (which
        would arguably be better).
        """
        assert len(hist_bins) == len(hist) + 1, "Incorrect bin specification"
        rng = np.random.RandomState(seed)
        sample = []
        h = hist
        norm = np.round(fsum(h), 2)
        assert norm <= 1 or int(norm) == norm, "Invalid histogram of size %f" % norm
        if norm >= 1:
            h = h/norm
            norm = 1
        hcum = np.cumsum(h)
        rns = rng.rand(k)
        for rn in rns:
            ind = next((i for i in range(0, len(hcum)) if rn < hcum[i]), np.nan)
            if np.isnan(ind):
                sample.append(np.nan)
            else:
                sample.append(rng.uniform(low=hist_bins[ind], high=hist_bins[ind+1]))
        return sample

    # TODO rewrite this to work more generically with all histograms
    @accepts(Self, Natural1, seed=Natural0)
    @requires("self.pdf_err()[0] == 0 and self.pdf_corr()[0] == 0") # TODO remove this after rewrite
    @returns(Sample)
    def resample(self, k=1, seed=0):
        """Generate a list of reaction times sampled from the PDF.

        `k` is the number of TRIALS, not the number of samples.  Since
        we are only showing the distribution from the correct trials,
        we guarantee that, for an identical seed, the sum of the two
        return values will be less than `k`.  If no non-decision
        trials exist, the sum of return values will be equal to `k`.

        This relies on the assumption that reaction time cannot be
        less than 0.

        Returns a tuple, where the first element is a list of correct
        reaction times, and the second element is a list of error
        reaction times.
        """
        # To sample from both correct and error distributions, we make
        # the correct answers be positive and the incorrect answers be
        # negative.
        pdf_domain = self.model.t_domain()
        assert pdf_domain[0] == 0, "Invalid PDF domain"
        combined_domain = list(reversed(np.asarray(pdf_domain)*-1)) + list(pdf_domain[1:])
        # Continuity correction doesn't work here.  Instead we say
        # that the histogram value at time dt belongs in the [0, dt]
        # bin, the value at time 2*dt belongs in the [dt, 2*dt] bin,
        # etc.
        assert self.pdf_err()[0] == 0 and self.pdf_corr()[0] == 0, "Invalid pdfs"
        combined_pdf = list(reversed(self.pdf_err()[1:]))+list(self.pdf_corr()[1:])
        sample = self._sample_from_histogram(np.asarray(combined_pdf)*self.model.dt, combined_domain, k, seed=seed)
        aa = lambda x : np.asarray(x)
        corr_sample = aa([x for x in sample if x >= 0])
        err_sample = aa([-x for x in sample if x < 0])
        non_decision = k - (len(corr_sample) + len(err_sample))
        conditions = {k : (aa([v]*len(corr_sample)), aa([v]*len(err_sample)), aa([v]*int(non_decision))) for k,v in self.conditions.items()}
        return Sample(corr_sample, err_sample, non_decision, **conditions)
