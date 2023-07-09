import unittest
from unittest import TestCase, main
from string import ascii_letters
import numpy as np
from itertools import groupby
from math import fsum
import pandas
import copy
import scipy.stats
import copy

from numpy import asarray as aa

import pyddm as ddm

import paranoid
paranoid.settings.Settings.set(enabled=True)

def fails(f, exception=BaseException):
    failed = False
    try:
        f()
    except exception as e:
        failed = True
    if failed == False:
        raise ValueError("Error, function did not fail")

class TestDependences(TestCase):
    def setUp(self):
        """Create fake models which act like models but are actually much simpler."""
        # Fake model which solves to be a uniform distribution
        class FakeUniformModel(ddm.Model):
            def solve(self, conditions={}, *args, **kwargs):
                choice_upper = self.t_domain()*0+.4/len(self.t_domain())
                choice_lower = self.t_domain()*0+.4/len(self.t_domain())
                undec = self.x_domain(conditions=conditions)*0+.2/len(self.x_domain(conditions=conditions))
                return ddm.Solution(choice_upper, choice_lower, self, conditions, undec)
        FakeUniformModel.solve_analytical = FakeUniformModel.solve
        FakeUniformModel.solve_numerical = FakeUniformModel.solve
        FakeUniformModel.solve_numerical_cn = FakeUniformModel.solve
        FakeUniformModel.solve_numerical_implicit = FakeUniformModel.solve
        FakeUniformModel.solve_numerical_explicit = FakeUniformModel.solve
        self.FakeUniformModel = FakeUniformModel
        # Fake model which solves to be a single point
        class FakePointModel(ddm.Model):
            def solve(self, conditions={}, *args, **kwargs):
                choice_upper = self.t_domain()*0
                choice_upper[1] = 1/256
                choice_lower = self.t_domain()*0
                choice_lower[1] = 255/256
                return ddm.Solution(choice_upper, choice_lower, self, conditions)
        FakePointModel.solve_analytical = FakePointModel.solve
        FakePointModel.solve_numerical = FakePointModel.solve
        FakePointModel.solve_numerical_cn = FakePointModel.solve
        FakePointModel.solve_numerical_implicit = FakePointModel.solve
        FakePointModel.solve_numerical_explicit = FakePointModel.solve
        self.FakePointModel = FakePointModel
        # Fake model which has all trials undecided
        class FakeUndecidedModel(ddm.Model):
            def solve(self, conditions={}, *args, **kwargs):
                choice_upper = self.t_domain()*0
                choice_lower = self.t_domain()*0
                undec = self.x_domain(conditions=conditions)*0+1/len(self.x_domain(conditions=conditions))
                return ddm.Solution(choice_upper, choice_lower, self, conditions, undec)
        FakeUndecidedModel.solve_analytical = FakeUndecidedModel.solve
        FakeUndecidedModel.solve_numerical = FakeUndecidedModel.solve
        FakeUndecidedModel.solve_numerical_cn = FakeUndecidedModel.solve
        FakeUndecidedModel.solve_numerical_implicit = FakeUndecidedModel.solve
        FakeUndecidedModel.solve_numerical_explicit = FakeUndecidedModel.solve
        self.FakeUndecidedModel = FakeUndecidedModel
    def test_Dependence_spec(self):
        """Ensure classes can inherit properly from Dependence"""
        # Instantiating directly fails
        fails(lambda : ddm.models.Dependence())
        # Fails without all properties
        class TestDepFail1(ddm.models.Dependence):
            pass
        fails(lambda : TestDepFail1())
        class TestDepFail2(ddm.models.Dependence):
            depname = "Depname"
        fails(lambda : TestDepFail2())
        class TestDepFail3(ddm.models.Dependence):
            depname = "Depname"
            name = "Name"
        fails(lambda : TestDepFail3())
        class TestDep(ddm.models.Dependence):
            depname = "Depname"
            name = "Name"
            required_parameters = []
        assert TestDep() is not None
    def test_Dependence_derived(self):
        """Ensure derived classes handle parameters properly"""
        class TestDep(ddm.models.Dependence):
            depname = "Test dependence"
        class TestDepComp(TestDep):
            name = "Test component"
            required_parameters = ["testparam1", "testparam2"]
            default_parameters = {"testparam2" : 10}
        # Not all params specified
        fails(lambda : TestDepComp())
        # Using default parameter
        assert TestDepComp(testparam1=5) is not None
        # Overriding the default parameter
        tdc = TestDepComp(testparam1=3, testparam2=4)
        assert tdc.testparam1 == 3
        assert tdc.testparam2 == 4
        assert tdc.required_conditions == []
        # Ensure class static variable holds
        tdc = TestDepComp(testparam1=7)
        assert tdc.testparam1 == 7
        assert tdc.testparam2 == 10
    def test_DependenceUses(self):
        """Test the _uses function and its cache"""
        class TestDrift(ddm.models.Drift):
            name = "Test"
            required_parameters = []
            def get_drift(self, x, t, **kwargs):
                return 0
        class TestDriftX(ddm.models.Drift):
            name = "TestX"
            required_parameters = []
            def get_drift(self, x, t, **kwargs):
                return x
        class TestDriftT(ddm.models.Drift):
            name = "TestT"
            required_parameters = []
            def get_drift(self, x, t, **kwargs):
                return t
        class TestDriftXT(ddm.models.Drift):
            name = "TestXT"
            required_parameters = []
            def get_drift(self, x, t, **kwargs):
                return t + x
        td = TestDrift()
        tdx = TestDriftX()
        tdt = TestDriftT()
        tdxt = TestDriftXT()
        assert not td._uses_x() and not td._uses_t()
        assert tdx._uses_x() and not tdx._uses_t()
        assert not tdt._uses_x() and tdt._uses_t()
        assert tdxt._uses_x() and tdxt._uses_t()
    def test_DriftReduces(self):
        """DriftLinear reduces to DriftConstant when x and t are 0"""
        drift_constant_instances = [e for e in ddm.models.DriftConstant._generate()]
        for cinst in drift_constant_instances:
            linst = ddm.models.DriftLinear(drift=cinst.get_drift(t=0), x=0, t=0)
            for t in [0, .1, .5, 1, 2, 10]:
                assert linst.get_drift(t=t, x=1) == cinst.get_drift(t=t, x=1)
    def test_NoiseReduces(self):
        """NoiseLinear reduces to NoiseConstant when x and t are 0"""
        noise_constant_instances = [e for e in ddm.models.NoiseConstant._generate()]
        for cinst in noise_constant_instances:
            linst = ddm.models.NoiseLinear(noise=cinst.get_noise(t=0), x=0, t=0)
            for t in [0, .1, .5, 1, 2, 10]:
                assert linst.get_noise(t=t, x=1) == cinst.get_noise(t=t, x=1)
    def test_ICArbitrary(self):
        """Arbitrary starting conditions from a distribution"""
        # Make sure we get out the same distribution we put in
        m = ddm.Model()
        unif = ddm.models.ICUniform()
        unif_a = ddm.models.ICArbitrary(unif.get_IC(m.x_domain({})))
        assert np.all(unif.get_IC(m.x_domain({})) == unif_a.get_IC(m.x_domain({})))
        point = ddm.models.ICPointSourceCenter()
        point_a = ddm.models.ICArbitrary(point.get_IC(m.x_domain({})))
        assert np.all(point.get_IC(m.x_domain({})) == point_a.get_IC(m.x_domain({})))
        # Make sure the distribution integrates to 1
        fails(lambda : ddm.models.ICArbitrary(aa([.1, .1, 0, 0, 0])))
        fails(lambda : ddm.models.ICArbitrary(aa([0, .6, .6, 0])))
        assert ddm.models.ICArbitrary(aa([1]))
    def test_ICRange(self):
        """Uniform distribution of starting conditions of arbitrary size centered at 0"""
        # Make sure it is the same as uniform in the limiting case
        icrange = ddm.models.ICRange(sz=1)
        icunif = ddm.models.ICUniform()
        params = dict(x=np.arange(-1, 1.0001, .01), dx=.01)
        assert np.all(np.isclose(icunif.get_IC(**params), icrange.get_IC(**params)))
        # Make sure it is the same as point source center when sz=0
        icpsc = ddm.models.ICPointSourceCenter()
        icrange = ddm.models.ICRange(sz=0)
        assert np.all(np.isclose(icpsc.get_IC(**params), icrange.get_IC(**params)))
        # For intermediate values, there should only be two values
        # generated, and it should be symmetric
        icrange = ddm.models.ICRange(sz=.444)
        ic = icrange.get_IC(x=np.arange(-.48, .48001, .02), dx=.02)
        assert np.all(np.isclose(ic, ic[::-1]))
        assert len(set(ic)) == 2
    def test_ICGaussian(self):
        """Gaussian distribution of starting conditions centered at 0"""
        # Make sure it integrates to 1
        icgauss1 = ddm.models.ICGaussian(stdev=.1)
        icgauss2 = ddm.models.ICGaussian(stdev=.9)
        params = dict(x=np.arange(-1, 1.0001, .01), dx=.01)
        assert np.all(np.isclose(np.sum(icgauss1.get_IC(**params)), 1))
        assert np.all(np.isclose(np.sum(icgauss2.get_IC(**params)), 1))
    def test_OverlayNone(self):
        """No overlay"""
        s = ddm.Model().solve()
        assert s == ddm.models.OverlayNone().apply(s)
        s = self.FakeUniformModel().solve()
        assert s == ddm.models.OverlayNone().apply(s)
        s = self.FakePointModel().solve()
        assert s == ddm.models.OverlayNone().apply(s)
    def test_OverlayUniformMixture(self):
        """Uniform mixture model overlay: a uniform distribution plus the model's solved distribution"""
        # Do nothing with 0 probability
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=1)).solve()
        smix = ddm.models.OverlayUniformMixture(umixturecoef=0).apply(s)
        assert s == smix
        # With mixture coef 1, integrate to 1
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=2), noise=ddm.models.NoiseConstant(noise=3)).solve()
        smix = ddm.models.OverlayUniformMixture(umixturecoef=1).apply(s)
        assert np.isclose(np.sum(smix.choice_upper) + np.sum(smix.choice_lower), 1, atol=1e-4)
        # Should not change uniform distribution
        s = self.FakeUniformModel(dt=.001).solve()
        assert s == ddm.models.OverlayUniformMixture(umixturecoef=.2).apply(s)
        # Don't change total probability
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=1)).solve()
        smix = ddm.models.OverlayUniformMixture(umixturecoef=.2).apply(s)
        assert np.isclose(np.sum(s.choice_upper) + np.sum(s.choice_lower),
                          np.sum(smix.choice_upper) + np.sum(smix.choice_lower))

    def test_OverlayPoissonMixture(self):
        """Poisson mixture model overlay: an exponential distribution plus the model's solved distribution"""
        # Do nothing with mixture coef 0
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=1)).solve()
        smix = ddm.models.OverlayPoissonMixture(pmixturecoef=0, rate=1).apply(s)
        assert s == smix
        # With mixture coef 1, integrate to 1
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=2), noise=ddm.models.NoiseConstant(noise=3)).solve()
        smix = ddm.models.OverlayPoissonMixture(pmixturecoef=1, rate=10).apply(s)
        assert np.isclose(np.sum(smix.choice_upper) + np.sum(smix.choice_lower), 1)
        # Should be monotonic decreasing on uniform distribution
        s = self.FakeUniformModel(dt=.001).solve()
        smix = ddm.models.OverlayPoissonMixture(pmixturecoef=.2, rate=1).apply(s)
        assert np.all([smix.choice_upper[i-1]-smix.choice_upper[i] > 0 for i in range(1, len(smix.choice_upper))])
        assert np.all([smix.choice_lower[i-1]-smix.choice_lower[i] > 0 for i in range(1, len(smix.choice_lower))])
        # Don't change total probability
        s = ddm.Model(ddm.models.DriftConstant(drift=1)).solve()
        smix = ddm.models.OverlayPoissonMixture(pmixturecoef=.2, rate=7).apply(s)
        assert np.isclose(np.sum(s.choice_upper) + np.sum(s.choice_lower),
                          np.sum(smix.choice_upper) + np.sum(smix.choice_lower))
    def test_OverlayNonDecision(self):
        """Non-decision time shifts the histogram"""
        # Should do nothing with no shift
        s = ddm.Model().solve(conditions={"amount": 2})
        assert s == ddm.models.OverlayNonDecision(nondectime=0).apply(s)
        # Shifts a single point distribution
        s = self.FakePointModel(dt=.01).solve(conditions={"amount": 2})
        sshift = ddm.models.OverlayNonDecision(nondectime=.01).apply(s)
        assert s.choice_upper[1] == sshift.choice_upper[2]
        assert s.choice_lower[1] == sshift.choice_lower[2]
        # Shift the other way
        s = self.FakePointModel(dt=.01).solve(conditions={"amount": 2})
        sshift = ddm.models.OverlayNonDecision(nondectime=-.01).apply(s)
        assert s.choice_upper[1] == sshift.choice_upper[0]
        assert s.choice_lower[1] == sshift.choice_lower[0]
        # Truncate when time bin doesn't align
        s = self.FakePointModel(dt=.01).solve(conditions={"amount": 2})
        sshift = ddm.models.OverlayNonDecision(nondectime=.019).apply(s)
        assert s.choice_upper[1] == sshift.choice_upper[2]
        assert s.choice_lower[1] == sshift.choice_lower[2]
        # Test subclassing
        class OverlayNonDecisionCondition(ddm.models.OverlayNonDecision):
            name = "condition_nondecision"
            required_parameters = ["nondectime"]
            required_conditions = ["amount"]
            def get_nondecision_time(self, conditions):
                return conditions['amount'] * self.nondectime
        sshift = ddm.models.OverlayNonDecision(nondectime=.02).apply(s)
        sshift2 = OverlayNonDecisionCondition(nondectime=.01).apply(s)
        assert np.all(sshift.choice_upper == sshift2.choice_upper)
        assert np.all(sshift.choice_lower == sshift2.choice_lower)
        
    def test_OverlayNonDecisionUniform(self):
        """Uniform-distributed non-decision time shifts the histogram"""
        # Should give the same results as OverlayNonDecision when halfwidth=0
        s = ddm.Model().solve(conditions={"amount": 2})
        for nondectime in [0, -.1, .01, .0099, .011111, 1]:
            ndunif = ddm.models.OverlayNonDecisionUniform(nondectime=nondectime, halfwidth=0).apply(s)
            ndpoint = ddm.models.OverlayNonDecision(nondectime=nondectime).apply(s)
            assert np.all(np.isclose(ndunif.choice_upper, ndpoint.choice_upper)), (nondectime, list(ndunif.choice_upper), list(ndpoint.choice_upper))
            assert np.all(np.isclose(ndunif.choice_lower, ndpoint.choice_lower))
        # Simple shift example
        s = self.FakePointModel(dt=.01).solve(conditions={"amount": 2})
        sshift = ddm.models.OverlayNonDecisionUniform(nondectime=.02, halfwidth=.01).apply(s)
        assert sshift.choice_upper[2] == sshift.choice_upper[3] == sshift.choice_upper[4]
        assert sshift.choice_lower[2] == sshift.choice_lower[3] == sshift.choice_lower[4]
        assert sshift.choice_upper[0] == sshift.choice_upper[1] == sshift.choice_upper[5] == 0
        assert sshift.choice_lower[0] == sshift.choice_lower[1] == sshift.choice_lower[5] == 0
        # Off-boundary and behind 0 example
        s = self.FakePointModel(dt=.01).solve(conditions={"amount": 2})
        sshift = ddm.models.OverlayNonDecisionUniform(nondectime=.021111, halfwidth=.033333).apply(s)
        assert sshift.choice_upper[0] == sshift.choice_upper[1]
        assert sshift.choice_lower[0] == sshift.choice_lower[1]
        assert len(set(sshift.choice_upper)) == 2
        assert len(set(sshift.choice_lower)) == 2
        # Test subclassing
        class OverlayNonDecisionUniformCondition(ddm.models.OverlayNonDecisionUniform):
            name = "condition_uniform_nondecision"
            required_parameters = ["nondectime", "halfwidth"]
            required_conditions = ["amount"]
            def get_nondecision_time(self, conditions):
                return conditions['amount'] * self.nondectime
        sshift = ddm.models.OverlayNonDecisionUniform(nondectime=.02, halfwidth=.01).apply(s)
        sshift2 = OverlayNonDecisionUniformCondition(nondectime=.01, halfwidth=.01).apply(s)
        assert np.all(sshift.choice_upper == sshift2.choice_upper)
        assert np.all(sshift.choice_lower == sshift2.choice_lower)
    def test_OverlayNonDecisionGamma(self):
        """Gamma-distributed non-decision time shifts the histogram"""
        # Should get back a gamma distribution from a delta spike
        s = self.FakePointModel(dt=.01).solve(conditions={"amount": 2})
        sshift = ddm.models.OverlayNonDecisionGamma(nondectime=.01, shape=1.3, scale=.002).apply(s)
        gamfn = scipy.stats.gamma(a=1.3, scale=.002).pdf(s.t_domain[0:-2])
        assert np.all(np.isclose(sshift.choice_upper[2:], gamfn/np.sum(gamfn)*s.choice_upper[1]))
        assert np.all(np.isclose(sshift.choice_lower[2:], gamfn/np.sum(gamfn)*s.choice_lower[1]))
        # Test subclassing
        class OverlayNonDecisionGammaCondition(ddm.models.OverlayNonDecisionGamma):
            name = "condition_gamma_nondecision"
            required_parameters = ["nondectime", "shape", "scale"]
            required_conditions = ["amount"]
            def get_nondecision_time(self, conditions):
                return conditions['amount'] * self.nondectime
        sshift = ddm.models.OverlayNonDecisionGamma(nondectime=.02, shape=1.3, scale=.002).apply(s)
        sshift2 = OverlayNonDecisionGammaCondition(nondectime=.01, shape=1.3, scale=.002).apply(s)
        assert np.all(sshift.choice_upper == sshift2.choice_upper)
        assert np.all(sshift.choice_lower == sshift2.choice_lower)
    def test_OverlaySimplePause(self):
        """Pause at some point in the trial and then continue, leaving 0 probability in the gap"""
        # Should do nothing with no shift
        s = ddm.Model().solve()
        assert s == ddm.models.OverlaySimplePause(pausestart=.4, pausestop=.4).apply(s)
        # Shift should make a gap in the uniform model
        s = self.FakeUniformModel().solve()
        smix = ddm.models.OverlaySimplePause(pausestart=.3, pausestop=.6).apply(s)
        assert len(set(smix.choice_upper).union(set(smix.choice_lower))) == 2
        assert len(list(groupby(smix.choice_upper))) == 3 # Looks like ----____----------
        # Should start with 0 and then go to constant with pausestart=.3
        s = self.FakeUniformModel(dt=.01).solve()
        smix = ddm.models.OverlaySimplePause(pausestart=0, pausestop=.05).apply(s)
        assert len(set(smix.choice_upper).union(set(smix.choice_lower))) == 2
        assert len(list(groupby(smix.choice_upper))) == 2 # Looks like ____----------
        assert np.all(smix.choice_upper[0:5] == 0) and smix.choice_upper[6] != 0
        # Truncate when time bin doesn't align
        s = self.FakePointModel(dt=.01).solve()
        sshift = ddm.models.OverlaySimplePause(pausestart=.01, pausestop=.029).apply(s)
        assert s.choice_upper[1] == sshift.choice_upper[2]
        assert s.choice_lower[1] == sshift.choice_lower[2]
    def test_OverlayBlurredPause(self):
        """Like OverlaySimplePause but with a gamma distribution on delay times"""
        # Don't change total probability when there are no undecided responses
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=1), T_dur=10).solve()
        smix = ddm.models.OverlayBlurredPause(pausestart=.3, pausestop=.6, pauseblurwidth=.1).apply(s)
        assert np.isclose(np.sum(s.choice_upper) + np.sum(s.choice_lower),
                          np.sum(smix.choice_upper) + np.sum(smix.choice_lower))
        # Make sure responses before the pause aren't affected
        s = self.FakePointModel(dt=.01).solve()
        sshift = ddm.models.OverlayBlurredPause(pausestart=.02, pausestop=.03, pauseblurwidth=.002).apply(s)
        assert s.choice_upper[1] == sshift.choice_upper[1] != 0
        assert s.choice_lower[1] == sshift.choice_lower[1] != 0
        # Make sure responses after look like a gamma distribution
        s = self.FakePointModel(dt=.01).solve()
        sshift = ddm.models.OverlayBlurredPause(pausestart=0, pausestop=.05, pauseblurwidth=.01).apply(s)
        positive = (sshift.choice_upper[2:] > sshift.choice_lower[1:-1]).astype(int) # Excluding first 0 point, should go from + to - slope only once
        assert positive[0] == 1 and positive[-1] == 0 and len(set(positive)) == 2
    def test_OverlayChain(self):
        """Combine multiple overlays in sequence"""
        # Combine with OverlayNone()
        s = self.FakePointModel(dt=.01).solve()
        o = ddm.models.OverlayChain(overlays=[
                ddm.models.OverlayNone(),
                ddm.models.OverlayNonDecision(nondectime=.01),
                ddm.models.OverlayNone()])
        sshift = o.apply(s)
        assert s.choice_upper[1] == sshift.choice_upper[2]
        assert s.choice_lower[1] == sshift.choice_lower[2]
        assert o.nondectime == .01
        o.nondectime = .3
        assert o.nondectime == .3
    def test_LossSquaredError(self):
        """Squared error loss function"""
        # Should be zero for empty sample when all undecided
        m = self.FakeUndecidedModel()
        s = ddm.Sample(aa([]), aa([]), undecided=1)
        assert ddm.models.LossSquaredError(sample=s, dt=m.dt, T_dur=m.T_dur).loss(m) == 0
        # Can also be determined precisely for the point model
        m = self.FakePointModel()
        sol = m.solve()
        err = ddm.models.LossSquaredError(sample=s, dt=m.dt, T_dur=m.T_dur).loss(m)
        assert np.isclose(err, np.sum(sol.choice_upper)**2 + np.sum(sol.choice_lower)**2)
    def test_LossLikelihood(self):
        """Likelihood loss function"""
        # We can calculate likelihood for this simple case
        m = self.FakePointModel(dt=.02)
        sol = m.solve()
        s = ddm.Sample(aa([.02]), aa([]))
        expected = -np.log(np.sum(sol.choice_upper)/m.dt)
        assert np.isclose(expected, ddm.models.LossLikelihood(sample=s, dt=m.dt, T_dur=m.T_dur).loss(m))
        # And for the uniform case we can assert equivalence
        m = self.FakeUniformModel()
        s1 = ddm.Sample(aa([.02, .05, .07, .12]), aa([.33, .21]))
        s2 = ddm.Sample(aa([.13, .1, .02]), aa([.66, .15, .89]))
        assert np.isclose(ddm.models.LossLikelihood(sample=s1, dt=m.dt, T_dur=m.T_dur).loss(m),
                          ddm.models.LossLikelihood(sample=s2, dt=m.dt, T_dur=m.T_dur).loss(m))
        # TODO I think this reveals we should be doing
        # (len(x_domain())-1) instead of len(x_domain()).  Multiple of 2 somewhere.
        # And it should not depend on dt since it is comparing to the pdf
        # m1 = self.FakeUniformModel(dt=.02)
        # m2 = self.FakeUniformModel(dt=.01)
        # print(m1.solve().pdf("_top"), m2.solve().pdf("_bottom"))
        # s = ddm.Sample(aa([.14, .1, .01]), aa([.66, .16, .89]))
        # assert np.isclose(ddm.models.LossLikelihood(sample=s, dt=m1.dt, T_dur=m1.T_dur).loss(m1),
        #                   ddm.models.LossLikelihood(sample=s, dt=m2.dt, T_dur=m2.T_dur).loss(m2))
    def test_BIC(self):
        """BIC loss function"""
        # -2*Likelihood == BIC for a sample size of 1
        m = self.FakePointModel(dt=.02)
        sol = m.solve()
        s = ddm.Sample(aa([.02]), aa([]))
        expected = -np.log(np.sum(sol.choice_upper)/m.dt)
        assert np.isclose(ddm.models.LossBIC(sample=s, dt=m.dt, T_dur=m.T_dur, nparams=1, samplesize=1).loss(m),
                          2*ddm.models.LossLikelihood(sample=s, dt=m.dt, T_dur=m.T_dur).loss(m))


class TestSample(TestCase):
    def setUp(self):
        # You need some gymnastics to get numpy to accept an array of
        # same-length tuples
        _tuple_same_length = np.empty(3, dtype=object)
        _tuple_same_length[:] = [(3,1,2), (3,3,3), (5,4,3)]
        self.samps = {
            # Empty sample
            "empty": ddm.Sample(aa([]), aa([]), 0),
            # Simple sample
            "simple": ddm.Sample(aa([1, 2]), aa([.5, .7]), 0),
            # Sample with conditions
            "conds": ddm.Sample(aa([1, 2, 3]), aa([]), 0,
                                cond1=(aa([1, 1, 2]), aa([]))),
            # Sample with conditions as strings
            "condsstr": ddm.Sample(aa([1, 2, 3]), aa([]), 0,
                                cond1=(aa(["x", "yy", "z z z"]), aa([])), choice_names=("x", "Y with space")),
            # Sample with conditions as tuples
            "condstuple": ddm.Sample(aa([1, 2, 3]), aa([]), 0,
                                     cond1=(aa([(3, 1, 2), (5, 5), (1, 1, 1, 1, 1)], dtype=object), aa([]))),
            # Sample with conditions as tuples which are the same length
            "condstuplesame": ddm.Sample(aa([1, 2, 3]), aa([]), 0,
                                cond1=(_tuple_same_length, aa([]))),
            # Sample with conditions and explicitly showing undecided
            "condsexp": ddm.Sample(aa([1, 2, 3]), aa([]), 0,
                                   cond1=(aa([1, 1, 2]), aa([]), aa([]))),
            # Sample with undecided
            "undec": ddm.Sample(aa([1, 2]), aa([.5, .7]), 2),
            # Sample with undecided and conditions
            "undeccond": ddm.Sample(aa([1, 2, 3]), aa([]), 3,
                                    cond1=(aa([1, 1, 2]), aa([]), aa([2, 2, 1]))),
            # For the adding test
            "adda": ddm.Sample(aa([1, 2]), aa([2, 3, 2]), 3,
                               cond1=(aa(["a", "a"]), aa(["a", "b", "b"]), aa(["a", "b", "b"]))),
            "addb": ddm.Sample(aa([1.5, 2, 1]), aa([1, 2, 3, 1]), 1,
                               cond1=(aa(["b", "b", "c"]), aa(["c", "c", "c", "c"]), aa(["d"]))),
            # Two conditions
            "two": ddm.Sample(aa([1]), aa([2]), 1,
                               conda=(aa(["a"]), aa(["b"]), aa(["a"])),
                               condb=(aa([1]), aa([2]), aa([2]))),
        }
    def test_add(self):
        """Adding two samples together"""
        s1 = self.samps["adda"]
        s2 = self.samps["addb"]
        s = s1 + s2
        assert len(s) == 16
        assert s.condition_names() == ["cond1"]
        assert s.condition_values("cond1") == ["a", "b", "c", "d"]
        assert s.prob_undecided() == .25
        assert s.prob(1) == 5/16
        assert s.prob(0) == 7/16
        # Try to add to the empty sample
        assert self.samps["empty"] + self.samps["undec"] == self.samps["undec"]
        assert self.samps["empty"] + self.samps["simple"] == self.samps["simple"]
    def test_eqality(self):
        """Two samples are equal iff they are the same"""
        # Equality and inequality with multiple conditions
        assert self.samps["adda"] != self.samps["addb"]
        assert self.samps["adda"] == self.samps["adda"]
    def test_condition_values(self):
        """Condition_values method"""
        assert self.samps["conds"].condition_values("cond1") == [1, 2]
        assert self.samps["condsexp"].condition_values("cond1") == [1, 2]
        assert self.samps["undeccond"].condition_values("cond1") == [1, 2]
        assert self.samps["adda"].condition_values("cond1") == ["a", "b"]
        assert self.samps["addb"].condition_values("cond1") == ["b", "c", "d"]
        assert self.samps["two"].condition_values("conda") == ["a", "b"]
        assert self.samps["two"].condition_values("condb") == [1, 2]
    def test_condition_combinations(self):
        """Condition combinations are a cartesian product of condition values"""
        def identical_conditions(c1, c2):
            for cond in c1:
                if cond not in c2:
                    return False
            for cond in c2:
                if cond not in c1:
                    return False
            return True
        # If we want nothing
        assert self.samps["conds"].condition_combinations([]) == [{}]
        # If nothing matches
        assert self.samps["conds"].condition_combinations(["xyz"]) == [{}]
        # If we want everything
        assert identical_conditions(self.samps["conds"].condition_combinations(None), [{"cond1": 1}, {"cond1": 2}])
        # Limit to one condition
        assert identical_conditions(self.samps["conds"].condition_combinations(["cond1"]), [{"cond1": 1}, {"cond1": 2}])
        # More conditions
        conds_two = self.samps["two"].condition_combinations()
        exp_conds_two = [{"conda": "a", "condb": 1},
                         {"conda": "b", "condb": 2},
                         {"conda": "a", "condb": 2}]
        assert identical_conditions(conds_two, exp_conds_two)
    def test_pdfs(self):
        """Produce valid distributions which sum to one"""
        dt = .02
        for n,s in self.samps.items():
            if n == "empty": continue
            assert np.isclose(fsum([fsum(s.pdf("_top", T_dur=4, dt=dt))*dt, fsum(s.pdf("_bottom", T_dur=4, dt=dt))*dt, s.prob_undecided()]), 1)
            assert np.isclose(fsum(s.pdf("_top", T_dur=4, dt=dt)*dt), s.prob("_top"))
            assert np.isclose(fsum(s.pdf("_bottom", T_dur=4, dt=dt)*dt), s.prob("_bottom"))
            if s.choice_names == ("correct", "error"):
                assert s.mean_decision_time() > 0
            if s.prob_undecided() == 0:
                assert s.prob("_top") == s.prob_forced("_top")
                assert s.prob("_bottom") == s.prob_forced("_bottom")
            assert len(s.pdf("_top", T_dur=4, dt=dt)) == len(s.t_domain(T_dur=4, dt=dt))
            s.cdf("_top")
            s.cdf("_bottom")
    def test_iter(self):
        """The iterator .items() goes through correct or error trials and their conditions"""
        itr = self.samps["conds"].items("_top")
        assert next(itr) == (1, {"cond1": 1})
        assert next(itr) == (2, {"cond1": 1})
        assert next(itr) == (3, {"cond1": 2})
        fails(lambda : next(itr), StopIteration)
        itr = self.samps["two"].items(correct=False)
        assert next(itr) == (2, {"conda": "b", "condb": 2})
        # Create a list to make sure we don't iterate past the end
        list(self.samps["conds"].items(correct=True))
        list(self.samps["conds"].items(correct=False))
        fails(lambda : list(self.samps["condsstr"].items(correct=True)))
        list(self.samps["condsstr"].items("_top"))
        list(self.samps["condsstr"].items(1))
        list(self.samps["condsstr"].items(0))
        list(self.samps["condsstr"].items(2))
        list(self.samps["condsstr"].items("_bottom"))
        list(self.samps["condsstr"].items("x"))
        list(self.samps["condsstr"].items("Y with space"))
    def test_subset(self):
        """Filter a sample by some conditions"""
        # Basic access
        assert len(self.samps['conds'].subset(cond1=2)) == 1
        assert len(self.samps['condsstr'].subset(cond1="z z z")) == 1
        assert len(self.samps['condstuple'].subset(cond1=(3,1,2))) == 1
        assert len(self.samps['condstuplesame'].subset(cond1=(3,1,2))) == 1
        # The elements being accessed
        assert list(self.samps['conds'].subset(cond1=1).choice_upper) == [1, 2]
        # An empty subset with two conditions
        assert len(self.samps['two'].subset(conda="b", condb=1)) == 0
        # A non-epty subset with two conditions
        assert len(self.samps['two'].subset(conda="a", condb=1)) == 1
        # Querying only one condition when more conditions exist
        assert len(self.samps['two'].subset(conda="a")) == 2
        # Query by list
        assert len(self.samps['two'].subset(conda=["a", "z"])) == 2
        # Query by function
        assert len(self.samps['two'].subset(conda=lambda x : True if x=="a" else False)) == 2
    def test_from_numpy_array(self):
        """Create a sample from a numpy array"""
        simple_ndarray = np.asarray([[1, 1], [.5, 0], [.7, 0], [2, 1]])
        assert ddm.Sample.from_numpy_array(simple_ndarray, []) == self.samps['simple']
        conds_ndarray = np.asarray([[1, 1, 1], [2, 1, 1], [3, 1, 2]])
        assert ddm.Sample.from_numpy_array(conds_ndarray, ["cond1"]) == self.samps['conds']
        assert ddm.Sample.from_numpy_array(conds_ndarray, ["cond1"]) == self.samps['condsexp']
    def test_from_pandas(self):
        """Create a sample from a pandas dataframe"""
        simple_df = pandas.DataFrame({'corr': [1, 0, 0, 1], 'resptime': [1, .5, .7, 2]})
        print(simple_df)
        assert ddm.Sample.from_pandas_dataframe(simple_df, 'resptime', 'corr') == self.samps['simple']
        cond_df = pandas.DataFrame({'c': [1, 1, 1], 'rt': [1, 2, 3], 'cond1': [1, 1, 2]})
        assert ddm.Sample.from_pandas_dataframe(cond_df, 'rt', 'c') == self.samps['conds']
        assert ddm.Sample.from_pandas_dataframe(cond_df, correct_column_name='c', rt_column_name='rt') == self.samps['condsexp']
        assert ddm.Sample.from_pandas_dataframe(cond_df, choice_column_name='c', rt_column_name='rt') == self.samps['condsexp']
        assert ddm.Sample.from_pandas_dataframe(cond_df, choice_column_name='c', rt_column_name='rt', choice_names=("c", "d")) != self.samps['condsexp']
        condsstr_df = pandas.DataFrame({'c': [1, 1, 1], 'rt': [1, 2, 3], 'cond1': ["x", "yy", "z z z"]})
        assert ddm.Sample.from_pandas_dataframe(condsstr_df, 'rt', 'c', choice_names=("x", "Y with space")) == self.samps['condsstr']
    def test_to_pandas(self):
        for sname,s in self.samps.items():
            if s.undecided == 0:
                if sname == "condsstr":
                    assert s == ddm.Sample.from_pandas_dataframe(s.to_pandas_dataframe("a", "b"), "a", "b", choice_names=("x", "Y with space"))
                else:
                    assert s == ddm.Sample.from_pandas_dataframe(s.to_pandas_dataframe("a", "b"), "a", "b")
            else:
                assert len(s.choice_upper)+len(s.choice_lower) == len(s.to_pandas_dataframe("a", "b", drop_undecided=True))

class TestSolution(TestCase):
    def setUp(self):
        class DriftSimple(ddm.Drift):
            name = "Test drift"
            required_conditions = ['coher']
            required_parameters = []
            def get_drift(self, conditions, **kwargs):
                return conditions["coher"]
        class DriftSimpleStringArg(ddm.Drift):
            name = "Test drift"
            required_conditions = ['type']
            required_parameters = []
            def get_drift(self, conditions, **kwargs):
                if conditions['type'] == "a":
                    return .3
                else:
                    return .1
        class DriftSimpleTupleArg(ddm.Drift):
            name = "Test drift"
            required_conditions = ['cohs']
            required_parameters = []
            def get_drift(self, conditions, **kwargs):
                return conditions['cohs'][0]
        # No undecided
        self.quick_ana = ddm.Model(T_dur=2, dt=.02).solve_analytical()
        # Includes undecided
        self.quick_cn = ddm.Model(T_dur=.5).solve_numerical_cn()
        # Includes undecided
        self.quick_imp = ddm.Model(T_dur=.5).solve_numerical_implicit()
        # No undecided, with parameters
        self.params_ana = ddm.Model(drift=DriftSimple(), T_dur=2.5, dt=.005).solve_analytical({"coher": .3})
        # Includes undecided, with parameters
        self.params_cn = ddm.Model(drift=DriftSimple(), T_dur=.5).solve_numerical_cn(conditions={"coher": .1})
        # Includes undecided, with parameters
        self.params_imp = ddm.Model(drift=DriftSimple(), T_dur=.5).solve_numerical_implicit(conditions={"coher": .1})
        # Dependence with a string argument
        self.params_strarg = ddm.Model(drift=DriftSimpleStringArg(), T_dur=.5).solve_analytical(conditions={"type": "a"})
        # Dependence with a tuple argument
        self.params_tuplearg = ddm.Model(drift=DriftSimpleTupleArg(), T_dur=.5).solve_analytical(conditions={"cohs": (.2, 1, 2)})
        self.quick_cn_b = ddm.Model(T_dur=.5, choice_names=("a", "b b")).solve_numerical_cn()
        self.all_sols = [self.quick_ana, self.quick_cn, self.quick_imp, self.params_ana, self.params_cn, self.params_imp, self.params_strarg, self.params_tuplearg, self.quick_cn_b]
        # Includes undecided
    def test_pdfs(self):
        """Make sure we produce valid distributions from solutions"""
        # For each test model
        for s in self.all_sols:
            dt = s.dt
            # Distribution sums to 1
            assert np.isclose(fsum([fsum(s.pdf("_top"))*dt, fsum(s.pdf("_bottom"))*dt, s.prob_undecided()]), 1)
            # Correct and error probabilities are sensible
            assert np.isclose(fsum(s.pdf("_top")*dt), s.prob("_top"))
            assert np.isclose(fsum(s.pdf("_bottom")*dt), s.prob("_bottom"))
            if s.choice_names == ("correct", "error"):
                assert s.mean_decision_time() > 0
            if s.prob_undecided() == 0:
                assert s.prob(1) == s.prob_forced(1)
                assert s.prob(2) == s.prob_forced(0)
            # Signed probabilities sum to 1
            if s.undec is not None:
                assert np.isclose(np.sum(s.prob_sign("_top")) + np.sum(s.prob_sign("_bottom")), 1, rtol=.005)
                assert np.sum(s.prob_sign("_top")) + np.sum(s.prob_sign("_bottom")) <= 1
            # Correct time domain
            assert len(s.pdf("_top")) == len(s.t_domain)
        self.quick_cn_b.pdf("a")
        self.quick_cn_b.pdf("b b")
        self.quick_cn_b.cdf("a")
        self.quick_cn_b.cdf("_bottom")
        self.quick_cn_b.prob("a")
        self.quick_cn_b.prob("b b")
        self.quick_cn_b.prob_forced("b b")
        # pdf_undec with pdf and pdf bottom sum to one if pdf_undec exists
        for s in [self.quick_cn, self.quick_imp, self.params_cn, self.params_imp]:
            dx = s.dx
            if s.undec is not None:
                # Allow better tolerance since accuracy isn't perfect for undecided pdf
                assert np.isclose(fsum([fsum(s.pdf("_top"))*dt, fsum(s.pdf("_bottom"))*dt, fsum(s.pdf_undec())*dx]), 1, atol=.001)
    def test_evaluate(self):
        for s in self.all_sols:
            assert s.evaluate(.3, True) > 0
            assert s.evaluate(.1, False) > 0
            assert s.evaluate(100, "_top") == 0
            assert s.evaluate(100, "_bottom") == 0
        self.quick_cn_b.evaluate(100, "a") == 0
        self.quick_cn_b.evaluate(100, "a") == 0

class TestTriDiagMatrix(TestCase):
    def setUp(self):
        self.matrices = [ddm.tridiag.TriDiagMatrix.eye(1)*4.1, # For fully collapsing bounds
                         ddm.tridiag.TriDiagMatrix.eye(3),
                         ddm.tridiag.TriDiagMatrix(diag=np.asarray([1, 2, 3]),
                                                   up=np.asarray([5, 1]),
                                                   down=np.asarray([1, 2])),
                         ddm.tridiag.TriDiagMatrix(diag=np.asarray([1.1, 2.6, -3.1]),
                                                   up=np.asarray([50, 1.6]),
                                                   down=np.asarray([.1, 2.4]))]
        self.scalars = [5.4, 9, 0, 1, -6]
    def test_multiply(self):
        for m in self.matrices:
            for s in self.scalars:
                #assert not np.any(((m * s).to_scipy_sparse() != m.to_scipy_sparse().dot(s)).todense())
                assert not np.any(((m * s).to_scipy_sparse() != (m.to_scipy_sparse()*s)).todense())
            for m2 in self.matrices:
                if m.shape == m2.shape:
                    assert not np.any(((m.dot(m2)) != m.to_scipy_sparse().dot(m2.to_scipy_sparse())).todense())
                    assert not np.any((m * m2).to_scipy_sparse() != m.to_scipy_sparse().multiply(m2.to_scipy_sparse()).todense())
    def test_add_inplace(self):
        ms = [copy.deepcopy(m) for m in self.matrices]
        for m,mo in zip(ms, self.matrices):
            m *= 1.4
            m *= mo
            assert m == (mo * 1.4) * mo
    def test_add(self):
        for m in self.matrices:
            #for s in self.scalars:
            #    np.sum((m + s).to_scipy_sparse() != m.to_scipy_sparse() + s)
            for m2 in self.matrices:
                if m.shape == m2.shape:
                    assert not np.any(((m + m2).to_scipy_sparse() != (m.to_scipy_sparse() + m2.to_scipy_sparse())).todense())
    def test_add_r(self):
        for m in self.matrices:
            #for s in self.scalars:
            #    np.sum((s + m).to_scipy_sparse() != s + m.to_scipy_sparse())
            for m2 in self.matrices:
                if m.shape == m2.shape:
                    assert not np.any(((m2 + m).to_scipy_sparse() != (m2.to_scipy_sparse() + m.to_scipy_sparse())).todense())
    def test_add_inplace(self):
        ms = [copy.deepcopy(m) for m in self.matrices]
        for m,mo in zip(ms, self.matrices):
            m += 1.4
            m += mo
            assert not m != (mo + 1.4) + mo
    def test_subtract(self):
        for m in self.matrices:
            #for s in self.scalars:
            #    np.sum((m - s).to_scipy_sparse() != m.to_scipy_sparse() + -s)
            for m2 in self.matrices:
                if m.shape == m2.shape:
                    assert not np.any(((m - m2).to_scipy_sparse() != (m.to_scipy_sparse() - m2.to_scipy_sparse())).todense())
    def test_subtract_r(self):
        for m in self.matrices:
            #for s in self.scalars:
            #    np.sum((s - m).to_scipy_sparse() != s - m.to_scipy_sparse())
            for m2 in self.matrices:
                if m.shape == m2.shape:
                    assert not np.any(((m2 - m).to_scipy_sparse() != (m2.to_scipy_sparse() - m.to_scipy_sparse())).todense())
    def test_subtract_inplace(self):
        ms = [copy.deepcopy(m) for m in self.matrices]
        for m,mo in zip(ms, self.matrices):
            m -= 1.4
            m -= mo
            assert not m != (mo - 1.4) - mo

class TestMisc(TestCase):
    def test_analytic_lin_collapse(self):
        """Make sure linearly collapsing bounds stops at 0"""
        # Will collapse to 0 by t=1
        b = ddm.models.bound.BoundCollapsingLinear(B=1, t=1)
        m = ddm.Model(bound=b, T_dur=2)
        s = m.solve_analytical()
        assert len(s.pdf("_top")) == len(m.t_domain())
        s = m.solve_analytical(force_python=True)
        assert len(s.pdf("_top")) == len(m.t_domain())
    def test_get_set_parameters_functions(self):
        """Test get_parameters, set_parameters, and get_parameter_names"""
        p1 = ddm.Fittable(minval=0, maxval=1)
        p2 = ddm.Fittable(minval=.3, maxval=.9, default=.4)
        m = ddm.Model(drift=ddm.DriftConstant(drift=p1), noise=ddm.NoiseLinear(noise=p2, x=.2, t=p1))
        print(m.get_model_parameters())
        assert all(id(a) == id(b) for a,b in zip(m.get_model_parameters(), [p1, p2]))
        assert all(a == b for a,b in zip(m.get_model_parameter_names(), ["drift/t", "noise"]))
        m.set_model_parameters(m.get_model_parameters())
        assert all(id(a) == id(b) for a,b in zip(m.get_model_parameters(), [p1, p2]))
        m.set_model_parameters([.5, .5])
        assert all(a == b for a,b in zip(m.get_model_parameters(), [.5, .5]))
    def test_fittable_kwargs(self):
        assert repr(ddm.Fittable(0,1)) == repr(ddm.Fittable(maxval=1, minval=0))
        assert repr(ddm.Fittable(0,1,.5)) == repr(ddm.Fittable(maxval=1, minval=0, default=.5))
        assert repr(ddm.Fittable(-np.inf, np.inf)) == repr(ddm.Fittable())
        f = ddm.Fittable(0,1,.5)
        assert repr(f) == repr(copy.copy(f))
        assert repr(f) == repr(copy.deepcopy(f))
        f = ddm.Fitted(0,minval=1,maxval=.5)
        assert repr(f) == repr(copy.copy(f))
        assert repr(f) == repr(copy.deepcopy(f))
        fails(lambda : ddm.Fittable(3))
        fails(lambda : ddm.Fittable(3,4,5,6))
    def test_model_stuff(self):
        m = ddm.Model(choice_names=("Aaa", "b b"), dx=.002, dt=.002, T_dur=5, drift=ddm.DriftConstant(drift=3), name='xxx')
        s = m.solve()
        assert m.choice_names == s.choice_names
        samp = s.resample(10)
        assert s.choice_names == m.choice_names
        str(m)
        from pyddm import Model, DriftConstant, NoiseConstant, BoundConstant, ICPointSourceCenter, OverlayNone
        assert m == eval(repr(m))
    def test_solve_all_conditions_parameterizations(self):
        class DriftCond(ddm.Drift):
            name = "Simple drift requires two conditions"
            required_conditions = ["c1", "c2"]
            required_parameters = []
            def get_drift(self, conditions, **kwargs):
                return conditions["c1"] * conditions["c2"]
        m = ddm.Model(drift=DriftCond())
        cond_combs = [{"c1": 2, "c2": 0.9}, {"c1": 2, "c2": 1.0}, {"c1": 2, "c2": 0.9}, {"c1": 2, "c2": 1.0}]
        samp_arr = np.array([[1, 0, conditions["c1"], conditions["c2"]] for conditions in cond_combs])
        samp = ddm.Sample.from_numpy_array(samp_arr, ["c1", "c2"])
        sols1 = ddm.functions.solve_all_conditions(m, sample=samp)
        sols2 = ddm.functions.solve_all_conditions(m, condition_combinations=cond_combs)
        for s1, s2 in zip(sols1.values(), sols2.values()):
            assert np.all(np.isclose(s1.pdf("_top"), s2.pdf("_top"), atol=1e-3, rtol=1e-3)), "solve_all_conditions parameterizations differed (correct RT)"
            assert np.all(np.isclose(s1.pdf("_bottom"), s2.pdf("_bottom"), atol=1e-3, rtol=1e-3)), "solve_all_conditions parameterizations differed (error RT)"
        with self.assertRaises(AssertionError):
            ddm.functions.solve_all_conditions(m, condition_combinations=cond_combs + [{"c1": 1, "c2": 1.0, "dummy": -1}])

class TestCSolver(TestCase):
    def test_numerical(self):
        assert ddm.model.HAS_CSOLVE, "C extension build failed"
        # Test all code paths
        models = [
            ddm.Model(),
            ddm.Model(bound=ddm.BoundCollapsingExponential(B=2.5, tau=.5)),
            ddm.Model(drift=ddm.DriftLinear(x=0, t=.5, drift=0)),
            ddm.Model(drift=ddm.DriftLinear(x=-.5, t=0, drift=.1)),
            ddm.Model(drift=ddm.DriftLinear(x=.5, t=.5, drift=.2), bound=ddm.BoundCollapsingExponential(B=1, tau=1)),
            ddm.Model(noise=ddm.NoiseLinear(x=0, t=.2, noise=.5)),
            ddm.Model(IC=ddm.ICPoint(x0=.3)),
            ddm.Model(IC=ddm.ICPointRatio(x0=.8), bound=ddm.BoundConstant(B=.3)),
            ddm.Model(noise=ddm.NoiseLinear(x=-.2, t=0, noise=.4), choice_names=("left", "right")),
            ddm.Model(noise=ddm.NoiseLinear(x=.2, t=.2, noise=.6), bound=ddm.BoundCollapsingLinear(B=1, t=1)),
            ]
        for i,m in enumerate(models):
            print(i)
            s1 = m.solve_numerical_implicit(force_python=True)
            s2 = m.solve_numerical_c()
            assert np.all(np.isclose(s1.pdf("_top"), s2.pdf("_top"), atol=2e-2, rtol=1e-2)), "Testing model id " + str(i)
            assert np.all(np.isclose(s1.pdf("_bottom"), s2.pdf("_bottom"), atol=2e-2, rtol=1e-2)), "Testing model id " + str(i)
    def test_analytic(self):
        assert ddm.analytic.HAS_CSOLVE, "C extension build failed"
        models = [
            ddm.Model(),
            ddm.Model(bound=ddm.BoundCollapsingLinear(B=2.5, t=2)),
            ddm.Model(bound=ddm.BoundCollapsingLinear(B=1.0, t=2)),
            ddm.Model(IC=ddm.ICPoint(x0=.3)),
            ddm.Model(IC=ddm.ICPoint(x0=.3), bound=ddm.BoundCollapsingLinear(B=1.0, t=2)),
            ]
        for i,m in enumerate(models):
            s1 = m.solve_analytical(force_python=True)
            s2 = m.solve_analytical(force_python=False)
            assert np.all(np.isclose(s1.pdf("_top"), s2.pdf("_top"), atol=1e-3, rtol=1e-3)), "Testing model id " + str(i)
            assert np.all(np.isclose(s1.pdf("_bottom"), s2.pdf("_bottom"), atol=1e-3, rtol=1e-3)), "Testing model id " + str(i)



# TODO test if there is no overlay, then corr + err + undecided = 1
# TODO test bounds that don't depend on t but do depend on conditions, mus like that, etc.
# TODO test solution.resample in integration testing
# TODO test loss parallelization?
