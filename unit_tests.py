import unittest
from unittest import TestCase, main
from string import ascii_letters
import numpy as np
from itertools import groupby

import ddm

def fails(f):
    failed = False
    try:
        f()
    except:
        failed = True
    if failed == False:
        raise ValueError("Error, function did not fail")

class TestDependences(TestCase):
    def setUp(self):
        class FakeUniformModel(ddm.Model):
            def solve(self, conditions={}, *args, **kwargs):
                corr = self.t_domain()*0+.4/len(self.t_domain())
                err = self.t_domain()*0+.4/len(self.t_domain())
                undec = self.x_domain(conditions=conditions)*0+.2/len(self.x_domain(conditions=conditions))
                return ddm.Solution(corr, err, self, conditions, undec)
        FakeUniformModel.solve_analytical = FakeUniformModel.solve
        FakeUniformModel.solve_numerical = FakeUniformModel.solve
        FakeUniformModel.solve_numerical_cn = FakeUniformModel.solve
        FakeUniformModel.solve_numerical_implicit = FakeUniformModel.solve
        FakeUniformModel.solve_numerical_explicit = FakeUniformModel.solve
        self.FakeUniformModel = FakeUniformModel
        class FakePointModel(ddm.Model):
            def solve(self, conditions={}, *args, **kwargs):
                corr = self.t_domain()*0
                corr[1] = .8
                err = self.t_domain()*0
                err[1] = .2
                return ddm.Solution(corr, err, self, conditions)
        FakePointModel.solve_analytical = FakePointModel.solve
        FakePointModel.solve_numerical = FakePointModel.solve
        FakePointModel.solve_numerical_cn = FakePointModel.solve
        FakePointModel.solve_numerical_implicit = FakePointModel.solve
        FakePointModel.solve_numerical_explicit = FakePointModel.solve
        self.FakePointModel = FakePointModel
        class FakeUndecidedModel(ddm.Model):
            def solve(self, conditions={}, *args, **kwargs):
                corr = self.t_domain()*0
                err = self.t_domain()*0
                undec = self.x_domain(conditions=conditions)*0+1/len(self.x_domain(conditions=conditions))
                return ddm.Solution(corr, err, self, conditions, undec)
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
            
    def test_DriftReduces(self):
        """Make sure DriftLinear reduces to DriftConstant when x and t are 0"""
        drift_constant_instances = [e for e in ddm.models.DriftConstant._generate()]
        for cinst in drift_constant_instances:
            linst = ddm.models.DriftLinear(drift=cinst.get_drift(t=0), x=0, t=0)
            for t in [0, .1, .5, 1, 2, 10]:
                assert linst.get_drift(t=t, x=1) == cinst.get_drift(t=t, x=1)
    def test_NoiseReduces(self):
        """Make sure NoiseLinear reduces to NoiseConstant when x and t are 0"""
        noise_constant_instances = [e for e in ddm.models.NoiseConstant._generate()]
        for cinst in noise_constant_instances:
            linst = ddm.models.NoiseLinear(noise=cinst.get_noise(t=0), x=0, t=0)
            for t in [0, .1, .5, 1, 2, 10]:
                assert linst.get_noise(t=t, x=1) == cinst.get_noise(t=t, x=1)
    def test_ICArbitrary(self):
        """Test arbitrary starting conditions from a distribution"""
        # Make sure we get out the same distribution we put in
        m = ddm.Model()
        unif = ddm.models.ICUniform()
        unif_a = ddm.models.ICArbitrary(unif.get_IC(m.x_domain({})))
        assert np.all(unif.get_IC(m.x_domain({})) == unif_a.get_IC(m.x_domain({})))
        point = ddm.models.ICPointSourceCenter()
        point_a = ddm.models.ICArbitrary(point.get_IC(m.x_domain({})))
        assert np.all(point.get_IC(m.x_domain({})) == point_a.get_IC(m.x_domain({})))
        # Make sure the distribution integrates to 1
        fails(lambda : ddm.models.ICArbitrary(np.asarray([.1, .1, 0, 0, 0])))
        fails(lambda : ddm.models.ICArbitrary(np.asarray([0, .6, .6, 0])))
        assert ddm.models.ICArbitrary(np.asarray([1]))
    def test_OverlayNone(self):
        s = ddm.Model().solve()
        assert s == ddm.models.OverlayNone().apply(s)
        s = self.FakeUniformModel().solve()
        assert s == ddm.models.OverlayNone().apply(s)
        s = self.FakePointModel().solve()
        assert s == ddm.models.OverlayNone().apply(s)
    def test_OverlayUniformMixture(self):
        # Do nothing with 0 probability
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=1)).solve()
        smix = ddm.models.OverlayUniformMixture(umixturecoef=0).apply(s)
        assert s == smix
        # With mixture coef 1, integrate to 1
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=2), noise=ddm.models.NoiseConstant(noise=3)).solve()
        smix = ddm.models.OverlayUniformMixture(umixturecoef=1).apply(s)
        assert np.isclose(np.sum(smix.corr) + np.sum(smix.err), 1)
        # Should not change uniform distribution
        s = self.FakeUniformModel(dt=.001).solve()
        assert s == ddm.models.OverlayUniformMixture(umixturecoef=.2).apply(s)
        # Don't change total probability
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=1)).solve()
        smix = ddm.models.OverlayUniformMixture(umixturecoef=.2).apply(s)
        assert np.isclose(np.sum(s.corr) + np.sum(s.err),
                          np.sum(smix.corr) + np.sum(smix.err))
    def test_OverlayPoissonMixture(self):
        # Do nothing with mixture coef 0
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=1)).solve()
        smix = ddm.models.OverlayPoissonMixture(pmixturecoef=0, rate=1).apply(s)
        assert s == smix
        # With mixture coef 1, integrate to 1
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=2), noise=ddm.models.NoiseConstant(noise=3)).solve()
        smix = ddm.models.OverlayPoissonMixture(pmixturecoef=1, rate=10).apply(s)
        assert np.isclose(np.sum(smix.corr) + np.sum(smix.err), 1)
        # Should be monotonic decreasing on uniform distribution
        s = self.FakeUniformModel(dt=.001).solve()
        smix = ddm.models.OverlayPoissonMixture(pmixturecoef=.2, rate=1).apply(s)
        assert np.all([smix.corr[i-1]-smix.corr[i] > 0 for i in range(1, len(smix.corr))])
        assert np.all([smix.err[i-1]-smix.err[i] > 0 for i in range(1, len(smix.err))])
        # Don't change total probability
        s = ddm.Model(ddm.models.DriftConstant(drift=1)).solve()
        smix = ddm.models.OverlayPoissonMixture(pmixturecoef=.2, rate=7).apply(s)
        assert np.isclose(np.sum(s.corr) + np.sum(s.err),
                          np.sum(smix.corr) + np.sum(smix.err))
    def test_OverlayNonDecision(self):
        # Should do nothing with no shift
        s = ddm.Model().solve()
        assert s == ddm.models.OverlayNonDecision(nondectime=0).apply(s)
        # Shifts a single point distribution
        s = self.FakePointModel(dt=.01).solve()
        sshift = ddm.models.OverlayNonDecision(nondectime=.01).apply(s)
        assert s.corr[1] == sshift.corr[2]
        assert s.err[1] == sshift.err[2]
        # Truncate when time bin doesn't align
        s = self.FakePointModel(dt=.01).solve()
        sshift = ddm.models.OverlayNonDecision(nondectime=.019).apply(s)
        assert s.corr[1] == sshift.corr[2]
        assert s.err[1] == sshift.err[2]
    def test_OverlaySimplePause(self):
        # Should do nothing with no shift
        s = ddm.Model().solve()
        assert s == ddm.models.OverlaySimplePause(pausestart=.4, pausestop=.4).apply(s)
        # Shift should make a gap in the uniform model
        s = self.FakeUniformModel().solve()
        smix = ddm.models.OverlaySimplePause(pausestart=.3, pausestop=.6).apply(s)
        assert len(set(smix.corr).union(set(smix.err))) == 2
        assert len(list(groupby(smix.corr))) == 3 # Looks like ----____----------
        # Should start with 0 and then go to constant with pausestart=.3
        s = self.FakeUniformModel(dt=.01).solve()
        smix = ddm.models.OverlaySimplePause(pausestart=0, pausestop=.05).apply(s)
        assert len(set(smix.corr).union(set(smix.err))) == 2
        assert len(list(groupby(smix.corr))) == 2 # Looks like ____----------
        assert np.all(smix.corr[0:5] == 0) and smix.corr[6] != 0
        # Truncate when time bin doesn't align
        s = self.FakePointModel(dt=.01).solve()
        sshift = ddm.models.OverlaySimplePause(pausestart=.01, pausestop=.029).apply(s)
        assert s.corr[1] == sshift.corr[2]
        assert s.err[1] == sshift.err[2]
    def test_OverlayBlurredPause(self):
        # Don't change total probability when there are no undecided responses
        s = ddm.Model(drift=ddm.models.DriftConstant(drift=1), T_dur=10).solve()
        smix = ddm.models.OverlayBlurredPause(pausestart=.3, pausestop=.6, pauseblurwidth=.1).apply(s)
        assert np.isclose(np.sum(s.corr) + np.sum(s.err),
                          np.sum(smix.corr) + np.sum(smix.err))
        # Make sure responses before the pause aren't affected
        s = self.FakePointModel(dt=.01).solve()
        sshift = ddm.models.OverlayBlurredPause(pausestart=.02, pausestop=.03, pauseblurwidth=.002).apply(s)
        assert s.corr[1] == sshift.corr[1] != 0
        assert s.err[1] == sshift.err[1] != 0
        # Make sure responses after look like a gamma distribution
        s = self.FakePointModel(dt=.01).solve()
        sshift = ddm.models.OverlayBlurredPause(pausestart=0, pausestop=.05, pauseblurwidth=.01).apply(s)
        positive = (sshift.corr[2:] > sshift.err[1:-1]).astype(int) # Excluding first 0 point, should go from + to - slope only once
        assert positive[0] == 1 and positive[-1] == 0 and len(set(positive)) == 2
    def test_OverlayChain(self):
        # Combine with OverlayNone()
        s = self.FakePointModel(dt=.01).solve()
        o = ddm.models.OverlayChain(overlays=[
                ddm.models.OverlayNone(),
                ddm.models.OverlayNonDecision(nondectime=.01),
                ddm.models.OverlayNone()])
        sshift = o.apply(s)
        assert s.corr[1] == sshift.corr[2]
        assert s.err[1] == sshift.err[2]
    def test_LossSquaredError(self):
        # Should be zero for empty sample when all undecided
        m = self.FakeUndecidedModel()
        s = ddm.Sample(np.asarray([]), np.asarray([]), undecided=1)
        assert ddm.models.LossSquaredError(sample=s, dt=m.dt, T_dur=m.T_dur).loss(m) == 0
        # Can also be determined precisely for the point model
        m = self.FakePointModel()
        sol = m.solve()
        err = ddm.models.LossSquaredError(sample=s, dt=m.dt, T_dur=m.T_dur).loss(m)
        assert np.isclose(err, np.sum(sol.corr)**2 + np.sum(sol.err)**2)
    def test_LossLikelihood(self):
        # We can calculate likelihood for this simple case
        m = self.FakePointModel(dt=.02)
        sol = m.solve()
        s = ddm.Sample(np.asarray([.02]), np.asarray([]))
        expected = -np.log(np.sum(sol.corr)/m.dt)
        assert np.isclose(expected, ddm.models.LossLikelihood(sample=s, dt=m.dt, T_dur=m.T_dur).loss(m))
        # And for the uniform case we can assert equivalence
        m = self.FakeUniformModel()
        s1 = ddm.Sample(np.asarray([.02, .05, .07, .12]), np.asarray([.33, .21]))
        s2 = ddm.Sample(np.asarray([.13, .1, .02]), np.asarray([.66, .15, .89]))
        assert np.isclose(ddm.models.LossLikelihood(sample=s1, dt=m.dt, T_dur=m.T_dur).loss(m),
                          ddm.models.LossLikelihood(sample=s2, dt=m.dt, T_dur=m.T_dur).loss(m))
        # TODO I think this reveals we should be doing
        # (len(x_domain())-1) instead of len(x_domain()).  Multiple of 2 somewhere.
        # And it should not depend on dt since it is comparing to the pdf
        # m1 = self.FakeUniformModel(dt=.02)
        # m2 = self.FakeUniformModel(dt=.01)
        # print(m1.solve().pdf_corr(), m2.solve().pdf_corr())
        # s = ddm.Sample(np.asarray([.14, .1, .01]), np.asarray([.66, .16, .89]))
        # assert np.isclose(ddm.models.LossLikelihood(sample=s, dt=m1.dt, T_dur=m1.T_dur).loss(m1),
        #                   ddm.models.LossLikelihood(sample=s, dt=m2.dt, T_dur=m2.T_dur).loss(m2))


# TODO test if there is no overlay, then corr + err + undecided = 1
# TODO test bounds that don't depend on t but do depend on conditions, mus like that, etc.
