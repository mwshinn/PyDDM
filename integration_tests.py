import unittest
from unittest import TestCase, main
import numpy as np
from math import fsum
import pandas

import ddm

SHOW_PLOTS = False

if SHOW_PLOTS:
    import ddm.plot
    import matplotlib.pyplot as plt

def _modeltest_numerical_vs_analytical(m, conditions={}, method=None, max_diff=.1, mean_diff=.05, prob_diff=.01):
    a = m.solve_analytical(conditions=conditions)
    if method is None:
        n = m.solve_numerical(conditions=conditions)
    elif method == "cn":
        n = m.solve_numerical_cn(conditions=conditions)
    elif method == "implicit":
        n = m.solve_numerical_implicit(conditions=conditions)
    elif method == "explicit":
        n = m.solve_numerical_explicit(conditions=conditions)
    if SHOW_PLOTS:
        ddm.plot.plot_solution_pdf(a)
        ddm.plot.plot_solution_pdf(n)
        plt.show()
    max_difference = np.max(np.abs(a.pdf_corr() - n.pdf_corr()))
    mean_difference = np.sum(np.abs(a.pdf_corr() - n.pdf_corr()))/len(m.t_domain())
    print(max_difference, mean_difference)
    assert max_difference < max_diff, "Maximum distance between correct distributions was too high"
    assert mean_difference < mean_diff, "Mean distance between correct distributions was too high"
    max_difference = np.max(np.abs(a.pdf_err() - n.pdf_err()))
    mean_difference = np.sum(np.abs(a.pdf_err() - n.pdf_err()))/len(m.t_domain())
    assert max_difference < max_diff, "Maximum distance between error distributions was too high"
    assert mean_difference < mean_diff, "Mean distance between error distributions was too high"
    assert abs(a.prob_correct() - n.prob_correct()) < prob_diff, "Correct probability was too different"
    assert abs(a.prob_error() - n.prob_error()) < prob_diff, "Error probability was too different"
    assert abs(a.prob_undecided() - n.prob_undecided()) < prob_diff, "Undecided probability was too different"


def _verify_param_match(dependence, parameter, m1, m2, tol=.1):
    p1 = getattr(m1.get_dependence(dependence), parameter)
    p2 = getattr(m2.get_dependence(dependence), parameter)
    assert abs(p1 - p2) < 0.1 * p1, "%s param from %s dependence doesn't match: %.4f != %.4f" % (parameter, dependence, p1, p2)

class TestSimulation(TestCase):
    """Numerical solutions should be close to analytical solutions"""
    def setUp(self):
        self.basic = ddm.Model(dx=.005, dt=.01, T_dur=2,
                               drift=ddm.DriftConstant(drift=.4),
                               noise=ddm.NoiseConstant(noise=1),
                               bound=ddm.BoundConstant(B=1))
        class NoiseCond(ddm.Noise):
            name = "Noise with a condition"
            required_conditions = ['cond']
            required_parameters = []
            def get_noise(self, **kwargs):
                return cond
        self.withcond = ddm.Model(noise=NoiseCond())
    def test_basic_cn(self):
        """Simple DDM, Crank-Nicolson"""
        _modeltest_numerical_vs_analytical(self.basic, method="cn")
    def test_basic_implicit(self):
        """Simple DDM"""
        _modeltest_numerical_vs_analytical(self.basic, method="implicit")
    def test_basic_explicit(self):
        """Simple DDM with explicit method.  For a reasonable runtime we need terrible numerics"""
        prev_dx = self.basic.dx
        prev_dt = self.basic.dt
        self.basic.dx = .05
        self.basic.dt = .001
        _modeltest_numerical_vs_analytical(self.basic, method="explicit",
                                           max_diff=.3, mean_diff=.2, prob_diff=.05)
        self.basic.dx = prev_dx
        self.basic.dt = prev_dt
    def test_overlay_chain_distribution_integrates_to_1(self):
        """Overlays integrate to 1"""
        m = ddm.Model(name="Overlay_test", drift=ddm.DriftConstant(drift=2), T_dur=5,
                      overlay=ddm.OverlayChain(overlays=[ddm.OverlayPoissonMixture(pmixturecoef=.2, rate=2),
                                                         ddm.OverlayUniformMixture(umixturecoef=.2),
                                                         ddm.OverlayNonDecision(nondectime=.2)]))
        s = m.solve()
        distsum = s.prob_correct() + s.prob_error()
        assert .99 < distsum < 1.0001, "Distribution doesn't sum to 1"
    def test_with_condition(self):
        """With conditions"""
        _modeltest_numerical_vs_analytical(self.basic, method="cn", conditions={"cond": .2})
        _modeltest_numerical_vs_analytical(self.basic, method="cn", conditions={"cond": .6})


class TestFit(TestCase):
    def test_fit_drift(self):
        """A simple one-parameter fit"""
        m = ddm.Model(name="DDM", drift=ddm.DriftConstant(drift=2))
        s = m.solve()
        sample = s.resample(10000)
        mfit = ddm.Model(name="DDM", drift=ddm.DriftConstant(drift=ddm.Fittable(minval=0, maxval=10)))
        ddm.fit_adjust_model(m=mfit, sample=sample)
        # Within 10%
        if SHOW_PLOTS:
            mfit.name = "Fitted solution"
            sfit = mfit.solve()
            plot_compare_solutions(s, sfit)
            plt.show()
        _verify_param_match("drift", "drift", m, mfit)
    def test_double_fit(self):
        """Fit different parameters in the same (or a different) model using a single Fittable object"""
        class NoiseDouble(ddm.Noise):
            name = "time-varying noise"
            required_parameters = ["noise1", "noise2"]
            def get_noise(self, **kwargs):
                if np.random.rand() > .5:
                    return self.noise1
                else:
                    return self.noise2
        class NoiseConstantButNot(ddm.Noise): # To avoid the numerical simulations
            name = "almost noise constant"
            required_parameters = ["noise"]
            def get_noise(self, **kwargs):
                return self.noise
        # Generate data
        m = ddm.Model(name="DDM",
                  drift=ddm.DriftConstant(drift=1),
                  noise=ddm.NoiseConstant(noise=1.7))
        s = m.solve_numerical() # Solving analytical and then fitting numerical may give a bias
        sample = s.resample(10000)
        mone = ddm.fit_model(sample, drift=ddm.DriftConstant(drift=1),
                             noise=NoiseConstantButNot(noise=ddm.Fittable(minval=.5, maxval=3)))
        sigs = ddm.Fittable(minval=.5, maxval=3)
        msam = ddm.fit_model(sample, drift=ddm.DriftConstant(drift=1),
                             noise=NoiseDouble(noise1=sigs,
                                               noise2=sigs))
        assert msam._noisedep.noise1 == msam._noisedep.noise2, "Fitting to be the same failed"
        assert abs(msam._noisedep.noise1 - mone._noisedep.noise) < 0.1 * mone._noisedep.noise

