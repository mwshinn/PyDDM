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


def _modeltest_pdf_evolution(m, conditions={}, max_diff=.1, max_deviation=.01):
    sol_with_evolution = m.solve_numerical_implicit(conditions=conditions, return_evolution=True)    
    sol_without_evolution = np.zeros((len(sol_with_evolution.model.x_domain(conditions)), len(sol_with_evolution.model.t_domain())))          
    sol_without_evolution[:,0] = m.IC(conditions=conditions)/m.dx
    for t_ind, t in enumerate(sol_with_evolution.model.t_domain()[1:]):
        T_dur_backup = m.T_dur
        m.T_dur = t
        sol = m.solve_numerical_implicit(conditions=conditions, return_evolution=False) 
        m.T_dur = T_dur_backup
        print("Shapes:", sol_without_evolution.shape, sol.pdf_undec().shape)
        sol_without_evolution[:,t_ind+1] = sol.pdf_undec()
    difference = sol_with_evolution.pdf_evolution() - sol_without_evolution
    max_difference = np.max(np.abs(difference))
    print(max_difference)
    sums = np.array([np.sum(sol_with_evolution.pdf_corr()[0:t]*m.dt) + np.sum(sol_with_evolution.pdf_err()[0:t]*m.dt) + np.sum(sol_with_evolution.pdf_evolution()[:,t]*m.dx) for t in range(1,len(sol_with_evolution.model.t_domain()))])
    print(np.max(np.abs(sums-1)))
    assert max_difference < max_diff, "Maximum distance between pdf evolutions was too high"
    assert np.max(np.abs(sums-1)) < max_deviation, "PDF does not sum up to 1"


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
            def get_noise(self, conditions, **kwargs):
                return conditions["cond"]
        self.withcond = ddm.Model(noise=NoiseCond())
        class FancyBounds(ddm.Bound):
            name = "Increasing/decreasing bounds"
            required_conditions = []
            required_parameters = []
            def get_bound(self, conditions, t, **kwargs):
                if t <= 1:
                    return 1 + t
                if t > 1:
                    return 2/t
        self.bound = ddm.Model(bound=FancyBounds())
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
    def test_collapsing_bounds(self):
        """Bounds collapse to zero"""
        m = ddm.Model(bound=ddm.BoundCollapsingLinear(B=1, t=2))
        _modeltest_numerical_vs_analytical(m, method="implicit", max_diff=.3, mean_diff=.2, prob_diff=.05)
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
        _modeltest_numerical_vs_analytical(self.withcond, method="cn", conditions={"cond": .2})
        _modeltest_numerical_vs_analytical(self.withcond, method="cn", conditions={"cond": .6})
    def test_bounds(self):
        self.bound.solve()
    def test_pdf_evolution(self):
        """PDF evolution in simple DDM"""
        _modeltest_pdf_evolution(self.basic)
        # Doesn't work here, but that's okay. In general, pdf_undec
        # implicitly determines size based on x_domain, which accounts
        # for increasing bounds by maximizing over t_domain.  But for
        # testing purposes here, we vary T_dur, which changes
        # t_domain, thus making the function return a different
        # maximum.
        # 
        # _modeltest_pdf_evolution(self.bound)
    def test_ICPoint(self):
        """Arbitrary pointwise initial condition"""
        m = ddm.Model(name='ICPoint_test',
              drift=ddm.DriftConstant(drift=2),
              noise=ddm.NoiseConstant(noise=1.5),
              bound=ddm.BoundConstant(B=1),
              IC=ddm.ICPoint(x0=-.25))
        _modeltest_numerical_vs_analytical(m, method="implicit", max_diff=.3, mean_diff=.2, prob_diff=.05)
    def test_ICPoint_collapsing_bounds(self):
        m = ddm.Model(name='ICPoint_BCollapsingLin_test',
              drift=ddm.DriftConstant(drift=2),
              noise=ddm.NoiseConstant(noise=1.5),
              bound=ddm.BoundCollapsingLinear(B=1,t=0.5),
              IC=ddm.ICPoint(x0=-.25))
        _modeltest_numerical_vs_analytical(m, method="implicit", max_diff=.3, mean_diff=.2, prob_diff=.05)


class TestFit(TestCase):
    def setUp(self):
        from integration_test_models import DriftCond
        self.DriftCond = DriftCond
        self.cond_m = ddm.Model(drift=self.DriftCond(param=1))
        self.cond_s = self.cond_m.solve(conditions={"cond": .1}).resample(4000) + \
                      self.cond_m.solve(conditions={"cond": 1}).resample(4000) + \
                      self.cond_m.solve(conditions={"cond": 2}).resample(4000)
    def test_fit_drift(self):
        """A simple one-parameter fit"""
        m = ddm.Model(name="DDM", drift=ddm.DriftConstant(drift=2))
        s = m.solve()
        sample = s.resample(10000)
        mfit = ddm.Model(name="DDM", drift=ddm.DriftConstant(drift=ddm.Fittable(minval=0, maxval=10)))
        ddm.fit_adjust_model(model=mfit, sample=sample)
        # Within 10%
        if SHOW_PLOTS:
            mfit.name = "Fitted solution"
            sfit = mfit.solve()
            plot_compare_solutions(s, sfit)
            plt.show()
        _verify_param_match("drift", "drift", m, mfit)
    def test_fit_with_condition(self):
        """A simple one-parameter fit with conditions"""
        m = self.cond_m
        s = self.cond_s
        mfit = ddm.Model(drift=self.DriftCond(param=ddm.Fittable(minval=.1, maxval=3)))
        ddm.fit_adjust_model(model=mfit, sample=s)
        # Within 10%
        if SHOW_PLOTS:
            mfit.name = "Fitted solution"
            sfit = mfit.solve()
            plot_compare_solutions(s, sfit)
            plt.show()
        _verify_param_match("drift", "param", m, mfit)
    def test_fit_with_condition_parallel(self):
        """A simple one-parameter fit with conditions, parallelized"""
        ddm.set_N_cpus(2)
        self.test_fit_with_condition()
        ddm.set_N_cpus(1)
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

