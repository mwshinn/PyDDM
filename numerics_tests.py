"""Conservation and accuracy regressions for the Fokker--Planck solvers."""
import unittest
from unittest.mock import patch
import pickle
import subprocess
import sys

import numpy as np

import pyddm as ddm
from pyddm.analytic import analytic_ddm


def solve(model, method, python=True):
    return model.solve_numerical(method=method, force_python=python)


def cn_density_error(solution, reference):
    # CN's first startup node remains quadrature weighted; subsequent nodal
    # PDF values, including the final one, are unweighted.
    reference = reference.copy()
    reference[:, 0] *= .5
    return solution.dt*np.abs(np.array([solution.pdf('_top'), solution.pdf('_bottom')])
                              - reference).sum()


class ZeroBound(ddm.Bound):
    name = 'Already collapsed'
    required_parameters = []

    def get_bound(self, **kwargs):
        return 0.


class IncreasingBound(ddm.Bound):
    name = 'Increasing test boundary'
    required_parameters = []

    def get_bound(self, t, **kwargs):
        return .3+.3*t


class TestFPConservation(unittest.TestCase):
    def test_solution_saved_before_pdf_corrections_can_be_loaded(self):
        solution = ddm.Model(dt=.01, T_dur=.5).solve_numerical('cn')
        # Older pickles contain neither attribute.
        del solution._pdf_correction
        del solution._choice_rescale
        restored = pickle.loads(pickle.dumps(solution))
        np.testing.assert_allclose(restored.pdf('correct'), restored.choice_upper/restored.dt)
        self.assertEqual(len(restored.sample(10, seed=1)), 10)

    def test_cn_final_pdf_is_unweighted_but_probabilities_are_conserved(self):
        model = ddm.Model(dx=.005, dt=.005, T_dur=.5)
        reference = model.solve_analytical()
        results = []
        for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
            solution = solve(model, 'cn', python)
            results.append(solution)
            for choice, mass in [('correct', solution.choice_upper), ('error', solution.choice_lower)]:
                self.assertAlmostEqual(solution.pdf(choice)[-1], reference.pdf(choice)[-1], delta=1e-5)
                self.assertAlmostEqual(solution.evaluate(.5, choice), reference.pdf(choice)[-1], delta=1e-5)
                self.assertAlmostEqual(mass[-1], .5*solution.pdf(choice)[-1]*model.dt, delta=1e-14)
                self.assertAlmostEqual(solution.cdf(choice)[-1], solution.prob(choice), delta=1e-14)
            self.assertAlmostEqual(solution.prob('correct') + solution.prob('error')
                                   + solution.prob_undecided(), 1, delta=1e-14)
            self.assertEqual(len(solution.sample(1000, seed=2)), 1000)
        if len(results) == 2:
            np.testing.assert_allclose(results[0].pdf('correct'), results[1].pdf('correct'), atol=1e-12, rtol=1e-10)
            np.testing.assert_allclose(results[0].choice_upper, results[1].choice_upper, atol=1e-12, rtol=1e-10)

    def test_cn_last_step_metadata_with_zero_final_flux(self):
        class StoppingNoise(ddm.Noise):
            name = 'Noise stops at the final time'
            required_parameters = []
            def get_noise(self, t, **kwargs): return 1. if t < .5 else 0.

        model = ddm.Model(dx=.02, dt=.01, T_dur=.5, noise=StoppingNoise())
        results = [solve(model, 'cn', python) for python in ([True, False] if ddm.model.HAS_CSOLVE else [True])]
        for solution in results:
            self.assertEqual(solution.pdf('correct')[-1], 0.)
            self.assertGreater(solution.pdf('correct')[-2], 0.)
            self.assertAlmostEqual(solution.pdf('correct')[-2]*model.dt,
                                   solution.choice_upper[-2], delta=1e-14)
        if len(results) == 2:
            np.testing.assert_allclose(results[0].pdf('correct'), results[1].pdf('correct'), atol=1e-12, rtol=1e-10)

    def test_cn_extension_last_step_is_optional(self):
        if not ddm.model.HAS_CSOLVE:
            self.skipTest('C extension unavailable')
        args = (np.array([0.]), 0, np.array([1.]), 0, np.array([1.]), 0,
                np.array([0., 1., 0.]), .1, .01, .5, 11, 1e-14)
        legacy = ddm.csolve.cn_time(*args)
        extended = ddm.csolve.cn_time(*args, 1)
        self.assertEqual(len(legacy), 3)
        self.assertEqual(len(extended), 4)
        self.assertEqual(extended[3], 10)
        for a, b in zip(legacy, extended):
            np.testing.assert_array_equal(a, b)

    def test_cn_final_pdf_during_startup_is_instantaneous_flux(self):
        for steps in [1, 2]:
            model = ddm.Model(dx=.005, dt=.0001, T_dur=steps*.0001,
                              bound=ddm.BoundConstant(B=.1), IC=ddm.ICPoint(x0=.09))
            reference = model._solve_numerical_python('cn', {}, return_evolution=True)
            matrix = model._fp_matrix(model.x_domain({}), model.t_domain()[-1], .1, {})
            lower, upper = matrix[1:]
            state = reference.evolution[:, -1]
            self.assertAlmostEqual(reference.pdf('correct')[-1], upper[1]*state[upper[0]], delta=1e-12)
            self.assertAlmostEqual(reference.pdf('error')[-1], lower[1]*state[lower[0]], delta=1e-12)
            if ddm.model.HAS_CSOLVE:
                compiled = model.solve_numerical('cn')
                np.testing.assert_allclose(compiled.pdf('correct'), reference.pdf('correct'), atol=1e-12, rtol=1e-10)
                np.testing.assert_allclose(compiled.pdf('error'), reference.pdf('error'), atol=1e-12, rtol=1e-10)

    def test_cn_pdf_correction_through_overlays_and_likelihood(self):
        overlays = [ddm.OverlayNone(), ddm.OverlayNonDecision(nondectime=.05),
                    ddm.OverlayNonDecision(nondectime=-.05), ddm.OverlayUniformMixture(umixturecoef=.2),
                    ddm.OverlayExponentialMixture(pmixturecoef=.2, rate=1),
                    ddm.OverlayNonDecisionUniform(nondectime=.01, halfwidth=.02),
                    ddm.OverlayNonDecisionGamma(nondectime=-.02, shape=2, scale=.01),
                    ddm.OverlaySimplePause(pausestart=.1, pausestop=.2),
                    ddm.OverlayBlurredPause(pausestart=.1, pausestop=.2, pauseblurwidth=.03),
                    ddm.OverlayChain(overlays=[ddm.OverlayNonDecision(nondectime=0),
                                               ddm.OverlayUniformMixture(umixturecoef=.2)])]
        for overlay in overlays:
            with self.subTest(overlay=type(overlay).__name__):
                model = ddm.Model(dx=.005, dt=.005, T_dur=.5, overlay=overlay)
                analytical = model.solve_analytical()
                for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
                    solution = solve(model, 'cn', python)
                    np.testing.assert_allclose(solution.pdf('correct'), analytical.pdf('correct'), atol=.0012, rtol=.002)
                    self.assertEqual(len(solution.sample(100, seed=3)), 100)

        model = ddm.Model(dx=.005, dt=.005, T_dur=.5)
        sample = ddm.Sample(np.array([.5]), np.array([]))
        loss = ddm.models.LossLikelihood(sample=sample, dt=model.dt, T_dur=model.T_dur, method='cn')
        self.assertAlmostEqual(loss.loss(model), -np.log(model.solve_analytical().pdf('correct')[-1]), delta=1e-5)

    def test_cn_correction_survives_partial_conditions(self):
        class ConditionDrift(ddm.Drift):
            name = 'Condition test drift'
            required_parameters = []
            required_conditions = ['level']
            def get_drift(self, conditions, **kwargs): return conditions['level']

        model = ddm.Model(dx=.01, dt=.005, T_dur=.5, drift=ConditionDrift())
        combined = ddm.solve_partial_conditions(model, conditions={'level': [0, .5]}, method='cn')
        a, b = [model.solve_numerical('cn', conditions={'level': level}) for level in [0, .5]]
        for choice in ['correct', 'error']:
            np.testing.assert_allclose(combined.pdf(choice), (a.pdf(choice)+b.pdf(choice))/2, atol=1e-12)
            self.assertAlmostEqual(combined.prob(choice), (a.prob(choice)+b.prob(choice))/2, delta=1e-12)

    def test_cn_correction_with_gddm_generated_overlays(self):
        def drift(x): return 0*x
        def delay(T): return np.exp(-((T+.01)/.02)**2)
        model = ddm.gddm(drift=drift, nondecision=delay, mixture_coef=.2,
                         dx=.005, dt=.005, T_dur=.5)
        reference_model = ddm.Model(dx=model.dx, dt=model.dt, T_dur=model.T_dur,
                                    overlay=model.get_dependence('overlay'))
        reference = reference_model.solve_analytical()
        for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
            solution = solve(model, 'cn', python)
            np.testing.assert_allclose(solution.pdf('correct'), reference.pdf('correct'), atol=.0012, rtol=.002)
            self.assertEqual(len(solution.sample(100, seed=3)), 100)

    def test_custom_overlay_can_transform_cn_pdf_correction(self):
        class ScaleOverlay(ddm.Overlay):
            name = 'Scale choice masses'
            required_parameters = []
            def apply(self, solution):
                result = ddm.Solution(.8*solution.choice_upper, .8*solution.choice_lower,
                                      solution.model, solution.conditions)
                return result._transform_pdf_correction(solution, lambda values, choice: .8*values)

        model = ddm.Model(dx=.005, dt=.005, T_dur=.5, overlay=ScaleOverlay())
        numerical = model.solve_numerical('cn')
        analytical = model.solve_analytical()
        self.assertAlmostEqual(numerical.pdf('correct')[-1], analytical.pdf('correct')[-1], delta=1e-5)

    def test_legacy_model_import_preserves_fittable_identity(self):
        # Importing the old plotting API previously loaded ddm.model as a
        # second module, so fitting rejected its otherwise valid Fittables.
        code = '''\
import ddm
from ddm.plot import *
from ddm.model import Fittable
from pyddm.model import Fittable as CurrentFittable
from pyddm.functions import Fittable as FitFunctionType
assert Fittable is CurrentFittable is FitFunctionType
'''
        result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plot_solution_and_compare_named_choices(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pyddm.plot as plot

        model = ddm.Model(dt=.01, T_dur=.1, choice_names=('left', 'right'))
        solution = model.solve_numerical(method='implicit')
        fig, ax = plt.subplots()
        try:
            plot.plot_solution_pdf(solution, ax=ax, choice='right')
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), solution.pdf('right'))
            plt.figure()
            plot.plot_compare_solutions(solution, solution)
            self.assertEqual(len(plt.gcf().axes), 2)
            self.assertEqual([len(a.lines) for a in plt.gcf().axes], [2, 2])
        finally:
            plt.close('all')

    def test_state_dependent_noise_trajectory_has_no_spurious_drift(self):
        # For dX=(1+.5X)dW and X(0)=0, the Itô mean is zero.  Pairing
        # equal-and-opposite increments removes Monte Carlo uncertainty.
        class FixedRandom:
            def __init__(self, seed): self.sign = 1 if seed == 1 else -1
            def choice(self, values, p): return 0.
            def randn(self): return self.sign

        model = ddm.Model(dx=.01, dt=.01, T_dur=.01,
            noise=ddm.NoiseLinear(noise=1, x=.5, t=.2), bound=ddm.BoundConstant(B=2))
        with patch('pyddm.model.np.random.RandomState', FixedRandom):
            positive = model.simulate_trial(seed=1, rk4=True)[-1]
            negative = model.simulate_trial(seed=2, rk4=True)[-1]
        self.assertAlmostEqual((positive+negative)/2, 0, delta=1e-14)

    def test_gddm_vector_only_coefficients_work_for_all_backends(self):
        # The gddm API promises x-dependent functions an array, including
        # when the solver asks for one boundary point or a trajectory point.
        def drift(x): return .2 + np.zeros(len(x)) - .1*x
        def noise(x): return .8 + np.zeros(len(x)) + .1*x
        model = ddm.gddm(drift=drift, noise=noise, mixture_coef=0,
                         dx=.02, dt=.005, T_dur=.2)
        for method in ['implicit', 'cn']:
            with self.subTest(method=method):
                p = solve(model, method, True)
                if ddm.model.HAS_CSOLVE:
                    c = solve(model, method, False)
                    np.testing.assert_allclose(p.choice_upper, c.choice_upper, atol=1e-12, rtol=1e-10)
                    np.testing.assert_allclose(p.choice_lower, c.choice_lower, atol=1e-12, rtol=1e-10)
        self.assertTrue(np.isfinite(model.simulate_trial(seed=1)).all())

    def test_nondecision_trajectories_are_shifted_and_seeded(self):
        from paranoid.settings import Settings
        previous = Settings.get('enabled')
        Settings.set(enabled=True)
        try:
            model = ddm.Model(dt=.01, T_dur=.5)
            trajectory = np.array([0., 1., 2.])
            overlays = [ddm.OverlayNonDecision(nondectime=.1),
                        ddm.OverlayNonDecisionUniform(nondectime=.1, halfwidth=.03),
                        ddm.OverlayNonDecisionGamma(nondectime=.1, shape=2, scale=.02)]
            for overlay in overlays:
                with self.subTest(overlay=type(overlay).__name__):
                    def apply(seed):
                        return overlay.apply_trajectory(trajectory, model, {}, seed=seed)
                    result = apply(1)
                    self.assertGreater(len(result), len(trajectory))
                    np.testing.assert_array_equal(result[-len(trajectory):], trajectory)
                    np.testing.assert_array_equal(result, apply(1))
                    if not isinstance(overlay, ddm.OverlayNonDecision):
                        self.assertGreater(len({len(apply(seed)) for seed in range(12)}), 1)
        finally:
            Settings.set(enabled=previous)

    def test_simulated_solution_respects_initial_absorption_and_metadata(self):
        model = ddm.Model(dx=.01, dt=.01, T_dur=.1, IC=ddm.ICPoint(x0=1),
                          choice_names=('yes', 'no'))
        np.testing.assert_array_equal(model.simulate_trial(seed=1), [1.])
        sample = model.simulated_solution(conditions={'coherence': 0}, size=5, seed=1)
        self.assertEqual(sample.choice_names, ('yes', 'no'))
        np.testing.assert_array_equal(sample.choice_upper, np.zeros(5))
        np.testing.assert_array_equal(sample.conditions['coherence'][0], np.zeros(5))
        self.assertEqual(sample.undecided, 0)

        collapsed = ddm.Model(dx=.01, dt=.01, T_dur=.1, bound=ZeroBound())
        sample = collapsed.simulated_solution(size=100, seed=1)
        self.assertEqual(sample.undecided, 0)
        self.assertGreater(len(sample.choice_upper), 25)
        self.assertGreater(len(sample.choice_lower), 25)
        np.testing.assert_array_equal(np.r_[sample.choice_upper, sample.choice_lower], np.zeros(100))

    def test_simulated_solution_tracks_decisions_before_nondeterministic_response_time(self):
        class FastGrowingBound(ddm.Bound):
            name = 'Fast growing test boundary'
            required_parameters = []
            def get_bound(self, t, **kwargs): return 1 + 3*t

        # The decision crosses the growing bound, but the bound is larger
        # than the final position by the time the response delay is over.
        model = ddm.Model(dx=.01, dt=.01, T_dur=1, bound=FastGrowingBound(),
            drift=ddm.DriftConstant(drift=8), noise=ddm.NoiseConstant(noise=1e-6),
            overlay=ddm.OverlayNonDecision(nondectime=.2))
        sample = model.simulated_solution(size=1, seed=1)
        self.assertEqual(len(sample.choice_upper), 1)
        self.assertEqual(sample.undecided, 0)

        # A negative shift shortens an undecided trajectory; that alone does
        # not mean that a boundary was reached.
        model = ddm.Model(dx=.01, dt=.01, T_dur=.1, noise=ddm.NoiseConstant(noise=1e-6),
                          overlay=ddm.OverlayNonDecision(nondectime=-.05))
        sample = model.simulated_solution(size=1, seed=1)
        self.assertEqual(sample.undecided, 1)

    def test_short_gddm_and_blurred_pause_edge_cases(self):
        # Default mixture density must also work when T_dur < dt.
        model = ddm.gddm(dt=.005, T_dur=.001)
        solution = model.solve()
        self.assertTrue(np.isfinite(solution.choice_upper).all())
        self.assertEqual(len(solution.sample(20, seed=1)), 20)

        model = ddm.Model(dt=.01, T_dur=.2)
        solution = model.solve()
        cases = [ddm.OverlayBlurredPause(pausestart=.3, pausestop=.4, pauseblurwidth=.05),
                 ddm.OverlayBlurredPause(pausestart=.2, pausestop=.4, pauseblurwidth=.05),
                 # This valid gamma shape is less than one and its PDF is
                 # singular at zero; its first-bin probability is finite.
                 ddm.OverlayBlurredPause(pausestart=.1, pausestop=.14, pauseblurwidth=.07)]
        for overlay in cases:
            with self.subTest(pause=overlay):
                result = overlay.apply(solution)
                self.assertTrue(np.isfinite(result.choice_upper).all())
                self.assertTrue(np.isfinite(result.choice_lower).all())
                self.assertGreaterEqual(result.prob_undecided(), 0)
        np.testing.assert_array_equal(cases[0].apply(solution).choice_upper, solution.choice_upper)

    def test_solution_roundoff_uses_same_sum_as_probability_validation(self):
        # Summing the choices separately gives 1 + one ulp, while summing
        # their elementwise sum rounds to exactly 1 on IEEE-754 doubles.
        upper = np.array([float.fromhex(x) for x in [
            '0x1.0b52418c2c5cep-5', '0x1.643fb848abbdep-3', '0x1.bf106fea6b418p-3']])
        lower = np.array([float.fromhex(x) for x in [
            '0x1.a1a805d4de91fp-5', '0x1.363d2ee820750p-2', '0x1.c4f6e824655b4p-3']])
        self.assertGreater(upper.sum()+lower.sum(), 1)
        self.assertLessEqual((upper+lower).sum(), 1)
        model = ddm.Model(dt=.005, T_dur=.01)
        solution = ddm.Solution(upper, lower, model, {})
        ddm.Solution._test(solution)

    def test_ratio_start_uses_initial_not_maximum_increasing_bound(self):
        # B(0)=.3, B(.5)=.45: a half-bound start is .15, not .225.
        for ratio in [-.5, .5]:
            settings = dict(dx=.005, dt=.005, T_dur=.5, bound=IncreasingBound(),
                            drift=ddm.DriftConstant(drift=.2))
            relative = ddm.Model(IC=ddm.ICPointRatio(x0=ratio), **settings)
            absolute = ddm.Model(IC=ddm.ICPoint(x0=.3*ratio), **settings)
            x = relative.x_domain({})
            self.assertAlmostEqual(x[np.argmax(relative.IC({}))], .3*ratio)
            self.assertEqual(relative.simulate_trial(seed=1, cutoff=False)[0], .3*ratio)
            for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
                with self.subTest(ratio=ratio, python=python):
                    a = solve(relative, 'implicit', python)
                    b = solve(absolute, 'implicit', python)
                    np.testing.assert_allclose(a.choice_upper, b.choice_upper, atol=1e-13, rtol=0)
                    np.testing.assert_allclose(a.choice_lower, b.choice_lower, atol=1e-13, rtol=0)
                    np.testing.assert_allclose(a.undec, b.undec, atol=1e-13, rtol=0)

    def test_bad_initial_condition_shape_raises_without_crashing(self):
        class BadInitial(ddm.InitialCondition):
            name = 'invalid_shape'
            required_parameters = []

            def get_IC(self, x, dx, conditions):
                return np.array(1.)

        model = ddm.Model(IC=BadInitial(), dx=.01, dt=.005, T_dur=.1)
        for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
            for method in ['implicit', 'cn']:
                with self.subTest(python=python, method=method):
                    with self.assertRaisesRegex(ValueError, 'matching the spatial grid'):
                        solve(model, method, python)

    def test_integer_initial_condition_does_not_truncate_probability(self):
        class IntegerInitial(ddm.InitialCondition):
            name = 'integer_point'
            required_parameters = []

            def get_IC(self, x, dx, conditions):
                result = np.zeros(len(x), dtype=int)
                result[len(x)//2] = 1
                return result

        kwargs = dict(dx=.05, dt=.005, T_dur=.2)
        integer = ddm.Model(IC=IntegerInitial(), **kwargs)
        reference = ddm.Model(IC=ddm.ICPointSourceCenter(), **kwargs)
        for method in ['implicit', 'cn']:
            for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
                with self.subTest(method=method, python=python):
                    a, b = solve(integer, method, python), solve(reference, method, python)
                    np.testing.assert_allclose(a.choice_upper, b.choice_upper, atol=1e-13, rtol=0)
                    np.testing.assert_allclose(a.choice_lower, b.choice_lower, atol=1e-13, rtol=0)

    def test_invalid_initial_probabilities_are_rejected(self):
        class BadInitial(ddm.InitialCondition):
            name = 'invalid_probability'
            required_parameters = ['kind']

            def get_IC(self, x, dx, conditions):
                result = np.zeros(len(x))
                if self.kind == 'nan': result[len(x)//2] = np.nan
                if self.kind == 'negative': result[0], result[len(x)//2] = -1, 2
                return result

        for kind in ['zero', 'nan', 'negative']:
            model = ddm.Model(IC=BadInitial(kind=kind), dx=.01, dt=.005, T_dur=.1)
            for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
                with self.subTest(kind=kind, python=python):
                    with self.assertRaisesRegex(ValueError, 'finite, nonnegative probabilities summing to one'):
                        solve(model, 'implicit', python)

    def test_sample_single_timepoint_with_and_without_initial_absorption(self):
        for IC in [ddm.ICPointSourceCenter(), ddm.ICPoint(x0=1)]:
            model = ddm.Model(dt=.005, dx=.01, T_dur=.001, IC=IC)
            for method in ['analytical', 'implicit', 'cn']:
                with self.subTest(initial=type(IC).__name__, method=method):
                    solution = (model.solve_analytical() if method=='analytical'
                                else model.solve_numerical(method=method))
                    sample = solution.sample(100, seed=3)
                    self.assertEqual(len(sample), 100)
                    if isinstance(IC, ddm.ICPoint):
                        self.assertEqual(len(sample.choice_upper), 100)
                        np.testing.assert_array_equal(sample.choice_upper, np.zeros(100))
                    else:
                        self.assertEqual(sample.undecided, 100)

    def test_cn_narrow_bounds_do_not_create_undecided_trials(self):
        for bound in [.1, .03]:
            model = ddm.Model(dx=.005, dt=.005, T_dur=2,
                              bound=ddm.BoundConstant(B=bound))
            for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
                with self.subTest(bound=bound, python=python):
                    sol = solve(model, 'cn', python)
                    self.assertLess(sol.prob_undecided(), 2e-11)
                    self.assertAlmostEqual(sol.prob('_top'), .5, delta=2e-11)
                    self.assertAlmostEqual(sol.prob('_bottom'), .5, delta=2e-11)

    def test_cn_conserves_before_clipping_or_renormalization(self):
        # Observe the raw state and flux, so final clipping/normalization
        # cannot hide lost probability. Exercise startup, ordinary steps,
        # fractional bounds, variable coefficients, and oscillatory tails.
        finish = ddm.Model._finish_numerical
        for varying in [False, True]:
            for bound, duration in [(.103, .005), (.103, .01), (.103, .015),
                                     (.103, .5), (.03, 2), (1.003, .5)]:
                model = ddm.Model(dx=.005, dt=.005, T_dur=duration,
                    bound=ddm.BoundConstant(B=bound),
                    drift=ddm.DriftLinear(drift=.4, x=.1, t=.2 if varying else 0),
                    noise=ddm.NoiseLinear(noise=1, x=.2, t=.1 if varying else 0))
                for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
                    with self.subTest(varying=varying, bound=bound, duration=duration, python=python):
                        raw = {}
                        def capture(self, upper, lower, conditions, undec=None, evolution=None, **kwargs):
                            raw['absorbed'] = upper.sum()+lower.sum()
                            if evolution is not None:
                                raw['survival'] = evolution[:, -1].sum()
                            return finish(self, upper, lower, conditions, undec, evolution, **kwargs)
                        with patch.object(ddm.Model, '_finish_numerical', capture):
                            if python:
                                model._solve_numerical_python('cn', {}, return_evolution=True)
                            else:
                                cn_time = ddm.csolve.cn_time
                                def capture_c(*args):
                                    result = cn_time(*args)
                                    raw['survival'] = result[2].sum()
                                    return result
                                with patch.object(ddm.csolve, 'cn_time', capture_c):
                                    solve(model, 'cn', False)
                        self.assertAlmostEqual(raw['absorbed']+raw['survival'], 1, delta=3e-12)

    def test_cn_preserves_genuine_undecided_probability(self):
        # Exact survival for Brownian motion between +/-1, including the
        # deterministic variance-time change for a time-dependent noise.
        for varying in [False, True]:
            model = ddm.Model(dx=.005, dt=.005, T_dur=.5,
                              noise=ddm.NoiseLinear(noise=1, x=0, t=.5 if varying else 0))
            t = model.t_domain()[-1]
            tau = t+.5*t*t+t**3/12 if varying else t
            k = 2*np.arange(100)+1
            survival = 4/np.pi*np.sum((-1.)**np.arange(100)/k*np.exp(-k*k*np.pi**2*tau/8))
            self.assertGreater(survival, .5)
            for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
                with self.subTest(varying=varying, python=python):
                    sol = solve(model, 'cn', python)
                    self.assertAlmostEqual(sol.prob_undecided(), survival, delta=5e-5)

    def test_explicit_absorption_uses_the_old_state(self):
        model = ddm.Model(dx=.1, dt=.005, T_dur=.005, IC=ddm.ICPoint(x0=.9))
        sol = solve(model, 'explicit')
        self.assertAlmostEqual(sol.choice_upper[1], .25)
        self.assertAlmostEqual(sol.undec.sum(), .75)
        self.assertAlmostEqual(sol.choice_upper.sum()+sol.choice_lower.sum()+sol.undec.sum(), 1)
        self.assertEqual(sol.undec[-1], 0)

    def test_explicit_and_implicit_with_stepwise_moving_bounds(self):
        class StepBound(ddm.Bound):
            name = 'step_bound'
            required_parameters = ['closes']

            def get_bound(self, t, **kwargs):
                if t < .1: return 1.
                if t < .15: return 0. if self.closes else .9
                return 1.1

        for closes in [False, True]:
            model = ddm.Model(dx=.05, dt=.0002, T_dur=.2, bound=StepBound(closes=closes))
            self.assertTrue(model.can_solve_explicit())
            explicit = solve(model, 'explicit', True)
            implicit = solve(model, 'implicit', False)
            for choice in ['_top', '_bottom']:
                self.assertLess(np.max(np.abs(explicit.cdf(choice)-implicit.cdf(choice))), .002)
            self.assertAlmostEqual(explicit.choice_upper.sum()+explicit.choice_lower.sum()+explicit.undec.sum(), 1, delta=1e-10)
            if closes:
                self.assertEqual(explicit.undec.sum(), 0)
                self.assertEqual(implicit.undec.sum(), 0)

    def test_variable_noise_preserves_mass_and_ito_moments(self):
        for dx in [.1, .05, .025]:
            model = ddm.Model(dx=dx, noise=ddm.NoiseLinear(noise=1, x=.5, t=0))
            x = model.x_domain({})
            matrix, _, _ = model._fp_matrix(x, 0, 1, {})
            initial = model.IC({})
            derivative = -matrix.dot(initial)
            self.assertAlmostEqual(derivative.sum(), 0, places=10)
            self.assertAlmostEqual(x @ derivative, 0, places=10)
            self.assertAlmostEqual(x**2 @ derivative, 1, places=10)

    def test_each_time_integrator_balances_its_flux(self):
        model = ddm.Model(dx=.1, dt=.0001, bound=ddm.BoundConstant(B=.93),
                          drift=ddm.DriftLinear(drift=.2, x=.1, t=.3),
                          noise=ddm.NoiseLinear(noise=1, x=.3, t=.2),
                          IC=ddm.ICPoint(x0=.9))
        x, initial = model.x_domain({}), model.IC({})
        old = model._fp_matrix(x, 0, .93, {})
        new = model._fp_matrix(x, model.dt, .93, {})
        for theta in [0, .5, 1]:
            following, lower, upper = model._fp_step(initial, old, new, model.dt, theta)
            self.assertAlmostEqual(following.sum()+lower+upper, 1, places=12)

    def test_mean_exit_time_has_no_boundary_residence_delay(self):
        # For driftless Brownian motion E[tau] = (B**2-x0**2)/sigma**2.
        # Integrating the semidiscrete survival probability is (-L)^-1 q0;
        # this avoids conflating the spatial error with time quadrature.
        for dx, bound in [(.1, 1.), (.05, 1.), (.05, 1.013), (.1, .037)]:
            model = ddm.Model(dx=dx, bound=ddm.BoundConstant(B=bound))
            x = model.x_domain({})
            matrix, lower, upper = model._fp_matrix(x, 0, bound, {})
            first, stop = model._fp_indices(x, bound)
            lifetime = matrix.splice(first, stop).spsolve(model.IC({})[first:stop])
            self.assertAlmostEqual(lifetime.sum(), bound**2, places=11)
            self.assertAlmostEqual(lower[1]*lifetime[0], .5, places=11)
            self.assertAlmostEqual(upper[1]*lifetime[-1], .5, places=11)

    def test_variable_noise_eventual_choice_probability(self):
        # A bounded zero-drift Ito diffusion is a martingale.  Starting at
        # zero between symmetric bounds, either exit must have probability .5.
        model = ddm.Model(dx=.05, dt=.01, T_dur=20,
                          noise=ddm.NoiseLinear(noise=1, x=.5, t=0))
        for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
            sol = solve(model, 'implicit', python)
            self.assertAlmostEqual(sol.choice_upper.sum(), .5, delta=1e-7)
            self.assertAlmostEqual(sol.choice_lower.sum(), .5, delta=1e-7)
            self.assertAlmostEqual(sol.choice_upper.sum()+sol.choice_lower.sum()+sol.undec.sum(), 1, places=11)

    def test_moving_boundary_conserves_evolution(self):
        for method in ['explicit', 'implicit']:
            # This short collapse is stable for explicit stepping, and
            # exercises boundaries between grid points at every time step.
            model = ddm.Model(dx=.1, dt=.001, T_dur=.01,
                              bound=ddm.BoundCollapsingLinear(B=1, t=.5),
                              IC=ddm.ICPoint(x0=.9))
            sol = model.solve_numerical(method=method, force_python=True, return_evolution=True)
            total = np.cumsum(sol.choice_upper+sol.choice_lower)+sol.evolution.sum(axis=0)
            np.testing.assert_allclose(total, 1, atol=2e-12, rtol=0)
            np.testing.assert_allclose(sol.evolution[:, -1], sol.undec, atol=2e-12, rtol=0)

    def test_boundary_collapse_to_zero_and_single_interior_point(self):
        for bound in [ddm.BoundConstant(B=.025), ZeroBound(),
                      ddm.BoundCollapsingLinear(B=.05, t=1)]:
            model = ddm.Model(dx=.05, dt=.001, T_dur=.06, bound=bound)
            p = solve(model, 'implicit')
            self.assertAlmostEqual(p.choice_upper.sum()+p.choice_lower.sum()+p.undec.sum(), 1, places=12)
            self.assertTrue(np.all(p.undec >= 0))
            if ddm.model.HAS_CSOLVE:
                c = solve(model, 'implicit', False)
                np.testing.assert_allclose(c.choice_upper, p.choice_upper, atol=1e-12, rtol=1e-11)
                np.testing.assert_allclose(c.undec, p.undec, atol=1e-12, rtol=1e-11)

    def test_initial_mass_on_boundary_is_absorbed_immediately(self):
        for method in ['explicit', 'implicit', 'cn']:
            model = ddm.Model(dx=.1, dt=.001, T_dur=.01, IC=ddm.ICPoint(x0=1))
            sol = solve(model, method)
            self.assertEqual(sol.choice_upper[0], 1)
            self.assertEqual(sol.choice_upper[1:].sum()+sol.choice_lower.sum(), 0)
            self.assertEqual(model.solve_analytical().choice_upper[0], 1)


class TestFPAccuracy(unittest.TestCase):
    def test_analytical_collapse_time_does_not_depend_on_evidence_units(self):
        times = np.arange(0, 1.5, .005)
        for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
            reference = np.array(analytic_ddm(.4, 1, .6, times, shift=.7,
                                               b_slope=-.8, force_python=python))
            for scale in [.5, 2.]:
                with self.subTest(python=python, scale=scale):
                    result = np.array(analytic_ddm(.4*scale, scale, .6*scale, times,
                        shift=.7, b_slope=-.8*scale, force_python=python))
                    self.assertTrue(np.isfinite(result).all())
                    np.testing.assert_allclose(result, reference, atol=1e-11, rtol=1e-10)
                    self.assertGreater(result[:, int(.6/.005)].sum(), 1e-8)
                    self.assertLess(result[:, int(.8/.005)].sum(), 1e-90)

    def test_analytical_arbitrary_timepoints_and_collapse_order(self):
        # The C implementation accepts only a uniform grid starting at zero;
        # public analytic_ddm also accepts irregular and unsorted timepoints.
        for times in [np.array([.02, .07, .11, .23]),
                      np.array([0., .02, .07, .11, .23]),
                      np.array([.8, .2, 0., .9, .5])]:
            with self.subTest(times=times):
                c = np.array(analytic_ddm(.4, 1.5, .6, times, shift=.7, b_slope=-.8))
                p = np.array(analytic_ddm(.4, 1.5, .6, times, shift=.7, b_slope=-.8, force_python=True))
                np.testing.assert_allclose(c, p, rtol=1e-12, atol=1e-12)
                self.assertTrue(np.all(c[:, times >= .75] < 1e-90))
                self.assertTrue(np.isfinite(c).all())

    def test_analytical_time_inputs_are_validated_and_not_modified(self):
        from pyddm.analytic import analytic_ddm_linbound
        times = np.array([0, 1, 2])
        for python in [True, False]:
            density = np.array(analytic_ddm(.4, 1, 1, times, force_python=python))
            self.assertTrue(np.isfinite(density).all())
            np.testing.assert_array_equal(times, [0, 1, 2])
        self.assertEqual(len(analytic_ddm_linbound(1, 0, -1, 0, [])), 0)
        for invalid in [[-1., 0.], [0., np.nan], [[0., .1]]]:
            with self.assertRaisesRegex(ValueError, 'finite nonnegative times'):
                analytic_ddm(.4, 1, 1, invalid)

    def test_analytical_model_subclass_with_irregular_times(self):
        class IrregularModel(ddm.Model):
            def t_domain(self): return np.array([0., .01, .03, .04])

        model = IrregularModel(dt=.01, T_dur=.04, drift=ddm.DriftConstant(drift=.4))
        actual = model.solve_analytical()
        expected = analytic_ddm(.4, 1, 1, model.t_domain(), force_python=True)
        np.testing.assert_allclose(actual.pdf('_top'), expected[0], atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(actual.pdf('_bottom'), expected[1], atol=1e-12, rtol=1e-10)

    def test_cached_cn_matches_equivalent_time_dependent_model(self):
        class TimedConstantDrift(ddm.DriftConstant):
            def _uses_t(self):
                return True

        for dx, dt, bound, duration in [(.05, .001, .025, .05),
                                        (.02, .01, .1, 2),
                                        (.005, .005, .1033, .5)]:
            common = dict(dx=dx, dt=dt, T_dur=duration, bound=ddm.BoundConstant(B=bound),
                          noise=ddm.NoiseLinear(noise=1, x=.2, t=0))
            cached = solve(ddm.Model(drift=ddm.DriftConstant(drift=.4), **common), 'cn')
            varying = solve(ddm.Model(drift=TimedConstantDrift(drift=.4), **common), 'cn')
            np.testing.assert_allclose(cached.choice_upper, varying.choice_upper, atol=5e-13, rtol=1e-9)
            np.testing.assert_allclose(cached.choice_lower, varying.choice_lower, atol=5e-13, rtol=1e-9)

    def test_tail_termination_preserves_probabilities(self):
        for bound, drift, duration in [(.1, .4, 2), (1.003, 4, 10)]:
            model = ddm.Model(dx=.01, dt=.005, T_dur=duration,
                              bound=ddm.BoundConstant(B=bound), drift=ddm.DriftConstant(drift=drift))
            for method in ['implicit', 'cn']:
                for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
                    truncated = solve(model, method, python)
                    with patch.object(ddm.parameters, 'numerical_mass_tolerance', 0):
                        complete = solve(model, method, python)
                    error = (np.abs(truncated.choice_upper-complete.choice_upper).sum()
                             + np.abs(truncated.choice_lower-complete.choice_lower).sum())
                    self.assertLess(error, 2e-12)
                    if truncated.undec is not None:
                        self.assertLess(np.abs(truncated.undec-complete.undec).sum(), 2e-12)
                        self.assertAlmostEqual(truncated.choice_upper.sum()+truncated.choice_lower.sum()
                                               + truncated.undec.sum(), 1, places=11)

    def test_saved_evolution_ignores_tail_cutoff(self):
        model = ddm.Model(dx=.02, dt=.005, T_dur=.3, bound=ddm.BoundConstant(B=.1))
        with patch.object(ddm.parameters, 'numerical_mass_tolerance', .1):
            sol = model.solve_numerical_implicit(force_python=True, return_evolution=True)
        with patch.object(ddm.parameters, 'numerical_mass_tolerance', 0):
            complete = model.solve_numerical_implicit(force_python=True, return_evolution=True)
        np.testing.assert_array_equal(sol.evolution, complete.evolution)
        np.testing.assert_allclose(np.cumsum(sol.choice_upper+sol.choice_lower)+sol.evolution.sum(axis=0),
                                   1, atol=2e-12, rtol=0)
        self.assertGreater(sol.evolution[:, -1].sum(), 0)

    def test_repeated_solves_reflect_changed_parameters(self):
        model = ddm.Model(dx=.02, dt=.005, T_dur=.4, drift=ddm.DriftConstant(drift=.2))
        for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
            for method in ['implicit', 'cn']:
                model.get_dependence('drift').drift = .2
                original = solve(model, method, python)
                model.get_dependence('drift').drift = -.2
                reversed_drift = solve(model, method, python)
                np.testing.assert_allclose(original.choice_upper, reversed_drift.choice_lower, atol=2e-12, rtol=0)
                self.assertGreater(np.max(np.abs(original.choice_upper-reversed_drift.choice_upper)), 1e-5)

    def test_explicit_checks_late_instability_even_after_absorption(self):
        # Exercise the runtime check, independently of paranoid's preflight.
        from paranoid.settings import Settings
        enabled = Settings.get('enabled')
        Settings.set(enabled=False)
        try:
            for start in [0, 1]:
                model = ddm.Model(dx=.05, dt=.001, T_dur=.5, IC=ddm.ICPoint(x0=start),
                                  noise=ddm.NoiseLinear(noise=.4, x=0, t=3))
                self.assertFalse(model.can_solve_explicit())
                with self.assertRaises(AssertionError):
                    solve(model, 'explicit')
        finally:
            Settings.set(enabled=enabled)

    def test_cn_temporal_order_with_constant_and_varying_noise(self):
        for varying in [False, True]:
            errors = []
            for dt in [.02, .01, .005]:
                model = ddm.Model(dx=.002, dt=dt, T_dur=2,
                                  noise=ddm.NoiseLinear(noise=1, x=0, t=.5 if varying else 0))
                t = model.t_domain()
                # Exact deterministic time change of Brownian motion.
                tau = t+.5*t*t+t**3/12 if varying else t
                reference = np.array(analytic_ddm(0, 1, 1, tau.copy(), force_python=True))
                if varying:
                    reference *= (1+.5*t)**2
                sol = solve(model, 'cn', python=not ddm.model.HAS_CSOLVE)
                errors.append(cn_density_error(sol, reference))
            self.assertGreater(errors[0]/errors[1], 3.5)
            self.assertGreater(errors[1]/errors[2], 3.5)

    def test_cn_first_rt_moment_has_no_half_step_shift(self):
        # Integral_0^T t*f(t) dt = Integral_0^T S(t) dt - T*S(T).
        # The Brownian survival expansion gives an independent reference,
        # including surviving trials. Depositing interval masses at their
        # right endpoints would introduce a first-order time shift here.
        duration = 2.
        k = 2*np.arange(2000)+1
        rates = k*k*np.pi**2/8
        coefficients = 4/np.pi*(-1.)**np.arange(len(k))/k
        reference = np.sum(coefficients*((1-np.exp(-rates*duration))/rates
                                        - duration*np.exp(-rates*duration)))
        for python in [True, False] if ddm.model.HAS_CSOLVE else [True]:
            errors = []
            for dt in [.02, .01, .005]:
                model = ddm.Model(dx=.001, dt=dt, T_dur=duration)
                sol = solve(model, 'cn', python)
                moment = np.sum(sol.t_domain*(sol.choice_upper+sol.choice_lower))
                errors.append(abs(moment-reference))
            with self.subTest(python=python):
                self.assertGreater(errors[0]/errors[1], 3.5)
                self.assertGreater(errors[1]/errors[2], 3.5)

    def test_off_grid_boundaries_match_their_actual_values(self):
        solutions = []
        for bound in [1.001, 1.009]:
            model = ddm.Model(dx=.01, dt=.001, T_dur=2, bound=ddm.BoundConstant(B=bound))
            sol = solve(model, 'cn', python=not ddm.model.HAS_CSOLVE)
            reference = model.solve_analytical(force_python=True)
            self.assertLess(cn_density_error(sol, np.array([reference.pdf('_top'), reference.pdf('_bottom')])), 8e-5)
            solutions.append(sol.choice_upper)
        self.assertGreater(np.max(np.abs(solutions[0]-solutions[1])), 1e-6)

    def test_off_grid_starting_point_matches_analytical(self):
        model = ddm.Model(dx=.01, dt=.001, T_dur=2, bound=ddm.BoundConstant(B=1.003),
                          IC=ddm.ICPoint(x0=.3))
        sol = solve(model, 'cn', python=not ddm.model.HAS_CSOLVE)
        reference = model.solve_analytical(force_python=True)
        self.assertLess(cn_density_error(sol, np.array([reference.pdf('_top'), reference.pdf('_bottom')])), 4e-4)

    def test_explicit_stability_checks_space_drift_and_boundary_distance(self):
        self.assertFalse(ddm.Model(dx=.1, dt=.005,
                         noise=ddm.NoiseLinear(noise=1, x=.9, t=0)).can_solve_explicit())
        self.assertFalse(ddm.Model(dx=.1, dt=.00001,
                         drift=ddm.DriftConstant(drift=20)).can_solve_explicit())
        self.assertFalse(ddm.Model(dx=.1, dt=.005,
                         bound=ddm.BoundConstant(B=1.001)).can_solve_explicit())
        self.assertTrue(ddm.Model(dx=.1, dt=.005).can_solve_explicit())


@unittest.skipUnless(ddm.model.HAS_CSOLVE, 'C extension not built')
class TestCAccuracy(unittest.TestCase):
    @unittest.skipUnless(ddm.model.HAS_CSOLVE, 'C extension not available')
    def test_c_solver_rejects_invalid_inputs_without_crashing(self):
        # Run in a subprocess so a regression to an invalid pointer access
        # fails the test instead of terminating the entire test runner.
        code = r'''
import numpy as np
from pyddm import csolve
drift = np.array([0.]); noise = np.array([1.]); bound = np.array([1.])
ic = np.zeros(21); ic[10] = 1.
for fn in [csolve.implicit_time, csolve.cn_time]:
    valid = [drift, 0, noise, 0, bound, 0, ic, .1, .005, .1, 21]
    assert all(np.isfinite(a).all() for a in fn(*valid))
    invalid = []
    for index, value in [(0, np.zeros((2,2))), (1, 3), (2, np.array([])),
                         (5, 1), (6, np.array(1.)), (6, np.zeros(20)),
                         (8, 0.), (9, 0.), (10, 0), (10, -1)]:
        args = valid.copy(); args[index] = value; invalid.append(args)
    for args in invalid:
        try: fn(*args)
        except (ValueError, TypeError): pass
        else: raise AssertionError(('invalid input accepted', args))
assert len(csolve.analytic_ddm_linbound(1., 0., -1., 0., 0, .005)) == 0
try: csolve.analytic_ddm_linbound(1., 0., -1., 0., -1, .005)
except ValueError: pass
else: raise AssertionError('negative analytic length accepted')
'''
        result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_python_c_exact_initial_absorption(self):
        # These outputs contain only exactly representable zero, one, and
        # one-half masses, so literal equality is appropriate here.
        for options in [dict(bound=ZeroBound()), dict(IC=ddm.ICPoint(x0=1))]:
            model = ddm.Model(dx=.02, dt=.001, T_dur=.05, **options)
            for method in ['implicit', 'cn']:
                with self.subTest(options=options, method=method):
                    p, c = solve(model, method), solve(model, method, False)
                    np.testing.assert_array_equal(p.choice_upper, c.choice_upper)
                    np.testing.assert_array_equal(p.choice_lower, c.choice_lower)
                    self.assertEqual(p.undec is None, c.undec is None)
                    if p.undec is not None:
                        np.testing.assert_array_equal(p.undec, c.undec)

    def test_python_c_agreement_all_coefficient_modes(self):
        # Exercise all 16 C drift/noise mode combinations and both numerical
        # methods. General floating-point outputs are not bitwise identical;
        # benchmarks/check_backend_parity.py reports exact equality separately.
        for drift_mode in range(4):
            for noise_mode in range(4):
                model = ddm.Model(dx=.02, dt=.005, T_dur=.4,
                    bound=ddm.BoundConstant(B=.993),
                    drift=ddm.DriftLinear(drift=.2, x=-.25 if drift_mode & 2 else 0,
                                          t=.3 if drift_mode & 1 else 0),
                    noise=ddm.NoiseLinear(noise=.8, x=.1 if noise_mode & 2 else 0,
                                          t=.15 if noise_mode & 1 else 0))
                for method in ['implicit', 'cn']:
                    with self.subTest(drift_mode=drift_mode, noise_mode=noise_mode, method=method):
                        entrypoint = 'implicit_time' if method == 'implicit' else 'cn_time'
                        with patch.object(ddm.csolve, entrypoint, wraps=getattr(ddm.csolve, entrypoint)) as call:
                            p = solve(model, method)
                            self.assertEqual(call.call_count, 0)
                            c = solve(model, method, False)
                            self.assertEqual(call.call_count, 1)
                        for choice in model.choice_names:
                            np.testing.assert_allclose(p.pdf(choice), c.pdf(choice), atol=1e-12, rtol=1e-10)
                        if p.undec is not None:
                            np.testing.assert_allclose(p.undec, c.undec, atol=1e-12, rtol=1e-10)

    def test_analytical_image_recurrence_for_moving_bounds(self):
        times = np.arange(0, 2.0001, .002)
        for bound in [.1, 1.]:
            for slope in [-1., -.35, 0., .2]:
                for drift in [-2., 0., 2.]:
                    for shift in [.1, .5, .8]:
                        c = analytic_ddm(drift, 1, bound, times.copy(), shift=shift, b_slope=slope)
                        p = analytic_ddm(drift, 1, bound, times.copy(), shift=shift,
                                         b_slope=slope, force_python=True)
                        np.testing.assert_allclose(c, p, atol=2e-12, rtol=5e-10)

    def test_analytical_short_long_time_transition(self):
        for bound in [.1, .7, 1.3]:
            times = np.arange(0, 2.0001, .001)
            for drift in [-2., 0., 2.]:
                for shift in [.03, .3, .5, .9]:
                    c = analytic_ddm(drift, 1, bound, times.copy(), shift=shift)
                    p = analytic_ddm(drift, 1, bound, times.copy(), shift=shift, force_python=True)
                    np.testing.assert_allclose(c, p, atol=2e-12, rtol=5e-10)

    def test_analytical_tail_against_independent_eigenfunction_expansion(self):
        bound, drift, t = .1, .4, .115
        times = np.arange(0, 2.0001, .001)
        c = analytic_ddm(drift, 1, bound, times.copy(), force_python=False)[0]
        p = analytic_ddm(drift, 1, bound, times.copy(), force_python=True)[0]
        k = np.arange(60)*2+1
        reference = np.pi/(4*bound**2)*np.sum((-1.)**np.arange(60)*k
                    *np.exp(-k**2*np.pi**2*t/(8*bound**2)))*np.exp(drift*bound-drift**2*t/2)
        self.assertAlmostEqual(c[115], reference, delta=1e-12)
        np.testing.assert_allclose(c, p, atol=1e-12, rtol=1e-10)

    def test_python_c_agreement_for_fractional_and_moving_boundaries(self):
        for bound in [ddm.BoundConstant(B=1.003), ddm.BoundCollapsingLinear(B=1.003, t=.3),
                      IncreasingBound()]:
            model = ddm.Model(dx=.02, dt=.005, T_dur=.4, bound=bound,
                              drift=ddm.DriftLinear(drift=.2, x=.1, t=.3),
                              noise=ddm.NoiseLinear(noise=1, x=.2, t=.1))
            for method in ['implicit', 'cn'] if model.can_solve_cn() else ['implicit']:
                p, c = solve(model, method), solve(model, method, False)
                np.testing.assert_allclose(p.choice_upper, c.choice_upper, atol=1e-12, rtol=1e-10)
                np.testing.assert_allclose(p.choice_lower, c.choice_lower, atol=1e-12, rtol=1e-10)
                if p.undec is not None:
                    np.testing.assert_allclose(p.undec, c.undec, atol=1e-12, rtol=1e-10)

    def test_single_point_tridiagonal_solve(self):
        from pyddm import csolve
        upper, lower, undec = csolve.implicit_time(np.array([0.]), 0, np.array([.1]), 0,
                  np.array([.01]), 0, np.array([1.]), .002, .001, .1, 3)
        self.assertAlmostEqual(undec[0], (1+.001*.1**2/.01**2)**-2)
        self.assertAlmostEqual(.001*(upper.sum()+lower.sum())+undec.sum(), 1)


if __name__ == '__main__':
    unittest.main()
