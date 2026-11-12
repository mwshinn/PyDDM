"""Conservation and accuracy regressions for the Fokker--Planck solvers."""
import unittest
from unittest.mock import patch

import numpy as np

import pyddm as ddm
from pyddm.analytic import analytic_ddm


def solve(model, method, python=True):
    return model.solve_numerical(method=method, force_python=python)


def cn_density_error(solution, reference):
    # CN stores quadrature-weighted mass at the existing time nodes. The
    # endpoint nodes have half weights; compare like-for-like with the PDF.
    reference = reference.copy()
    reference[:, [0, -1]] *= .5
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
                        def capture(self, upper, lower, conditions, undec=None, evolution=None):
                            raw['absorbed'] = upper.sum()+lower.sum()
                            if evolution is not None:
                                raw['survival'] = evolution[:, -1].sum()
                            return finish(self, upper, lower, conditions, undec, evolution)
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
