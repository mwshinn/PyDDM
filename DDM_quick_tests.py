# Invoke with pytest DDM_quick_tests.py
#
# These are some quick tests to make sure things are still running
# smoothly.  This does not check all possibilities.  It is not
# intended as a "proof" that the software is working bug-free, only as
# a first-pass to see if a change to the code caused a major feature
# to break.
#
# Some known limitations:
# 
# - When the bounds are very close to 0, or when standard deviation is
#   very small, the numerical and analytical solutions will not match
#   up very well.
# - Sometimes the fitting code doesn't work.  I attribute this to it
#   getting into a local maximum, but I have not investigated further.

import numpy as np
import numpy
import ddm
from ddm import *
from ddm.plot import *
import ddm.models as models
from ddm.models import *
from ddm.models.drift import DriftConstant, Drift
from ddm.models.noise import NoiseConstant, Noise
from ddm.models.bound import BoundConstant
from ddm.models.overlay import OverlayChain, OverlayPoissonMixture, OverlayUniformMixture
from ddm.functions import fit_model

SHOW_PLOTS = False

if SHOW_PLOTS:
    import matplotlib.pyplot as plt

# ========== Utility functions ==============    

def _modeltest_numerical_vs_analytical(m, max_diff=.1, mean_diff=.05, prob_diff=.01):
    a = m.solve_analytical()
    n = m.solve_numerical()
    if SHOW_PLOTS:
        plot_solution_pdf(a)
        plot_solution_pdf(n)
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

# ============ Actual tests =================


def test_verify_ddm_analytic_close_to_numeric_params1():
    m = Model(dx=.005, dt=.01, T_dur=2,
              drift=DriftConstant(drift=0),
              noise=NoiseConstant(noise=1),
              bound=BoundConstant(B=1))
    _modeltest_numerical_vs_analytical(m)

def test_verify_ddm_analytic_close_to_numeric_params2():
    m = Model(dx=.005, dt=.01, T_dur=2,
              drift=DriftConstant(drift=1),
              noise=NoiseConstant(noise=1),
              bound=BoundConstant(B=1))
    _modeltest_numerical_vs_analytical(m)

def test_verify_ddm_analytic_close_to_numeric_params3():
    m = Model(dx=.001, dt=.0005, T_dur=2,
              drift=DriftConstant(drift=1),
              noise=NoiseConstant(noise=.05),
              bound=BoundConstant(B=1))
    _modeltest_numerical_vs_analytical(m, max_diff=1)

def test_verify_ddm_analytic_close_to_numeric_params4():
    m = Model(dx=.005, dt=.01, T_dur=2,
              drift=DriftConstant(drift=.1),
              noise=NoiseConstant(noise=1),
              bound=BoundConstant(B=.6))
    _modeltest_numerical_vs_analytical(m, max_diff=1)



# TODO Test to make sure increasing mean/varince decreases decision time, etc.

def test_fit_simple_ddm():
    m = Model(name="DDM", dt=.01,
              drift=DriftConstant(drift=2),
              noise=NoiseConstant(noise=1),
              bound=BoundConstant(B=1))
    s = m.solve()
    sample = s.resample(10000)
    mfit = fit_model(sample, drift=DriftConstant(drift=Fittable(minval=0, maxval=10)))
    # Within 10%
    if SHOW_PLOTS:
        mfit.name = "Fitted solution"
        sfit = mfit.solve()
        plot_compare_solutions(s, sfit)
        plt.show()

    _verify_param_match("drift", "drift", m, mfit)

# def test_fit_constant_drift_constant_noise():
#     m = Model(name="DDM",
#               drift=DriftConstant(drift=1.1),
#               noise=NoiseConstant(noise=.3),
#               bound=BoundConstant(B=1))
#     s = m.solve()
#     sample = s.resample(10000)
#     mfit = fit_model(sample,
#                      drift=DriftConstant(drift=Fittable(minval=0.01, maxval=10)),
#                      noise=NoiseConstant(noise=Fittable(minval=0.01, maxval=5)),
#                      bound=BoundConstant(B=1))
#     if SHOW_PLOTS:
#         mfit.name = "Fitted solution"
#         sfit = mfit.solve()
#         plot_compare_solutions(s, sfit)
#         plt.show()
    
#     _verify_param_match("drift", "drift", mfit, m)
#     _verify_param_match("noise", "noise", m, mfit)


# def test_fit_linear_drift_constant_noise():
#     m = Model(name="DDM", dt=.01,
#               drift=DriftLinear(drift=1, x=0, t=.3),
#               noise=NoiseConstant(noise=.3),
#               bound=BoundConstant(B=1))
#     s = m.solve()
#     sample = s.resample(10000)
#     mfit = fit_model(sample,
#                      drift=DriftLinear(drift=Fittable(minval=0.01, maxval=10), x=0, t=Fittable(minval=-5, maxval=5)),
#                      noise=NoiseConstant(noise=Fittable(minval=0.01, maxval=5)))
#     if SHOW_PLOTS:
#         mfit.name = "Fitted solution"
#         sfit = mfit.solve()
#         plot_compare_solutions(s, sfit)
#         plt.show()
    
#     assert abs(m._driftdep.drift - mfit._driftdep.drift) < 0.1 * m._driftdep.drift
#     assert abs(m._noisedep.noise - mfit._noisedep.noise) < 0.1 * m._noisedep.noise
#     assert abs(m._driftdep.t - mfit._driftdep.t) < 0.1 * m._driftdep.t
    

# def test_fit_linear_drift_linear_noise():
#     m = Model(name="DDM", dt=.01,
#               drift=DriftLinear(drift=1, x=0, t=.3),
#               noise=NoiseLinear(noise=.3, t=.7, x=0),
#               bound=BoundConstant(B=1))
#     s = m.solve()
#     sample = s.resample(10000)
#     mfit = fit_model(sample,
#                      drift=DriftLinear(drift=Fittable(minval=0.01, maxval=10), x=0, t=.3),
#                      noise=NoiseLinear(noise=Fittable(minval=0.01, maxval=5), t=Fittable(minval=-2, maxval=2), x=0))
#     if SHOW_PLOTS:
#         mfit.name = "Fitted solution"
#         sfit = mfit.solve()
#         plot_compare_solutions(s, sfit)
#         plt.show()
    
#     assert abs(m._driftdep.drift - mfit._driftdep.drift) < 0.1 * m._driftdep.drift
#     assert abs(m._noisedep.noise - mfit._noisedep.noise) < 0.1 * m._noisedep.noise
#     assert abs(m._driftdep.t - mfit._driftdep.t) < 0.1 * m._driftdep.t
    
# ============ Testing specific features =================

# Make sure we can fit different parameters in the same (or a
# different) model using a single Fittable object
class NoiseDouble(Noise):
    name = "time-varying noise"
    required_parameters = ["noise1", "noise2"]
    def get_noise(self, t, conditions, **kwargs):
        if numpy.random.rand() > .5:
            return self.noise1
        else:
            return self.noise2

class NoiseConstantButNot(Noise): # To avoid the numerical simulations
    name = "almost noise constant"
    required_parameters = ["noise"]
    def get_noise(self, t, conditions, **kwargs):
        return self.noise

# def test_shared_parameter_fitting_samemodel():
#     # Generate data
#     m = Model(name="DDM",
#               drift=DriftConstant(drift=1),
#               noise=NoiseConstant(noise=1.7))
#     s = m.solve_numerical() # Solving analytical and then fitting numerical gives a big bias
#     sample = s.resample(10000)
#     mone = fit_model(sample, drift=DriftConstant(drift=1),
#                      noise=NoiseConstantButNot(noise=Fittable(minval=.5, maxval=3)))
#     sigs = Fittable(minval=.5, maxval=3)
#     msam = fit_model(sample, drift=DriftConstant(drift=1),
#                      noise=NoiseDouble(noise1=sigs,
#                                        noise2=sigs))
#     print(msam._noisedep)
#     print(mone._noisedep)
#     assert msam._noisedep.noise1 == msam._noisedep.noise2, "Fitting to be the same failed"
#     assert abs(msam._noisedep.noise1 - mone._noisedep.noise) < 0.1 * mone._noisedep.noise


class DriftPowerTime(Drift):
    name = "drift power with time"
    required_parameters = ["drift", "power"]
    def get_drift(self, t, conditions, **kwargs):
        return t**self.power * self.drift

class NoisePowerTime(Noise):
    name = "noise power with time"
    required_parameters = ["noise", "power"]
    def get_noise(self, t, conditions, **kwargs):
        return t**self.power * self.noise

# def test_shared_parameter_fitting_diffmodel():
#     # Generate data
#     m = Model(name="DDM", 
#               drift=DriftPowerTime(drift=1, power=1.3),
#               noise=NoisePowerTime(noise=1, power=1.3))
#     s = m.solve_numerical() # Solving analytical and then fitting numerical gives a big bias
#     sample = s.resample(10000)
#     powers = Fittable(minval=1, maxval=2)
#     msam = fit_model(sample, drift=DriftPowerTime(drift=1, power=powers), 
#                      noise=NoisePowerTime(noise=1, power=powers))
#     print(msam)
#     assert msam._noisedep.power == msam._driftdep.power, "Fitting to be the same failed"
#     _verify_param_match("noise", "power", m, msam)

def test_shared_parameter_fitting_diffmodel_thirdvar():
    # Generate data
    m = Model(name="DDM", 
              drift=DriftPowerTime(drift=1.1, power=1.3),
              noise=NoisePowerTime(noise=1, power=1.3))
    s = m.solve_numerical() # Solving analytical and then fitting numerical gives a big bias
    sample = s.resample(10000)
    powers = Fittable(minval=1, maxval=2)
    msam = fit_model(sample, drift=DriftPowerTime(drift=Fittable(minval=.5, maxval=2), power=powers), 
                     noise=NoisePowerTime(noise=1, power=powers))
    print(msam)
    assert msam._noisedep.power == msam._driftdep.power, "Fitting to be the same failed"
    _verify_param_match("noise", "power", msam, m)
    _verify_param_match("drift", "drift", m, msam)

# Test the overlays

# def test_poisson_overlay():
#     m = Model(name="Poisson_test", drift=DriftConstant(drift=1),
#               overlay=OverlayPoissonMixture(mixturecoef=.1, rate=.3), dt=.004)
#     s = m.solve_numerical()
#     sample = s.resample(10000)
#     f = fit_model(sample, drift=DriftConstant(drift=Fittable(minval=0, maxval=3)),
#                   overlay=OverlayPoissonMixture(mixturecoef=Fittable(minval=.001, maxval=.2),
#                                                 rate=Fittable(minval=.1, maxval=1)))
#     plot.plot_compare_solutions(s, f.solve_numerical())
#     print(f)
#     _verify_param_match("drift", "drift", m, f)
#     _verify_param_match("overlay", "mixturecoef", m, f)
#     _verify_param_match("overlay", "rate", m, f)

def test_no_overlay():
    m = Model(name="Overlay", drift=DriftConstant(drift=2), overlay=OverlayNone())
    s = m.solve_numerical()
    sample = s.resample(10000)
    f = fit_model(sample, drift=DriftConstant(drift=Fittable(minval=0, maxval=3)))
    plot_compare_solutions(s, f.solve_numerical())
    print(f)
    _verify_param_match("drift", "drift", m, f)

def test_uniform_overlay():
    m = Model(name="Overlay", drift=DriftConstant(drift=2), overlay=OverlayUniformMixture(umixturecoef=.1))
    s = m.solve_numerical()
    sample = s.resample(10000)
    f = fit_model(sample, drift=DriftConstant(drift=Fittable(minval=0, maxval=3)),
                  overlay=OverlayUniformMixture(umixturecoef=Fittable(minval=.001, maxval=.5)))
    plot_compare_solutions(s, f.solve_numerical())
    print(f)
    _verify_param_match("drift", "drift", m, f)
    _verify_param_match("overlay", "umixturecoef", m, f)

# See how sensitive a fitting method is to a single outlier.  Here, we
# add one outlier to the error trials near the end of the time window.
# def test_parameter_sensitivity_poisson():
#     m = Model(name="Poisson_test", drift=DriftConstant(drift=4), noise=NoiseConstant(noise=.5),
#               overlay=OverlayPoissonMixture(mixturecoef=.2, rate=.2), dt=.001, dx=.001)
#     s = m.solve_numerical()
#     sample = s.resample(10000)
#     sample.err[0] = 1.9
#     f = fit_model(sample, drift=DriftConstant(drift=Fittable(minval=0, maxval=6)), noise=NoiseConstant(noise=.5),
#                   overlay=OverlayPoissonMixture(mixturecoef=Fittable(minval=.001, maxval=.5),
#                                                 rate=Fittable(minval=.1, maxval=10)), lossfunction=LossBIC, dt=.001, dx=.001)
#     plot.plot_compare_solutions(s, f.solve_numerical())
#     print(f)
#     _verify_param_match("drift", "drift", m, f)
#     _verify_param_match("overlay", "mixturecoef", m, f)
#     _verify_param_match("overlay", "rate", m, f)


# In the following tests, we set drift to be high enough such that as
# little fraction as possible goes past T-dur, i.e. then we get fewer
# undecided trials, which bring the probability down from 1.

def test_overlay_uniform_distribution_integrates_to_1():
    m = Model(name="Overlay_test", drift=DriftConstant(drift=2), overlay=OverlayUniformMixture(umixturecoef=.2))
    s = m.solve_numerical()
    distsum = s.prob_correct() + s.prob_error()
    assert .98 < distsum < 1.0001, "Distribution doesn't sum to 1"

def test_overlay_poisson_distribution_integrates_to_1():
    m = Model(name="Overlay_test", drift=DriftConstant(drift=2), overlay=OverlayPoissonMixture(pmixturecoef=.2, rate=2))
    s = m.solve_numerical()
    distsum = s.prob_correct() + s.prob_error()
    assert .98 < distsum < 1.0001, "Distribution doesn't sum to 1"

def test_overlay_chain_distribution_integrates_to_1():
    m = Model(name="Overlay_test", drift=DriftConstant(drift=2),
              overlay=OverlayChain(overlays=[OverlayPoissonMixture(pmixturecoef=.2, rate=2),
                                             OverlayUniformMixture(umixturecoef=.2)]))
    s = m.solve_numerical()
    distsum = s.prob_correct() + s.prob_error()
    assert .98 < distsum < 1.0001, "Distribution doesn't sum to 1"
