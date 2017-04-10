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
from ddm import *
from ddm.plot import *

SHOW_PLOTS = True

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





# ============ Actual tests =================


def test_verify_ddm_analytic_close_to_numeric_params1():
    m = Model(dx=.005, dt=.01, T_dur=2,
              mu=MuConstant(mu=0),
              sigma=SigmaConstant(sigma=1),
              bound=BoundConstant(B=1))
    _modeltest_numerical_vs_analytical(m)

def test_verify_ddm_analytic_close_to_numeric_params2():
    m = Model(dx=.005, dt=.01, T_dur=2,
              mu=MuConstant(mu=1),
              sigma=SigmaConstant(sigma=1),
              bound=BoundConstant(B=1))
    _modeltest_numerical_vs_analytical(m)

def test_verify_ddm_analytic_close_to_numeric_params3():
    m = Model(dx=.001, dt=.0005, T_dur=2,
              mu=MuConstant(mu=1),
              sigma=SigmaConstant(sigma=.05),
              bound=BoundConstant(B=1))
    _modeltest_numerical_vs_analytical(m, max_diff=1)

def test_verify_ddm_analytic_close_to_numeric_params4():
    m = Model(dx=.005, dt=.01, T_dur=2,
              mu=MuConstant(mu=.1),
              sigma=SigmaConstant(sigma=1),
              bound=BoundConstant(B=.6))
    _modeltest_numerical_vs_analytical(m, max_diff=1)



# TODO Test to make sure increasing mean/varince decreases decision time, etc.

def test_fit_simple_ddm():
    m = Model(name="DDM", dt=.01,
              mu=MuConstant(mu=2),
              sigma=SigmaConstant(sigma=1),
              bound=BoundConstant(B=1))
    s = m.solve()
    sample = s.resample(10000)
    mfit = fit_model(sample, mu=MuConstant(mu=Fittable(minval=0, maxval=10)))
    # Within 10%
    if SHOW_PLOTS:
        mfit.name = "Fitted solution"
        sfit = mfit.solve()
        plot_compare_solutions(s, sfit)
        plt.show()

    assert abs(m._mudep.mu - mfit._mudep.mu) < 0.1 * m._mudep.mu

def test_fit_constant_mu_constant_sigma():
    m = Model(name="DDM", dt=.01,
              mu=MuConstant(mu=.1),
              sigma=SigmaConstant(sigma=1.1),
              bound=BoundConstant(B=1))
    s = m.solve()
    sample = s.resample(10000)
    mfit = fit_model(sample,
                     mu=MuConstant(mu=Fittable(minval=0.01, maxval=10)),
                     sigma=SigmaConstant(sigma=Fittable(minval=0.01, maxval=5)),
                     bound=BoundConstant(B=1))
    if SHOW_PLOTS:
        mfit.name = "Fitted solution"
        sfit = mfit.solve()
        plot_compare_solutions(s, sfit)
        plt.show()

    assert abs(m._mudep.mu - mfit._mudep.mu) < 0.1 * m._mudep.mu
    assert abs(m._sigmadep.sigma - mfit._sigmadep.sigma) < 0.1 * m._sigmadep.sigma


# def test_fit_linear_mu_constant_sigma():
#     m = Model(name="DDM", dt=.01,
#               mu=MuLinear(mu=1, x=0, t=.3),
#               sigma=SigmaConstant(sigma=.3),
#               bound=BoundConstant(B=1))
#     s = m.solve()
#     sample = s.resample(10000)
#     mfit = fit_model(sample,
#                      mu=MuLinear(mu=Fittable(minval=0.01, maxval=10), x=0, t=Fittable(minval=-5, maxval=5)),
#                      sigma=SigmaConstant(sigma=Fittable(minval=0.01, maxval=5)))
#     if SHOW_PLOTS:
#         mfit.name = "Fitted solution"
#         sfit = mfit.solve()
#         plot_compare_solutions(s, sfit)
#         plt.show()
    
#     assert abs(m._mudep.mu - mfit._mudep.mu) < 0.1 * m._mudep.mu
#     assert abs(m._sigmadep.sigma - mfit._sigmadep.sigma) < 0.1 * m._sigmadep.sigma
#     assert abs(m._mudep.t - mfit._mudep.t) < 0.1 * m._mudep.t
    

# def test_fit_linear_mu_linear_sigma():
#     m = Model(name="DDM", dt=.01,
#               mu=MuLinear(mu=1, x=0, t=.3),
#               sigma=SigmaLinear(sigma=.3, t=.7, x=0),
#               bound=BoundConstant(B=1))
#     s = m.solve()
#     sample = s.resample(10000)
#     mfit = fit_model(sample,
#                      mu=MuLinear(mu=Fittable(minval=0.01, maxval=10), x=0, t=.3),
#                      sigma=SigmaLinear(sigma=Fittable(minval=0.01, maxval=5), t=Fittable(minval=-2, maxval=2), x=0))
#     if SHOW_PLOTS:
#         mfit.name = "Fitted solution"
#         sfit = mfit.solve()
#         plot_compare_solutions(s, sfit)
#         plt.show()
    
#     assert abs(m._mudep.mu - mfit._mudep.mu) < 0.1 * m._mudep.mu
#     assert abs(m._sigmadep.sigma - mfit._sigmadep.sigma) < 0.1 * m._sigmadep.sigma
#     assert abs(m._mudep.t - mfit._mudep.t) < 0.1 * m._mudep.t
    
# ============ Testing specific features =================

# Make sure we can fit different parameters in the same (or a
# different) model using a single Fittable object
class SigmaDouble(ddm.Sigma):
    name = "time-varying sigma"
    required_parameters = ["sigma1", "sigma2"]
    def get_sigma(self, t, conditions, **kwargs):
        if numpy.random.rand() > .5:
            return self.sigma1
        else:
            return self.sigma2

class SigmaConstantButNot(ddm.Sigma):
    name = "almost sigma constant"
    required_parameters = ["sigma"]
    def get_sigma(self, t, conditions, **kwargs):
        return self.sigma

def test_shared_parameter_fitting_samemodel():
    # Generate data
    m = Model(name="DDM", 
              mu=MuConstant(mu=1),
              sigma=SigmaConstant(sigma=1.7))
    s = m.solve_numerical() # Solving analytical and then fitting numerical gives a big bias
    sample = s.resample(10000)
    mone = fit_model(sample, mu=MuConstant(mu=1),
                     sigma=SigmaConstantButNot(sigma=Fittable(minval=.5, maxval=3)))
    sigs = Fittable(minval=.5, maxval=3)
    msam = fit_model(sample, mu=MuConstant(mu=1), 
                     sigma=SigmaDouble(sigma1=sigs,
                                       sigma2=sigs))
    print(msam._sigmadep)
    print(mone._sigmadep)
    assert msam._sigmadep.sigma1 == msam._sigmadep.sigma2, "Fitting to be the same failed"
    assert abs(msam._sigmadep.sigma1 - mone._sigmadep.sigma) < 0.1 * mone._sigmadep.sigma


class MuPowerTime(ddm.Mu):
    name = "mu power with time"
    required_parameters = ["mu", "power"]
    def get_mu(self, t, conditions, **kwargs):
        return t**self.power * self.mu

class SigmaPowerTime(ddm.Sigma):
    name = "sigma power with time"
    required_parameters = ["sigma", "power"]
    def get_sigma(self, t, conditions, **kwargs):
        return t**self.power * self.sigma

def test_shared_parameter_fitting_diffmodel():
    # Generate data
    m = Model(name="DDM", 
              mu=MuPowerTime(mu=1, power=1.3),
              sigma=SigmaPowerTime(sigma=1, power=1.3))
    s = m.solve_numerical() # Solving analytical and then fitting numerical gives a big bias
    sample = s.resample(10000)
    powers = Fittable(minval=1, maxval=2)
    msam = fit_model(sample, mu=MuPowerTime(mu=1, power=powers), 
                     sigma=SigmaPowerTime(sigma=1, power=powers))
    print(msam)
    assert msam._sigmadep.power == msam._mudep.power, "Fitting to be the same failed"
    assert abs(msam._sigmadep.power - m._sigmadep.power) < 0.1 * m._sigmadep.power

def test_shared_parameter_fitting_diffmodel_thirdvar():
    # Generate data
    m = Model(name="DDM", 
              mu=MuPowerTime(mu=1.1, power=1.3),
              sigma=SigmaPowerTime(sigma=1, power=1.3))
    s = m.solve_numerical() # Solving analytical and then fitting numerical gives a big bias
    sample = s.resample(10000)
    powers = Fittable(minval=1, maxval=2)
    msam = fit_model(sample, mu=MuPowerTime(mu=Fittable(minval=.5, maxval=2), power=powers), 
                     sigma=SigmaPowerTime(sigma=1, power=powers))
    print(msam)
    assert msam._sigmadep.power == msam._mudep.power, "Fitting to be the same failed"
    assert abs(msam._sigmadep.power - m._sigmadep.power) < 0.1 * m._sigmadep.power
    assert abs(msam._mudep.mu - m._mudep.mu) < 0.1 * m._mudep.mu
