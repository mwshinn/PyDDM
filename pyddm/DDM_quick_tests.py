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
    m1 = Model(name="DDM", 
               mu=MuConstant(mu=2),
               sigma=SigmaConstant(sigma=1),
               bound=BoundConstant(B=1))
    s1 = m1.solve()
    sample = s1.resample(10000)
    non_decision = 10000-(len(sample[0])+len(sample[1]))
    m1fit = fit_model_stable(sample[0], sample[1], non_decision,
                             mu=MuConstant(mu=Fittable(minval=0)))
    # Within 10%
    if SHOW_PLOTS:
        m1fit.name = "Fitted solution"
        s1fit = m1fit.solve()
        plot_compare_solutions(s1, s1fit)
        plt.show()

    assert abs(m1._mudep.mu - m1fit._mudep.mu) < 0.1 * m1._mudep.mu

def test_fit_constant_mu_constant_sigma():
    m2 = Model(name="DDM",
               mu=MuConstant(mu=.1),
               sigma=SigmaConstant(sigma=1.1),
               bound=BoundConstant(B=1))
    s2 = m2.solve()
    sample = s2.resample(10000)
    non_decision = 10000-(len(sample[0])+len(sample[1]))
    m2fit = fit_model_stable(sample[0], sample[1], non_decision,
                             mu=MuConstant(mu=Fittable(minval=0.01)),
                             sigma=SigmaConstant(sigma=Fittable(minval=0.01)),
                             bound=BoundConstant(B=1))
    if SHOW_PLOTS:
        m2fit.name = "Fitted solution"
        s2fit = m2fit.solve()
        plot_compare_solutions(s2, s2fit)
        plt.show()

    assert abs(m2._mudep.mu - m2fit._mudep.mu) < 0.1 * m2._mudep.mu
    assert abs(m2._sigmadep.sigma - m2fit._sigmadep.sigma) < 0.1 * m2._sigmadep.sigma


def test_fit_linear_mu_constant_sigma():
    m3 = Model(name="DDM", 
               mu=MuLinear(mu=1, x=0, t=.3),
               sigma=SigmaConstant(sigma=.3),
               bound=BoundConstant(B=1))
    s3 = m3.solve()
    sample = s3.resample(10000)
    non_decision = 10000-(len(sample[0])+len(sample[1]))
    s3 = m3.solve()
    m3fit = fit_model_stable(sample[0], sample[1], non_decision,
                             mu=MuLinear(mu=Fittable(minval=0.01), x=0, t=Fittable()),
                             sigma=SigmaConstant(sigma=Fittable(minval=0.01)))
    if SHOW_PLOTS:
        m3fit.name = "Fitted solution"
        s3fit = m3fit.solve()
        plot_compare_solutions(s3, s3fit)
        plt.show()
    
    assert abs(m3._mudep.mu - m3fit._mudep.mu) < 0.1 * m3._mudep.mu
    assert abs(m3._sigmadep.sigma - m3fit._sigmadep.sigma) < 0.1 * m3._sigmadep.sigma
    assert abs(m3._mudep.t - m3fit._mudep.t) < 0.1 * m3._mudep.t
    
