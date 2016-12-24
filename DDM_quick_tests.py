# Invoke with pytest DDM_quick_tests.py
#
# TODO not all of these pass yet, in particular, the analytic vs
# numeric ones.  Find out if this is a bug or too low tolerance.

import numpy as np
from DDM_model import *
from DDM_plot import *
from DDM_functions import fit_model, fit_model_stable

SHOW_PLOTS = True

if SHOW_PLOTS:
    import matplotlib.pyplot as plt


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
    _modeltest_numerical_vs_analytical(m)

def test_verify_ddm_analytic_close_to_numeric_params4():
    m = Model(dx=.005, dt=.01, T_dur=2,
              mu=MuConstant(mu=.4),
              sigma=SigmaConstant(sigma=1),
              bound=BoundConstant(B=.3))
    _modeltest_numerical_vs_analytical(m)



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


# TODO Test to make sure increasing mean/varince decreases decision time, etc.

def test_fit_simple_ddm():
    m1 = Model(name="DDM", 
               mu=MuConstant(mu=1),
               sigma=SigmaConstant(sigma=1),
               bound=BoundConstant(B=1))

    s1 = m1.solve()

    m1fit = fit_model_stable([s1.pdf_corr(), s1.pdf_err()],
                             mu=MuConstant(mu=Fittable()))

    assert abs(m1._mudep.mu - m1fit._mudep.mu) < .01

    if SHOW_PLOTS:
        m1fit.name = "Fitted solution"
        s1_fit = m1fit.solve()
        plot_solution_pdf(s1)
        plot_solution_pdf(s1_fit)
        plt.legend()
        plt.show()

def test_fit_constant_mu_constant_sigma():
    m2 = Model(name="DDM",
               mu=MuConstant(mu=1.1),
               sigma=SigmaConstant(sigma=.3),
               bound=BoundConstant(B=1))

    s2 = m2.solve()
    m2fit = fit_model_stable([s2.pdf_corr(), s2.pdf_err()],
                             mu=MuConstant(mu=Fittable(minval=0)),
                             sigma=SigmaConstant(sigma=Fittable(minval=0)))

    assert abs(m2._mudep.mu - m2fit._mudep.mu) < .01
    assert abs(m2._sigmadep.sigma - m2fit._sigmadep.sigma) < .01
    
    if SHOW_PLOTS:
        s2_fit = m2fit.solve()
        plot_solution_pdf(s2)
        plot_solution_pdf(s2_fit)
        plt.legend()
        plt.show()


def test_fit_linear_mu_constant_sigma():
    m3 = Model(name="DDM", 
               mu=MuLinear(mu=1, x=0, t=.3),
               sigma=SigmaConstant(sigma=.3),
               bound=BoundConstant(B=1))

    s3 = m3.solve()

    m3fit = fit_model_stable([s3.pdf_corr(), s3.pdf_err()],
                             mu=MuLinear(mu=Fittable(minval=0), x=0, t=Fittable()),
                             sigma=SigmaConstant(sigma=Fittable(minval=0)))

    assert abs(m3._mudep.mu - m3fit._mudep.mu) < .01
    assert abs(m3._sigmadep.sigma - m3fit._sigmadep.sigma) < .01
    assert abs(m3._mudep.t - m3fit._mudep.t) < .01
    
    if SHOW_PLOTS:
        m3fit.name = "Fitted solution"
        s3_fit = m3fit.solve()
        plot_solution_pdf(s3)
        plot_solution_pdf(s3_fit)
        plt.legend()
        plt.show()
