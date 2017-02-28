from .model import *
import matplotlib.pyplot as plt

def plot_solution_pdf(sol, ax=None, correct=True):
    """Plot the PDF of the solution.

    - `ax` is optionally the matplotlib axis on which to plot.
    - If `correct` is true, we draw the distribution of correct
      answers.  Otherwise, we draw the error distributions.

    This does not return anything, but it plots the PDF.  It does not
    show it, and thus requires a call to plt.show() to see.
    """
    if ax == None:
        ax = plt.gca()

    if correct == True:
        ts = sol.pdf_corr()
    else:
        ts = sol.pdf_err()
    
    ax.plot(sol.model.t_domain(), ts, label=sol.model.name)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('PDF (normalized)')
    
def plot_solution_cdf(sol, ax=None, correct=True):
    """Plot the CDF of the solution.

    - `ax` is optionally the matplotlib axis on which to plot.
    - If `correct` is true, we draw the distribution of correct
      answers.  Otherwise, we draw the error distributions.

    This does not return anything, but it plots the CDF.  It does not
    show it, and thus requires a call to plt.show() to see.
    """
    if ax == None:
        ax = plt.gca()

    if correct == True:
        ts = sol.cdf_corr()
    else:
        ts = sol.cdf_err()
    
    ax.plot(sol.model.t_domain(), ts, label=sol.model.name)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('CDF (normalized)')
    
def plot_fit_diagnostics(model, sample):
    """Compare actual data to the best fit model of the data.

    - `model` should be the Model object fit from `rt_data`.
    - `sample` should be a Sample object describing the data
    """
    T_dur = model.T_dur
    dt = model.dt
    total_samples = len(sample)
    print(T_dur, dt, total_samples, len(sample.corr), len(sample.err))
    data_hist_corr = np.histogram(sample.corr, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0] # dt/2 terms are for continuity correction
    data_hist_err = np.histogram(sample.err, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0]
    # First create an empty array the same length as the histogram.
    # Then, for each type of model in the sample, add it to the plot
    # weighted by its frequency of occuring.
    model_corr = np.histogram([], bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0].astype(float) # dt/2 terms are for continuity correction
    model_err = np.histogram([], bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0].astype(float)
    for conds in sample.condition_combinations(): # TODO make None the default in the API
        print(conds)
        subset = sample.subset(**conds)
        sol = model.solve(conditions=conds)
        model_corr += len(subset)/len(sample)*sol.pdf_corr()
        model_err += len(subset)/len(sample)*sol.pdf_err()
        print(sum(model_corr)+sum(model_err))
    plt.subplot(2, 1, 1)
    plt.plot(model.t_domain(), np.asarray(data_hist_corr)/total_samples/dt, label="Data") # Divide by samples and dt to scale to same size as solution pdf
    plt.plot(model.t_domain(), model_corr, label="Fit")
    print(sum(data_hist_corr/total_samples/dt)+sum(data_hist_err/total_samples/dt))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(model.t_domain(), np.asarray(data_hist_err)/total_samples/dt, label="Data")
    plt.plot(model.t_domain(), model_err, label="Fit")

def plot_compare_solutions(s1, s2):
    """Compare two model solutions to each other.

    `s1` and `s2` should be solution objects.  This will display a
    pretty picture of the correct and error distribution pdfs.
    """
    plt.subplot(2, 1, 1)
    plot_solution_pdf(s1)
    plot_solution_pdf(s2)
    plt.legend()
    plt.subplot(2, 1, 2)
    plot_solution_pdf(s1, correct=False)
    plot_solution_pdf(s2, correct=False)
