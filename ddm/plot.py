# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import copy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .model import *
from .parameters import dt as default_dt, dx as default_dx
from .functions import solve_partial_conditions

def plot_solution_pdf(sol, ax=None, correct=True):
    """Plot the PDF of the solution.

    - `ax` is optionally the matplotlib axis on which to plot.
    - If `correct` is true, we draw the distribution of correct
      answers.  Otherwise, we draw the error distributions.

    This does not return anything, but it plots the PDF.  It does not
    show it, and thus requires a call to plt.show() to see.
    """
    if ax is None:
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
    if ax is None:
        ax = plt.gca()

    if correct == True:
        ts = sol.cdf_corr()
    else:
        ts = sol.cdf_err()
    
    ax.plot(sol.model.t_domain(), ts, label=sol.model.name)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('CDF (normalized)')
    

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

                            
def plot_decision_variable_distribution(model, conditions={}, resolution=.1, figure=None):
    """Show the distribution of the decision variable.

    Show the intermediate distributions for the decision variable.
    `model` should be the model to plot, and `conditions` should be
    the conditions over which to plot it.  `resolution` should be the
    timestep of the plot (NOT of the model).  Optionally, `figure` is
    an existing figure on which to make the plot.

    Note that currently this is O(1/n) with resolution which is quite
    slow.  The default resolution of 0.1 seconds should strike a good
    balance between precision and runtime.
    
    Also, note that for clarity of the visualization, the square root
    of the distribution is plotted instead of the distribution itself.
    Without this, it is quite difficult to see the evolving
    distribution because the distribution of histogram values is
    highly skewed.

    Finally, note that this routine always uses the implicit method
    because it gives the most reliable histograms for the decision
    variable.  (Crank-Nicoloson tends to oscillate.)
    """


    # Generate the distributions.  Note that this is extremely
    # inefficient (it is O(n) with resolution and should be O(1) with
    # resolution) so this should be improved someday...
    hists = []
    old_T_dur = model.T_dur
    hists.append(model.get_dependence("IC").get_IC(x=model.x_domain(conditions=conditions), conditions=conditions))
    for i in range(1, int(old_T_dur/resolution)+1):
        print(i*resolution)
        model.T_dur = i*resolution
        s = model.solve_numerical_implicit(conditions=conditions)
        hists.append(s.pdf_undec())
        top = s.pdf_corr()
        bot = s.pdf_err()
    model.T_dur = old_T_dur
    # Plot the output
    f = figure if figure is not None else plt.figure()
    # Set up three axes, with one in the middle and two on the borders
    gs = plt.GridSpec(7, 1, wspace=0, hspace=0)
    ax_main = f.add_subplot(gs[1:-1,0])
    ax_top = f.add_subplot(gs[0,0], sharex=ax_main)
    ax_bot = f.add_subplot(gs[-1,0], sharex=ax_main)
    # Show the relevant data on those axes
    ax_main.imshow(np.sqrt(np.flipud(np.transpose(hists))), aspect='auto', interpolation='bicubic')
    ax_top.plot(np.linspace(0, len(hists)-1, len(top)), top, clip_on=False)
    ax_bot.plot(np.linspace(0, len(hists)-1, len(top)), -bot, clip_on=False)
    # Make them look decent
    ax_main.axis("off")
    ax_top.axis("off")
    ax_bot.axis("off")
    # Set axes to be the right size
    maxval = np.max([top, bot])
    ax_top.set_ylim(0, maxval)
    ax_bot.set_ylim(-maxval, 0)
    return f

def plot_fit_diagnostics(model=None, sample=None, fig=None, conditions=None, data_dt=None, method=None):
    """Visually assess model fit.

    This function plots a model on top of data, primarily for the
    purpose of assessing the model fit.  The plot can be configured
    with the following arguments:

    - `model` - The model object to plot.  None of the parameters
      should be "Fittable" instances, they should all be either
      "Fitted" or numbers.
    - `sample` - A sample, normally the sample used to fit the model.
    - `fig` - A matplotlib figure object.  If not passed, the current
      figure will be used.
    - `conditions` - Optionally restrict the conditions of the model
      to those specified, in a format which could be passed to
      Sample.subset.
    - `data_dt` - Bin size to use for the data histogram.  Defaults to
      the model's dt.
    - `method` - Optionally the method to use to solve the model,
      either "analytical", "numerical" "cn", "implicit", "explicit",
      or None (auto-select, the default).
    """
    # Avoid stupid errors with mutable objects
    if conditions is None:
        conditions = {}
    # Create a figure if one is not given
    if fig is None:
        fig = plt.gcf()
    
    # We use these a lot, hence the shorthand
    if model:
        dt = model.dt if not data_dt else data_dt
        T_dur = model.T_dur
        t_domain = model.t_domain()
    elif sample:
        dt = .01 if not data_dt else data_dt # sample dt
        T_dur = max(sample)
        t_domain = np.linspace(0, T_dur, T_dur/dt+1)
    else:
        raise ValueError("Must specify non-empty model or sample in arguments")
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    # If a sample is given, plot it behind the model.
    sample_cond = sample.subset(**conditions)
    if sample:
        data_hist_top = np.histogram(sample_cond.corr, bins=int(T_dur/dt)+1, range=(0-dt/2, T_dur+dt/2))[0]
        data_hist_bot = np.histogram(sample_cond.err, bins=int(T_dur/dt)+1, range=(0-dt/2, T_dur+dt/2))[0]
        total_samples = len(sample_cond)
        data_t_domain = np.linspace(0, T_dur, T_dur/dt+1)
        ax1.plot(data_t_domain, np.asarray(data_hist_top)/total_samples/dt, label="Data", alpha=.5)
        ax2.plot(data_t_domain, np.asarray(data_hist_bot)/total_samples/dt, label="Data", alpha=.5)
    if model:
        s = solve_partial_conditions(model, sample_cond, conditions=conditions, method=method)
        ax1.plot(t_domain, s.pdf_corr(), lw=2, color='red')
        ax2.plot(t_domain, s.pdf_err(), lw=2, color='red')
    ax1.axis([0, T_dur, 0, None])
    ax2.axis([0, T_dur, 0, None])
    ax1.set_title("Correct RTs")
    ax2.set_title("Error RTs")
    pt = fig.suptitle("")
    plt.tight_layout()


# TODO sample is not optional
def model_gui(model,
              sample=None,
              pool=None,
              data_dt=None,
              plot=plot_fit_diagnostics):
    """Mess around with model parameters visually.

    This allows you to see how the model `model` would be affected by
    various changes in parameter values.  It also allows you to easily
    plot `sample` conditioned on different conditions.

    First, the sample is optional.  If provided, it will be displayed
    in the background.

    Second, the function `plot` allows you to change what is plotted.
    By default, it is plot_fit_diagnostics.  If you would like to
    define your own custom function, it must take four keyword
    arguments: "model", the model to plot, "sample", an optional
    (defaulting to None) Sample object to potentially compare to the
    model, "fig", an optional (defaulting to None) matplotlib figure
    to plot on, and "conditions", the conditions selected for
    plotting.  It should not return anything, but it should draw the
    figure on "fig".

    Because sometimes the model is run in very high resolution,
    `data_dt` allows you to set the bin width for `sample`.

    Some of this code is taken from `fit_model`.
    """
    # Loop through the different components of the model and get the
    # parameters that are fittable.  Save the "Fittable" objects in
    # "params".  Since the name is not saved in the parameter object,
    # save them in a list of the same size called "paramnames".  (We
    # can't use a dictonary because some parameters have the same
    # name.)  Create a list of functions to set the value of these
    # parameters, named "setters".
    if model:
        components_list = [model.get_dependence("drift"),
                           model.get_dependence("noise"),
                           model.get_dependence("bound"),
                           model.get_dependence("IC"),
                           model.get_dependence("overlay")]
        # All of the conditions required by at least one of the model
        # components.
        required_conditions = list(set([x for l in components_list for x in l.required_conditions]))
    elif sample:
        components_list = []
        required_conditions = sample.condition_names()
    else:
        print("Must define model, sample, or both")
        return
    
    
    params = [] # A list of all of the Fittables that were passed.
    setters = [] # A list of functions which set the value of the corresponding parameter in `params`
    paramnames = [] # The names of the parameters
    for component in components_list:
        for param_name in component.required_parameters: # For each parameter in the model
            pv = getattr(component, param_name) # Parameter value in the object
            if isinstance(pv, Fittable): # If this was fit (or can be fit) via optimization
                # Create a function which sets each parameter in the
                # list to some value `a` for model `x`.  Note the
                # default arguments to the function are necessary here
                # to preserve scope.  Without them, these variables
                # would be interpreted in the local scope, so they
                # would be equal to the last value encountered in the
                # loop.
                def setter(x,a,pv=pv,component=component,param_name=param_name):
                    if not isinstance(a, Fittable):
                        a = pv.make_fitted(a)
                    setattr(x.get_dependence(component.depname), param_name, a)
                    # Return the fitted instance so we can chain it.
                    # This way, if the same Fittable object is passed,
                    # the same Fitted object will be in both places in
                    # the solution.
                    return a 
                
                # If we have the same Fittable object in two different
                # components inside the model, we only want the
                # Fittable object in the list "params" once, but we
                # want the setter to update both.  We use 'id' because
                # we only want this to be the case with an identical
                # parameter object, not just an identical name/value.
                if id(pv) in map(id, params):
                    pind = list(map(id, params)).index(id(pv))
                    oldsetter = setters[pind]
                    # This is a hack way of executing two functions in
                    # a single function call while passing forward the
                    # same argument object (not just the same argument
                    # value)
                    newsetter = lambda x,a,setter=setter,oldsetter=oldsetter : oldsetter(x,setter(x,a))
                    setters[pind] = newsetter
                    paramnames[pind] += "/"+param_name # "/" for cosmetics for multiple parameters
                else: # This setter is unique (so far)
                    params.append(pv)
                    setters.append(setter)
                    paramnames.append(param_name)
    # Since we don't want to modify the original model, duplicate it,
    # and then use that base model in the optimization routine.  (We
    # can't duplicate it earlier in this function or else duplicated
    # parameters will have separate setters since they will no
    # longer have the same id.
    m = copy.deepcopy(model) if model else None
    
    # Grid of the Fittables, replacing them with the default values.
    x_0 = [] # Default parameter values
    for p,s in zip(params, setters):
        # Save the default
        default = p.default()
        x_0.append(default)
        # Set the default
        s(m, default)
    
    # Initialize the TK (tkinter) subsystem.
    root = tk.Tk()    
    root.wm_title("Model: %s" % m.name if m else "Data")
    root.grid_columnconfigure(1, weight=0)
    root.grid_columnconfigure(2, weight=2)
    root.grid_columnconfigure(3, weight=1)
    root.grid_columnconfigure(4, weight=0)
    root.grid_rowconfigure(0, weight=1)
    
    # Creates a widget for a matplotlib figure.  Anything drawn to
    # this figure can be displayed by calling canvas.draw().
    fig = Figure()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=0, column=2, sticky="nswe")
    fig.text(.5, .5, "Loading...")
    canvas.show()
    canvas.draw()
    
    def update():
        """Redraws the plot according to the current parameters of the model
        and the selected conditions."""
        current_conditions = {c : condition_vars_values[i][condition_vars[i].get()] for i,c in enumerate(required_conditions) if condition_vars[i].get() != "All"}
        fig.clear()
        plot(model=m, fig=fig, sample=sample, conditions=current_conditions, data_dt=data_dt)
        canvas.draw()
    
    def value_changed():
        """Calls update() if the real time checkbox is checked.  Triggers when a value changes on the sliders or the condition radio buttons"""
        if real_time.get() == True:
            update()
    
    # Draw the radio buttons allowing the user to select conditions
    frame_params_container = tk.Canvas(root, bd=2, width=100)
    frame_params_container.grid(row=0, column=0, sticky="nesw")
    scrollbar_params = tk.Scrollbar(root, command=frame_params_container.yview)
    scrollbar_params.grid(row=0, column=1, sticky="ns")
    frame_params_container.configure(yscrollcommand = scrollbar_params.set)

    frame = tk.Frame(master=frame_params_container)
    windowid_params = frame_params_container.create_window((0,0), window=frame, anchor='nw')
    # Get the sizing right
    def adjust_window_params(e, wid=windowid_params, c=frame_params_container):
        c.configure(scrollregion=frame_params_container.bbox('all'))
        c.itemconfig(wid, width=e.width)
    frame_params_container.bind("<Configure>", adjust_window_params)

    
    #frame = tk.Frame(master=root)
    #frame.grid(row=0, column=0, sticky="nw")
    condition_names = required_conditions
    if required_conditions is not None:
        condition_names = [n for n in condition_names if n in required_conditions]
    condition_vars = [] # Tk variables for condition values (set by radio buttons)
    condition_vars_values = [] # Corresponds to the above, but with numerical values instead of strings
    for i,cond in enumerate(condition_names):
        lframe = tk.LabelFrame(master=frame, text=cond)
        lframe.pack(expand=True, anchor=tk.W)
        thisvar = tk.StringVar()
        condition_vars.append(thisvar)
        b = tk.Radiobutton(master=lframe, text="All", variable=thisvar, value="All", command=value_changed)
        b.pack(anchor=tk.W)
        for cv in sample.condition_values(cond):
            b = tk.Radiobutton(master=lframe, text=str(cv), variable=thisvar, value=cv, command=value_changed)
            b.pack(anchor=tk.W)
        condition_vars_values.append({str(cv) : cv for cv in sample.condition_values(cond)})
        thisvar.set("All")
    
    # And now create the sliders.  While we're at it, get rid of the
    # Fittables, replacing them with the default values.
    if params: # Make sure there is at least one parameter
        # Allow a scrollbar
        frame_sliders_container = tk.Canvas(root, bd=2, width=200)
        frame_sliders_container.grid(row=0, column=3, sticky="nsew")
        scrollbar = tk.Scrollbar(root, command=frame_sliders_container.yview)
        scrollbar.grid(row=0, column=4, sticky="ns")
        frame_sliders_container.configure(yscrollcommand = scrollbar.set)
        
        # Construct the region with sliders
        frame_sliders = tk.LabelFrame(master=frame_sliders_container, text="Parameters")
        windowid = frame_sliders_container.create_window((0,0), window=frame_sliders, anchor='nw')
        # Get the sizing right
        def adjust_window(e, wid=windowid, c=frame_sliders_container):
            c.configure(scrollregion=frame_sliders_container.bbox('all'))
            c.itemconfig(wid, width=e.width)
        frame_sliders_container.bind("<Configure>", adjust_window)
    widgets = [] # To set the value programmatically in, e.g., set_defaults
    for p,s,name in zip(params, setters, paramnames):
        # Calculate slider constraints
        minval = p.minval if p.minval > -np.inf else None
        maxval = p.maxval if p.maxval < np.inf else None
        slidestep = (maxval-minval)/200 if maxval and minval else .01
        # Function for the slider change.  A hack to execute both the
        # value changed function and set the value in the model.
        onchange = lambda x,s=s : [s(m, float(x)), value_changed()]
        # Create the slider and set its value
        slider = tk.Scale(master=frame_sliders, label=name, from_=minval, to=maxval, resolution=slidestep, orient=tk.HORIZONTAL, command=onchange)
        slider.set(default)
        slider.pack(expand=True, fill="both")
        widgets.append(slider)
        
    def set_defaults():
        """Set default slider (model parameter) values"""
        for w,default,s in zip(widgets,x_0,setters):
            w.set(default)
            s(m, default)
        update()
    
    # Draw the buttons and the real-time checkbox
    real_time = tk.IntVar()
    c = tk.Checkbutton(master=frame, text="Real-time", variable=real_time)
    c.pack(expand=True, fill="both")
    b = tk.Button(master=frame, text="Update", command=update)
    b.pack(expand=True, fill="both")
    b = tk.Button(master=frame, text="Reset", command=set_defaults)
    b.pack(expand=True, fill="both")
    
    root.update()
    set_defaults()
    frame_params_container.configure(scrollregion=frame_params_container.bbox('all'))
    tk.mainloop()
    return m
