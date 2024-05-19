# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

import logging
import numpy as np
import sys
import traceback
import time
import scipy.stats
from paranoid.settings import Settings as paranoid_settings
from .logger import logger as _logger

# A workaround for a bug on Mac related to FigureCanvasTKAgg
if 'matplotlib.pyplot' in sys.modules and sys.platform == 'darwin':
    _gui_compatible = False
    _logger.warning("model_gui function unavailable.  To use model_gui, please import pyddm.plot " \
        "before matplotlib.pyplot.")
else:
    _gui_compatible = True
    if sys.platform == 'darwin':
        import matplotlib
        matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

try:
    import tkinter as tk
except:
    print("Tk unavailable, model_gui will not work, but model_gui_jupyter may still work.")

import copy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .model import *
from .parameters import dt as default_dt, dx as default_dx
from .functions import solve_partial_conditions



def plot_solution_pdf(sol, ax=None, choice=None, correct=True):
    """Plot the PDF of the solution.

    - `ax` is optionally the matplotlib axis on which to plot.
    - If `correct` is true, we draw the distribution of correct
      answers.  Otherwise, we draw the error distributions.

    This does not return anything, but it plots the PDF.  It does not
    show it, and thus requires a call to plt.show() to see.
    """
    if correct is not None:
        assert choice is None, "Either choice or correct argument must be None"
        assert sol.choice_names == ("correct", "error")
        choice = sol.choice_names[0] if correct else sol.choice_names[1]
    else:
        assert choice is not None, "Choice and correct arguments cannot both be None"

    if ax is None:
        ax = plt.gca()

    ts = sol.pdf(choice)
    ax.plot(sol.t_domain, ts, label=sol.model_name)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f'{choice} PDF (normalized)')
    
def plot_solution_cdf(sol, ax=None, choice=None, correct=None):
    """Plot the CDF of the solution.

    - `ax` is optionally the matplotlib axis on which to plot.
    - If `correct` is true, we draw the distribution of correct
      answers.  Otherwise, we draw the error distributions.

    This does not return anything, but it plots the CDF.  It does not
    show it, and thus requires a call to plt.show() to see.
    """
    if correct is not None:
        assert choice is None, "Either choice or correct argument must be None"
        choice = sol.choice_names[0] if correct else sol.choice_names[1]
    else:
        assert choice is not None, "Choice and correct arguments cannot both be None"
    if ax is None:
        ax = plt.gca()

    ts = sol.cdf(choice)
    
    ax.plot(sol.t_domain, ts, label=sol.model_name)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f'{choice} CDF (normalized)')
    

def plot_compare_solutions(s1, s2):
    """Compare two model solutions to each other.

    `s1` and `s2` should be solution objects.  This will display a
    pretty picture of the correct and error distribution pdfs.
    """
    plt.subplot(2, 1, 1)
    plot_solution_pdf(s1, choice=s1.choice_names[0])
    plot_solution_pdf(s2, choice=s2.choice_names[0])
    plt.legend()
    plt.subplot(2, 1, 2)
    plot_solution_pdf(s1, choice=s1.choice_names[1])
    plot_solution_pdf(s2, choice=s2.choice_names[1])

                            
def plot_decision_variable_distribution(model, conditions={}, resolution=.1, figure=None):
    """Show the distribution of the decision variable.

    Show the intermediate distributions for the decision variable.
    `model` should be the model to plot, and `conditions` should be
    the conditions over which to plot it.  `resolution` should be the
    timestep of the plot (NOT of the model).  Optionally, `figure` is
    an existing figure on which to make the plot.
    
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
    s = model.solve_numerical_implicit(conditions=conditions, return_evolution=True)
    hists = s.pdf_evolution()
    top = s.pdf("_top")
    bot = s.pdf("_bottom")
    # Plot the output
    f = figure if figure is not None else plt.figure()
    # Set up three axes, with one in the middle and two on the borders
    gs = plt.GridSpec(7, 1, wspace=0, hspace=0)
    ax_main = f.add_subplot(gs[1:-1,0])
    ax_top = f.add_subplot(gs[0,0], sharex=ax_main)
    ax_bot = f.add_subplot(gs[-1,0], sharex=ax_main)
    # Show the relevant data on those axes
    ax_main.imshow(np.log(.0001+np.flipud(hists)), aspect='auto', interpolation='bicubic')
    ax_top.plot(range(0, len(model.t_domain())), top, clip_on=False)
    ax_bot.plot(range(0, len(model.t_domain())), -bot, clip_on=False)
    # Make them look decent
    ax_main.axis("off")
    ax_top.axis("off")
    ax_bot.axis("off")
    # Set axes to be the right size
    maxval = np.max([top, bot])
    ax_top.set_ylim(0, maxval)
    ax_bot.set_ylim(-maxval, 0)
    return f

def plot_fit_diagnostics(model=None, sample=None, fig=None, conditions=None, data_dt=.01, method=None):
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
    - `data_dt` - Bin size to use for the data histogram.  Defaults
      to 0.01.
    - `method` - Optionally the method to use to solve the model,
      either "analytical", "numerical" "cn", "implicit", "explicit",
      or None (auto-select, the default).
    """
    # Avoid stupid warnings with mutable objects
    if conditions is None:
        conditions = {}
    # Create a figure if one is not given
    if fig is None:
        fig = plt.gcf()
    
    # If we just pass a sample and no model, set appropriate T_dur and adjust data_dt if necessary
    if model:
        T_dur = model.T_dur
        if model.dt > data_dt:
            data_dt = model.dt
    elif sample:
        T_dur = max(sample)
    else:
        raise ValueError("Must specify non-empty model or sample in arguments")
    ax1 = fig.add_axes([.12, .56, .85, .43])
    ax2 = fig.add_axes([.12, .13, .85, .43], sharex=ax1)
    ax2.invert_yaxis()
    # If a sample is given, plot it behind the model.
    if sample:
        sample = sample.subset(**conditions)
        t_domain_data = np.linspace(0, T_dur, int(T_dur/data_dt+1))
        data_hist_top = np.histogram(sample.choice_upper, bins=int(T_dur/data_dt)+1, range=(0-data_dt/2, T_dur+data_dt/2))[0]
        data_hist_bot = np.histogram(sample.choice_lower, bins=int(T_dur/data_dt)+1, range=(0-data_dt/2, T_dur+data_dt/2))[0]
        total_samples = len(sample)
        ax1.fill_between(t_domain_data, np.asarray(data_hist_top)/total_samples/data_dt, label="Data", alpha=.5, color=(.5, .5, .5))
        ax2.fill_between(t_domain_data, np.asarray(data_hist_bot)/total_samples/data_dt, label="Data", alpha=.5, color=(.5, .5, .5))
        toplabel,bottomlabel = sample.choice_names
    if model:
        s = solve_partial_conditions(model, sample, conditions=conditions, method=method)
        ax1.plot(model.t_domain(), s.pdf("_top"), lw=2, color='k')
        ax2.plot(model.t_domain(), s.pdf("_bottom"), lw=2, color='k')
        toplabel,bottomlabel = model.choice_names
    # Set up nice looking plots
    for ax in [ax1, ax2]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    height = max(ax1.axis()[3], ax2.axis()[2])
    ax1.yaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(4))
    ax2.yaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(4))
    ax2.plot([0, T_dur], [0, 0], color="k", linestyle="--", linewidth=.5)
    ax1.axis([0, T_dur, 0, height])
    ax2.axis([0, T_dur, height, 0])
    ax2.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(.5))
    ax2.xaxis.set_minor_locator(plt.matplotlib.ticker.MultipleLocator(.25))
    # Easiest way I could find to prevent zero from being printed
    # twice without resorting to ax2.set_yticks(ax2.get_yticks()[1:]),
    # which makes it such that the axes don't rescale at the same rate
    class NonZeroScalarFormatter(plt.matplotlib.ticker.ScalarFormatter):
        def __call__(self, x, pos=None):
            if x == 0:
                return ""
            else:
                return super().__call__(x, pos)
    ax1.yaxis.set_major_formatter(NonZeroScalarFormatter())
    for l in ax1.get_xticklabels():
        l.set_visible(False)
    ax1.spines['left'].set_position(('outward', 10))
    ax2.spines['left'].set_position(('outward', 10))
    ax2.spines['bottom'].set_position(('outward', 10))
    ax1.set_ylabel(toplabel+" RTs")
    ax2.set_ylabel(bottomlabel+" RTs")
    ax2.set_xlabel("Time (s)")
    pt = fig.suptitle("")


def model_gui(model,
              sample=None,
              data_dt=.01,
              plot=plot_fit_diagnostics,
              conditions=None,
              verify=False):
    """Mess around with model parameters visually.

    This allows you to see how the model `model` would be affected by
    various changes in parameter values.  It also allows you to easily
    plot `sample` conditioned on different conditions.  A sample is
    required so that model_gui knows the conditions to include and the
    ratio of these conditions.

    The function `plot` allows you to change what is plotted.
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

    For performance purposes, Paranoid Scientist verification is
    disabled when running this function.  Enable it by setting the
    `verify` argument to True.

    Some of this code is taken from `fit_model`.
    """
    # WARNING: This function REALLY needs to be refactored.  See
    # model_gui_jupyter for an example of how this could look.
    assert _gui_compatible == True, "Due to a OSX bug in matplotlib," \
        " matplotlib's backend must be explicitly set to TkAgg. To avoid" \
        " this, please import pyddm.plot BEFORE matplotlib.pyplot."
    # Make sure either a sample or conditions are specified.
    assert not model.required_conditions or (sample or conditions), \
        "If a sample is not passed, conditions must be passed through the 'conditions' argument."
    # Disable paranoid for this
    paranoid_state = paranoid_settings.get('enabled')
    if paranoid_state and not verify:
        paranoid_settings.set(enabled=False)
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
        if sample:
            sample_condition_values = {cond: sample.condition_values(cond) for cond in required_conditions}
        else:
            assert all(c in conditions.keys() for c in required_conditions), \
                "Please pass all conditions needed by the model in the 'conditions' argument."
            sample_condition_values = {c : (list(sorted(conditions[c])) if isinstance(c, list) else conditions[c]) for c in required_conditions}
    elif sample:
        components_list = []
        required_conditions = sample.condition_names()
        sample_condition_values = {cond: sample.condition_values(cond) for cond in required_conditions}
    else:
        _logger.error("Must define model, sample, or both")
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
    canvas.draw()
    
    def update():
        """Redraws the plot according to the current parameters of the model
        and the selected conditions."""
        current_conditions = {c : condition_vars_values[i][condition_vars[i].get()] for i,c in enumerate(required_conditions) if condition_vars[i].get() != "All"}
        # If any conditions were "all", they will not be in current
        # conditions.  Here, we update current_conditions with any
        # conditions which were specified in the conditions argument,
        # implying they are not in the sample.
        if conditions is not None:
            for k,v in conditions.items():
                if k not in current_conditions.keys():
                    current_conditions[k] = v
        fig.clear()
        # If there was an error, display it instead of a plot
        try:
            plot(model=m, fig=fig, sample=sample, conditions=current_conditions, data_dt=data_dt)
        except:
            fig.clear()
            fig.text(0, 1, traceback.format_exc(), horizontalalignment="left", verticalalignment="top")
            canvas.draw()
            raise
        canvas.draw()
    
    def value_changed():
        """Calls update() if the real time checkbox is checked.  Triggers when a value changes on the sliders or the condition radio buttons"""
        if real_time.get() == True:
            update()
    
    # Draw the radio buttons allowing the user to select conditions
    frame_params_container = tk.Canvas(root, bd=2, width=110)
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
        for cv in sample_condition_values[cond]:
            b = tk.Radiobutton(master=lframe, text=str(cv), variable=thisvar, value=str(cv), command=value_changed)
            b.pack(anchor=tk.W)
        condition_vars_values.append({str(cv) : cv for cv in sample_condition_values[cond]})
        thisvar.set("All")
    print(condition_vars_values)
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
    real_time.set(1)
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
    # Re-enable paranoid
    if paranoid_state and not verify:
        paranoid_settings.set(enabled=True)
    return m


def model_gui_jupyter(model,
                      sample=None,
                      data_dt=.01,
                      plot=plot_fit_diagnostics,
                      conditions=None,
                      verify=False):
    """Mess around with model parameters visually in a Jupyter notebook.

    This function is equivalent to model_gui, but displays in a
    Jupyter notebook with controls.  It does nothing when called
    outside a Jupyter notebook.
    """
    # Exit if we are not in a Jupyter notebook.  Note that it is not
    # possible to reliably detect whenther you are in a Jupyter
    # notebook (e.g. in jupyter-console vs ipython) but that is a
    # fundamental design flaw which the Jupyter developers have
    # carefully enforced.
    try:
        get_ipython
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except (NameError, ImportError):
        return
    # Set up conditions
    if model:
        # All of the conditions required by at least one of the model
        # components.
        required_conditions = model.required_conditions
        if sample:
            sample_condition_values = {cond: sample.condition_values(cond) for cond in required_conditions}
        else:
            assert all(c in conditions.keys() for c in required_conditions), \
                "Please pass all conditions needed by the model in the 'conditions' argument."
            sample_condition_values = {c : (list(sorted(conditions[c])) if isinstance(c, list) else conditions[c]) \
                                       for c in required_conditions}
    elif sample:
        components_list = []
        required_conditions = sample.condition_names()
        sample_condition_values = {cond: sample.condition_values(cond) for cond in required_conditions}
    else:
        _logger.error("Must define model, sample, or both")
        return
    # Set up params
    params = model.get_model_parameters()
    default_params = [p.default() for p in params]
    param_names = model.get_model_parameter_names()
    # Update the plot, as a callback function
    def draw_update(**kwargs):
        conditions = {}
        parameters = {}
        # To make this work with the ipython library, we prefix
        # parameters starting with a "_p_" and conditions starting
        # with a "_c_".  Here we detect what is what, and strip away
        # the prefix.
        if not util_widgets[0].value and not util_widgets[2].value:
            print("Update to see new plot")
            return
        for k,v in kwargs.items():
            if k.startswith("_c_"):
                conditions[k[3:]] = v
            elif k.startswith("_p_"):
                parameters[k[3:]] = v
        ordered_parameters = [parameters[p] for p in param_names]
        model.set_model_parameters(ordered_parameters)
        clear_output(wait=True)
        plot(model=model, sample=sample, conditions=conditions, data_dt=data_dt)
        # Set the "update" button back to False, but don't trigger a redraw
        changes_tmp = util_widgets[2]._trait_notifiers['value']['change']
        util_widgets[2]._trait_notifiers['value']['change'] = []
        util_widgets[2].value = False
        util_widgets[2]._trait_notifiers['value']['change'] = changes_tmp
        plt.show()
    def draw(*args, **kwargs):
        util_widgets[2].value = True
    # Reset to default values
    def reset(*args, **kwargs):
        for w,d in zip(param_widgets,default_params):
            # Temporarily disable callbacks and then re-enable after
            # setting the value.  This prevents redraw after changing
            # each parameter.
            changes_tmp = w._trait_notifiers['value']['change']
            w._trait_notifiers['value']['change'] = []
            w.value = d
            w._trait_notifiers['value']['change'] = changes_tmp
        # Now run the redraw only once
        draw()

    # Set up all of the widgets we will use to control the plot
    param_widgets = [widgets.FloatSlider(min=p.minval,
                                         max=p.maxval,
                                         value=dp,
                                         description=name,
                                         continuous_update=False,
                                         step=(p.maxval-p.minval)/100)
                     for p,name,dp in zip(params,param_names,default_params)]
    condition_widgets = [widgets.Dropdown(options=[("All", sample_condition_values[name])]+
                                                  [(c, [c]) for c in sample_condition_values[name]],
                                          value=sample_condition_values[name],
                                          description=name)
                         for name in required_conditions]
    util_widgets = [widgets.Checkbox(value=True, description="Real-time"),
                    widgets.Button(description='Reset to defaults'),
                    widgets.ToggleButton(description='Update')]
    util_widgets[1].on_click(reset)

    # Make three columns: parameters, conditions, and buttons/settings
    layout = widgets.HBox([widgets.VBox(param_widgets),
                           widgets.VBox(condition_widgets),
                           widgets.VBox(util_widgets)])

    # Add prefixes to parameters/conditions (see "draw" function)
    allargs = {**{"_p_"+n:p for n,p in zip(param_names,param_widgets)},
               **{"_c_"+n:c for n,c in zip(required_conditions,condition_widgets)},
               **{"_update_": util_widgets[2]}}
    # Run the display
    out = widgets.interactive_output(draw_update, allargs)
    return display(layout, out)

def plot_psychometric(condition_across, split_by_condition=None, resolution=11, forced_choice=True):
    """Create a psychometric function plot for use in the model GUI.

    `condition_across` specifies the x axis for the psychometric function.

    `split_by_condition` is (optionally) the name of a condition to split the
    psychometric curve.  For instance, if there are two types of trials in your
    dataset and you would like to compare them, you can set this to the name of
    the condition which denotes the different trial type.

    `resolution` specifies how finely spaced points to plot when computing the
    model's psychometric function.  Larger numbers will look smoother but be
    slower.

    `forced_choice` determines how to handle undecided trials.  When True,
    choices which run out of time are made randomly, with 50% probability of
    either choice.
    """
    def _plot_psychometric(model=None, sample=None, fig=None, conditions={}, data_dt=None, method=None):
        colour_cycle = [plt.cm.Set1(i) for i in range(0, 8)]*10
        # Create a figure if one is not given
        if fig is None:
            fig = plt.gcf()
        ax = fig.add_subplot(111)
        for i,split_cond in (enumerate(sorted(set(sample.condition_values(split_by_condition)))) if split_by_condition is not None else [(0,None)]):
            x_sim = []
            x_data = []
            y_sim = []
            y_data = []
            ci_data = []
            cohs = sorted(set(sample.condition_values(condition_across)))
            for coh in cohs:
                if sample:
                    if split_by_condition is not None:
                        matchingconds = {split_by_condition: split_cond, condition_across: coh}
                    else:
                        matchingconds = {condition_across: coh}
                    matchingsample = sample.subset(**matchingconds)
                    if len(conditions) > 0:
                        matchingsample = matchingsample.subset(**conditions)
                    if len(matchingsample) == 0: continue
                    x_data.append(coh)
                    if forced_choice:
                        y_data.append(matchingsample.prob_forced("upper"))
                    else:
                        y_data.append(matchingsample.prob("upper"))
                    ci_data.append(_binom_ci(matchingsample))
            model_cohs = np.linspace(np.min(cohs), np.max(cohs), resolution)
            for coh in model_cohs:
                matchingconds = conditions.copy()
                if split_by_condition is not None:
                    matchingconds[split_by_condition] = split_cond
                matchingconds[condition_across] = coh
                if model:
                    x_sim.append(coh)
                    s = solve_partial_conditions(model, sample=sample, conditions=matchingconds, method=method)
                    if forced_choice:
                        y_sim.append(s.prob_forced("upper"))
                    else:
                        y_sim.append(s.prob("upper"))
            if model:
                label = {"label": f"{split_by_condition}={split_cond}"} if split_by_condition is not None else {}
                ax.plot(x_sim, y_sim, c=colour_cycle[i], clip_on=False, linestyle='-', linewidth=1, **label)
            if sample:
                ax.errorbar(x_data, y_data, yerr=ci_data, c=colour_cycle[i], clip_on=False, linestyle=' ', marker='o', markersize=3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel(condition_across)
        ax.set_yticks([0, .5, 1])
        #ax.set_yticklabels(["", "0.25", "0.5", "0.75", ""])
        ax.set_ylabel(f"P({model.choice_names[0]})")
        if split_by_condition is not None:
            ax.legend()
    return _plot_psychometric

def plot_chronometric(condition_across, split_by_condition=None, resolution=11):
    """Create a chronometric function plot for use in the model GUI.

    `condition_across` specifies the x axis for the psychometric function.

    `split_by_condition` is (optionally) the name of a condition to split the
    psychometric curve.  For instance, if there are two types of trials in your
    dataset and you would like to compare them, you can set this to the name of
    the condition which denotes the different trial type.

    `resolution` specifies how finely spaced points to plot when computing the
    model's psychometric function.  Larger numbers will look smoother but be
    slower.
    """
    # This code is largely copied from plot_psychometric
    def _plot_chronometric(model=None, sample=None, fig=None, conditions={}, data_dt=None, method=None):
        colour_cycle = [plt.cm.Set1(i) for i in range(0, 8)]*10
        # Create a figure if one is not given
        if fig is None:
            fig = plt.gcf()
        ax = fig.add_subplot(111)
        for i,split_cond in (enumerate(sorted(set(sample.condition_values(split_by_condition)))) if split_by_condition is not None else [(0,None)]):
            x_sim = []
            x_data = []
            y_sim = []
            y_data = []
            ci_data = []
            cohs = sorted(set(sample.condition_values(condition_across)))
            for coh in cohs:
                if sample:
                    if split_by_condition is not None:
                        matchingconds = {split_by_condition: split_cond, condition_across: coh}
                    else:
                        matchingconds = {condition_across: coh}
                    matchingsample = sample.subset(**matchingconds)
                    if len(conditions) > 0:
                        matchingsample = matchingsample.subset(**conditions)
                    if len(matchingsample) == 0: continue
                    x_data.append(coh)
                    y_data.append(matchingsample.mean_rt())
                    ci_data.append(_binom_ci(matchingsample))
            model_cohs = np.linspace(np.min(cohs), np.max(cohs), resolution)
            for coh in model_cohs:
                matchingconds = conditions.copy()
                if split_by_condition is not None:
                    matchingconds[split_by_condition] = split_cond
                matchingconds[condition_across] = coh
                if model:
                    x_sim.append(coh)
                    s = solve_partial_conditions(model, sample=sample, conditions=matchingconds, method=method)
                    y_sim.append(s.mean_rt())
            if model:
                label = {"label": f"{split_by_condition}={split_cond}"} if split_by_condition is not None else {}
                ax.plot(x_sim, y_sim, c=colour_cycle[i], clip_on=False, linestyle='-', linewidth=1, **label)
            if sample:
                ax.errorbar(x_data, y_data, yerr=ci_data, c=colour_cycle[i], clip_on=False, linestyle=' ', marker='o', markersize=3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel(condition_across)
        ax.set_ylabel("Mean RT (s)")
        if split_by_condition is not None:
            ax.legend()
    return _plot_chronometric

def plot_bound_shape(model, sample=None, fig=None, conditions=None, data_dt=None, method=None):
    """Plot the shape of the bound.

    Indended for use in the model GUI."""
    # We will ignore most of these arguments
    if fig is None:
        fig = plt.gcf()
    b = model.get_dependence("bound").get_bound(model.t_domain(), conditions=conditions)
    if isinstance(b, (float, int)):
        b = model.t_domain()*0+b
    ax1 = fig.add_axes([.12, .56, .85, .43])
    ax2 = fig.add_axes([.12, .13, .85, .43], sharex=ax1)
    ax2.invert_yaxis()
    ax1.plot(model.t_domain(), b, color='k', lw=2)
    ax2.plot(model.t_domain(), b, color='k', lw=2)
    ax1.set_ylim(0, None)
    ax2.set_ylim(None, 0)
    for ax in [ax1, ax2]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

def _binom_ci(sample, alpha=.05):
    """Confidence interval using Gaussian approximation for binomial RVs"""
    corr = len(sample.choice_upper)
    err = len(sample.choice_lower)
    z = scipy.stats.norm.ppf(1-alpha/2)
    p = corr/(corr+err)
    return z*np.sqrt(p*(1-p)/(corr+err))
