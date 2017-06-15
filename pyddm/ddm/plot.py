from .model import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
from .parameters import *
from .functions import solve_partial_conditions

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
    
def plot_fit_diagnostics(model, sample, samescale=False):
    """Compare actual data to the best fit model of the data.

    - `model` should be the Model object fit from `rt_data`.
    - `sample` should be a Sample object describing the data
    - `samescale` should be set to True if the error trials
      distribution should have the same axis as the correct trials
      distribution
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
    print(model_corr)
    plt.plot(model.t_domain(), np.asarray(data_hist_corr)/total_samples/dt, label="Data") # Divide by samples and dt to scale to same size as solution pdf
    plt.plot(model.t_domain(), model_corr, label="Fit")
    axis_corr = plt.axis()
    print(sum(data_hist_corr/total_samples/dt)+sum(data_hist_err/total_samples/dt))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(model.t_domain(), np.asarray(data_hist_err)/total_samples/dt, label="Data")
    plt.plot(model.t_domain(), model_err, label="Fit")
    plt.axis(axis_corr)

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

# TODO this is very messy and needs some serious cleanup

# This is a wrapper to fit the old interface.  For compatibility
# purposes only.  Depreciated.
def play_with_model(sample=None,
                    show_loss=None,
                    synchronous=False,
                    default_model=None,
                    conditions={},
                    mu=MuConstant(mu=0),
                    sigma=SigmaConstant(sigma=1),
                    bound=BoundConstant(B=1),
                    IC=ICPointSourceCenter(),
                    task=TaskFixedDuration(),
                    dt=dt, dx=dx, 
                    overlay=OverlayNone(),
                    pool=None,
                    name="fit_model",
                    samescale=True):
    if default_model:
        return model_gui(default_model)
    
    if sample:
        T_dur = np.ceil(max(sample)/dt)*dt
    else:
        T_dur = 2
    assert T_dur < 30, "Too long of a simulation... are you using milliseconds instead of seconds?"
    # For optimization purposes, create a base model, and then use
    # that base model in the optimization routine.  First, set up the
    # model with all of the Fittables inside.  Deep copy on the entire
    # model is a shortcut for deep copying each individual component
    # of the model.
    m = copy.deepcopy(Model(name=name, mu=mu, sigma=sigma, bound=bound, IC=IC, task=task, overlay=overlay, T_dur=T_dur, dt=dt, dx=dx))
    return model_gui(m, sample=sample, pool=pool, show_loss=show_loss, samescale=samescale, synchronous=synchronous, dt=dt)


def model_gui(model,
              sample=None,
              show_loss=None,
              synchronous=False,
              pool=None,
              samescale=True):
    """Mess around with model parameters visually.

    This allows you to see how a model would be affected by various
    changes in parameter values.  Its arguments are exactly the same
    as `fit_model`, with a few exceptions:

    First, the sample is optional.  If provided, it will be displayed
    in the background.

    Second, if a sample is given, and if `show_loss` is set to a Loss
    object, the plot will show the value of the loss function when
    plotting.  Note that this is very slow because it resolves the
    model.

    Third, `synchronous` specifies whether the user is required to
    push the `update` button after making changes to the model.

    Finally, perhaps most importantly, `conditions` specifies the
    conditions to use for this model.  You can't (currently) toggle
    between conditions, so it is necessary to specify them beforehand.

    Most of this code is taken from `fit_model`.
    """
    dt = model.dt
    # Loop through the different components of the model and get the
    # parameters that are fittable.  Save the "Fittable" objects in
    # "params".  Create a list of functions to set the value of these
    # parameters, named "setters".
    components_list = [model.get_dependence("mu"),
                       model.get_dependence("sigma"),
                       model.get_dependence("bound"),
                       model.get_dependence("IC"),
                       model.get_dependence("task"),
                       model.get_dependence("overlay")]
    required_conditions = list(set([x for l in components_list for x in l.required_conditions]))

    params = [] # A list of all of the Fittables that were passed.
    setters = [] # A list of functions which set the value of the corresponding parameter in `params`
    getters = [] # A list of functions which get the value of the corresponding parameter in `params`
    paramnames = [] # The names of the parameters
    for component in components_list:
        for param_name in component.required_parameters:
            pv = getattr(component, param_name) # Parameter value in the object
            if isinstance(pv, Fittable):
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
                
                getter = lambda x,component=component,param_name=param_name : getattr(x.get_dependence(component.depname), param_name)
                # If we have the same Fittable object in two different
                # components inside the model, we only want the Fittable
                # object in the list "params" once, but we want the setter
                # to update both.
                if id(pv) in map(id, params):
                    pind = list(map(id, params)).index(id(pv))
                    oldsetter = setters[pind]
                    # This is a hack way of executing two functions in
                    # a single function call while passing forward the
                    # same argument object (not just the same argument
                    # value)
                    newsetter = lambda x,a,setter=setter,oldsetter=oldsetter : oldsetter(x,setter(x,a)) 
                    setters[pind] = newsetter
                    paramnames[pind] += "/"+param_name
                else: # This setter is unique (so far)
                    params.append(pv)
                    setters.append(setter)
                    paramnames.append(param_name)
                    getters.append(getter)

    # For optimization purposes, create a base model, and then use
    # that base model in the optimization routine.  First, set up the
    # model with all of the Fittables inside.  Deep copy on the entire
    # model is a shortcut for deep copying each individual component
    # of the model.
    m = copy.deepcopy(model)

    fig, ax = plt.subplots()
    plt.subplots_adjust(right=.75, left=.28)

    T_dur = model.T_dur
    #s = m.solve()
    use_correct = True
    if sample:
        if show_loss:
            lf = show_loss(sample, required_conditions=required_conditions,
                              pool=pool, T_dur=T_dur, dt=dt,
                              nparams=len(params), samplesize=len(sample))
        sample_cond = sample
        data_hist_top = np.histogram(sample_cond.corr, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0]
        data_hist_bot = np.histogram(sample_cond.err, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0]
        total_samples = len(sample_cond)

        plt.subplot(211)
        histl_top, = plt.plot(m.t_domain(), np.asarray(data_hist_top)/total_samples/dt, label="Data", alpha=.5)
        plt.subplot(212)
        histl_bot, = plt.plot(m.t_domain(), np.asarray(data_hist_bot)/total_samples/dt, label="Data", alpha=.5)
    plt.subplot(211)
    l_top, = plt.plot(m.t_domain(), np.zeros(m.t_domain().shape), lw=2, color='red')
    plt.subplot(212)
    l_bot, = plt.plot(m.t_domain(), np.zeros(m.t_domain().shape), lw=2, color='red')
    plt.axis([0, m.T_dur, 0, None])
    pt = fig.suptitle("")

    height = .7/(len(setters)+3)
    if height > .2:
        height = .2

    # Make a set of radio buttons for each condition
    condition_axes = [] # Matplotlib axis objects for condition radio buttons
    condition_radios = [] # Matplotlib RadioButtons objects for condition radio buttons
    condition_names = sample.condition_names()
    if required_conditions is not None:
        condition_names = [n for n in condition_names if n in required_conditions]
    for i, cond in enumerate(condition_names):
        cax = plt.axes([0.025, 0.5-.1*len(condition_names)+.2*i, 0.23, 0.15])
        labels = [str(cond)+"="+str(cv) for cv in sample.condition_values(cond)]
        labels.append("All")
        radio_cond = RadioButtons(cax, tuple(labels), active=len(labels)-1)
        condition_axes.append(cax)
        condition_radios.append(radio_cond)

    def radio_val(val): # Strips the "xxx=" part off and gives a numeric value
        return eval(val.split("=")[1]) if "=" in val else None
    axupdate = plt.axes([0.8, 1-1/(len(setters)+4), 0.15, height])
    buttonupdate = Button(axupdate, 'Update', hovercolor='0.975')
    axreset = plt.axes([0.8, 1-2/(len(setters)+4), 0.15, height])
    buttonreset = Button(axreset, 'Reset', hovercolor='0.975')
    def update(event=None):
        print(condition_names)
        current_conditions = {c : radio_val(condition_radios[i].value_selected) for i,c in enumerate(condition_names) if condition_radios[i].value_selected != "All"}
        print(current_conditions)
        sample_cond = sample.subset(**current_conditions)
        s = solve_partial_conditions(m, sample_cond, conditions=current_conditions)
        scale_fact = len(sample_cond)/len(sample)
        print(s.pdf_corr())
        if show_loss and sample:
            print("Computing loss")
            pt.set_text("loss="+str(lf.loss(m)))
        l_top.set_ydata(s.pdf_corr()*scale_fact)
        l_bot.set_ydata(s.pdf_err()*scale_fact)
        corrhist = np.histogram(sample_cond.corr, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0]/total_samples/dt
        print(corrhist)
        errhist = np.histogram(sample_cond.err, bins=T_dur/dt+1, range=(0-dt/2, T_dur+dt/2))[0]/total_samples/dt
        histl_top.set_ydata(corrhist)
        histl_bot.set_ydata(errhist)
        topmax = max(max(s.pdf_corr()*scale_fact), max(corrhist))
        botmax = max(max(s.pdf_err()*scale_fact), max(errhist))
        plt.subplot(211)
        plt.ylim(0, topmax*1.1)
        plt.subplot(212)
        if samescale:
            plt.ylim(0, topmax*1.1)
        else:
            plt.ylim(0, botmax*1.1)
        fig.canvas.draw_idle()
    buttonupdate.on_clicked(update)
    for radio in condition_radios:
        radio.on_clicked(update)

    
    # And now get rid of the Fittables, replacing them with the
    # default values.  
    x_0 = []
    constraints = [] # List of (min, max) tuples.  min/max=None if no constraint.
    axes = [] # We don't need the axes or widgets variables, but if we don't save them garbage collection screws things up
    widgets = []
    for p,s,g,i,name in zip(params, setters, getters, range(0, len(setters)), paramnames):
        default = p.default()
        s(m, default)
        minval = p.minval if p.minval > -np.inf else None
        maxval = p.maxval if p.maxval < np.inf else None
        constraints.append((minval, maxval))
        x_0.append(default)
        ypos = 1-(i+3)/(len(setters)+4)
        axes.append(plt.axes([0.8, ypos, 0.15, height]))
        widgets.append(Slider(axes[-1], name, p.minval, p.maxval, valinit=default))
        widgets[-1].on_changed(lambda val, s=s, m=m : [s(m, val), update(None) if synchronous else None])

    def set_defaults(event=None):
        nonlocal synchronous
        oldsync = synchronous
        synchronous = False
        for w in widgets:
            w.reset()
        synchronous = oldsync
        update(None)

    buttonreset.on_clicked(set_defaults)
    update(None)
    plt.show()
    return m
