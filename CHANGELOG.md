# Version 0.5.1

Released December 4, 2020

## Minor changes

- Option to suppress output in differential evolution
- Option to drop undecided trials when converting a sample to a pandas dataframe

## Bug fixes

- Fixes a bug in which Solution.resample can sometimes give negative RTs under
  certain conditions
- Fixes a bug in likelihood-based fitting methods which produced incorrect
  results when fitting with undecided trials
- Fixes a bug in condition_combinations which made fitting very slow for Samples
  with many conditions

# Version 0.5.0

Released: September 15, 2020

## New features

- Jupyter notebook compatibility

    In addition to the classic model GUI, it is now possible to
    explore models interactively in Jupyter notebooks using the
    "ddm.plot.model_gui_jupyter" function.  With this change, PyDDM
    also now offers online interactive examples.

- Analytical solutions for arbitrary point source initial conditions (thanks Nathan!)

    Analytical solutions are now implemented for point source initial
    conditions at points other than x=0.  The analytical solver will
    automatically detect if you have specified a compatible model, and
    if so, it will solve it analytically instead of numerically.  This
    leads to speed boosts of several orders of magnitude for these
    models.

- Prescriptive warning messages

    When possible, warnings now give details of how to correct the
    model, for example, by decreasing dx or dt.

## Bug fixes

- Fix broken robust loss functions
- Example code no longer gives warning messages
- Fixed integer overflows on Windows
- Fixed underflow warning messages on model simulation for specific
  models (thanks Nathan!)

## Breaking changes

- The "get_model_parameters" function now does not return repeated
  parameters.  So, if two parameters in your model share the same
  Fittable object, meaning they are fit together, only one copy of
  this will be listed in calls to "get_model_parameters".
  Corresponding changes were made to "set_model_parameters".

# Version 0.4.0

Released: June 19, 2020

## New features

- Trial-wise trajectory support for Overlays

    Overlays are an important feature of the GDDM, but previously they
    were not supported on trial-wise trajectory simulations.  Now, it
    is possible to define the function "apply_trajectory" in the
    Overlay object if the overlay can be applied to a trajectory
    simulation.

- Details of model fits are preserved

    After running a model fit, it is useful to know the parameters of
    the fit, the objective function value, the methods used for the
    fit, etc.  This information is now easily-accessible from within
    fit models as a FitResult object, with clear documentation on how
    to use it.

- Performance enhancements for Crank-Nicolson and analytical solutions

    Moderate speedups for the Crank-Nicolson method, and a roughly one
    order of magnitude speedup for analytical solutions

- Samples can be exported

    Previously, it was possible to create a new Sample object from a
    pandas DataFrame. Now, it is also possible to do the reverse, i.e.
    convert an existing Sample object to a pandas DataFrame.

## Bug fixes

- Documentation links now refer to the stable version instead of the
  development version.
- Fixed bug when bounds collapse to zero
- Sample objects were internally inconsistent when imported through
  pandas

## Other

- New option to suppress diagnostic text (thanks Arkady!)
- LossRobustBIC and LossRobustLikelihood now provide shortcuts for
  uniform distribution overlays.
- Added a new implementation of biased reward for compatibility with
  fittable bounds (thanks Nathan!)
- get_model_loss function as a shortcut for finding the value of a
  given loss function for a given model
- New function to get list of model parameter names
- Function to compute mean decision time for samples
- More informative error messages

## Breaking changes

- The command-line argument "method" now has a different meaning, and
  will throw an error if used for the previous purpose.  This made
  terminology for "fitting_method" and "method" more consistent: now
  "fitting_method" refers to the optimization routine
  (e.g. differential evolution) whereas "method" refers to the
  numerical algorithm (e.g. backward Euler)
- The "returnEvolution" arguement is now called "return_evolution".

# Version 0.3.0

Released: October 20, 2019

## New features

- Increasing bounds

    Previously, PyDDM supported collapsing bounds, but bounds were
    required to be monotonically decreasing.  This restriction was
    lifted, allowing bounds to both increase and decrease over time
    according to any arbitrary function.
    
- Tracking decision variable evolution (thanks Stefan!)

    The evolution of the decision variable distribution may be tracked
    over time by passing a flag to the solver routine.

- Custom optimizers

    Previously, PyDDM required optimizers (e.g. differential evolution
    or the Nelder-Mead simplex algorithm) to be hard-coded.  Now,
    these algorithms can be passed as arguments to the
    "fit_adjust_model" function.

- Improvements in the model GUI

    The model GUI now has nicer-looking plots, and is more transparent
    when errors occur in the model.  Additionally, it now supports
    plotting a model when a sample is not present.

## Bug fixes

- Under certain conditions, parallel simulations were computationally
  inefficient.
- ICPoint had invalid arguments for the get_IC function
- simulate_trial cut off when reaching the boundary, contrary to the
  function documentation

## Other

- The PyDDM cookbook now provides several simple examples of how to
  use PyDDM in a variety of situations.
- Particle simulations may now use either RK4 or Euler's method
- Several new model components were added
- Error messages were improved
