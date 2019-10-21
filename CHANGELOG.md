# Version 0.3

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
