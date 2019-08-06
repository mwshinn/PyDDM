FAQs
====

How do I know if my model will run analytically or numerically?
---------------------------------------------------------------

The function :func:`~ddm.model.Model.solve` will automatically choose the best
solver for your model.  Solves, in order of preference, are

1. Analytical
2. Crank-Nicolson
3. Backward Euler (implicit method)

The analytical solver requires that Drift and Noise do not depend on
time or particle position, and that the initial position is fixed at
zero (:class:`.ICPointSourceCenter`).  Additionally, they require either Bounds
which do not depend on time, or alternatively linearly collapsing
bounds using :class:`.BoundCollapsingLinear`.  (Parameterized linearly
collapsing bounds are not currently supported.)

The Crank-Nicolson solver requires that Bounds do not depend on time,
i.e. models with collapsing bounds are not supported.

The Backward Euler (implicit method) solver is compatible with all
models and thus serves as a fallback.

Particle simulations and Forward Euler (explicit method) are also
available, but must be explicitly called via
:func:`~ddm.model.Model.solve_numerical_explicit` and
:func:`~ddm.model.Model.simulated_solution`.  They will never be chosen
automatically.

For custom models, these are specified by including ``x`` or ``t`` in
the argument list.


What arguments do :func:`~ddm.models.drift.Drift.get_drift`, :func:`~ddm.models.noise.Noise.get_noise`, etc. take?
------------------------------------------------------------------------------------------------------------------

The most appropriate solver is selected by PyDDM by examining the
variables on which different model components depend.  For models
components built-in to PyDDM, this is checked automatically.

For custom model components, this is found based on the arguments
taken by the relevant model function.  Internally, the time is passed
as the number ``t`` keyword argument, the position is passed as a
number or vector ``x``, and the trial conditions are passed as a
dictionary ``conditions``.  Thus, by including any of these variables
in the method definition, we thereby depend on that parameter.
(Likewise, all methods should end with ``**kwargs`` to not throw an
error when other parameters are passed, but should not access the
``kwargs`` variable directly.)

For example, suppose our experiment consisted of two blocks, one
associated with high rewards and the other associated with low
rewards, and we hypothesize that different bounds are used in these
two cases.  We could create the following Bound object which allows
the bounds in the two blocks to be fit independently::

  class BoundReward(ddm.Drift):
      name = "Reward-modulated bounds"
      required_conditions = ["block"]
      required_parameters = ["bound1", "bound2"]
      def get_bound(self, conditions, **kwargs):
          if conditions["block"] == 1:
              return self.bound1
          elif conditions["block"] == 2:
              return self.bound2

Notice how the ``get_bound`` function does not depend on ``t``.
However, we could in theory also define the method ``get_bound`` as::

  def get_bound(self, t, conditions, **kwargs):
      ...

Even if the variable ``t`` is never used inside ``get_bound``, PyDDM
would interpret this to mean that the function depends on time.  Thus,
while this will give the expected result, it will not allow PyDDM to
properly optimize.

Alternatively, suppose a separate hypothesis whereby the block
described above modulates the rate of collapse in exponentially
collapsing bounds.  This could be modeled as::

  import numpy as np
  class BoundReward(ddm.Drift):
      name = "Reward-modulated bounds"
      required_conditions = ["block"]
      required_parameters = ["rate1", "rate2", "B0"]
      def get_bound(self, t, conditions, **kwargs):
          if conditions["block"] == 1:
              rate = self.rate1
          elif conditions["block"] == 2:
              rate = self.rate2
          return self.B0 * np.exp(-rate*t)

In this case, our bound does depend on ``t``, so it **must** be
included in the function signature.

Why do I get "Paranoid" errors?
-------------------------------

`Paranoid Scientist <http://paranoid-scientist.readthedocs.io>`_ is a
library for verifying the accuracy of scientific software.  It is used
to check the entry and exit conditions of functions.

Paranoid Scientist will, overall, decrease the probability of an
undetected error by increasing the number of bugs overall.  Some
common errors are:

- When a particular model parametrization causes numerical instability
  at the given dx and dt.  This can cause probability distributions
  which go below zero.
- When numerical issues are amplified in the model, making the
  distribution integrate to more than 1 (plus floating point error).
- When dx and dt are too small for Crank-Nicolson and oscillations
  occur in the distribution.

If this becomes a problem during model fitting, it can be disabled
with::

  import paranoid as pns
  pns.settings.Settings.set(enabled=False)

When performing final simulations for the paper, it is recommended to
keep re-enable Paranoid Scientist, since turning it off may mask
numerical issues.

Can PyDDM fit hierarchical models?
----------------------------------

No, PyDDM cannot fit hierarchical models.  This need is already
addressed by the `hddm package <https://github.com/hddm-devs/hddm/>`_.
Due to limited resources, we do not plan to add support for
hierarchical models, but you are welcome to implement the feature
yourself and submit a pull request on Github.  If you plan to
implement this feature, please let us know so we can help you get
familiar with the code.
