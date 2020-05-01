Recipes for Collapsing Bounds
=============================

General use of bounds
~~~~~~~~~~~~~~~~~~~~~

Both linearly collapsing bounds (:class:`.BoundCollapsingLinear`) and
exponentially collapsing bounds (:class:`.BoundCollapsingExponential`)
already exist in PyDDM.  For example::

  from ddm import Model
  from ddm.models import BoundCollapsingLinear, BoundCollapsingExponential
  model1 = Model(bound=BoundCollapsingExponential(B=1, tau=2))
  model2 = Model(bound=BoundCollapsingLinear(B=1, t=.2))


.. _bound-step:

Step function collapsing bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to make collapsing bounds of any shape.  For
example, the following describes bounds which collapse in discrete
steps of a particular length:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start BoundCollapsingStep
   :end-before: # End BoundCollapsingStep


Try it out with::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(bound=BoundCollapsingStep(B0=Fittable(minval=.5, maxval=1.5),
                                          stepheight=Fittable(minval=0, maxval=.49),
                                          steplength=Fittable(minval=0, maxval=2)),
                dx=.01, dt=.01)
  model_gui(model)

.. _bound-weibull-cdf:

Weibull CDF collapsing bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Weibull function is a popular choice for collapsing bounds.  (See,
e.g., `Hawkins et al. (2015) <https://doi.org/10.1523/JNEUROSCI.2410-14.2015>`_.)  This can be
implemented using:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start BoundCollapsingWeibull
   :end-before: # End BoundCollapsingWeibull

Try it out with::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(bound=BoundCollapsingWeibull(a=Fittable(minval=1, maxval=2),
                                             aprime=Fittable(minval=0, maxval=1),
                                             lam=Fittable(minval=0, maxval=2),
                                             k=Fittable(minval=0, maxval=5)),
                dx=.01, dt=.01)
  model_gui(model)

(Note that in `Hakwins et al. (2015)
<https://doi.org/10.1523/JNEUROSCI.2410-14.2015>`_, diffusion goes
from [0,1], whereas our diffusion goes from [-1,1].  Thus, the 0.5
term was removed.)


.. _bound-exp-delay:

Delayed collapsing bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following implements exponentially collapsing bounds with a delay:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start BoundCollapsingExponentialDelay
   :end-before: # End BoundCollapsingExponentialDelay


.. _bound-speedacc:

Bounds which depend on task conditions (e.g. speed vs accuracy tradeoff)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an example of bounds which depend on a task conditions, we assume a
task in which a subject is cued before the stimulus about whether to
prioritize speed or accuracy.  The following model could test the
hypothesis that the subject changes their integration bound to be high
for the accuracy condition and low for the speed condition.

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start BoundSpeedAcc
   :end-before: # End BoundSpeedAcc

Try it out with::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(bound=BoundSpeedAcc(Bacc=Fittable(minval=.5, maxval=1.5),
                                    Bspeed=Fittable(minval=0, maxval=1)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"speed_trial": [0, 1]})


.. _bound-increase:

Increasing bounds
~~~~~~~~~~~~~~~~~

In addition to collapsing bounds, PyDDM also supports increasing
bounds, or bounds which both increase and decrease over time.  Note
that performance is proportional to the maximum size of the bounds, so
very large bounds should be avoided.

For example, the following bounds are constant from t=0 until t=1,
increase from t=1 until t=1.2, decrease from t=1.2 until t=1.4, and
then are again constant::

  import ddm
  class BoundIncreasingAndDecreasing(ddm.Bound):
      name = "Increasing bound"
      required_conditions = []
      required_parameters = []
      def get_bound(self, t, *args, **kwargs):
          if t > 1 and t < 1.4:
              return 1 + (.2-abs(t-1.2))*3
          else:
              return 1

