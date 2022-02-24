Recipes for Initial Conditions
==============================

General use of initial conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initial conditions can be included in the model by passing it directly
to the Model object.  For example, for a uniform distribution centered
at 0, do::

  from ddm import Model
  from ddm.models import ICRange
  model = Model(IC=ICRange(sz=.2))

.. _ic-biased:

Biased Initial Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~

Often we want to model a side bias, either those which arise naturally or those
introduced experimentally through asymmetric reward or stimulus probabilities.
The most popular way of modeling a side bias is to use a starting position which
is closer to the boundary representing that side.

To do this, we must first include a condition in our dataset describing whether
the correct answer was the left side or the right side.  Suppose we have a
sample which has the ``left_is_correct`` condition, which is 1 if the correct
answer is on the left side, and 0 if the correct answer is on the right side.
Now, we can define an :class:`.InitialCondition` object which uses this
information.  We do this by defining a :meth:`~.InitialCondition.get_IC` method.
This method should generate a discretized probability distribution for the
starting position.  Here, we want this distribution to be a single point ``x0``,
the sign of which (positive or negative) depends on whether the correct answer
is on the left or right side.  The function receives the support of the
distribution in the ``x`` argument.  We can model this with:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start ICPointSideBias
   :end-before: # End ICPointSideBias

Then we can compare the distribution of left-correct trials to those of
right-correct trials::

  from ddm import Model
  from ddm.plot import plot_compare_solutions
  import matplotlib.pyplot as plt
  model = Model(IC=ICPointSideBias(x0=.3))
  s1 = model.solve(conditions={"left_is_correct": 1})
  s2 = model.solve(conditions={"left_is_correct": 0})
  plot_compare_solutions(s1, s2)
  plt.show()

We can also see these directly in the model GUI::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(IC=ICPointSideBias(x0=Fittable(minval=0, maxval=1)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"left_is_correct": [0, 1]})

To more accurately represent the initial condition, we can 
linearly approximate the probability density function at the two 
neighboring grids of the initial position:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start ICPointSideBiasInterp
   :end-before: # End ICPointSideBiasInterp

Try it out with::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(IC=ICPointSideBiasInterp(x0=Fittable(minval=0, maxval=1)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"left_is_correct": [0, 1]})

In practice, these are very similar, but the latter gives a smoother
derivative, which may be useful for gradient-based fitting methods
(which are not used by default).


.. _ic-ratio:

Fixed ratio instead of fixed value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When fitting both the initial condition and the bound height, it can
be preferable to express the initial condition as a proportion of the total
distance between the bounds. This ensures that the initial condition will always
stay within the bounds, preventing errors in fitting.

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start ICPointSideBiasRatio
   :end-before: # End ICPointSideBiasRatio

Try it out with:: 

  from ddm import Model, Fittable
  from ddm.models import BoundConstant
  from ddm.plot import model_gui
  model = Model(IC=ICPointSideBiasRatio(x0=Fittable(minval=-1, maxval=1)),
                bound=BoundConstant(B=Fittable(minval=.1, maxval=2)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"left_is_correct": [0, 1]})

.. _ic-biased-range:

Biased Initial Condition Range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start ICPointRange
   :end-before: # End ICPointRange

Try it out with constant drift using::

  from ddm import Model, Fittable, DriftConstant
  from ddm.plot import model_gui
  model = Model(drift=DriftConstant(drift=1),
                IC=ICPointRange(x0=Fittable(minval=0, maxval=.5),
                                sz=Fittable(minval=0, maxval=.49)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"left_is_correct": [0, 1]})

.. _ic-cauchy:

Cauchy-distributed Initial Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start ICCauchy
   :end-before: # End ICCauchy

Try it out with::

  from ddm import Model, Fittable, BoundCollapsingLinear
  from ddm.plot import model_gui
  model = Model(IC=ICCauchy(scale=Fittable(minval=.001, maxval=.3)),
                bound=BoundCollapsingLinear(t=0, B=1),
                dx=.01, dt=.01)
  model_gui(model)
