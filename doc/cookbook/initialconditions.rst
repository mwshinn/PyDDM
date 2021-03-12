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

When rewards or stimulus probabilities are asymmetric, a common
paradigm is to start the trial with a bias towards one side.  Suppose
we have a sample which has the ``highreward`` condition set to either
0 or 1, describing whether the correct answer is the high or low
reward side.  We must define the :meth:`~.InitialCondition.get_IC`
method in an :class:`.InitialCondition` object which generates a
discredited probability distribution on the model's position grid
domain.  We can model this with:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start ICPointRew
   :end-before: # End ICPointRew

Then we can compare the high reward distribution to the low reward
distribution::

  from ddm import Model
  from ddm.plot import plot_compare_solutions
  import matplotlib.pyplot as plt
  model = Model(IC=ICPointRew(x0=.3))
  s1 = model.solve(conditions={"highreward": 1})
  s2 = model.solve(conditions={"highreward": 0})
  plot_compare_solutions(s1, s2)
  plt.show()

We can also see these directly in the model GUI::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(IC=ICPointRew(x0=Fittable(minval=0, maxval=1)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"highreward": [0, 1]})

To more accurately represent the initial condition, we can 
linearly approximate the probability density function at the two 
neighboring grids of the initial position:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start ICPointRewInterp
   :end-before: # End ICPointRewInterp

Try it out with::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(IC=ICPointRewInterp(x0=Fittable(minval=0, maxval=1)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"highreward": [0, 1]})

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
   :start-after: # Start ICPointRewRatio
   :end-before: # End ICPointRewRatio

Try it out with:: 

  from ddm import Model, Fittable
  from ddm.models import BoundConstant
  from ddm.plot import model_gui
  model = Model(IC=ICPointRewRatio(x0=Fittable(minval=-1, maxval=1)),
                bound=BoundConstant(B=Fittable(minval=.1, maxval=2)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"highreward": [0, 1]})

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
  model_gui(model, conditions={"highreward": [0, 1]})

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
