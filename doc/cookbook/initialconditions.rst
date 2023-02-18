Recipes for Initial Conditions
==============================

General use of initial conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initial conditions can be included in the model by passing it directly
to the Model object.  For example, for a uniform distribution centered
at 0, do::

  from pyddm import Model
  from pyddm.models import ICRange
  model = Model(IC=ICRange(sz=.2))

.. _ic-biased:

Biased starting point
~~~~~~~~~~~~~~~~~~~~~

Often we want to model a side bias, either those which arise naturally or those
introduced experimentally through asymmetric reward or stimulus probabilities.
The most popular way of modeling a side bias is to use a starting position which
is closer to the boundary representing that side.

There are two ways to do this in PyDDM, :ref:`which are equivalent but
implemented in different ways <howto-stimulus-coding>`.  In the first, the upper
and lower boundaries of the diffusion process represent the correct and
incorrect answer.  This scheme, sometimes called "accuracy coding", is the
default for PyDDM.  However, it requires a calculation for each trial to
determine whether the bias is towards the upper or lower boundary.  In the
second way, the upper and lower boundaries represent distinct choices, e.g.,
"left" and "right".  This scheme, sometimes called "stimulus coding", must be
manually activated in PyDDM for a given Model and Sample by choosing names for
the boundaries.  This allows a much simpler (built-in) InitialCondition object
to be used, but it also makes it harder to visualise performance independent of
side.

For these examples, we will assume the two chices are "left" and "right", and
that we are implementing side bias to the left or right.

Accuracy coding for biased starting point
-----------------------------------------

Here, the upper boundary will be "the correct response" (whether that is left or
right on a given trial) and the lower boundary will be "the incorrect response".
This means the drift rate should always be positive.

First, we must first include a condition in our dataset describing whether the
correct answer was the left side or the right side.  Suppose we have a sample
which has the ``left_is_correct`` condition, which is 1 if the (true underlying)
correct answer is on the left side, and 0 if the (true underlying) correct
answer is on the right side.  Now, we can define an :class:`.InitialCondition`
object which uses this information.  We do this by defining a
:meth:`~.InitialCondition.get_IC` method.  This method should generate a
discretized probability distribution for the starting position.  Here, we want
this distribution to be a single point ``x0``, the sign of which (positive or
negative) depends on whether the correct answer is on the left or right side.
The function receives the support of the distribution in the ``x`` argument.  We
can model this with:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start ICPointSideBias
   :end-before: # End ICPointSideBias

Then we can compare the distribution of left-correct trials to those of
right-correct trials::

  from pyddm import Model
  from pyddm.plot import plot_compare_solutions
  import matplotlib.pyplot as plt
  model = Model(IC=ICPointSideBias(x0=.3))
  s1 = model.solve(conditions={"left_is_correct": 1})
  s2 = model.solve(conditions={"left_is_correct": 0})
  plot_compare_solutions(s1, s2)
  plt.show()

We can also see these directly in the model GUI::

  from pyddm import Model, Fittable
  from pyddm.plot import model_gui
  model = Model(IC=ICPointSideBias(x0=Fittable(minval=0, maxval=1)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"left_is_correct": [0, 1]})


Stimulus coding for biased starting point
-----------------------------------------

Instead of flipping the bias towards the upper or lower boundary, we can define
the upper boundary as the left choice and the lower boundary as the right
choice.  Then, we need to code the choices differently.  This can be achieved
with two modifications: one to the sample, and the other to the model.

For the Sample, instead of specifying choices by whether the choice was correct
or not, we instead need to define them by whether the choice was to the left or
the right.  **This requires a different input format for the data.** We can use
the "choice_names" argument as follows::

    samp = pyddm.Sample.from_pandas_dataframe(df, choice_column_name='choice_side',
                                                  rt_column_name='rt',
                                                  choice_names=("Left", "Right"))

This means that the choices specified in ``df['choice_side']`` are 1 if the
choice is to the left side and 0 if it is to the right side.  Other conditions
in the data may need to change their representation as well, e.g., the coding of
stimulus strength/coherence. 

Then, when creating our model, we must do::

    model = pyddm.Model(..., IC=ICPoint(x0=Fittable(minval=-1, maxval=1)), ..., choice_names=("Left", "Right"))

The "choice_names" variable in the model must match the "choice_names" variable
in the sample.  This construction can be used for either :class:`.ICPoint` or
:class:`.ICPointRatio`.

We can also try this out directly in the model GUI.  Notice how the GUI no
longer labels the sides as "correct" and "error", but instead, as "left" and
"right"::

  from pyddm import Model, Fittable, ICPoint
  from pyddm.plot import model_gui
  model = Model(IC=ICPoint(x0=Fittable(minval=-1, maxval=1)),
                dx=.01, dt=.01, choice_names=("Left", "Right"))
  model_gui(model)


Interpolation of starting points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Individual starting points may fall between a grid spacing when fitting data.
If this becomes a problem (e.g., with large dx), it is possible to linearly
approximate the probability density function at the two neighboring grids of the
initial position.  For instance, to do this for the biased initial conditions
above:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start ICPointSideBiasInterp
   :end-before: # End ICPointSideBiasInterp

Try it out with::

  from pyddm import Model, Fittable
  from pyddm.plot import model_gui
  model = Model(IC=ICPointSideBiasInterp(x0=Fittable(minval=0, maxval=1)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"left_is_correct": [0, 1]})

In practice, it is very similar, but the interpolated version gives a smoother
derivative, which may be useful for gradient-based fitting methods (which are
not used by default).


.. _ic-ratio:

Fixed ratio instead of fixed value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When fitting both the initial condition and the bound height, it can be
preferable to express the initial condition as a proportion of the total
distance between the bounds.  This ensures that the initial condition will
always stay within the bounds, preventing errors in fitting.  The "ratio" analog
of :class:`.ICPoint` is :class:`.ICPointRatio`, which is built-in to PyDDM.  If
you want to make it depend on conditions, you can do the following:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start ICPointSideBiasRatio
   :end-before: # End ICPointSideBiasRatio

Try it out with:: 

  from pyddm import Model, Fittable
  from pyddm.models import BoundConstant
  from pyddm.plot import model_gui
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

  from pyddm import Model, Fittable, DriftConstant
  from pyddm.plot import model_gui
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

  from pyddm import Model, Fittable, BoundCollapsingLinear
  from pyddm.plot import model_gui
  model = Model(IC=ICCauchy(scale=Fittable(minval=.001, maxval=.3)),
                bound=BoundCollapsingLinear(t=0, B=1),
                dx=.01, dt=.01)
  model_gui(model)
