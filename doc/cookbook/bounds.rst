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
example, the following describes bounds which collapse according to a
step function:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start BoundCollapsingStep
   :end-before: # End BoundCollapsingStep


Then we can use this in a model with::

  from ddm import Model
  model = Model(bound=BoundCollapsingStep(B0=1, stepheight=.1, steplength=.1))

.. _bound-weibull-cdf:

Weibull CDF collapsing bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Weibull function is a popular choice for collapsing bounds.  (See,
e.g., `Hawkins et al. (2015)
<https://doi.org/10.1523/JNEUROSCI.2410-14.2015 >`_.)  This can be
implemented using:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start BoundCollapsingWeibull
   :end-before: # End BoundCollapsingWeibull


(Note that in `Hakwins et al. (2015)
<https://doi.org/10.1523/JNEUROSCI.2410-14.2015>`_, diffusion goes
from [0,1], whereas our diffusion goes from [-1,1].  Thus, the 0.5
term was removed.)

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
