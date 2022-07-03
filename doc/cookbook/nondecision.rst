Recipes for Non-Decision Time
==============================

General use of non-decision time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to incorporate a non-decision time is to use the
built-in :class:`.OverlayNonDecision`, for example::

  from pyddm import Model, Fittable, OverlayNonDecision
  from pyddm.plot import model_gui
  model = Model(overlay=OverlayNonDecision(nondectime=Fittable(minval=0, maxval=.8)),
                dx=.01, dt=.01)
  model_gui(model)


.. _nd-gaussian:

Gaussian-distributed non-decision time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start OverlayNonDecisionGaussian
   :end-before: # End OverlayNonDecisionGaussian

Try it out with::

  from pyddm import Model, Fittable, OverlayNonDecision
  from pyddm.plot import model_gui
  model = Model(overlay=OverlayNonDecisionGaussian(
                    nondectime=Fittable(minval=0, maxval=.8),
                    ndsigma=Fittable(minval=0, maxval=.8)),
                dx=.01, dt=.01)
  model_gui(model)

.. _nd-lr:

Different non-decision time for left and right trials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we show an example of a non-decision time which depends on both the trial
conditions and mulitiple parameters.  In this case, we use a different
non-decision time based on whether a stimulus was presented on the left or right
side of the subject.

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start OverlayNonDecisionLR
   :end-before: # End OverlayNonDecisionLR

Try it out with::

  from pyddm import Model, Fittable, OverlayNonDecision
  from pyddm.plot import model_gui
  model = Model(overlay=OverlayNonDecisionLR(
                    nondectimeL=Fittable(minval=0, maxval=.8),
                    nondectimeR=Fittable(minval=0, maxval=.8)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"side": [0, 1]})

