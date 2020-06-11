Drift and Noise Models
======================

.. _drift-leak:

Leaky/Unstable integrator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Leaky/unstable integrators are implemented in :class:`.DriftLinear`.
For a leaky integrator, set the parameter ``x`` to be less than 0.
For an unstable integrator, set the parameter ``x`` to be greater
than 0.  For example::

  from ddm import Model
  from ddm.models import DriftLinear
  model = Model(drift=DriftLinear(drift=0, t=.2, x=.1))

Try it out with::

  from ddm import Model, Fittable, DriftLinear
  from ddm.plot import model_gui
  model = Model(drift=DriftLinear(drift=Fittable(minval=0, maxval=2),
                                  t=Fittable(minval=0, maxval=2),
                                  x=Fittable(minval=-1, maxval=1)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"frequency": [0, 4, 8]})

  
.. _drift-sine:

Sine wave evidence
~~~~~~~~~~~~~~~~~~

We use evidence in the form of a sine wave as an example of how to
construct a new model class.

Suppose we have a task where evidence varies according to a sine wave
which has a different frequency on different trials.  The frequency is
a feature of the task, and will be the same for all components of the
model.  Thus, it is a "condition".  By contrast, how strongly the
animal weights the evidence is not observable and only exists internal
to the model.  It is a "parameter", or something that we must fit to
the data.  This model can then be defined as:

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start DriftSine
   :end-before: # End DriftSine
		  
In this case, ``frequency`` is externally provided per trial, thus
defined as a condition.  By contrast, ``scale`` is a parameter to fit,
and is thus defined as a parameter.  We then use the DriftSine class
to define model::

  from ddm import Model
  model = Model(name='Sine-wave evidences',
	            drift=DriftSine(scale=0.5))
  sol = model.solve(conditions={"frequency": 5})
  
The model is solved and the result is saved in the variable "sol",
where the :meth:`probability correct <.Solution.prob_correct>`, the
:meth:`reaction time distribution <.Solution.pdf_corr>`, and other
outputs could be retrieved. Finally, note that the conditions, being
externally defined (e.g. trial-by-trial), must be input during the
call to model.solve. The parameters, such as offset, are defined
within the respective classes.  Depending on the context, it could be
either a constant (as done here) or as a :class:`.Fittable` object, if
fitting to data is required.

Try it out with::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(drift=DriftSine(scale=Fittable(minval=0, maxval=2)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"frequency": [0, 4, 8]})


.. _drift-coh:

Coherence-dependent drift rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start DriftCoherence
   :end-before: # End DriftCoherence

Try it out with::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(drift=DriftCoherence(driftcoh=Fittable(minval=0, maxval=1)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"coh": [0, .25, .5]})


.. _drift-coh-rew:

Biased coherence-dependent drift rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To model tasks in which some trials are expected to have a biased
drift rate (e.g. tasks with a reward bias or strong trial history
effects), we may want the drift rate to be biased in a certain
direction (called "me" in some models) depending on task parameters.
In the case of reward bias, whereby high reward trials have a higher
drift rate, we need two task parameters: one describing the coherence
of the trial ("coh") and one indicator of whether the trial was high
reward (1) or low reward (0) ("highreward").  Then we have to
parameters: the amount by which coherence impacts the drift rate
("driftcoh") and the amount of reward bias ("rewbias").

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start DriftCoherenceRewBias
   :end-before: # End DriftCoherenceRewBias

Try it out with::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(drift=DriftCoherenceRewBias(
                          driftcoh=Fittable(minval=0, maxval=1),
                          rewbias=Fittable(minval=0, maxval=1)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"coh": [0, .25, .5], "highreward": [0, 1]})


.. _drift-coh-leak:

Coherence-dependent drift rate with leak
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start DriftCoherenceLeak
   :end-before: # End DriftCoherenceLeak

Try it out with::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(drift=DriftCoherenceLeak(driftcoh=Fittable(minval=0, maxval=1),
                                         leak=Fittable(minval=-1, maxval=1)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"coh": [0, .25, .5]})


.. _drift-gain-function:

Urgency gain function
~~~~~~~~~~~~~~~~~~~~~

To implement a gain function, we want to scale both the drift rate and
the noise by some function.  First, let's define such a function to be
a simple linear ramp.

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start urgency_gain
   :end-before: # End urgency_gain

Now we can define a Drift model component which uses this.

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start DriftUrgencyGain
   :end-before: # End DriftUrgencyGain

We also need to define a Noise model component which uses it.  This
allows us to keep the signal-to-noise ratio (SNR) fixed.

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start NoiseUrgencyGain
   :end-before: # End NoiseUrgencyGain

Finally, here is an example of how we would use this.  Note that it
utilizes :ref:`shared parameters <howto-shared-params>`::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  gain_start = Fittable(minval=0, maxval=1)
  gain_slope = Fittable(minval=0, maxval=2)
  m = Model(drift=DriftUrgencyGain(snr=Fittable(minval=0, maxval=2),
                                   gain_start=gain_start,
                                   gain_slope=gain_slope),
            noise=NoiseUrgencyGain(gain_start=gain_start,
                                   gain_slope=gain_slope),
            dt=.01, dx=.01)
  model_gui(model=m)



.. _drift-uniform:

Across-trial variability in drift rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While across-trial variability in drift rate is possible in PyDDM, it
is not efficient or ergonomic.  Unlike variability in starting
position or non-decision time, the distribution of drift rates must be
discretized, and each must be run separately.  Here, we demonstrate
how to do this for a uniform distribution.

In order to run such a model, first you must prepare your Sample by
running it through the following function.  This makes one duplicate
of each of your data points according to each discretization point.
As a result, note that that likelihood, BIC, and other summary
statistics about the data or fit quality may be inaccurate.

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start prepare_sample_for_variable_drift
   :end-before: # End prepare_sample_for_variable_drift

After using the above function to generate a new sample, use a class
such as the following, which provides a uniformly-distributed drift
rate.

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start DriftUniform
   :end-before: # End DriftUniform

This can be used like any other Drift class.  For example::

  from ddm import Model, Fittable
  from ddm.plot import model_gui
  model = Model(drift=DriftUniform(drift=Fittable(minval=0, maxval=2),
                                   width=Fittable(minval=0, maxval=2)),
                dx=.01, dt=.01)
  model_gui(model, conditions={"driftnum": list(range(0, 11))})

