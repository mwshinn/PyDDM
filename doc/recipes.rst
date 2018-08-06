Examples and Recipes
=========================

Here are a list of common model features and how to implement them in
PyDDM.  If you have a simple and clear example, please contact us to
have it added.


Collapsing Bounds
~~~~~~~~~~~~~~~~~

Both linearly collapsing bounds (:class:`.BoundCollapsingLinear`) and
exponentially collapsing bounds (:class:`.BoundCollapsingExponential`)
already exist in PyDDM.  For example::

  from ddm import Model
  from ddm.models import BoundCollapsingLinear, BoundCollapsingExponential
  model1 = Model(bound=BoundCollapsingExponential(B=1, tau=2))
  model2 = Model(bound=BoundCollapsingLinear(B=1, t=.2))

It is also possible to make collapsing bounds of any shape.  For
example, the following describes bounds which collapse according to a
step function::

  from ddm.models import Bound
  class BoundCollapsingStep(Bound):
      name = "Step collapsing bounds"
      required_conditions = []
      required_parameters = ["B0", "stepheight", "steplength"]
      def get_bound(self, t, **kwargs):
          stepnum = t//self.steplength
          step = self.B0 - stepnum * self.stepheight
          return max(step, 0)

Then we can use this in a model with::

  from ddm import Model
  model = Model(bound=BoundCollapsingStep(B0=1, stepheight=.1, steplength=.1))


Biased Initial Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~

When rewards or stimulus probabilities are asymmetric, a common
paradigm is to start the trial with a bias towards one side.  Suppose
we have a sample which has the ``highreward`` condition set to either
0 or 1, describing whether the correct answer is the high or low
reward side.  We must define the :meth:`~.InitialCondition.get_IC`
method in an :class:`.InitialCondition` object which generates a
discredited probability distribution on the model's position grid
domain.  We can model this with::

  from ddm.models import InitialCondition
  import numpy as np
  class ICPoint(InitialCondition):
      name = "A dirac delta function at a position dictated by reward."
      required_parameters = ["x0"]
      required_conditions = ["highreward"]
      def get_IC(self, x, dx, conditions):
          start = np.round(self.x0/dx)
          if not conditions['highreward']:
              start = -start
          shift_i = int(start + (len(x)-1)/2)
          assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
          pdf = np.zeros(len(x))
          pdf[shift_i] = 1. # Initial condition at x=self.x0.
          return pdf

Then we can compare the high reward distribution to the low reward
distribution::

  from ddm import Model
  from ddm.plot import plot_compare_solutions
  import matplotlib.pyplot as plt
  model = Model(IC=ICPoint(x0=.3))
  s1 = model.solve(conditions={"highreward": 1})
  s2 = model.solve(conditions={"highreward": 0})
  plot_compare_solutions(s1, s2)
  plt.show()

Lapse rates for model fits
~~~~~~~~~~~~~~~~~~~~~~~~~~

When fitting models, especially when doing so with likelihood, it is
useful to have a constant lapse rate in the model to prevent the
likelihood from being negative inifinity.  PyDDM has two useful
built-in lapse rates for this which are used as mixture models: an
:class:`Exponential lapse rate <.OverlayPoissonMixture>` (according
to a Poisson process, the recommended method) and the :class:`Uniform
lapse rate <.OverlayUniformMixture>` (which is more common in the
literature).  These can be introduced with::

  from ddm import Model
  from ddm.models import OverlayPoissonMixture, OverlayUniformMixture
  model1 = Model(overlay=OverlayPoissonMixture(pmixturecoef=.05, rate=1))
  model2 = Model(overlay=OverlayUniformMixture(umixturecoef=.05))

If another overlay is to be used, such as
:class:`.OverlayNonDecision`, then an :class:`.OverlayChain` object
must be used::

  from ddm import Model
  from ddm.models import OverlayPoissonMixture, OverlayNonDecision, OverlayChain
  model = Model(overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=.2),
                                               OverlayPoissonMixture(pmixturecoef=.05, rate=1)]))


Leaky/Unstable integrator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Leaky/unstable integrators are implemented in :class:`.DriftLinear`.
For a leaky integrator, set the parameter ``x`` to be less than 0.
For an unstable integrator, set the parameter ``x`` to be greater
than 0.  For example::

  from ddm import Model
  from ddm.models import DriftLinear
  model = Model(drift=DriftLinear(drift=0, t=.2, x=.1))

Shared parameters
~~~~~~~~~~~~~~~~~

In order to use the same parameter for multiple different components
of the model, pass the same :class:`.Fittable` instance to both.  As a
concrete example, suppose we want both the drift rate and the variance
to increase by some factor ``boost`` at time ``tboost``.  We could make
:class:`.Drift` and :class:`.Noise` objects as follows::

  from ddm.models import Drift, Noise
  class DriftBoost(Drift):
      name = "Drift with a time-delayed boost"
      required_parameters = ["driftbase", "driftboost", "tboost"]
      required_conditions = []
      def get_drift(self, t, conditions, **kwargs):
          if t < self.tboost:
              return self.driftbase
          elif t >= self.tboost:
              return self.driftbase * self.driftboost
  
  class NoiseBoost(Noise):
      name = "Noise with a time-delayed boost"
      required_parameters = ["noisebase", "noiseboost", "tboost"]
      required_conditions = []
      def get_drift(self, t, conditions, **kwargs):
          if t < self.tboost:
              return self.noisebase
          elif t >= self.tboost:
              return self.noisebase * self.noiseboost

Now, we can define a model to fit with::

  from ddm import Model, Fittable
  t_boost = Fittable(minval=0, maxval=1)
  boost = Fittable(minval=1, maxval=3)
  m = Model(drift=DriftBoost(driftbase=Fittable(minval=.1, maxval=3),
                       driftboost=boost,
                       tboost=t_boost),
            noise=NoiseBoost(noisebase=Fittable(minval=.2, maxval=1.5),
                       noiseboost=boost,
                       tboost=t_boost),
            T_dur=3, dt=.001, dx=.001)

This will ensure that the value of ``driftboost`` is always equal to the
value of ``noiseboost``, and that the value of ``tboost`` in Drift is always
equal to the value of ``tboost`` in Noise.

Note that this is **not the same** as::

  m = Model(drift=DriftBoost(driftbase=Fittable(minval=.1, maxval=3),
                       driftboost=Fittable(minval=1, maxval=3),
                       tboost=Fittable(minval=0, maxval=1)),
            noise=NoiseBoost(noisebase=Fittable(minval=.2, maxval=1.5),
                       noiseboost=Fittable(minval=1, maxval=3),
                       tboost=Fittable(minval=0, maxval=1)),
            T_dur=3, dt=.001, dx=.001)

In the latter case, ``driftboost`` and ``noiseboost`` will be fit to
different values, and the two ``tboost`` parameters will not be equal.

Parallelization
~~~~~~~~~~~~~~~

PyDDM has built-in support for parallelization if `pathos
<https://pypi.python.org/pypi/pathos>`_ is installed.

To use parallelization, first set up the parallel pool::

  from pathos.multiprocessing import Pool
  pool = Pool(3) # Fit with 3 cpus

Then, pass the ``pool`` object to the :func:`fit_adjust_model` function;
for example, to parallelize the example from the quickstart::

  fit_model_rs = fit_adjust_model(sample=roitman_sample, model=model_rs, pool=pool)
  
There are a few caveats with parallelization:

1. It is only possible to run fits in parallel if they are on the same
   computer.  It is not possible to fit across multiple nodes in a
   cluster, for example.
2. Due to a bug in pathos, all model components must be **defined in a
   separate file** and then imported.
3. Only models with many conditions will be sped up by
   parallelization.  The cardinality of the cartesian product of the
   conditions is the maximum number of CPUs that will have an effect:
   for example, if you have four coherence conditions, a right vs left
   condition, and a high vs low reward condition, then after :math:`4
   \times 2 \times 2 = 16` CPUs, there will be no benefit to
   increasing the number of CPUs.


Pulse paradigm
~~~~~~~~~~~~~~

The pulse paradigm, where evidence is presented for a fixed amount of
time only, is common in behavioral neuroscience.  For simplicity, let
us first model it without coherence dependence::

  from ddm.models import Drift
  class DriftPulse(Drift):
      name = "Drift for a pulse paradigm"
      required_parameters = ["start", "duration", "drift"]
      required_conditions = []
      def get_drift(self, t, conditions, **kwargs):
          if self.start <= t <= self.start + self.duration:
              return self.drift
          return 0

Here, ``drift`` is the strength of the evidence integration during the
pulse, ``start`` is the time of the pulse onset, and ``duration`` is the
duration of the pulse.

This can easily be modified to make it coherence dependent, where
``coherence`` is the coherence in the :class:`.Sample`::

  from ddm.models import Drift
  class DriftPulseCoh(Drift):
      name = "Drift for a coherence-dependent pulse paradigm"
      required_parameters = ["start", "duration", "drift"]
      required_conditions = ["coherence"]
      def get_drift(self, t, conditions, **kwargs):
          if self.start <= t <= self.start + self.duration:
              return self.drift * conditions["coherence"]
          return 0

Sine wave evidence
~~~~~~~~~~~~~~~~~~

Suppose we have a task where evidence varies according to
a sine wave which has a different frequency on different trials::

  import numpy as np
  class DriftSine(ddm.Drift):
      name = "Sine-wave bounds"
      required_conditions = ["frequency"]
      required_parameters = ["offset"]
      def get_drift(self, t, conditions, **kwargs):
          return np.sin(t*conditions["frequency"]*2*np.pi)+self.offset
