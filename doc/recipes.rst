Examples and Recipes
=========================

Here are a list of common model features and how to implement them in
PyDDM.  If you have a simple and clear example, please contact us to
have it added.


Collapsing Bounds
~~~~~~~~~~~~~~~~~

Both linearly collapsing bounds (:class:`.BoundCollapsingLinear`) and
exponentially collapsing bounds (:class:`.BoundCollapsingExponential`)
already exist in PyDDM.  

Leaky/Unstable integrator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Leaky/unstable integrators are implemented in :class:`.DriftLinear`.  For
a leaky integrator, set the parameter `x` to be less than 0.  For an
unstable integrator, set the parameter `x` to be greater than 0.

Shared parameters
~~~~~~~~~~~~~~~~~

In order to use the same parameter for multiple different components
of the model, pass the same :class:`.Fittable` instance to both.  As a
concrete example, suppose we want both the drift rate and the variance
to increase by some factor `boost` at time `tboost`.  We could make
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
 
This will ensure that the value of `driftboost` is always equal to the
value of `noiseboost`, and that the value of `tboost` in Drift is always
equal to the value of `tboost` in Noise.
            
Note that this is **not the same** as::

  m = Model(drift=DriftBoost(driftbase=Fittable(minval=.1, maxval=3),
                       driftboost=Fittable(minval=1, maxval=3),
                       tboost=Fittable(minval=0, maxval=1)),
            noise=NoiseBoost(noisebase=Fittable(minval=.2, maxval=1.5),
                       noiseboost=Fittable(minval=1, maxval=3),
                       tboost=Fittable(minval=0, maxval=1)),
            T_dur=3, dt=.001, dx=.001)

In the latter case, `driftboost` and `noiseboost` will be fit to
different values, and the two `tboost` parameters will not be equal.

Parallelization
~~~~~~~~~~~~~~~

PyDDM has built-in support for parallelization if `pathos
<https://pypi.python.org/pypi/pathos>`_ is installed.

To use parallelization, first set up the parallel pool::

  from pathos.multiprocessing import Pool
  pool = Pool(3) # Fit with 3 cpus

Then, pass the `pool` object to the :func:`fit_adjust_model` function;
for example, to parallelize the example from the quickstart::

  fit_model_rs = fit_adjust_model(sample=roitman_sample, m=model_rs, pool=pool)
  
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

Here, `drift` is the strength of the evidence integration during the
pulse, `start` is the time of the pulse onset, and `duration` is the
duration of the pulse.

This can easily be modified to make it coherence dependent, where
`coherence` is the coherence in the :class:`.Sample`::

  from ddm.models import Drift
  class DriftPulseCoh(Drift):
      name = "Drift for a coherence-dependent pulse paradigm"
      required_parameters = ["start", "duration", "drift"]
      required_conditions = ["coherence"]
      def get_drift(self, t, conditions, **kwargs):
          if self.start <= t <= self.start + self.duration:
              return self.drift * conditions["coherence"]
          return 0

