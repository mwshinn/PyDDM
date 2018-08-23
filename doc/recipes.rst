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

To more accurately represent the initial condition, we can 
linearly approximate the probability density function at the two 
neighbouring grids of the initial position::

  from ddm.models import InitialCondition
  import numpy as np
  class ICPoint(InitialCondition):
      name = "A dirac delta function at a position dictated by reward."
      required_parameters = ["x0"]
      required_conditions = ["highreward"]
      def get_IC(self, x, dx, conditions):
          start_in = np.floor(self.x0/dx)
		  start_out = np.sign(start_in)*(np.abs(start_in)+1)
		  w_in = np.abs(start_out - self.x0/dx)
		  w_out = np.abs(self.x0/dx - start_in)
          if not conditions['highreward']:
              start_in = -start_in
              start_out = -start_out
          shift_in_i = int(start_in + (len(x)-1)/2)
          shift_out_i = int(start_out + (len(x)-1)/2)
		  if w_in>0:
			assert shift_in_i>= 0 and shift_in_i < len(x), "Invalid initial conditions"
		  if w_out>0:
			assert shift_out_i>= 0 and shift_out_i < len(x), "Invalid initial conditions"
          pdf = np.zeros(len(x))
          pdf[shift_in_i] = w_in # Initial condition at the inner grid next to x=self.x0.
          pdf[shift_out_i] = w_out # Initial condition at the outer grid next to x=self.x0.
          return pdf

  
Lapse rates for model fits
~~~~~~~~~~~~~~~~~~~~~~~~~~

When fitting models, especially when doing so with likelihood, it is
useful to have a constant lapse rate in the model to prevent the
likelihood from being negative infinity.  PyDDM has two useful
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
      def get_noise(self, t, conditions, **kwargs):
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
``coherence`` is a condition in the :class:`.Sample`::

  from ddm.models import Drift
  class DriftPulseCoh(Drift):
      name = "Drift for a coherence-dependent pulse paradigm"
      required_parameters = ["start", "duration", "drift"]
      required_conditions = ["coherence"]
      def get_drift(self, t, conditions, **kwargs):
          if self.start <= t <= self.start + self.duration:
              return self.drift * conditions["coherence"]
          return 0
		  
Alternatively, drift can be set at a default value independent of
coherence, and changed during the pulse duration::

  from ddm.models import Drift
  class DriftPulse(Drift):
      name = "Drift for a pulse paradigm, with baseline drift"
      required_parameters = ["start", "duration", "drift", "drift0"]
      required_conditions = []
      def get_drift(self, t, conditions, **kwargs):
          if self.start <= t <= self.start + self.duration:
              return self.drift
          return self.drift0

		  
Psychophysical Kernel paradigm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the psychophysical kernel paradigm, random time-varying but on average 
unbiased stimuli are presented on a trial-by-trial basis to quantify the 
weight a given time point has on behavioural choice. 

In particular, consider a sequence of coherences ``coh_t_list``, generated 
by randomly sampling from a pool of coherences ``coh_list_PK`` for 
``Tdur = 2`` seconds every ``dt_PK = 0.05`` seconds::

  coh_list = np.array([-25.6, -12.8, -6.4, 6.4, 12.8, 25.6])
  Tdur = 2
  dt_PK=0.05
  i_coh_t_list = np.random.randint(len(coh_list), size=int(Tdur/dt_PK))
  coh_t_list = [0.01*coh_list[i] for i in i_coh_t_list]

If the conversion from coherence to "drift" is known (e.g. by fitting 
other tasks), one can model the DDM with this sequence of evidence::

  from ddm.models import Drift
  class DriftPK(Drift):
      name = "PK drifts"
      required_conditions = ["coh_t_list", "dt_PK"]
      required_parameters = ["drift"]
      def get_drift(self, t, conditions, **kwargs):
          return self.drift**0.01*conditions["coh_t_list"][int(t/conditions["dt_PK"])]
	
Running the same process over multiple trials, we can use reverse correlation 
to obtain the impact of stimuli at each time-step on the final choice.
(Note: the following step is slow, as sufficiently many trials is needed to 
ensure each stimulus strength at each time-step is considered)::

  import numpy as np
  from ddm import Model
  from ddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture
  from ddm.functions import display_model
  n_rep=1000
  coh_list = np.array([-25.6, -12.8, -6.4, 6.4, 12.8, 25.6])
  Tdur = 2
  dt_PK=0.05
  PK_Mat = np.zeros((int(Tdur/dt_PK), len(coh_list)))
  PK_n   = np.zeros((int(Tdur/dt_PK), len(coh_list)))                                 
  for i_rep in range(n_rep):                                                                                    
      i_coh_t_list = np.random.randint(len(coh_list), size=int(Tdur/dt_PK))
      coh_t_list = [0.01*coh_list[i] for i in i_coh_t_list]
      model = Model(name='PK',
          drift=DriftPK(drift=2.2),
          noise=NoiseConstant(noise=1.5),
          bound=BoundConstant(B=1.1),
          overlay=OverlayNonDecision(nondectime=.1),
          dx=.001, dt=.01, T_dur=2)
      sol = model.solve(conditions={"coh_t_list": coh_t_list, "dt_PK": dt_PK})
      for i_t in range(int(Tdur/dt_PK)):
          PK_Mat[i_t, i_coh_t_list[i_t]] += sol.prob_correct() - sol.prob_error()
          PK_n[i_t, i_coh_t_list[i_t]] += 1
  PK_Mat = PK_Mat/PK_n

Where ``n_rep`` is the number trials. ``PK_Mat`` is known as the
psychophysical matrix. Normalizing by coherence and averaging across
stimuli (for each time-step), one obtains the psychophysical kernel
``PK``::
   
  for i_coh in range(len(coh_list)):
      PK_Mat[:,i_coh] /= coh_list[i_coh]
  PK = np.mean(PK_Mat, axis=1)

	
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
the data.  This model can then be defined as::

  import numpy as np
  from ddm.models import Drift
  class DriftSine(Drift):
      name = "Sine-wave drifts"
      required_conditions = ["frequency"]
      required_parameters = ["scale"]
      def get_drift(self, t, conditions, **kwargs):
          return np.sin(t*conditions["frequency"]*2*np.pi)*self.scale
		  
In this case, ``frequency`` is externally provided per trial, thus
defined as a condition.  By contrast, ``scale`` is a parameter to fit,
and is thus defined as a parameter.  We then use the DriftSine class
to define model::

  from ddm import Model
  model = Model(name='Sine-wave evidences',
	            drift=DriftSine(scale=0.5))
  sol = model.solve(conditions={"frequency": 5})
  
The model is solved and the result is saved in the variable sol, where
the :meth:`probability correct <.Solution.prob_correct>`, the
:meth:`reaction time distribution <.Solution.pdf_corr>`, and other
outputs could be retrieved. Finally, note that the conditions, being
externally defined (e.g. trial-by-trial), must be input during the
call to model.solve. The parameters, such as offset, are defined
within the respective classes.  Depending on the context, it could be
either a constant (as done here) or as a :class:`.Fittable` object, if
fitting to data is required.


Alternative fitting methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fitting methods can be modified by changing the loss function or by
changing the algorithm used to optimize the loss function.  The
default loss function is likelihood (via
:class:`~.models.loss.LossLikelihood`).  Squared error (via
:class:`~.models.loss.LossSquaredError`) and BIC (via
:class:`~.models.loss.LossBIC`) are also available.

As an example, to fit the "Simple example" from the quickstart guide,
do::

  fit_adjust_model(samp, model_fit,
                   method="differential_evolution",
                   lossfunction=LossSquaredError)

The default fitting method is differential evolution, but alternatives
include "simple" (by approximating the gradient and using steepest
descent), "simplex" (the Nelder-Mead simplex algorithm), and "basin"
(for the basin hopping algorithm).  For example, to use the
Nelder-Mead simplex algorithm, do::

  fit_adjust_model(samp, model_fit,
                   method="simplex",
                   lossfunction=LossBIC)

At this time, PyDDM does not support custom optimization methods, but
loss functions may be defined by extending
:class:`~.models.loss.LossFunction`.  Please see the API documentation
for more information on how to create custom loss functions.


Fitting with undecided trials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to correct and incorrect trials, some trials may go beyond
the time allowed for a decision.  The effect of these trials is
usually minor due to the design of task paradigms, but PyDDM is
capable of using these values within its fitting procedures.

Currently, the functions which import Sample objects from numpy arrays
do not support undecided trials; thus, to include undecided trials in
a sample, they must be passed directly to the Sample constructor in a
more complicated form.

To construct a sample with undecided trials, first create a Numpy
array of correct RTs and incorrect RTs in units of seconds, and count
the number of undecided trials.  Then, for each task conditions,
create a tuple containing three elements.  The first element should be
a Numpy array with the task condition value for each associated
correct RT, the second should be the same but for error trials, and
the final element should be a Numpy array in no particular order with
a number of elements equal to the undecided trials, with one
corresponding to each undecided trial.

Consider the following example with "reward" as the task condition. We
suppose there is one correct trial with a reward of 3 and an RT of
0.3s, one error with a reward of 2 and an RT of 0.5s, and two
undecided trials with rewards of 1 and 2::

  sample = Sample(np.asarray([0.3]), np.asarray([0.5]), 2,
                  reward=(np.asarray([3]), np.asarray([2]), np.asarray([1, 2])))
                                                                   
A sample created using this method can be used the same way as one
created using :meth:`~.Sample.from_numpy_array` or
:meth:`~.Sample.from_pandas_dataframe`.
