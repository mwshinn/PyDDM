Task paradigms
==============

.. _paradigm-pulse:

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

		  
.. _paradigm-pk:

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

