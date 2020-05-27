How-to guides
=============

.. _howto-shared-params:

Shared parameters
~~~~~~~~~~~~~~~~~

In order to use the same parameter for multiple different components
of the model, pass the same :class:`.Fittable` instance to both.  As a
concrete example, suppose we want both the drift rate and the standard
deviation to increase by some factor ``boost`` at time ``tboost``.  We
could make :class:`.Drift` and :class:`.Noise` objects as follows::

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


.. _howto-parallel:

Parallelization
~~~~~~~~~~~~~~~

PyDDM has built-in support for parallelization if `pathos
<https://pypi.python.org/pypi/pathos>`_ is installed.

To use parallelization, first set up the parallel pool::

  from ddm import set_N_cpus
  set_N_cpus(4)

Then, PyDDM will automatically parallelize the fitting routines.  For
example, just call::

  fit_model_rs = fit_adjust_model(sample=roitman_sample, model=model_rs)
  
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
4. It is possible but not recommended to set the number of CPUs to be
   greater than the number of physical CPU cores on the machine.  This
   will cause a slight reduction in performance.

.. _howto-fit-custom-algorithm:
	
Fitting models with custom algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As described in :func:`.fit_adjust_model`, three different algorithms
can be used to fit models.  The default is differential evolution,
which we have observed to be robust for models with large numbers of
parameters.

Other methods can be used by passing the "method" argument to
:func:`.fit_adjust_model` or :func:`.fit_model`.  This can take one of
several values:

- "simplex": Use the Nelder-Mead simplex method
- "simple": Gradient descent
- "basin": Use Scipy's `basin hopping algorithm
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_.
- A function can be passed to use this function as a custom objective
  function.

For example, to fit the model in the quickstart using the Nelder-Mead
simplex method, you can do::

  fit_model_rs = fit_adjust_model(sample=roitman_sample, model=model_rs, method="simplex")


.. _howto-evolution:

Retrieve the evolving pdf of a solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting return_evolution=True in solve_numerical() will (with methods "implicit" 
and "explicit" only) return an M-by-N array (as part of the Solution) whose 
columns contain the cross-sectional pdf for every time step::

  sol = model.solve_numerical_implicit(conditions=conditions, return_evolution=True)
  sol.pdf_evolution()
     
This is equivalent to (but much faster than)::
  
    sol = np.zeros((len(model.x_domain(conditions)), len(model.t_domain())))          
    sol[:,0] = model.IC(conditions=conditions)/model.dx
    for t_ind, t in enumerate(model.t_domain()[1:]):
        T_dur_backup = model.T_dur
        model.T_dur = t
        ans = model.solve_numerical_implicit(conditions=conditions, return_evolution=False) 
        model.T_dur = T_dur_backup
        sol[:,t_ind+1] = ans.pdf_undec()    
        
Note that::

    
    sum(pdf_corr()[0:t]*dt) + sum(pdf_err()[0:t]*dt) + sum(pdf_evolution()[:,t]*dx) = 1
