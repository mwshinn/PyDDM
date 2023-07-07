FAQs
====

How do I know if my model will run analytically or numerically?
---------------------------------------------------------------

The function :func:`~pyddm.model.Model.solve` will automatically choose the best
solver for your model.  Solvers, in order of preference, are

1. Analytical
2. Crank-Nicolson
3. Backward Euler (implicit method)

The analytical solver requires that Drift and Noise do not depend on
time or particle position, and that the initial position is fixed at
zero (:class:`.ICPointSourceCenter`).  Additionally, they require either Bounds
which do not depend on time, or alternatively linearly collapsing
bounds using :class:`.BoundCollapsingLinear`.  (Parameterized linearly
collapsing bounds are not currently supported.)

The Crank-Nicolson solver requires that Bounds do not depend on time,
i.e. models with collapsing bounds are not supported.

The Backward Euler (implicit method) solver is compatible with all
models and thus serves as a fallback.

Particle simulations and Forward Euler (explicit method) are also
available, but must be explicitly called via
:func:`~pyddm.model.Model.solve_numerical_explicit` and
:func:`~pyddm.model.Model.simulated_solution`.  They will never be chosen
automatically.

For custom models, these are specified by including ``x`` or ``t`` in
the argument list.


What arguments do :func:`~pyddm.models.drift.Drift.get_drift`, :func:`~pyddm.models.noise.Noise.get_noise`, etc. take?
------------------------------------------------------------------------------------------------------------------

The most appropriate solver is selected by PyDDM by examining the
variables on which different model components depend.  For models
components built-in to PyDDM, this is checked automatically.

For custom model components, this is found based on the arguments
taken by the relevant model function.  Internally, the time is passed
as the number ``t`` keyword argument, the position is passed as a
number or vector ``x``, and the trial conditions are passed as a
dictionary ``conditions``.  Thus, by including any of these variables
in the method definition, we thereby depend on that parameter.
(Likewise, all methods should end with ``**kwargs`` to not throw an
error when other parameters are passed, but should not access the
``kwargs`` variable directly.)

For example, suppose our experiment consisted of two blocks, one
associated with high rewards and the other associated with low
rewards, and we hypothesize that different bounds are used in these
two cases.  We could create the following Bound object which allows
the bounds in the two blocks to be fit independently::

  class BoundReward(pyddm.Bound):
      name = "Reward-modulated bounds"
      required_conditions = ["block"]
      required_parameters = ["bound1", "bound2"]
      def get_bound(self, conditions, **kwargs):
          if conditions["block"] == 1:
              return self.bound1
          elif conditions["block"] == 2:
              return self.bound2

Notice how the ``get_bound`` function does not depend on ``t``.
However, we could in theory also define the method ``get_bound`` as::

  def get_bound(self, t, conditions, **kwargs):
      ...

Even if the variable ``t`` is never used inside ``get_bound``, PyDDM
would interpret this to mean that the function depends on time.  Thus,
while this will give the expected result, it will not allow PyDDM to
properly optimize.

Alternatively, suppose a separate hypothesis whereby the block
described above modulates the rate of collapse in exponentially
collapsing bounds.  This could be modeled as::

  import numpy as np
  class BoundReward(pyddm.Bound):
      name = "Reward-modulated bounds"
      required_conditions = ["block"]
      required_parameters = ["rate1", "rate2", "B0"]
      def get_bound(self, t, conditions, **kwargs):
          if conditions["block"] == 1:
              rate = self.rate1
          elif conditions["block"] == 2:
              rate = self.rate2
          return self.B0 * np.exp(-rate*t)

In this case, our bound does depend on ``t``, so it **must** be
included in the function signature.

Why do I get "Paranoid" errors?
-------------------------------

`Paranoid Scientist <http://paranoid-scientist.readthedocs.io>`_ is a
library for verifying the accuracy of scientific software.  It is used
to check the entry and exit conditions of functions.

Paranoid Scientist will, overall, decrease the probability of an
undetected error by increasing the number of bugs overall.  Some
common errors are:

- When a particular model parametrization causes numerical instability
  at the given dx and dt.  This can cause probability distributions
  which go below zero.
- When numerical issues are amplified in the model, making the
  distribution integrate to more than 1 (plus floating point error).
- When dx and dt are too small for Crank-Nicolson and oscillations
  occur in the distribution.

If this becomes a problem during model fitting, it can be disabled
with::

  import paranoid as pns
  pns.settings.Settings.set(enabled=False)

When performing final simulations for the paper, it is recommended to
keep re-enable Paranoid Scientist, since turning it off may mask
numerical issues.

Can PyDDM fit hierarchical models?
----------------------------------

No, PyDDM cannot fit hierarchical models.  This need is already
addressed by the `hddm package <https://github.com/hddm-devs/hddm/>`_.
Due to limited resources, we do not plan to add support for
hierarchical models, but you are welcome to implement the feature
yourself and submit a pull request on Github.  If you plan to
implement this feature, please let us know so we can help you get
familiar with the code.

What is the difference between LossLikelihood and LossRobustLikelihood or LossBIC and LossRobustBIC?
----------------------------------------------------------------------------------------------------

Maximum likelihood in general is not good at handling probabilities of
zero.  When performing fitting using maximum likelihood (or
equivalently, BIC), the fit will fail if there are any times at which
the likelihood is zero.  If there is even one trial in the
experimental data which falls into a region where the simulated
probability distribution is zero, then the likelihood of the model
under that data is zero, and hence negative log likelihood is
infinity.  (See Ratcliff and Tuerlinckx (2002) for a more complete
discussion.)  In practice, there can be several locations where the
likelihood is theoretically zero.  For example, the non-decision time
by definition should have no responses.  However, data are noisy, and
some responses may be spurious.  This means that when fitting with
likelihood, the non-decision time cannot be any longer than the
shortest response in the data.  Clearly this is not acceptable.

PyDDM has two ways of circumventing this problem.  The most robust
way is to fit the data with a mixture model.  Here, the DDM process is
mixed with another distribution (called a "lapse", "contaminant", or
"outlier" distribution) which represent responses which came from a
non-DDM process.  Traditionally :class:`a uniform distribution
<.OverlayUniformMixture>` has been used, but PyDDM also offers the
option of using :class:`an exponential distribution
<.OverlayExponentialMixture>`, which has the benefit of providing a flat
lapse rate hazard function.  If you would also like to have a
non-decision time, you may need to :class:`chain together multiple
overlays <.OverlayChain>`.

The easier option is to use the :class:`LossRobustLikelihood
<.LossRobustLikelihood>` loss function.  This imposes a minimum value for the
likelihood.  In theory, it is similar to imposing a uniform distribution, but
with an unspecified mixture probability.  It will give nearly identical results
as LossLikelihood if there are no invalid results, but due to the minimum it
imposes, it is more of an approximation than the true likelihood.

Why do I get oscillations in my simulated RT distribution?
----------------------------------------------------------

Oscillations occur in the Crank-Nicolson method when your dt is too
large.  Try decreasing dt.  You should almost never use a dt larger
than .01, but smaller values are ideal.

Why is PyDDM so fast?
~~~~~~~~~~~~~~~~~~~~~

First, the core routines of PyDDM are written in optimized C.  We are
continuously tuning and refining our code to maximize performance.

Second, PyDDM by default does not (convieniently) support drift rate
variability.  This is a major performance bottleneck on all major DDM packages,
since it is hard to use mathematical derivations to optimize it.  Instead, a
model must be solved many times with different drift rates each time, slowing
down solving by about one order of magnitude.

Third, PyDDM automatically selects the appropriate solver for your model, based
on whether different aspects of your model depend on time.  For many complicated
models, PyDDM is able to find a strategy which uses an analytic solver.

Fourth, parallelization is easy, using the :func:`.set_N_cpus` function.  This
makes models with many conditions execute independently on different CPUs.

Why is PyDDM so slow?
---------------------

Your model may be slow for a number of different reasons.

- **You have a lot of conditions** -- Each time you solve the model (e.g. by
  calling :meth:`.Model.solve`), PyDDM internally needs to simulate one pdf per
  potential combination of conditions.  For example, if you are using 200
  different conditions values, then PyDDM will need to simulate 200 different
  pdfs for each call you make to :meth:`.Model.solve`.  Minimizing the number of
  conditions will thus lead to substantial speedups.
- **Your numerics (dx and dt) are too small** -- Larger values of dx and dt can
  lead to imprecise estimations of the response time distribution.  Therefore,
  be cautious when adjusting dx and dt.  As a rule of thumb, dx and dt should
  almost always be smaller than 0.01 and larger than 0.0001.  Setting them to
  0.005 is a good place to start.  If dx and dt are larger than 0.01, your
  estimated response time distribution will be inaccurate, and if dx and dt are
  smaller than 0.0001, solving the model will be extremely slow.  Usually, as a
  heuristic, PyDDM works best if dx and dt are approximately equal.
- **The C solver is not working properly** -- You can confirm that the C solver
  is operating by ensuring the variable ``pyddm.model.HAS_CSOLVE`` is True.  If
  there was an error installing the C solver when installing PyDDM, PyDDM will
  still run, but it will be 10-100x slower.

While simulations in PyDDM are fast, fitting models in PyDDM may be slower than
other packages.  This is because PyDDM uses differential evolution as a fitting
algorithm.  This algorithm rarely fails to maximize likelihood; however, it
requires more iterations than a gradient-based approach.  For simple models, it
*may* be possible to use the "simplex" method, but if you do, **please check to
make sure your models converge to a consistent global minimum**.  You can do
this by running several models using both the "simplex" method and
"differential_evolution" and confirming the results are the same, or by running
parameter recovery experiments on your model.  Since all GDDMs are different,
there can be no general guidance for how to ensure model convergence.

How many trials do I need to fit a GDDM to data?
------------------------------------------------

Since the GDDM is a framework rather than a specific model, there is no firm
minimum number of trials you need to fit a GDDM.  All GDDMs are different, and
so different models, fitting procedures, and objective functions could require
different sample sizes.

However, in general, there cannot be a "minimum sample size", because the more
data available, the more precise the parameters estimates will be.  Therefore,
the required sample size depends on how much variability one is willing to
tolerate in the parameter estimates.  This is true for other packages as well,
and so when other packages make claims about minimum sample size, these
estimates should be interpreted as rough guides of what people tend to use
rather than interpreted literally.

However, PyDDM makes it easy to test parameter recovery, which can be considered
a gold standard for determining the required sample size.  This allows you to
determine how many trials you need in order to get the parameter variability
you're willing to tolerate.  The idea is to build the model you want to fit,
choose reasonable-ish default parameters, and then simulate several trials from
that model using the :meth:`.Solution.resample` method.  After you simulate
these trials for different sample sizes, you fit the same model (but with
Fittable parameters) to the generated data. Then, you can find how close the
parameter estimates are to the actual parameters when you have different sample
sizes.

Does PyDDM support HDDM's "stimulus coding"?
--------------------------------------------

Yes, see :ref:`howto-stimulus-coding`.


Does PyDDM allow non-discrete conditions?
-----------------------------------------

Yes.  Conditions can be any Python object, including a number, a list, an array,
or a string.  This allows, for example, attention DDMs.

PyDDM runs fastest when there are a smaller number of conditions.  However,
PyDDM is frequently used for models where there is a separate condition for each
trial.  For instance, it is possible to have drift rate depend on other
observations, such as eye movements or electrophysiological signals.  See
:ref:`momenttomoment` for an example.

While PyDDM is able to do this faster than most other software packages, PyDDM
is fastest when there are fewer conditions.  (The execution time increases
linearly with the number of conditions.)  PyDDM can also parallelize this with
no extra effort required by the user to make it even faster.

Unfortunately, there are limits to this speed.  According to the two standard
solver methodologies (both supported by PyDDM), it is either possible to
simulate individual diffusion trajectories, or to solve the Fokker-Planck
equation separately for each trial.  If PyDDM isn't fast enough, the only (as
of 2022) way to make simulations with many conditions run faster is to simulate
many instances and then train a deep neural network on the RT distribution.
There is a way to do this in HDDM, described in `Fengler et al (2022)
<https://elifesciences.org/articles/65074>`_.  No such feature is currently
planned in PyDDM.


When should I use RobustLikelihood or RobustBIC?
------------------------------------------------

RobustLikelihood and RobustBIC are almost identical to Likelihood and BIC, but
they have a uniform distribution mixture model built in.  (More specifically, it
sets a "minimum value" for the log likelihood by adding a small constant term to
it.)  This is to avoid infinite likelihoods where the distribution is zero.  If
you are already using a mixture model (e.g. OverlayUniformMixture or
OverlayExponentialMixture), then you should not use RobustLikelihood or
RobustBIC.

If you compare the likelihood or BIC of two models using the robust versions,
keep in mind that you are actually comparing the mixture model.  This is
necessary for likelihood estimation and therefore occurs in other packages as
well, which refer to it as the probability of "contaminant RTs" (fast-dm) or
"outliers" (HDDM).
