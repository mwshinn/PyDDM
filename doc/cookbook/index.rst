PyDDM Cookbook
===============

Here are a list of examples of common model features and how to
implement them in PyDDM.  **If you created an example or model in
PyDDM and would like it to be added to the cookbook, please** `send
it to us <mailto:maxwell.shinn@yale.edu>`_ **so we can add it**.
Include the author(s) of the example or model and optionally a
literature reference so that we can give you proper credit and direct
users to your paper!

:download:`Download cookbook.py (all models in the cookbook) <../downloads/cookbook.py>`

Task paradigms
~~~~~~~~~~~~~~

Here are some examples of potential task paradigms that can be
simulated with PyDDM.

- :ref:`A pulse paradigm <paradigm-pulse>`

- :ref:`Evidence oscillating in a sine wave <drift-sine>`

- :ref:`A psychophysical kernel paradigm <paradigm-pk>`

- Something else :doc:`(Write your own, using these as a guide.) <driftnoise>`


Drift and noise
~~~~~~~~~~~~~~~

I want:

- :ref:`Leaky or unstable integration <drift-leak>`

- :class:`A general Ornstein-Uhlenbeck process <.DriftLinear>`

- :ref:`A drift rate which depends linearly on a task parameter (e.g. coherence) <drift-coh>`

- :ref:`A biased drift dependent on a task condition (e.g. reward or choice history) <drift-coh-rew>`

- :ref:`A leaky integrator with a drift rate which depends linearly on a task parameter (e.g. coherence) <drift-coh-leak>`

- :ref:`An urgency gain function <drift-gain-function>`

- :ref:`Drift rate variability (uniform distribution) <drift-uniform>`

- Something else :doc:`(Write your own, using these as a guide.) <driftnoise>`


Collapsing bounds (or time-varying bounds)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I instead want bounds which:

- :class:`Are constant over time <.BoundConstant>`

- :class:`Collapse linearly <.BoundCollapsingLinear>`

- :class:`Collapse exponentially <.BoundCollapsingExponential>`

- :ref:`Collapse exponentially after a delay <bound-exp-delay>`

- :ref:`Collapse according to a Weibull CDF <bound-weibull-cdf>`

- :ref:`Collapse according to a step function <bound-step>`

- :ref:`Vary based on task conditions, e.g. for a speed vs accuracy task <bound-speedacc>`

- :ref:`Increase over time <bound-increase>`

- Something else :doc:`(Write your own, using these as a guide) <bounds>`


Initial conditions
~~~~~~~~~~~~~~~~~~

I don't want my initial conditions to be :class:`A single point
positioned in the middle of the bounds <.ICPointSourceCenter>` (the
default).  Instead, I want my initial conditions to be:

- A single point

  - :class:`A single point at an arbitrary location <.ICPoint>`

  - :ref:`A single point at an arbitrary location with a sign which depends on task conditions <ic-biased>`

  - :ref:`A single point determined as a ratio of bound height (useful for fitting bound height) <ic-ratio>`

- A uniform distribution

  - :class:`A uniform distribution across all potential starting positions <.ICUniform>`

  - :class:`A uniform distribution of arbitrary length centered in the middle of the bounds <.ICRange>`

  - :ref:`A uniform distribution of arbitrary length with a center at an arbitrary location wtih sign determined by task conditions <ic-biased-range>`

- :class:`A Gaussian distribution centered in the middle of the bounds <.ICGaussian>`

- :ref:`A Cauchy distribution <ic-cauchy>`

- :func:`A specific distribution which does not change based on task parameters <.ICArbitrary>`

- Something else :doc:`(Write your own, using these as a guide) <initialconditions>`


Non-decision time
~~~~~~~~~~~~~~~~~

I want to use a non-decision time which is:

- :class:`Fixed at a single value <.OverlayNonDecision>`

- :class:`A uniform distribution <.OverlayNonDecisionUniform>`

- :class:`A gamma distribution <.OverlayNonDecisionGamma>`

- :ref:`A Gaussian distribution <nd-gaussian>`

- :ref:`A different value for left and right choices <nd-lr>`

- Something else :doc:`(Write your own, using these as a guide) <nondecision>`


Mixture models (Contaminant RTs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I want to fit a distribution which has contaminant RTs distributed
according to:

- :class:`A uniform distribution <.OverlayUniformMixture>`

- :class:`An exponential distribution (corresponding to a Poisson process) <.OverlayExponentialMixture>`

- Something else (Write your own)


Objective functions
~~~~~~~~~~~~~~~~~~~

I don't want to use the default recommended objective function
(:class:`negative log likelihood <.LossLikelihood>`) but would rather
use:

- :class:`Squared error <.LossSquaredError>`

- :class:`BIC <.LossBIC>`

- :ref:`Mean RT and P(correct) <loss-means>`

- :ref:`Something which takes undecided trials into account <loss-undecided>`

- Something else :doc:`(Write your own, using these as a guide) <loss>`

(Note that changing the objective function to something other than
likelihood will not speed up model fitting.)

Fitting methods
~~~~~~~~~~~~~~~

I don't want to fit using the default recommended method (differential
evolution), but would rather fit using:

- :ref:`Nelder-Mead <howto-fit-custom-algorithm>`

- :ref:`Basin hopping <howto-fit-custom-algorithm>`

- :ref:`Gradient descent <howto-fit-custom-algorithm>`

(While using a fitting method other than differential evolution will
likely reduce the time needed for fitting models, other methods may
not offer robust parameter estimation for high-dimensional models.)

Models from specific papers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

*If you have a paper which used PyDDM, please* `send us your model <mailto:maxwell.shinn@yale.edu>`_ *so we can include them here!*

- :doc:`papers/shinn2020`

- :doc:`papers/degee2020`

Other recipes
~~~~~~~~~~~~~

I want to:

- :doc:`Fit a model to data <../quickstart>`

- :ref:`Share parameters between two different models <howto-shared-params>`

- :class:`Use multiple "Overlay" objects in the same model <.OverlayChain>`

- :ref:`Run models in parallel using multiple CPUs <howto-parallel>`

- :ref:`Retrieve the evolving pdf of a solution <howto-evolution>`


.. toctree::
   :caption: See also:
   :maxdepth: 1

   howto
   paradigms
   driftnoise
   bounds
   initialconditions
   nondecision
   lapse
   loss

