Lapse rates for model fits
==========================

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


