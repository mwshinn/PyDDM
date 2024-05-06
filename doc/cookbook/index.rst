PyDDM Cookbook
==============

Below are recipies for how to get specific models and features to work in PyDDM.
If you have not yet worked through the :doc:`Quickstart guide <../quickstart>`,
please do that first!  This will give you the fundamentals for understanding the
models below, demonstrated using leaky integration, collapsing bounds, and a
coherence-dependent drift rate.

Common models
~~~~~~~~~~~~~

- :ref:`Attention DDM (ADDM) <addm>`
- :ref:`Reinforcement learning DDM (RL-DDM) <rlddm>`
- :ref:`DDM with parameter variability ("full DDM") <full-ddm>`
- :ref:`Weibull collapsing bounds <weibull-bounds>`
- :ref:`Multi-sensory integration <multisensory-drift>`

Model components
~~~~~~~~~~~~~~~~

- :ref:`Biased drift rate <biased-drift>`
- :ref:`Changing drift rate <changing-drift>`
- :ref:`Non-linear drift rate <nonlinear-drift>`
- :ref:`Unique moment-to-moment drift rate on each trial (e.g., matching EEG) <drift-moment-to-moment>`
- :ref:`Biased starting position <biased-starting-position>`
- :ref:`Starting position variability <starting-point-variability>`
- :ref:`Non-decision time variability <non-decision-variability>`
- :ref:`Urgency signal as a multiplicative gain <urgency-gain>`
- :ref:`Attractor states <attractors>`

How-to
~~~~~~

- :ref:`Parallelization <howto-parallel>`
- :ref:`Stimulus coding vs accuracy coding <howto-stimulus-coding>`
- :ref:`Change the fitting algorithm <howto-fit-custom-algorithm>`
- :ref:`Visualize the evolution of the PDF over time <howto-evolution>`
- `Change the objective function used for fitting <loss>`_

Object-oriented interface
~~~~~~~~~~~~~~~~~~~~~~~~~

The object-oriented interface is no longer recommended.  It is not (and never
will be) deprecated, but it is more complicated to use than the :func:`.gddm`
function.

- :doc:`Quickstart guide for the object-oriented interface <../ooquickstart>`
- :doc:`Cookbook for the object-oriented interface <oocookbook>`
