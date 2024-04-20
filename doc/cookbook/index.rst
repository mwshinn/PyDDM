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
- :ref:`Weibull collapsing bounds <weibull-bounds>`
- :ref:`DDM with parameter variability ("full DDM") <full-ddm>`

Model components
~~~~~~~~~~~~~~~~

- :ref:`Biased drift rate <biased-drift>`
- :ref:`Changing drift rate <changing-drift>`
- :ref:`Biased starting position <biased-starting-position>`
- :ref:`Starting point variability <starting-point-variability>`
- :ref:`Non-decision time variability <non-decision-variability>`
- :ref:`Multi-sensory integration <multisensory-drift>`
- :ref:`Non-linear coherence dependent drift rate <nonlinear-drift>`
- :ref:`Urgency signal as a multiplicative gain <urgency-gain>`
- :ref:`Unique moment-to-moment drift rate on each trial (e.g., matching EEG) <drift-moment-to-moment>`

How-to
~~~~~~

- `Change the objective function used for fitting <loss>`_
- :ref:`Change the fitting algorithm <howto-fit-custom-algorithm>`
- :ref:`Parallelization <howto-parallel>`
- :ref:`Visualize the evolution of the PDF over time <howto-evolution>`
- :ref:`Stimulus coding vs accuracy coding <howto-stimulus-coding>`

Object-oriented interface
~~~~~~~~~~~~~~~~~~~~~~~~~

The object-oriented interface is no longer recommended.  It is not (and never
will be) deprecated, but it is more complicated to use than auto_model.

- :doc:`Quickstart guide for the object-oriented interface <../ooquickstart>`
- :doc:`Cookbook for the object-oriented interface <oocookbook>`
