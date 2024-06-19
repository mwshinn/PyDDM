Common models
=============

.. _weibull-bounds:

Weibull collapsing bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

A popular choice for collapsing bounds is the Weibull function.  You can
implement this in PyDDM with::

    import numpy as np
    m = pyddm.gddm(bound=lambda t,a,aprime,lam,k : a - (1 - np.exp(-(t/lam)**k)) * (a - aprime),
                   parameters={"a": (1,2), "aprime": (0,1), "lam": (0,2), "k": (0,5)})
    pyddm.plot.model_gui(m)

We can visualize the shape of the bounds using a special view function for the
model GUI::

    pyddm.plot.model_gui(m, plot=pyddm.plot.plot_bound_shape)
       
    
.. _rlddm:

Reinforcement learning DDM (RL-DDM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`Download rlddm.py (code for this example) <../downloads/rlddm.py>`

The RL-DDM allows reinforcement learning models to be combined with the DDM.
These models use update rules from traditional RL models, but use the DDM to
calculate the likelihood (instead of a softmax function).

The RL-DDM in PyDDM needs to use a custom loss function which implements the
reinforcement learning algorithm of choice.  In this example, we use
Rescorla-Wagner with the learning rate "alpha" parameter, but any model with any
number of parameters is possible.

We will simulate data from a two-armed bandit and then fit a model.  Each
participant is simulated for multiple sessions.  In addition to using PyDDM for
the fitting in the final model, we also use PyDDM for simulation in order to
determine response time.  The simulation with alpha=.1 and drift rate
(driftscale) = 1 can be written as:

.. literalinclude:: ../downloads/rlddm.py
   :language: python
   :start-after: # BEGIN SIMULATION_CODE
   :end-before: # END SIMULATION_CODE

And some example data::

    _     choice        RT  trial  session  reward
    0          0  1.154669      0        0       1
    1          0  1.078064      1        0       0
    ...      ...       ...    ...      ...     ...
    1995       0  0.949016    995        1       1
    1996       1  0.957841    996        1       1

As you can see, the PyDDM model used for the simulations is pretty simple.  The
real magic is in the loss function.  For fitting these data, we can use the
following loss function, which implements Rescorla-Wagner for a two-armed
bandit.  To implement a different model, this is what you need to edit.

.. literalinclude:: ../downloads/rlddm.py
   :language: python
   :start-after: # BEGIN LOSS
   :end-before: # END LOSS

While this loss function works, fitting is extremely slow.  This is because
PyDDM must simulate a DDM for every single different "deltaq" value (qright -
qleft).  This is because ΔQ is a floating point number, which is unlikely to be
the same for any two trials.  However, much of this precision may not be needed.
If we instead truncate ΔQ at two decimal digits, we can greatly reduce the
number of DDMs that must be simulated.  We can do this using the following loss
function:

.. literalinclude:: ../downloads/rlddm.py
   :language: python
   :start-after: # BEGIN FASTLOSS
   :end-before: # END FASTLOSS


Finally, build and fit the model:

.. literalinclude:: ../downloads/rlddm.py
   :language: python
   :start-after: # BEGIN MODEL
   :end-before: # END MODEL

.. note::
    The RL-DDM depends on "deltaq" as a condition, even though this does not
    (and cannot) exist in the data.  This is because we set it in the loss
    function as we iterate through the trials, running an RL model with the
    given parameters.  In practice, this means that you do not have to
    explicitly pass deltaq as a condition when fitting, because the loss
    function calls the model with the appropriate deltaq.  However, when
    simulating trials or viewing the model using the model GUI, you must provide
    a deltaq.

And after fitting, we can see the parameters are similar to what we simulated,
indicating our model fitting procedure recovers these parameters::

    >>> m.parameters()
    {'drift': {'driftscale': Fitted(0.9622368901610503, minval=0, maxval=8),
               'alpha': Fitted(0.12131174557396074, minval=0, maxval=1)},
     'noise': {'noise': 1},
     'bound': {'B': 1},
     'IC': {'x0': 0},
     'overlay': {'nondectime': 0, 'umixturecoef': 0.02}}


.. _addm:

Attention DDM (ADDM)
~~~~~~~~~~~~~~~~~~~~

The ADDM changes the drift rate based on the location of the subject's gaze.
Here, we will model the formulation given in `Krajbich et al
(2012) <https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2012.00193/full>`_,
where the participant must choose whether to purchase a given item, shown on the
screen, at a given price, also shown on the scren.  This assumes the subject is
always looking at either the item or the price.  The integration process
represents the value of the object, and crossing the upper boundary indicates a
decision to purchase.

We will assume our data are in a Pandas dataframe of the following form::

    _ choice   RT  value  price                     fixation
           1  0.8      3      4     (0, 1, 1, 0, 0, 1, 0, 0)
           1  0.9      2      1  (1, 0, 0, 0, 1, 1, 1, 1, 1)

Here, "value" is the value of the item for the participant, "price" is the price
displayed on the screen during this trial, "choice" is 1 if the participant made
the purchase or 0 if they did not, and RT is the response time in seconds.
"fixation" is a sequence of values of 1 (if the participant is looking at the
item) or 0 (if the participant is looking at the price) in 0.1 sec intervals
(given by the BINSIZE variable).

We set up the sample with::

    import pandas
    df = pandas.DataFrame([{"choice": 1, "RT": .8, "value": 3, "price": 4, "fixation": (0,1,1,0,0,1,0,0)},
                           {"choice": 1, "RT": .9, "value": 2, "price": 1, "fixation": (1,0,0,0,1,1,1,1,1)}])
    samp = pyddm.Sample.from_pandas_dataframe(df, choice_column_name="choice", rt_column_name="RT", choice_names=("Purchase", "Don't purchase"))
    BINSIZE = .1

The model is then::

    def drift_func(value, price, fixation, d, theta, t):
        current_fixation = fixation[min(int(t//BINSIZE), len(fixation)-1)]
        if current_fixation == 1:
            return d * (value - price * theta)
        else:
            return -d * (price - value * theta)
    m = pyddm.gddm(drift=drift_func, noise=.1, bound="b", nondecision="tnd", mixture_coef=.02,
                   parameters={"d": (0, 2), "theta": (0, 1), "b": (.5, 2), "tnd": (0, .5)},
                   conditions=["value", "price", "fixation"])
    pyddm.plot.model_gui(m, sample=samp)

Note that there will be many conditions shown in the left hand side of the model
GUI if used on data with more trials.


.. _multisensory-drift:

Multi-sensory integration
~~~~~~~~~~~~~~~~~~~~~~~~~

Drift rate can be determined by multiple signals.  For instance, if we have
vestibular and tactile evidence which add linearly, the model would be::

    m = pyddm.gddm(drift=lambda tact,tact_scale,vest,vest_scale : tact*tact_scale + vest*vest_scale,
                   parameters={"tact_scale": (0, 3), "vest_scale": (0, 3)},
                   conditions=["vest", "tact"])
    pyddm.plot.model_gui(m, conditions={"vest": [-1, 0, 1], "tact": [-1, -.5, 0, .5, 1]})


.. _full-ddm:

DDM with parameter variability ("full DDM")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A version of the DDM known as the "full DDM" utilizes a variable starting
position with uniform starting position variability, uniform variability in
non-decision time, and Gaussian-distributed variability in drift rate.  This
model is attractive because it was the first model to explain differences
between the shape of correct and error RT distributions.

The first three of these four extensions to the DDM are simple to implement in
PyDDM, and can be accomplished using the following model::

    import scipy.stats
    m = pyddm.gddm(drift="drift",
                   noise=0.1,
                   bound="bound",
                   starting_position=lambda x0,x0_width,x : scipy.stats.uniform(x0-x0_width/2, x0_width).pdf(x),
                   nondecision=lambda T,tnd,tnd_width : scipy.stats.uniform(tnd-tnd_width/2, tnd_width).pdf(T),
                   parameters={"drift": (-2, 2),
                               "bound": (.3, 3),
                               "x0": (-.7, .7),
                               "x0_width": (.01, .29),
                               "tnd": (.1, .5),
                               "tnd_width": (.01, .1)},
                   T_dur=5.0)

However, drift rate variability is difficult in PyDDM and is really not
recommended.  If you absolutely must, you can use the following, but make sure
you know what you are doing and how this will impact your model.  (E.g., you
will get incorrect likelihood outputs.)

.. warning::

   Seriously, don't use drift rate variability unless you absolutely have to.
   It is unsupported and improved support will never be added to PyDDM.  If
   drift rate variability is an essential and integral part of your research,
   PyDDM is not a good choice.

Unlike variability in starting position or non-decision time, the distribution
of drift rates must be discretized, and each must be run separately.  Here, we
demonstrate how to do this for a uniform distribution.  To do this for a normal
distribution, as in the full DDM, you need to add weights according to a
Gaussian distribution PDF.

In order to run such a model, first you must prepare your Sample by running it
through the following function.  This makes one duplicate of each of your data
points according to each discretization point.  As a result, note that that
likelihood, BIC, and other summary statistics about the data or fit quality may
be inaccurate.

.. literalinclude:: ../downloads/cookbook.py
   :language: python
   :start-after: # Start prepare_sample_for_variable_drift
   :end-before: # End prepare_sample_for_variable_drift

Then, once you have prepared your sample, you can use the following model::

    import scipy.stats
    def drift_uniform_distribution(drift, width, driftnum):
        stepsize = width/(RESOLUTION-1)
        mindrift = drift - width/2
        return mindrift + stepsize*driftnum
    m = pyddm.gddm(drift=drift_uniform_distribution,
                   noise=0.1,
                   bound="bound",
                   starting_position=lambda x0,x0_width,x : scipy.stats.uniform(x0-x0_width/2, x0_width).pdf(x),
                   nondecision=lambda T,tnd,tnd_width : scipy.stats.uniform(tnd-tnd_width/2, tnd_width),
                   parameters={"drift": (-2, 2),
                               "bound": (.3, 3),
                               "x0": (-.7, .7),
                               "x0_width": (.01, .59),
                               "tnd": (.1, .5),
                               "tnd_width": (.01, .2)})
