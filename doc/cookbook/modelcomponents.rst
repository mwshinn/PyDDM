Model components
================

.. _changing-drift:

Changing signal strength
~~~~~~~~~~~~~~~~~~~~~~~~

Suppose the strength of the signal starts out at zero and then switches to some
fixed value at t=500 ms, and each trial in the Sample has a "coherence"
condition, specifying the signal strength.  Then we can write::

    m = pyddm.gddm(drift=lambda t,coherence,drift_multiplier : 0 if t<.5 else coherence*drift_multiplier,
                   parameters={"drift_multiplier": (0, 4)},
                   conditions=["coherence"])
    pyddm.plot.model_gui(m, conditions={"coherence": [0, .25, .5]})

Alternatively, suppose the stimulus strength switches once after t1 seconds and
again t2 seconds later.  Now, we need conditions in the Sample to specify t1,
t2, coh1, coh2, and coh3.  Since this is a longer function, for convenience, we
will write it as a separate function instead of as a lambda function.  We can
write::

    def coherence_changes(t, coh1, coh2, coh3, drift_multiplier, t1, t2):
        if t<t1:
            return coh1*drift_multiplier
        elif t<t1+t2:
            return coh2*drift_multiplier
        else:
            return coh3*drift_multiplier
    m = pyddm.gddm(drift=coherence_changes,
                   parameters={"drift_multiplier": (-3,3)},
                   conditions=["coh1", "coh2", "coh3", "t1", "t2"])
    pyddm.plot.model_gui(m, conditions={"coh1": [0, .5, 1], "coh2": [0, .5, 1], "coh3": [0, .5, 1], "t1": [.2, .4, .6], "t2": [.2, .4, .6]})

Note that for each of these, to fit the model, conditions would be specified in
the Sample containing the data.  For the purpose of plotting with the model gui,
we specify a few predetermined values.

.. _urgency-gain:

Urgency signal through a multiplicative gain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's suppose we have some urgency function which controls the gain and
increases over time.  For this demonstration, we will use an exponential
function.  We want this to scale both the drift rate and the noise.  In this
example, the drift rate is scaled by signal strength.  We can do that with::

    def urgency(gain, lmbda, t):
        return gain*np.exp(lmbda*t)
    m = pyddm.gddm(drift=lambda t,g,l,signal_strength,drift_scale: urgency(g, l, t)*signal_strength*drift_scale,
                   noise=lambda g,l,t : urgency(g, l, t),
                   parameters={"l": (.1, 10), "g": (.1, 1), "drift_scale": (.01, 5)},
                   conditions=["signal_strength"])
    pyddm.plot.model_gui(m, conditions={"signal_strength": [0, .4, .8]})

Of course, the urgency signal can be as simple as a linear function, and can
depend on any parameters you like.

.. _nonlinear-drift:

Non-linear coherence-dependent drift rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The relationship between drift rate and coherence (signal strength) can be any
arbitrarily complicated non-linear relationship.  For instance, to fit a
quadratic function::

    m = pyddm.gddm(drift=lambda coh,b1,b2 : coh*b1 + coh**2*b2,
                   parameters={"b1": (-3, 3), "b2": (-3, 3)},
                   conditions=["coh"])
    pyddm.plot.model_gui(m, conditions={"coh": [0, .4, .8]})

.. _attractors:

Attractor states
~~~~~~~~~~~~~~~~

PyDDM can simulate attractor state dynamics.  This is accomplished using the "x"
argument to drift, as with leaky or unstable integration.  For instance, to put
a stable fixed point at each bound and an unstable fixed point at the origin::

    m = pyddm.gddm(drift=lambda x,driftscale,b : -driftscale*(x-b)*(x+b)*x,
                   bound="b",
                   parameters={"driftscale": (-5, 5),  "b": (.5, 3)})
    pyddm.plot.model_gui(m, sample=samp)

In this case, when the parameter "dirftscale" goes negative, the fixed points at
the bounds will change from stable to unstable, and vice versa for the fixed
point at the origin.

.. _drift-moment-to-moment:

Unique moment-to-moment drift rate on each trial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In many situations we may wish to model drift rate as a vector which varies
across time, and is unique for each trial.  For example, we may hypothesize that
drift rate is modulated by EEG activity, pupil diameter, skin conductance, etc.
We can do this by setting a condition to be a tuple, and then constructing an
appropriate drift function.

In the following example, we will adjust the gain of evidence accumulation by
moment-to-moment estimates of arousal levels, using the "arousal" variable,
discretized to 200 ms.  Since the participant will have responded at different
times, we only have a limited number of measurements of arousal levels, and this
may be a different number of measurements for each trial.

Suppose we construct a sample with three trials::
    
    df = pandas.DataFrame({"choice_correct": [1, 0, 1], "arousal": [(.5, .7, .8, .9), (.2, .3, .4, .1, .1), (1.1, 1.3)], "RT": [.65, .94, .30]})
    samp = pyddm.Sample.from_pandas_dataframe(df, rt_column_name="RT", choice_column_name="choice_correct")

Which gives::

    _  choice_correct                    arousal    RT
    0               1       (0.5, 0.7, 0.8, 0.9)  0.65
    1               0  (0.2, 0.3, 0.4, 0.1, 0.1)  0.94
    2               1                 (1.1, 1.3)  0.30

We can use this in a PyDDM model with the following::

    def find_drift(t, arousal_vec):
        t_bin = int(t // .2) # 200 ms binning
        t_bin = min(len(arousal_vec)-1, t_bin) # Never go past the final arousal bin
        return arousal_vec[t_bin]
    m = pyddm.gddm(drift=lambda t,arousal_scale,arousal : find_drift(t, arousal)*arousal_scale,
                   parameters={"arousal_scale": (0, 2)},
                   conditions=["arousal"])
    pyddm.plot.model_gui(m, samp, data_dt=.2)

(Here, data_dt indicates the bin size for the data.  Since we only have three
data points, we want the bins to be large so we can see the model distribution
better.)


.. _biased-drift:

Biased drift rate
~~~~~~~~~~~~~~~~~

The drift rate or starting point may be biased towards one option, such as the
option with a higher prior probability or with a higher reward upon correct
choice.  Modelling this depends on how we define our boundaries.

In stimulus coding, if the upper boundary is the choice with the bias (e.g., the
option with the higher reward), then the bias is always in the same direction.
The signal strength may then be positive or negative, and it may change trial to
trial (e.g., motion coherence).  Thus, a model could be::

    m = pyddm.gddm(drift=lambda coh,driftmultiplier,bias : coh*driftmultiplier + bias,
                   parameters={"driftmultiplier": (0, 2), "bias": (0, 2)},
                   conditions=["coh"],
                   choice_names=("High reward probability", "Low reward probability"))
    pyddm.plot.model_gui(m, conditions={"coh": [-2, -1, 0, 1, 2]})
                        

In accuracy coding, if the upper boundary is the correct choice and the lower
boundary is the incorrect choice, we need to flip the bias depending on whether
the biased choice is correct or incorrect.  However, in this case, the motion
coherence should always be positive.  Thus, a model could be::

    m = pyddm.gddm(drift=lambda coh,driftmultiplier,bias,biascorrect : coh*driftmultiplier + bias*(1 if biascorrect else -1),
                   parameters={"driftmultiplier": (0, 2), "bias": (0, 2)},
                   conditions=["coh", "biascorrect"],
                   choice_names=("Correct", "Error"))
    pyddm.plot.model_gui(m, conditions={"coh": [0, 1, 2], "biascorrect": [0, 1]})


.. _biased-starting-position:

Biased starting position
~~~~~~~~~~~~~~~~~~~~~~~~

Just as in the case of the :ref:`biased drift rate <biased-drift>`, we need to
implement this slightly differently for stimulus vs accuracy coding.

For stimulus coding, where the bias is towards one of the two stimuli, it is easy::

    m = pyddm.gddm(starting_position="bias",
                   parameters={"bias": (-1, 1)},
                   choice_names=("High reward probability", "Low reward probability"))
    pyddm.plot.model_gui(m)

For accuracy coding, we need to switch the direction of the bias based on
whether the choice with the bias was correct on the given trial::

    m = pyddm.gddm(starting_position=lambda bias,biascorrect: bias if biascorrect else -bias,
                   parameters={"bias": (-1, 1)},
                   conditions=["biascorrect"],
                   choice_names=("Correct", "Error"))
    pyddm.plot.model_gui(m, conditions={"biascorrect": [0, 1]})

.. _starting-point-variability:

Starting point variability
~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of starting point coming from a fixed point, it can also come from a
distribution.  To do this, the "starting_point" in :func:`.gddm` can accept the
variable "x", which is the domain of the distribution, i.e., all of the
potential starting points over which the distribution is defined.  Then it must
output a vector of the same length as x describing the probability density at
each point.  (Since the starting point is not fixed in general, and may even
vary trial to trial, the length of x may be different on each trial.)  If the
output does not sum to 1, it will be renormalised.

Or the normal distribution::

    import scipy.stats
    import numpy as np
    m = pyddm.gddm(starting_position=lambda mu,sigma,x: scipy.stats.norm(mu,sigma).pdf(x),
                   parameters={"mu": (-.5, .5), "sigma": (.01, .3)})
    pyddm.plot.model_gui(m)

Or the uniform distribution::

    import numpy as np
    m = pyddm.gddm(starting_position=lambda x: np.ones(len(x))/len(x))
    pyddm.plot.model_gui(m)

Or the beta distribution::

    import scipy.stats
    import numpy as np
    m = pyddm.gddm(starting_position=lambda a,b,x: scipy.stats.beta(a,b).pdf((x-np.min(x))/(np.max(x)-np.min(x))),
                   parameters={"a": (.001, 10), "b": (.001, 10)})
    pyddm.plot.model_gui(m)

Note that the parameters of these distributions also offer a way to implement a
:ref:`starting point bias <biased-starting-position>`.


.. _non-decision-variability:

Non-decision time variability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to being a fixed value, the non-decision time can be a distribution.
To do this, your non-decision time function should accept the parameter T (as
the capital letter), which is the vector of possible non-decision times.  The
function should then return a vector of the same length of T, containing the
density at each point.  Note that T consists of positive *and negative* numbers,
as it may be useful to have negative non-decision times for some experiments.
If you don't want negative non-decision times, make sure the density is zero for
T<0.

For instance, the following is a non-decision time which follows a normal
distribution::

    import scipy.stats
    m = pyddm.gddm(noise=2,
                   nondecision=lambda T,mu,sigma : scipy.stats.norm(mu, sigma).pdf(T),
                   parameters={"sigma": (.01, .1), "mu": (0, 1)})
    pyddm.plot.model_gui(m)

For instance, the following is a non-decision time which follows a gamma
distribution::

    import scipy.stats
    m = pyddm.gddm(noise=2,
                   nondecision=lambda T,min_t,shape,scale : scipy.stats.gamma(shape, min_t, scale).pdf(T),
                   parameters={"shape": (1, 5), "scale": (.01, 1), "min_t": (0, 1)})
    pyddm.plot.model_gui(m)
