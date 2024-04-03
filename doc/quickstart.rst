Quick Start Guide
=================

Hello, world!
-------------

To get started, let's simulate a basic model and plot it.  For simplicity, we
will use all of the model defaults: a drift rate of 0, a noise standard
deviation of 1, a bound of 1, a starting point in the middle of the bounds, a
non-decision time of 0, no mixture model.  (More on each of these later.)

.. literalinclude:: downloads/helloworld.py
   :language: python

.. image:: images/helloworld.png

Congratulations!  You've successfully simulated your first model!
Let's dig in a bit so that we can define more useful models.

:download:`Download this full example <downloads/helloworld.py>`

.. note::

   There are two ways to build models with PyDDM: the classic object-oriented
   interface, and "auto_model".  Here, we will discuss auto_model: it supports
   almost all of the features of the classic interface, but is much easier to
   use.  We will discuss the differences in detail later in this tutorial.


For the rest of this tutorial, we will slowly build a generalized drift
diffusion model (GDDM), validate the model, and then fit the model to an open
dataset.  As we do, we will illustrate the main concepts of PyDDM to so you can
start building your own models.

A model with fixed parameters
-----------------------------

First, let's see how we can modify the pieces of the model given above.  For
consistency, we will refer to these (drift, noise, bound, starting position,
non-decision time, and mixture coefficient) as Components. The auto_model
function allows us to pass values for different parameters.  The most important
parameters are:

- **drift**: the drift rate, i.e., the amount of evidence that would be
  accumulated in 1 sec if there was no noise.
- **noise**: the noise level, i.e., the standard deviation of noise.  By
  tradition, this is frequently set to 0.1 or 1.0.  If not specified, it
  defaults to 1.0.
- **bound**: the height of the boundary on each side.  (So, the total separation
  of the boundaries will be twice this value.)
- **starting_position**: the initial position of the diffusion process.  A value
  of 1 indicates a starting position on the upper boundary, -1 on the bottom
  boundary, and 0 in between.  (This differs from some other packages, which
  define 0 as the lower boundary and 1 as the upper boundary.)
- **nondecision**: the non-decision time, in units of seconds.  Both positive
  and negative values are valid.
- **mixture_coef**: By default, PyDDM returns an RT distribution which is a
  mixture model of the DDM RT distribution and the uniform distribution.  This
  parameter defines the ratio of uniform distribution to DDM RT distribution.
  Set to 0 to disable the mixture model.  By default, this is 0.02.  (Mixture
  models assist with model fitting when using maximum likelihood, which we will
  discuss later in this tutorial.)

So, for a model with a drift rate of 0.5, a noise level of 1.0, a bound of 0.6,
a starting position of 0.3 (slightly biased towards the upper bound), and a
non-decision time of 0.2 sec, the model would be:

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start ModelDef
   :end-before: # End ModelDef

We can show information about our model using the show() function:

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start ShowModel
   :end-before: # End ShowModel

This shows::

    Model information:
    Choices: 'correct' (upper boundary), 'error' (lower boundary)
    Drift component DriftConstant:
        constant
        Fixed parameters:
        - drift: 0.500000
    Noise component NoiseConstant:
        constant
        Fixed parameters:
        - noise: 1.000000
    Bound component BoundConstant:
        constant
        Fixed parameters:
        - B: 0.600000
    IC component ICPointRatio:
        An arbitrary starting point expressed as a proportion of the distance between the bounds.
        Fixed parameters:
        - x0: 0.300000
    Overlay component OverlayChain:
        Overlay component OverlayNonDecision:
            Add a non-decision by shifting the histogram
            Fixed parameters:
            - nondectime: 0.200000
        Overlay component OverlayUniformMixture:
            Uniform distribution mixture model
            Fixed parameters:
            - umixturecoef: 0.020000

Once we have a model, we can simulate or "solve" the model.  Solution objects
represent the probability distribution functions over time for choices
associated with upper and lower bound crossings.  We saw one of these, "s", in
the first example.  They can also be used to generate artificial data for
parameter recovery simulations.  We will discuss this shortly.

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start ModelSolve
   :end-before: # End ModelSolve

Fitting model parameters
------------------------

We can also define a model with free parameters - these parameters can be fit to
data, or tweaked using the PyDDM GUI.  Instead of passing numbers to these
arguments, we can instead pass the names of parameters to fit.  Then, at the
end, we use the "parameters" argument to define the parameters.  This argument
should be a dictionary, where the key of the dictionary is the name of the
parameter, and the value is a tuple containing the minimum and maximum value of
the parameter.  For example, to fit the drift rate, the noise level, and the
non-decision time, but keep the bound fixed at 1.0, we can write:

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start ModelFittableDef
   :end-before: # End ModelFittableDef

.. note::

  In the examples here, the ranges of the parameters have been chosen
  arbitrarily.  You may have to adjust the ranges to something that works for
  your model and data.  Usually it is better to have a range that is too large
  than too small, but an extremely large range will slow down fitting.  Once you
  perform the fitting, make sure to check your fitted parameters are not close
  to the minimum or maximum values you set.  If they are, you may need a bigger
  range.

Before fitting the model, we can visualize this model using the model GUI to
make sure it behaves in the way we expect.  This is usually a good sanity check
when constructing a new model, as well as a useful way to get an intuition of
your model's behavior.

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start ModelGui
   :end-before: # End ModelGui

.. note::
  If the model GUI does not display the controls and the simulated RT
  distributions, see :ref:`help_model_gui`.

To demonstrate how to fit this model to data, let's simulate data from the
fixed-parameter model we made above.  By doing so, we will perform a *parameter
recovery* experiment.  Since the models are the same except for the free
parameters, we can see if fitting the free parameters to data from the fixed
parameters *recovers* the original fixed parameters.  If not, there may be
something wrong with our model!

Remember the "sol" solution object we created?  We will use this to simulate
data.  Since this contains the "solved" probability distributions from the
model, we can sample from this distribution to create simulated data.  Simulated
data and experimental data are both captured by a "Sample" object.  Later in
this tutorial, we will also see how to fit free parameetrs to experimental data
rather than simulated data.

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start Resample
   :end-before: # End Resample

Now we can fit the model to the simulated data.  By default, PyDDM uses
differential evolution to perform the fit: it is a slower algorithm than
gradient-based methods, but is much more likely to find the best fitting
parameters in complex models.  We use BIC as a loss function.

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start Fit
   :end-before: # End Fit

The result of model.show() is::

    Model information:
    Choices: 'correct' (upper boundary), 'error' (lower boundary)
    Drift component DriftConstant:
        constant
        Fitted parameters:
        - drift: 0.533938
    Noise component NoiseConstant:
        constant
        Fixed parameters:
        - noise: 1.000000
    Bound component BoundConstant:
        constant
        Fitted parameters:
        - B: 0.601329
    IC component ICPointRatio:
        An arbitrary starting point expressed as a proportion of the distance between the bounds.
        Fitted parameters:
        - x0: 0.279759
    Overlay component OverlayChain:
        Overlay component OverlayNonDecision:
            Add a non-decision by shifting the histogram
            Fixed parameters:
            - nondectime: 0.200000
        Overlay component OverlayUniformMixture:
            Uniform distribution mixture model
            Fixed parameters:
            - umixturecoef: 0.020000
    Fit information:
        Loss function: BIC
        Loss function value: 5911.712503639885
        Fitting method: differential_evolution
        Solver: auto
        Other properties:
            - nparams: 3
            - samplesize: 10000
            - mess: ''

As we can see, the parameters for drift, noise, and non-decision time are close
to the values from the model that generated the data.

.. note::

  If the model recovery fails, i.e., if the fit parameters are different than
  the ones you used, there are a few things to check before concluding that
  there as an issue with your model:

  - Ensure the simulation is long enough to capture the mass of the
    distribution.  You can do this by increasing T_dur.
  - Ensure that the simulation accuracy is suffirient for your model.  In
    practice, this means checking that dx and dt are small enough.
  - Ensure you are not fitting multiple redundant parameters.  For example, in
    the classic DDM, noise and boundary are redundant, so at least one of these
    parameters must be fixed.

  You will learn more about these in the upcomming section :ref:`other_params`.

If we want to use these programmatically, we can use the
:meth:`~.Model.parameters` function, like:

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start Parameters
   :end-before: # End Parameters

This will return a collection of :class:`~.model.Fitted` objects.  These objects
can be used anywhere in Python as if they are normal numbers/floats.  (Actually,
:class:`~.model.Fitted` is a subclass of "float"!)

We can also examine different properties of the fitting process in the
:class:`~.fitresult.FitResult` object.  For instance, to get the value of the
loss function, we can do:

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start Lossval
   :end-before: # End Lossval

We can also draw a plot visualizing the fit.  We can use one of PyDDM's
convenience methods, :func:`~.plot.plot_fit_diagnostics`.  We have to import
pyddm.plot separately.

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start Plot
   :end-before: # End Plot

.. image:: images/simple-fit.png

Using the :class:`.Solution` object ``sol`` we have access to a number of other
useful functions.  For instance, we can find the probability of a response using
:meth:`~.Solution.prob`, such as ``sol.prob("correct")`` for the probability of
a correct response, or the entire histogram of responses using
:meth:`~.Solution.pdf`, such as ``sol.pdf("error")`` for the distribution of
errors.

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start Probs
   :end-before: # End Probs

See :class:`the Solution object documentation <.Solution>` for more
such functions.

We could also named the upper and lower boundary as something else (e.g. "left"
and "right" response), sometimes called "stimulus coding".  To do this, we need
to pass the "choice_names" parameter to the Sample and the Model object.  See
:ref:`the section on stimulus coding <howto-stimulus-coding>`

Using functions as parameters
-----------------------------

In the last section, we saw how components in PyDDM can be specified by a
parameter.  We did this by assigning the name of the parameter (as a string) to
the component.   Specifically, we wrote

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start ModelFittableDef
   :end-before: # End ModelFittableDef

More generally, **components can be defined by functions**.  To do this, the
name of the parameter should be an argument of the function.  Then, the function
can be used interchangably anywhere you would pass the name of a parameter.  So,
our definition is equivalent to the following definition.

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start ModelFittableAltDef
   :end-before: # End ModelFittableAltDef

Note that "lambda" is a Python feature which allows us to define short
"anonymous" functions.  The important thing here is that the names of the
argument is the same as the name of the parameters.  We can define functions in
several different ways in Python, and any of these ways will work as long as the
argument is the name of the parameter.  So, for example, the following is also
equivalent:

.. literalinclude:: downloads/simple.py
   :language: python
   :start-after: # Start ModelFittableAlt2Def
   :end-before: # End ModelFittableAlt2Def


Functions can be useful because they allow us to specify more complex
relationships among parameters.  While here we have used very simple functions,
any valid Python function can be used.  Next, we will see additional ways that
functions can be used to specify simple or complex models.


Specifying trial properties through Conditions
----------------------------------------------

In most experiments, trials are not identical.  Different trials may have
different experimental conditions.  For instance, some trials in a random dot
motion task may have stronger motion coherence than others, providing different
amounts of evidence that participants can use to make a decision.  Or, as
another example, trials may come from different blocks, where each block has
different stimulus properties.  In PyDDM, we refer to these differences in
trials as "conditions".

**Conditions and parameters are the two primary ways we define model components
in PyDDM.** We previously saw how we can define components using any Python
function which depends on parameters.  To do this, we pass the parameters as
arguments to the function. In a similar way, conditions can also be passed to
these functions.  So, through this, components can also depend on conditions.
If we do this, we need to list all of the conditions we used at the end of the
model in the argument "conditions".

For example, suppose we have a model where the drift rate is linearly related to
the stimulus coherence, or the signal strength.  So, for some parameter
"drift_rate_scale" and stimulus coherence condition "coh", we have the following
model:

.. literalinclude:: downloads/gddms.py
   :language: python
   :start-after: # Start condition model
   :end-before: # End condition model

Now, the model can only be solved if we tell the model what values for the
conditions we want to use.  If solving the model using the solve() method, we
must specify to the solve() method what conditions we want to use.  For example::

    m.solve(conditions={"coh": .3})

If we fit the model using a Sample object, the Sample must have the given
conditions in it.  When we simulate data with the resample() function, we can
simulate from multiple conditions by solving the model for different sets of
conditions, calling resample() on each Solution object, and then combining them
by adding them together.  For example,

.. literalinclude:: downloads/gddms.py
   :language: python
   :start-after: # Start solve multiple conditions
   :end-before: # End solve multiple conditions

If we want to use the model GUI, we must tell it what conditions to use, by
passing in the "conditions" variable as a dictionary, where keys are the names
of the conditions which index lists of possible values for that condition.
Alternatively, we can pass a sample object to determine the appropriate
conditions automatically.  For example,

.. literalinclude:: downloads/gddms.py
   :language: python
   :start-after: # Start conditions model gui
   :end-before: # End conditions model gui

Lastly, when we fit the model, we must pass a sample object which has the
appropriate conditions.  If it does not, it will raise an error.


Generalized drift diffusion models
----------------------------------

In addition to depending on parameters and conditions, there are two more
variables which can be used in the models.  First, the drift, noise, and bounds
can all depend on time.  This is accomplished by adding "t" (for time) as an
argument, alongside the parameters and conditions.  "t" represents the time in
the simulation.  So, a model with a linearly-collapsing bound and a drift rate
which increases exponentially over time can be given by:

.. literalinclude:: downloads/gddms.py
   :language: python
   :start-after: # Start drift bounds gddm
   :end-before: # End drift bounds gddm

Second, the drift and the noise can depend on location along the decision
variable axis.  This allows implementing model features such as leaky
integration, unstable integration, or attractor states.  For instance, leaky
integration can be implemented with

.. literalinclude:: downloads/gddms.py
   :language: python
   :start-after: # Start leaky gddm
   :end-before: # End leaky gddm

Similar notation can be used for a dependence of noise on position.

.. note::

  Variability in parameters, namely distributions of starting position and
  non-decision time, is possible to implement in PyDDM, but cannot be
  implemented with auto_model.  Use the object-oriented interface for this.
  Variability in drift rate is possible but difficult and not recommended.  See
  the :doc:`ooquickstart` for more information.`

.. _other_params:

Additional properties of models
-------------------------------

There are also a few additional parameters which are useful in controlling model
output.  These cannot be functions, so they cannot depend on parameters,
conditions, time, or position.

- **Simulation duration**: The "T_dur" argument specifies the duration to
  simulate, in units of seconds.  By default, it is 2 seconds.
- **Meaning of the upper and lower bound**: The "choice_names" argument
  specifies the identity of the upper and lower boundaries, given as a tuple of
  two strings.  By default, this is ("correct", "error") for tasks with a ground
  truth, but can also be changed to any desired value.  For instance, ("left",
  "right"), ("high value", "low value"), or ("green", "blue").  This is often
  called "accuracy coding" and "stimulus coding".  See See :ref:`the section on
  stimulus coding <howto-stimulus-coding>` for more information.
- **Model name**: The name of the model is given by the "name" argument.  This
  can be helpful when displaying or saving model output, especially if you are
  fitting multiple models.
- **Numerical precision**: The "dt" argument is the timestep for solving the
  model.  By default, it is set to 0.005.  Likewise, the "dx" argument is the
  discritized accuracy of the integrator when solving the model.  If your model
  gives strange results, consider making dt and dx smaller.  As a rule of thumb
  for most models, dx and dt should have similar values for optimal performance.



.. _quickstart-roitman:

Loading data from a CSV file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we load data from the open dataset by Roitman and Shadlen
(2002).  This dataset can be `downloaded here
<https://shadlenlab.columbia.edu/resources/RoitmanDataCode.html>`_ and the
relevant data extracted :download:`with our script
<downloads/extract_roitman.py>`.  The processed CSV file can be
:download:`downloaded directly <downloads/roitman_rts.csv>`.

The CSV file generated from this looks like the following:

====== ===== ===== ======= =========
monkey rt    coh   correct trgchoice
====== ===== ===== ======= =========
1      0.355 0.512 1.0     2.0
1      0.359 0.256 1.0     1.0
1      0.525 0.128 1.0     1.0
====== ===== ===== ======= =========


We can load and process the CSV file in a similar way as the original paper

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :start-after: # Start Load
   :end-before: # End Load

Once we have the data, we must create a "Sample" object, telling PyDDM how to
access the data.  Any extra columns in the Pandas dataframe will be available as
conditions.

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :start-after: # Start Sample
   :end-before: # End Sample

This gives an output Sample object with the conditions "monkey", "coh", and
"trgchoice".

Note that this examples requires `pandas <https://pandas.pydata.org/>`_ to be
installed.

Loading data from a numpy array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data can also be loaded from a numpy array.  For example, let's load the above
data and perform the same operations directly in numpy:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :start-after: # Start Numpy Load
   :end-before: # End Numpy Load

We can confirm that these two methods of loading data produce the same results:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :start-after: # Start Test
   :end-before: # End Test

In practice, you only need to load data either using the numpy method or the
pandas method - this is just a demonstration of each method.

Fitting a model to data
~~~~~~~~~~~~~~~~~~~~~~~

Now that we have loaded these data, we can fit a model to them.  First
we will fit a DDM, and then we will fit a GDDM.

Since this task involves motion coherence, let's scale the drift rate by the
motion coherence.  We will fit the drift rate, bound, and non-decision time to
the data.

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :start-after: # Start Model
   :end-before: # End Model

.. note::

  If you see "Warning: renormalizing model solution from X to 1." for some X,
  this is okay as long as X is close (:math:`<10^{-5}` or so) to 1.0 or as long
  as this is seen early in the fitting procedure.  If it is larger or seen
  towards the end of the fitting procedure, consider using smaller dx or dt in
  the simulation.  This indicates numerical imprecision in the simulation.

Now we fit the model and plot the result:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :start-after: # Start display model
   :end-before: # End display model

This gives the following output (which may vary slightly, since the
fitting algorithm is stochastic)::

    Model information:
    Choices: 'correct' (upper boundary), 'error' (lower boundary)
    Drift component DriftEasy:
        easy_drift
        Fitted parameters:
        - driftcoh: 11.002888
    Noise component NoiseConstant:
        constant
        Fixed parameters:
        - noise: 1.000000
    Bound component BoundConstant:
        constant
        Fitted parameters:
        - B: 0.739045
    IC component ICPointRatio:
        An arbitrary starting point expressed as a proportion of the distance between the bounds.
        Fixed parameters:
        - x0: 0.000000
    Overlay component OverlayChain:
        Overlay component OverlayNonDecision:
            Add a non-decision by shifting the histogram
            Fitted parameters:
            - nondectime: 0.317882
        Overlay component OverlayUniformMixture:
            Uniform distribution mixture model
            Fixed parameters:
            - umixturecoef: 0.020000
    Fit information:
        Loss function: Negative log likelihood
        Loss function value: 158.04434128408155
        Fitting method: differential_evolution
        Solver: auto
        Other properties:
            - nparams: 3
            - samplesize: 2611
            - mess: ''

And that's it!  We fit our model!


Plotting the fit
~~~~~~~~~~~~~~~~

We can also graphically evaluate the quality of the fit.  We can plot
and save a graph:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :start-after: # Start Plot
   :end-before: # End Plot

.. image:: images/roitman-fit.png

This model does not seem to fit the data very well.

We can alternatively explore this with the PyDDM's model GUI:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :start-after: # Start Gui
   :end-before: # End Gui

.. image:: images/model-gui.png

See :doc:`modelgui` for more info.

:download:`Download this full example <downloads/roitman_shadlen.py>`


Improving the fit
-----------------

Let's see if we can improve the fit by including additional GDDM model
components.  We will include exponentially collapsing bounds and use a leaky or
unstable integrator instead of a perfect integrator.  We can implement these as
we did above.

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :start-after: # Start leak model
   :end-before: # End leak model

Before fitting this model, let's look at it in the model GUI::

  from pyddm.plot import model_gui
  model_gui(model_leak, sample=roitman_sample)

We can fit this and save it as an image using the following.  Note that this may
take a while (several minutes) due to the increased number of parameters and
because the earlier examples were able to use the analytical solver but the
present example must use backward Euler.  For all coherences, the fit is:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :start-after: # Start leak model show
   :end-before: # End leak model show

This gives the following model::

    Model information:
    Choices: 'correct' (upper boundary), 'error' (lower boundary)
    Drift component DriftEasy:
        easy_drift
        Fitted parameters:
        - driftcoh: 9.649262
        - leak: 0.253203
    Noise component NoiseConstant:
        constant
        Fixed parameters:
        - noise: 1.000000
    Bound component BoundEasy:
        easy_bound
        Fitted parameters:
        - bound_base: 2.270000
        - invtau: 2.124669
    IC component ICPointRatio:
        An arbitrary starting point expressed as a proportion of the distance between the bounds.
        Fixed parameters:
        - x0: 0.000000
    Overlay component OverlayChain:
        Overlay component OverlayNonDecision:
            Add a non-decision by shifting the histogram
            Fitted parameters:
            - nondectime: 0.170556
        Overlay component OverlayUniformMixture:
            Uniform distribution mixture model
            Fixed parameters:
            - umixturecoef: 0.020000
    Fit information:
        Loss function: Negative log likelihood
        Loss function value: -326.5744733973074
        Fitting method: differential_evolution
        Solver: auto
        Other properties:
            - nparams: 5
            - samplesize: 2611
            - mess: ''

.. image:: images/leak-collapse-fit.png

Going further
-------------

In this tutorial, we have only just scratched the surface of what is possible in
PyDDM.  See the :doc:`cookbook/index` for more examples.  Also see the
:doc:`ooquickstart` or :doc:`apidoc/index` to learn about the object oriented
API for constructing models in PyDDM.

Summary
-------

PyDDM can simulate models and generate artificial data, or it can fit
them to data.  Below are high-level overviews for how to accomplish
each.

To simulate models and generate artificial data:

1. Define a model using "auto_model" or the object-oriented API.  Here, we
   focused on "auto_model".  Models may depend on parameters, conditions, time,
   and space.
2. Simulate the model using the :meth:`.Model.solve()` method to generate a
   :class:`.Solution` object.  If you have multiple conditions, you must run
   :meth:`.Model.solve()` separately for each set of conditions and generate
   separate :class:`.Solution` objects.
3. Run the :meth:`.Solution.resample()` method of the :class:`.Solution` object
   to generate a :class:`.Sample`.  If you have multiple :class:`.Solution`
   objects (for multiple task conditions), you will need to generate multiple
   :class:`.Sample` objects as well.  These can be added together with the "+"
   operator to form one single :class:`.Sample` object.


To fit a model to data:

1. Define a model with at least one free parameter, using "auto_model" or the
   object-oriented API.  Here, we focused on "auto_model".  Models may depend on
   parameters, conditions, time, and space.
2. Create a :class:`.Sample` object, either using "from_pandas_dataframe" or
   "from_numpy_array".  Ensure that the conditions used in the model are present
   in the data.
3. Visualize your model using the model GUI to ensure it has the behavior you
   expect.  You may need to pass the sample you plan to fit, or else the
   "conditions" variable, to the model GUI function.
4. Run :meth:`.Model.fit` on the model and the sample.  Optionally
   specify a :class:`loss function <.LossFunction>` other than the default
   (which uses BIC).  After fitting, the the model will contain :class:`.Fitted`
   objects, that contain the fitted values.
5. View the output by calling :meth:`.Model.show` and the model GUI on the
   model.  The value of the loss function is accessible via
   :meth:`.Model.get_fit_result` and the parameters via
   :meth:`.Model.parameters`.
