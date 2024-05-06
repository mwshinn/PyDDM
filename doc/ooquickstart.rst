Object-oriented interface tutorial
==================================

There are two ways of interacting with PyDDM: the :func:`.gddm` function, and
the Object Oriented API.  :func:`.gddm` is much simpler, but is not compatible
with some specialized models.  The object oriented API allows you to access the
full power of PyDDM in these edge cases.

Please read the :doc:`quickstart`, for building models with :func:`.gddm`,
before this tutorial.


Hello, world!
-------------

The object-oriented "Hello World" is almost the same as that for :func:`.gddm`.
The only difference is the use of :class:`.Model` instead of :func:`.gddm`.
:class:`.Model` will be used extensively in the rest of this tutorial.

.. literalinclude:: downloads/oohelloworld.py
   :language: python

.. image:: images/helloworld.png


:download:`Download this full example <downloads/oohelloworld.py>`



Simple example
--------------

Let's use the object-oriented API to simulate a simple DDM with constant drift.
First, it shows how to build a model and then uses it to generate artificial
data.  After the artificial data has been generated, it fits a new model to
these data and shows that the parameters are similar.

:class:`.Model` is the object which represents a DDM.  Its default
behavior can be changed through individual model components which can be
mixed and matched.

- :class:`.Drift` specifies the drift rate.  In GDDMs, this may change over time
  or across spatial positions.  It defaults to a constant drift of zero.
- :class:`.Noise` specifies the standard deviation of the noise.  Most DDMs fix
  this to a constant, such as 0.1 or 1.  However, in GDDMs, it may also change over time
  or across spatial positions.  In PyDDM, it defaults to a constant of 1.0.
- :class:`.Bound` specifies the shape of the boundaries which terminate the
  decision process.  In traditional DDMs, this will be a constant, often fit to
  the data.  However, in GDDMs, it may also increase or decrease in time.  It
  defaults to a constant value of 1.0.
- :class:`.InitialCondition`, also known as starting point, may be a single
  point or a distribution of starting points.  In PyDDM, it defaults to a single
  point centered between the bounds.
- :class:`.Overlay` allows specifying many things, such as the non-decision
  time.  It also allows specifying mixture models when performing likelihood
  fitting.  In general, it allows the distribution of RTs to be modified after
  the end of the simulation.  It defaults to no no overlay, i.e., no
  non-decision time and no mixture model.  This encapsulates both "nondecision"
  and "mixture_coef" from :func:`.gddm`.

Each of these model components may take parameters which are (usually) unique to
that specific model component.  These are specified by the model component.  For
instance, the :class:`.DriftConstant` class takes the "drift" parameter, which
should be passed to the constructor.  Similarly, :class:`.NoiseConstant` takes
the parameter "noise" to determine the standard deviation of the drift process,
and :class:`.OverlayNonDecision` takes "nondectime", the non-decision time
(efferent/afferent/apparatus delay) in seconds.  Some model components, such as
:class:`.ICPointSourceCenter` which represents a starting point initial
condition directly in between the bounds, does not take any parameters.

For example, the following is a DDM with drift 2.2, noise 1.5, bound
1.1, and a 100ms non-decision time.  It is simulated for 2 seconds
(``T_dur``) with reasonable timestep and grid size (``dt`` and
``dx``).  Once we define the model, the :meth:`~.Model.solve` function
runs the simulation.  This model can be described as shown below:

.. literalinclude:: downloads/oosimple.py
   :language: python
   :lines: 4-13, 17-18

Solution objects represent the probability distribution functions over time for
choices associated with upper and lower bound crossings.  By default, this is
"correct" and "error" responses, respectively. As before, we can generate
psuedo-data from this solved model with the :meth:`~.Solution.resample`
function:

.. literalinclude:: downloads/oosimple.py
   :language: python
   :lines: 22
  
To fit the outputs, we first create a model with special :class:`.Fittable`
objects in all the parameters we would like to fit.  We specify the range of
each of these objects as a hint to the optimizer; this is mandatory for some but
not all optimization methods.  Then, we run the :meth:`Model.fit` function,
which will convert the :class:`.Fittable` objects to :class:`.Fitted` objects
and find a value for each which collectively minimizes the objective function.

Here, we use the same model as above, since we know the form the model
is supposed to have.  We fit the model to the generated data using BIC
as a loss function and differential evolution to optimize the
parameters:

.. literalinclude:: downloads/oosimple.py
   :language: python
   :lines: 26-38

We can display the newly-fit parameters with the
:func:`.Model.show` function:

.. literalinclude:: downloads/oosimple.py
   :language: python
   :lines: 40

This shows that the fitted value of drift is 2.2096, which is close to
the value of 2.2 we used to simulate it.  Similarly, noise fits to
1.539 (compared to 1.5) and nondectime (non-decision time) to 0.1193
(compared to 0.1).  The fitting algorithm is stochastic, so exact
values may vary slightly::

  Model Simple model (fitted) information:
  Drift component DriftConstant:
      constant
      Fitted parameters:
      - drift: 2.209644
  Noise component NoiseConstant:
      constant
      Fitted parameters:
      - noise: 1.538976
  Bound component BoundConstant:
      constant
      Fixed parameters:
      - B: 1.100000
  IC component ICPointSourceCenter:
      point_source_center
      (No parameters)
  Overlay component OverlayNonDecision:
      Add a non-decision by shifting the histogram
      Fitted parameters:
      - nondectime: 0.119300
  Fit information:
      Loss function: BIC
      Loss function value: 562.1805934500456
      Fitting method: differential_evolution
      Solver: auto
      Other properties:
          - nparams: 3
          - samplesize: 1000
          - mess: ''

Note that :class:`~.model.Fittable` objects are a subclass of
:class:`~.model.Fitted` objects, except they don't have a value.

These models function the same way as those created by :func:`.gddm`.  The only
difference between these two interfaces is the way the models are
constructed. Everything you do with the model after creating it, including the
use of Sample and Solution objects, is identical.  Therefore, we will focus on
model construction for the rest of this tutorial.

:download:`Download this full example <downloads/oosimple.py>`
           
Working with data
-----------------

(`View a shortened interactive version of this tutorial. <https://colab.research.google.com/github/mwshinn/PyDDM/blob/master/doc/notebooks/pyddm_gddm_short_tutorial.ipynb>`_)


.. _oo-quickstart-roitman:

We load the data the same way we did in the quickstart tutorial.

.. literalinclude:: downloads/oo_roitman_shadlen.py
   :language: python
   :lines: 6-23


Now that we have loaded these data, we can fit a model to them.  First
we will fit a DDM, and then we will fit a GDDM.

First, we want to let the drift rate vary with the coherence.  To do
so, we must subclass :class:`.Drift`.  Each subclass must contain a name
(a short description of how drift varies), required parameters (a list of
the parameters that must be passed when we initialize our subclass,
i.e. parameters which are passed to the constructor), and required
conditions (a list of conditions that must be present in any data when
we fit data to the model).  We can easily define a model that fits our
needs:

.. literalinclude:: downloads/oo_roitman_shadlen.py
   :language: python
   :lines: 58-66

Because we are fitting with likelihood, we must include a baseline
lapse rate to avoid taking the log of 0.  Traditionally this is
implemented with a uniform distribution, but PyDDM can also use an
exponential distribution using
:class:`~.models.overlay.OverlayPoissonMixture` (representing a
Poisson process lapse rate), as we use here.  However, since we also
want a non-decision time, we need to use two Overlay objects.  To
accomplish this, we can use an :class:`~.models.overlay.OverlayChain`
object.  Then, we can construct a model which uses this and fit the
data to the model:

.. literalinclude:: downloads/oo_roitman_shadlen.py
   :language: python
   :lines: 70-90

Finally, we can display the fit parameters with the following command:

.. literalinclude:: downloads/oo_roitman_shadlen.py
   :language: python
   :lines: 91

This gives the following output (which may vary slightly, since the
fitting algorithm is stochastic)::

  Model Roitman data, drift varies with coherence information:
  Drift component DriftCoherence:
      Drift depends linearly on coherence
      Fitted parameters:
      - driftcoh: 10.388292
  Noise component NoiseConstant:
      constant
      Fixed parameters:
      - noise: 1.000000
  Bound component BoundConstant:
      constant
      Fitted parameters:
      - B: 0.744209
  IC component ICPointSourceCenter:
      point_source_center
      (No parameters)
  Overlay component OverlayChain:
      Overlay component OverlayNonDecision:
          Add a non-decision by shifting the histogram
          Fitted parameters:
          - nondectime: 0.312433
      Overlay component OverlayPoissonMixture:
          Poisson distribution mixture model (lapse rate)
          Fixed parameters:
          - pmixturecoef: 0.020000
          - rate: 1.000000
  Fit information:
      Loss function: Negative log likelihood
      Loss function value: 199.3406727870083
      Fitting method: differential_evolution
      Solver: auto
      Other properties:
          - nparams: 3
          - samplesize: 2611
          - mess: ''

Or, to access them within Python instead of printing them,

.. literalinclude:: downloads/oo_roitman_shadlen.py
   :language: python
   :lines: 92

As before, we can graphically evaluate the quality of the fit.  We can plot
and save a graph:

.. literalinclude:: downloads/oo_roitman_shadlen.py
   :language: python
   :lines: 95-99

.. image:: images/roitman-fit-oo.png

We can also explore this with the PyDDM's model GUI:

.. literalinclude:: downloads/oo_roitman_shadlen.py
   :language: python
   :lines: 104

.. image:: images/model-gui.png


Improving the fit
-----------------

As before, let's improve the fit by including additional model components.  We
will include exponentially collapsing bounds and use a leaky or unstable
integrator instead of a perfect integrator.

To use a coherence-dependent leaky or unstable integrator, we can
build a drift model which incorporates the position of the decision
variable to either increase or decrease drift rate.  This can be
accomplished by making ``get_drift`` depend on the argument ``x``.

.. literalinclude:: downloads/oo_roitman_shadlen.py
   :language: python
   :lines: 110-117

Collapsing bounds are already included in PyDDM, and can be accessed
with :class:`~.models.bound.BoundCollapsingExponential`.

Thus, the full model definition is

.. literalinclude:: downloads/oo_roitman_shadlen.py
   :language: python
   :lines: 120-136

Before fitting this model, let's look at it in the model GUI::

  from pyddm.plot import model_gui
  model_gui(model_leak, sample=roitman_sample)
           
Again, we can fit this and save it as an image using the following:

.. literalinclude:: downloads/oo_roitman_shadlen.py
   :language: python
   :lines: 140-142

.. image:: images/leak-collapse-fit-oo.png

:download:`Download this full example <downloads/oo_roitman_shadlen.py>`

Going further
-------------

Just as we created DriftCoherence above (by inheriting from :class:`.Drift`)
to modify the drift rate based on coherence, we can modify other
portions of the model.  See :doc:`cookbook/index` for more examples.  Also
see the :doc:`apidoc/index` for more specific details about overloading
classes.

Summary
-------

PyDDM can simulate models and generate artificial data, or it can fit
them to data.  Below are high-level overviews for how to accomplish
each.

To define models in the object-oriented API:

1. Optionally, define unique components of your model. Models are
   modular, and allow specifying a dynamic drift rate, noise level,
   diffusion bounds, starting position of the integrator, or
   post-simulation modifications to the RT histogram.  Many common
   models for these are included by default, but for advance
   functionality you may need to subclass :class:`.Drift`,
   :class:`.Noise`, :class:`.Bound`, :class:`.InitialCondition`, or
   :class:`.Overlay`.  These model components may depend on
   "conditions", i.e. prespecified values associated with the
   behavioral task which change from trial to trial (e.g. stimulus
   coherence), or "parameters", i.e. values which apply to all trials
   and should be fit to the subject.
2. Define a model.  Models are represented by creating an instance of the
   :class:`.Model` class, and specifying the model components to use for it.
   These model component can either be :doc:`the model components included in
   PyDDM <apidoc/dependences>` or ones you created in step 1.  Parameters for
   the model components must either be specified expicitly or else set to a
   :class:`.Fittable` instance, for example "Fittable(minval=0, maxval=1)".
3. Simulate and fit the model, as with :func:`.gddm`.
