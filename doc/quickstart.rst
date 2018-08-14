Quick Start guide
=================

Hello, world!
-------------

To get started, let's simulate a basic model and plot it.  For
simplicity, we will use all of the model defaults.

.. literalinclude:: downloads/helloworld.py
   :language: python

.. image:: images/helloworld.png

Congratulations!  You've successfully simulated your first model!
Let's dig in a bit so that we can define more useful models.

:download:`Download this full example <downloads/helloworld.py>`

Simple example
--------------

The following simulates a simple DDM with constant drift.  First, it
shows how to build a model and then uses it to generate artificial
data.  After the artificial data has been generated, it fits a new
model to these data and shows that the parameters are similar.

:class:`.Model` is the object which represents a DDM.  Its default
behavior can be changed through :class:`.Drift`, :class:`.Noise`,
:class:`.Bound`, :class:`.InitialCondition`, and :class:`.Overlay`
objects, which specify the behavior of each of these model components.
Each model must have one of each of these, but defaults to a simple
case for each.  These determine how it calculates the drift rate
(:class:`.Drift`), diffusion coefficient (:class:`.Noise`), shape of the
integration boundaries (:class:`.Bound`), initial particle
distribution (:class:`.InitialCondition`), and any other modifications
to the generated solution (:class:`.Overlay`).

Each of these model components may take "parameters" which are
(usually) unique to that specific model component.  For instance, the
:class:`.DriftConstant` class takes the "drift" parameter, which
should be passed to the constructor.  Similarly,
:class:`.NoiseConstant` takes the parameter "noise" to determine the
standard deviation of the drift process, and
:class:`.OverlayNonDecision` takes "nondectime", the non-decision time
(efferent/afferent/apparatus delay) in seconds.  By contrast,
:class:`.ICPointSourceCenter` does not take any parameters.

For example, the following is a DDM with drift 2.2, noise 1.5, bound
1.1, and a 100ms non-decision time.  It is simulated for 2 seconds
(``T_dur``) with reasonable timestep and grid size (``dt``
and ``dx``).  Once we define the model, the :meth:`~.Model.solve`
function runs the simulation.  This can be computed as shown below:

.. literalinclude:: downloads/simple.py
   :language: python
   :lines: 4-13, 17-18

Solution objects represent correct and erred probability distribution 
functions over time. We can generate psuedo-data from this solved 
model with the :meth:`~.Solution.resample` function:

.. literalinclude:: downloads/simple.py
   :language: python
   :lines: 22
  
To fit the outputs, we first create a model with special
:class:`.Fittable` objects in all the parameters we would like to
fit.  We specify the range of each of these objects as a hint to the
optimizer; this is mandatory for some but not all optimization
methods.  Then, we run the :func:`.fit_adjust_model` function, which
will convert the :class:`.Fittable` objects to :class:`.Fitted`
objects and find a value for each which collectively minimizes the
objective function.

Here, we use the same model as above, since we know the form the model
is supposed to have.  We fit the model to the generated data using BIC:

.. literalinclude:: downloads/simple.py
   :language: python
   :lines: 26-38

We can display the newly-fit parameters with the
:func:`~.functions.display_model` function:

.. literalinclude:: downloads/simple.py
   :language: python
   :lines: 40

This shows that the fitted value of drift is 2.2096, which is close to
the value of 2.2 we used to simulate it.  Similarly, noise fits to
1.539 (compared to 1.5) and nondectime (non-decision time) to 0.1193
(compared to 0.1)::

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

Note that the fit is not perfect due to the finite size of the 
resampled data. We can also draw a plot visualizing the fit.  Unlike our first
example, we will now use one of PyDDM's convenience methods,
:func:`~.plot.plot_fit_diagnostics`:

.. literalinclude:: downloads/simple.py
   :language: python
   :lines: 43-47

.. image:: images/simple-fit.png

Using the :class:`.Solution` object ``sol`` we have access to a number
of other useful functions.  For instance, we can display the
probability of a correct response using
:meth:`~.Solution.prob_correct` or the entire histogram of errors
using :meth:`~.Solution.pdf_err`.

.. literalinclude:: downloads/simple.py
   :language: python
   :lines: 49-50

See :class:`the Solution object documentation <.Solution>` for more
such functions.

:download:`Download this full example <downloads/simple.py>`
           
Working with data
-----------------

Loading data from a CSV file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we load data from the open dataset by Roitman and
Shadlen (2002).  This dataset can be `downloaded here
<https://shadlenlab.columbia.edu/resources/RoitmanDataCode.html>`_ and
the relevant data extracted :download:`with our script
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


It is fairly easy then to load and process the CSV file:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :lines: 6-23

This gives an output sample with the conditions "monkey", "coh", and
"trgchoice".

Note that this examples requires `pandas
<https://pandas.pydata.org/>`_ to be installed.

Loading data from a numpy array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data can also be loaded from a numpy array.  For example, let's load
the above data without first loading it into pandas:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :lines: 28-50

We can confirm that these two methods of loading data produce the same results:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :lines: 53

Fitting a model to data
~~~~~~~~~~~~~~~~~~~~~~~

Now that we have loaded these data, we can fit a model to them.

First, we want to let the drift rate vary with the coherence.  To do
so, we must subclass :class:`.Drift`.  Each subclass must contain a name
(a short description of how drift varies), required parameters (a list of
the parameters that must be passed when we initialize our subclass,
i.e. parameters which are passed to the constructor), and required
conditions (a list of conditions that must be present in any data when
we fit data to the model).  We can easily define a model that fits our
needs:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :lines: 58-66

Then, we can construct a model which uses this and fit the data to the
model:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :lines: 70-90

Finally, we can display the fit parameters with the following command:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :lines: 91

This gives the following output (which may vary slightly, since the
fitting algorithm is stochastic)::

  Model Roitman data, drift varies with coherence information:
  Drift component DriftCoherence:
      Drift depends linearly on coherence
      Fitted parameters:
      - driftcoh: 10.364161
  Noise component NoiseConstant:
      constant
      Fixed parameters:
      - noise: 1.000000
  Bound component BoundConstant:
      constant
      Fitted parameters:
      - B: 0.744062
  IC component ICPointSourceCenter:
      point_source_center
      (No parameters)
  Overlay component OverlayChain:
      Overlay component OverlayNonDecision:
          Add a non-decision by shifting the histogram
          Fitted parameters:
          - nondectime: 0.313715
      Overlay component OverlayPoissonMixture:
          Poisson distribution mixture model (lapse rate)
          Fixed parameters:
          - pmixturecoef: 0.020000
          - rate: 1.000000

         
Plotting the fit
~~~~~~~~~~~~~~~~

We can also graphically evaluate the quality of the fit.  We can plot
and save a graph:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :lines: 95-99

.. image:: images/roitman-fit.png

We can alternatively explore this with the PyDDM's model GUI:

.. literalinclude:: downloads/roitman_shadlen.py
   :language: python
   :lines: 104

.. image:: images/model-gui.png

See :doc:`modelgui` for more info.

:download:`Download this full example <downloads/roitman_shadlen.py>`

Going further
-------------

Just as we created DriftCoherence above (by inheriting from :class:`.Drift`)
to modify the drift rate based on coherence, we can modify other
portions of the model.  See :doc:`recipes` for more examples.  Also
see the :doc:`apidoc/index` for more specific details about overloading
classes.
