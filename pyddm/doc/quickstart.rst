Quick Start guide
=================

Simple example
--------------

The following simulates a simple DDM with constant drift.
:class:`.Model` is the object which represents a DDM.  Its default
behavior can be changed through :class:`.Mu`, :class:`.Sigma`,
:class:`.Bound`, :class:`.InitialCondition`, and :class:`.Overlay`
objects, which specify the behavior of each of these model components.
For example, the following is a DDM with drift 2.2, noise 1.5, bound
1.1, and a 100ms non-decision time.  It is simulated for 2 seconds.
It can be represented by::

  from ddm import Model
  from ddm.models import MuConstant, SigmaConstant, BoundConstant, OverlayDelay
  from ddm.functions import fit_adjust_model, display_model

  model = Model(name='Simple model',
                mu=MuConstant(mu=2.2),
                sigma=SigmaConstant(sigma=1.5),
                bound=BoundConstant(B=1.1),
                overlay=OverlayDelay(delay=.1),
                dx=.001, dt=.01, T_dur=2)

  display_model(model)
  sol = model.solve()

Solution objects represent PDFs of the solved model.  We can generate
data from this solved model with::

  samp = sol.resample(1000)
  
To fit the outputs, we first create a model with special
:class:`.Fittable` objects in all the parameters we would like to be
fit.  We specify the range of each of these objects as a hint to the
optimizer; this is mandatory for some but not all optimization
methods.  Then, we run the :func:`.fit_adjust_model` function, which
will convert the :class:`.Fittable` objects to :class:`.Fitted`
objects and find a value for each which collectively minimizes the
objective function.

Here, we use the same model as above, since we know the form the model
is supposed to have.  We fit the model to the generated data using BIC::

  from ddm import Fittable
  from ddm.models import LossBIC
  from ddm.functions import fit_adjust_model
  model_fit = Model(name='Simple model (fitted)',
                    mu=MuConstant(mu=Fittable(minval=0, maxval=4)),
                    sigma=SigmaConstant(sigma=Fittable(minval=.5, maxval=4)),
                    bound=BoundConstant(B=1.1),
                    overlay=OverlayDelay(delay=Fittable(minval=0, maxval=1)),
                    dx=.001, dt=.01, T_dur=2)

  fit_adjust_model(samp, model_fit,
                   method="differential_evolution",
                   lossfunction=LossBIC)

  display_model(model_fit)

Working with data
-----------------

Loading data from a CSV file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we load data from the open dataset by Roitman and
Shadlen.

The CSV file looks like the following:

====== ===== ===== ======= =========
monkey rt    coh   correct trgchoice
====== ===== ===== ======= =========
1      0.355 0.512 1.0     2.0
1      0.359 0.256 1.0     1.0
1      0.525 0.128 1.0     1.0
====== ===== ===== ======= =========


It is fairly easy then to load and process the CSV file::

  from ddm import Sample
  import pandas
  with open("roitman_rts.csv", "r") as f:
      df_rt = pandas.read_csv(f)
  
  df_rt = df_rt[df_rt["monkey"] == 1] # Only monkey 1
  
  # Remove short and long RTs, as in 10.1523/JNEUROSCI.4684-04.2005.
  # This is not strictly necessary, but is performed here for
  # compatibility with this study.
  df_rt = df_rt[df_rt["rt"] > .1] # Remove trials less than 100ms
  df_rt = df_rt[df_rt["rt"] < 1.65] # Remove trials greater than 1650ms
  
  # Create a sample object from our data.  This is the standard input
  # format for fitting procedures.  Since RT and correct/error are
  # both mandatory columns, their names are specified by command line
  # arguments.
  roitman_sample = Sample.from_pandas_dataframe(df_rt, rt_column_name="rt", correct_column_name="correct")

This gives an output sample with the conditions "monkey", "coh", and
"trgchoice".

Note that this examples requires `pandas
<https://pandas.pydata.org/>`_ to be installed.

Loading data from a numpy array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data can also be loaded from a numpy array.  For example, let's load
the above data without first loading it into pandas::

  from ddm import Sample
  import numpy as np
  with open("roitman_rts.csv", "r") as f:
      M = np.loadtxt(f, delimiter=",", skiprows=1)
  
  # RT data must be the first column and correct/error must be the
  # second column.
  rt = M[:,1].copy() # Use .copy() because np returns a view
  corr = M[:,3].copy()
  monkey = M[:,0].copy()
  M[:,0] = rt
  M[:,1] = corr
  M[:,3] = monkey
  
  conditions = ["coh", "monkey", "trgchoice"]
  roitman_sample2 = Sample.from_numpy_array(M, conditions)

We can confirm that these two methods of loading data produce the same results::

  assert roitman_sample == roitman_sample2

Fitting a model to data
~~~~~~~~~~~~~~~~~~~~~~~

Now that we have loaded these data, we can fit a model to them.

First, we want to let the drift rate vary with the coherence.  To do
so, we must subclass :class:`.Mu`.  Each subclass must contain a name
(a short description of how mu varies), required parameters (a list of
the parameters that must be passed when we initialize our subclass,
i.e. parameters which are passed to the constructor), and required
conditions (a list of conditions that must be present in any data when
we fit data to the model).  We can easily define a model that fits our
needs::

  import ddm.models
  class MuCoherence(ddm.models.Mu):
      name = "Drift depends linearly on coherence"
      required_parameters = ["mucoh"] # <-- Parameters we want to include in the model
      required_conditions = ["coh"] # <-- Task parameters ("conditions"). Should be the same name as in the sample.
      
      # We must always define the get_mu function, which is used to compute the instantaneous value of mu.
      def get_mu(self, conditions, **kwargs):
          return self.mucoh * conditions['coh']

Then, we can construct a model which uses this and fit the data to the
model::

  from ddm import Model, Fittable
  from ddm.functions import fit_adjust_model, display_model
  from ddm.models import SigmaConstant, BoundConstant, OverlayChain, OverlayDelay, OverlayPoissonMixture
  model_rs = Model(name='Roitman data, mu varies with coherence',
                   mu=MuCoherence(mucoh=Fittable(minval=0, maxval=20)),
                   sigma=SigmaConstant(sigma=1),
                   bound=BoundConstant(B=Fittable(minval=.1, maxval=1.5)),
                   # Since we can only have one overlay, we use
                   # OverlayChain to string together multiple overlays.
                   # They are applied sequentially in order.  OverlayDelay
                   # implements a non-decision time by shifting the
                   # resulting distribution of response times by
                   # `delaytime` seconds.
                   overlay=OverlayChain(overlays=[OverlayDelay(delaytime=Fittable(minval=0, maxval=.4)),
                                                  OverlayPoissonMixture(pmixturecoef=.02,
                                                                        rate=1)]),
                   dx=.001, dt=.01, T_dur=2)
  
  # Fitting this will also be fast because PyDDM can automatically
  # determine that MuCoherence will allow an analytical solution.
  fit_model_rs = fit_adjust_model(sample=roitman_sample, m=model_rs)
  display_model(fit_model_rs)

Plotting the fit
~~~~~~~~~~~~~~~~

We can also evaluate the quality of the fit.  We can plot and save a
graph::

  import ddm.plot
  import matplotlib.pyplot as plt
  ddm.plot.plot_fit_diagnostics(model=fit_model_rs, sample=roitman_sample)
  plt.savefig("roitman-fit.png")
  plt.show()

We can alternatively explore this with the PyDDM's model GUI::

  ddm.plot.model_gui(model=fit_model_rs, sample=roitman_sample)

See :doc:`modelgui` for more info.

Going further
-------------

Just as we created MuCoherence above (by inheriting from :class:`.Mu`)
to modify the drift rate based on coherence, we can modify other
portions of the model.  See :doc:`recipes` for more examples.  Also
see the :doc:`apidoc/index` for more specific details about overloading
classes.
