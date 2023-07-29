PyDDM - A generalized drift diffusion model simulator
=====================================================

PyDDM is a simulator and modeling framework for **generalized drift-diffusion
models** (GDDM or DDM), with a focus on cognitive neuroscience.  

Key features include:

- **Fast** solutions for drift-diffusion models (DDM) and generalized
  drift-diffusion models (GDDM)
- **Easy and flexible syntax** for building models
- **Arbitrary Python functions define parameters** for drift rate, noise,
  bounds, non-decision time, and starting position
- **Graphical interface** for exploring new models
- Multiprocessor support

`Interactive online demo on Google Colab
<https://colab.research.google.com/github/mwshinn/PyDDM/blob/master/doc/notebooks/interactive_demo.ipynb>`_.

Start with the :doc:`tutorial <quickstart>`.  To see what PyDDM is
capable of, and for example models, see the :doc:`cookbook/index`.
Also see the :doc:`faqs` for more information.

Release annoucments are posted on the `pyddm-announce mailing list
<https://www.freelists.org/list/pyddm-announce>`_ and on `github
<https://github.com/mwshinn/pyddm>`_.

Please note that PyDDM is still beta software so you may experience
some glitches or uninformative error messages.  Please report any
problems to the `bug tracker <https://github.com/mwshinn/pyddm/issues>`_.

Examples
--------

To simulate a simple DDM::

    import pyddm
    import matplotlib.pyplot as plt
    model = pyddm.auto_model(drift=2, noise=1.5, bound=1.3, starting_position=.1, nondecision=.1)
    plt.plot(model.solve().pdf("upper_bound"))

To fit data to a simple DDM::

    import pyddm, pandas
    model = pyddm.auto_model(drift="driftrate", bound="B", starting_position="x0", nondecision="ndtime",
                            parameters={"driftrate": (-1, 1), "B": (.5, 2), "x0": (-.5, .5), "ndtime": (0, .5)})
    df = pandas.from_csv("your_data_here.csv")
    sample = pyddm.Sample.from_pandas_dataframe(df, rt_column_name="rt", choice_column_name="correct")
    model.fit(sample)

To use PyDDM's GUI to visualize a complex model with leaky integration, a
constant drift rate, exponentially collapsing bounds, and a variable starting
position::

    import pyddm
    import pyddm.plot
    import numpy as np
    model = pyddm.auto_model(drift=lambda x,leak,driftrate : driftrate - x*leak,
                            bound=lambda t,initial_B,collapse_rate : initial_B * np.exp(-collapse_rate*t),
                            starting_position="x0",
                            parameters={"leak": (0, 2),
                                        "driftrate": (-3, 3),
                                        "initial_B": (.5, 1.5),
                                        "collapse_rate": (0, 10),
                                        "x0": (-.9, .9)})

    pyddm.plot.model_gui(model) # If not using a Jupyter notebook, or...
    pyddm.plot.model_gui_jupyter(model) # If using a Jupyter notebook




.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installing
   quickstart
   ooquickstart
   cookbook/index
   modelgui
   apidoc/index
   faqs
   contact



