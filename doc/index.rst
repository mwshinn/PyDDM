PyDDM - A generalized drift diffusion model simulator
=====================================================

PyDDM is a simulator and modeling framework for **generalized
drift-diffusion models** (GDDM or DDM), with a focus on cognitive
neuroscience.

Key features include:

- **Fast solutions to generalized drift-diffusion models**, allowing
  data-fitting with a large number of parameters
- Fokker-Planck equation solved numerically using Crank-Nicolson and
  backward Euler methods for **likelihood fitting on the full
  distribution**
- **Arbitrary functions for parameters** drift rate, noise, bounds,
  and initial position distribution
- Arbitrary loss function and fitting method for parameter fitting
- **Multiprocessor support**
- **Optional GUI** for debugging and gaining an intuition for
  different models
- Convenient and extensible object oriented API allows building models
  in a component-wise fashion
- Verified accuracy of simulations using software verification
  techniques

Start with the :doc:`tutorial <quickstart>`.  To see what PyDDM is
capable of, and for example models, see the :doc:`cookbook/index`.
Also see the :doc:`faqs` for more information.  You can also `try an
interactive online demo on Google Colab
<https://colab.research.google.com/github/mwshinn/PyDDM/blob/master/doc/notebooks/pyddm_demo_leaky_collapse.ipynb>`_.

Release annoucments are posted on the `pyddm-announce mailing list
<https://www.freelists.org/list/pyddm-announce>`_ and on `github
<https://github.com/mwshinn/pyddm>`_.

Please note that PyDDM is still beta software so you may experience
some glitches or uninformative error messages.  Please report any
problems to the `bug tracker <https://github.com/mwshinn/pyddm/issues>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installing
   quickstart
   cookbook/index
   modelgui
   apidoc/index
   faqs
   contact



