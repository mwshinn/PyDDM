PyDDM
=================================

Overview
--------

PyDDM is a simulator and modeling framework for extended drift-diffusion 
models (DDM), with a focus on cognitive neuroscience.

Key features include:

- Fast solutions to extended drift-diffusion models, allowing data-fitting 
  with a large set of parameters
- Fokker-Planck equation solved numerically using Crank-Nicolson and 
  backward Euler methods (Analytical solutions, forward Euler, and 
  particle simulations also available)
- Arbitrary functions for drift rate, noise, bounds, and initial
  position distribution
- Arbitrary loss function and fitting method for parameter fitting
- Optional multiprocessor support
- Optional GUI for debugging and gaining an intuition for different
  models
- Convenient and extensible object oriented API allows building models
  in a component-wise fashion
- Verified accuracy of simulations using novel program verification
  techniques

Start with the :doc:`tutorial <quickstart>` or see the :doc:`faqs` for more information.
For potentially useful model fragments, see :doc:`recipes`

Please note that PyDDM is still beta software so you may experience
some glitches or uninformative error messages.  Please report any
problems to the `bug tracker <https://github.com/mwshinn/pyddm/issues>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installing
   quickstart
   modelgui
   recipes
   faqs
   apidoc/index

:ref:`genindex`   

