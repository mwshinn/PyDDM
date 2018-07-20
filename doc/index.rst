PyDDM
=================================

Overview
--------

PyDDM is a simulator and modeling framework for drift-diffusion models
(DDM), with a focus on cognitive neuroscience.

Key features include:

- Models solved numerically using Crank-Nicolson to solve the
  Fokker-Planck equation (Backward Euler, analytical solutions, and
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

See the
`documentation <https://pyddm.readthedocs.io/en/latest/index.html>`_,
`FAQs <https://pyddm.readthedocs.io/en/latest/faqs.html>`_,
or
`tutorial <https://pyddm.readthedocs.io/en/latest/quickstart.html>`_
for more information.

Please note that PyDDM is still beta software so you may experience
some glitches or uninformative error messages.  Please report any
problems to the `bug tracker <https://github.com/mwshinn/pyddm/issues>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   features
   installing
   quickstart
   modelgui
   recipes
   faqs
   apidoc/index

:ref:`genindex`   

