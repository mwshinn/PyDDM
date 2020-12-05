# PyDDM - A drift-diffusion model simulator

[![Build Status](https://travis-ci.com/mwshinn/PyDDM.svg?branch=master)](https://travis-ci.com/mwshinn/PyDDM)

# Overview

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
[documentation](https://pyddm.readthedocs.io/en/latest/index.html),
[FAQs](https://pyddm.readthedocs.io/en/latest/faqs.html), or
[tutorial](https://pyddm.readthedocs.io/en/latest/quickstart.html) for
more information.  You can also sign up for [release announcements by
email](https://www.freelists.org/list/pyddm-announce).

Please note that PyDDM is still beta software so you may experience
some glitches or uninformative error messages.


## Installation

Normally, you can install with:

    $ pip install pyddm

If you are in a shared environment (e.g. a cluster), install with:

    $ pip install pyddm --user

If installing from source, [download the source code](https://github.com/mwshinn/PyDDM), extract, and do:

    $ python3 setup.py install


## System requirements

- Python 3.5 or above
- Numpy version 1.9.2 or higher
- Scipy version 0.15.1 or higher
- Matplotlib
- [Paranoid Scientist](<https://github.com/mwshinn/paranoidscientist>)
- Pathos (optional, for multiprocessing support)


## Contact

Please report bugs to <https://github.com/mwshinn/pyddm/issues>.  This
includes any problems with the documentation.  PRs for bugs are
greatly appreciated.

Feature requests are currently not being accepted due to limited
resources.  If you implement a new feature in PyDDM, please do the
following before submitting a PR on Github:

- Make sure your code is clean and well commented
- If appropriate, update the official documentation in the docs/
  directory
- Ensure there are Paranoid Scientist verification conditions to your
  code
- Write unit tests and optionally integration tests for your new
  feature (runtests.sh)
- Ensure all existing tests pass

For all other questions or comments, contact maxwell.shinn@yale.edu.


## License

All code is available under the MIT license.  See LICENSE.txt for more
information.
