# PyDDM - Generalized drift-diffusion models for Python

# Overview

PyDDM is a simulator and modeling framework for generalized drift-diffusion
models (DDM). Key features include:

- **Fast** solutions for drift-diffusion models (DDM) and generalized
  drift-diffusion models (GDDM)
- **Easy and flexible syntax** for building models
- **Arbitrary Python functions define parameters** for drift rate, noise,
  bounds, non-decision time, and starting position
- **Graphical interface** for exploring new models
- Multiprocessor support

See the [documentation](https://pyddm.readthedocs.io/en/latest/index.html),
[FAQs](https://pyddm.readthedocs.io/en/latest/faqs.html), or
[tutorial](https://pyddm.readthedocs.io/en/latest/quickstart.html) for more
information.  If you want to try it out before installing, visit the
[interactive online
demo](https://colab.research.google.com/github/mwshinn/PyDDM/blob/master/doc/notebooks/interactive_demo.ipynb).
See the [Github Forums](https://github.com/mwshinn/PyDDM/discussions) for help
from the PyDDM community.  You can also sign up for [release announcements by
email](https://www.freelists.org/list/pyddm-announce) (a couple emails per
year).

## Examples

To simulate a simple DDM:

```python
from pyddm import gddm
import matplotlib.pyplot as plt
model = gddm(drift=2, noise=1.5, bound=1.3, starting_position=.1, nondecision=.1)
plt.plot(model.solve().pdf("upper_bound"))
```

To fit data to a simple DDM:

```python
import pyddm, pandas
model = pyddm.gddm(drift="driftrate", noise=1, bound="B", starting_position="x0", nondecision="ndt",
                   parameters={"driftrate": (-1, 1), 
                               "B": (.5, 2),
                               "x0": (-.5, .5),
                               "ndt": (0, .5)})
data = pandas.from_csv("your_data_here.csv")
sample = pyddm.Sample.from_pandas_dataframe(df, rt_column_name="rt", choice_column_name="correct")
model.fit(sample)
```

To use PyDDM's GUI to visualize a complex model with leaky integration, a
constant drift rate, exponentially collapsing bounds, and a variable starting
position:

```python
import pyddm
import pyddm.plot
import numpy as np
model = pyddm.gddm(drift=lambda x,leak,driftrate : driftrate - x*leak,
                   bound=lambda t,initial_B,collapse_rate : initial_B * np.exp(-collapse_rate*t),
                   starting_position="x0",
                   parameters={"leak": (0, 2),
                               "driftrate": (-3, 3),
                               "initial_B": (.5, 1.5),
                               "collapse_rate": (0, 10),
                               "x0": (-.9, .9)})

pyddm.plot.model_gui(model) # If not using a Jupyter notebook, or...
pyddm.plot.model_gui_jupyter(model) # If using a Jupyter notebook
```

[![PyDDM Model GUI](https://github.com/mwshinn/PyDDM/blob/master/doc/images/jupyter-model-gui-animation.gif?raw=true)](https://colab.research.google.com/github/mwshinn/PyDDM/blob/master/doc/notebooks/interactive_demo.ipynb)

## Installation

Normally, you can install with:

    $ pip install pyddm

If you are in a shared environment (e.g. a cluster), install with:

    $ pip install pyddm --user

If installing from source, [download the source code](https://github.com/mwshinn/PyDDM), extract, and do:

    $ python3 setup.py install


## System requirements

- Python 3.6 or above
- Numpy version 1.9.2 or higher
- Scipy version 0.16.0 or higher
- Matplotlib
- [Paranoid Scientist](<https://github.com/mwshinn/paranoidscientist>)
- Pathos (optional, for multiprocessing support)
- To install from source, you will need a C compiler (If you don't already have
  one, the easiest way to install one may be by installing Cython.)  This is
  not necessary if installing from pip.


## Contact

For help on using PyDDM, see the [Github
Forums](https://github.com/mwshinn/PyDDM/discussions).

Please report bugs to <https://github.com/mwshinn/pyddm/issues>.  This
includes any problems with the documentation.  Pull Requests for bugs are
greatly appreciated.

Feature requests are currently not being accepted due to limited
resources.  If you implement a new feature in PyDDM, please do the
following before submitting a Pull Request on Github:

- Make sure your code is clean and well commented
- If appropriate, update the official documentation in the docs/
  directory
- Ensure there are Paranoid Scientist verification conditions to your
  code
- Write unit tests and optionally integration tests for your new
  feature (runtests.sh)
- Ensure all existing tests pass

For all other questions or comments, contact m.shinn@ucl.ac.uk.


## License

All code is available under the MIT license.  See LICENSE.txt for more
information.
