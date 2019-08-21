Installation
============

System requirements
-------------------

- Python 3.5 or higher
- `Scipy/numpy <https://www.scipy.org/>`_
- `Paranoid scientist <https://github.com/mwshinn/paranoidscientist>`_ (``pip install paranoid-scientist``)
- For plotting features, `matplotlib <https://matplotlib.org/>`_
- For parallelization support, `pathos <https://pypi.python.org/pypi/pathos>`_

Installation
------------

Normally, you can install with::

    pip install pyddm

If you are in a shared environment (e.g. a cluster), install with::

    pip install pyddm --user

If installing from source, download, extract, and do::

    python3 setup.py install

Getting the source
------------------

`Source code available on Github <https://github.com/mwshinn/PyDDM>`_.

Contributing
------------

Please report bugs to https://github.com/mwshinn/pyddm/issues.  This
includes any problems with the documentation.  Fixes (in the form of
pull requests) for bugs are greatly appreciated.

Feature requests are currently not being accepted due to limited
resources, however if you implement the feature yourself we are open
to accepting it in PyDDM.  If you implement a new feature in PyDDM,
please do the following before submitting a pull request on Github:

- Make sure your code is clean and well commented
- If appropriate, update the official documentation in the ``docs/``
  directory
- Ensure there are Paranoid Scientist verification conditions to your
  code (if appropriate)
- Write unit tests and optionally integration tests for your new
  feature (please add them to ``unit_tests.py`` and
  ``integration_tests.py``)
- Ensure all existing tests pass (``runtests.sh`` returns without
  error)

For all other questions or comments, contact `Max Shinn <mailto:maxwell.shinn@yale.edu>`_.
