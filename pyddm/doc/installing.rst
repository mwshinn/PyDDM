Installation
============

System requirements
-------------------

- Python 3.4 or higher
- `Scipy/numpy <https://www.scipy.org/>`_
- `Paranoid scientist <https://github.com/mwshinn/paranoidscientist>`_ (``pip install paranoid-scientist``)
- For plotting features, `matplotlib <https://matplotlib.org/>`_

Installation
------------

Normally, you can install with::

    pip install pyddm

If you are in a shared environment (e.g. a cluster), install with::

    pip install pyddm --user

If installing from source, download, extract, and do::

    python3 setup.py install

Testing
-------

To ensure that everything works as expected, run::

    python3 tests/test.py
