from distutils.core import setup

from ddm._version import __version__

setup(
    name = 'PyDDM',
    version = __version__,
    description = 'Extensible drift diffusion modeling for Python',
    author = 'Max Shinn, Norman Lam',
    maintainer = 'Max Shinn',
    maintainer_email = 'maxwell.shinn@yale.edu',
    packages = ['ddm', 'ddm.models'],
    py_modules = ['ddm.functions', 'ddm.model', 'ddm.parameters', 'ddm.analytic', 'ddm.plot'],
    requires = ['numpy', 'scipy', 'matplotlib']
)
