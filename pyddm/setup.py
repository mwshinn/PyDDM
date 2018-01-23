from distutils.core import setup

setup(
    name = 'pyddm',
    version = '1.0',
    description = 'Extensible drift diffusion modeling for Python',
    author = 'Norman Lam',
    maintainer = 'Max Shinn',
    maintainer_email = 'maxwell.shinn@yale.edu',
    packages = ['ddm', 'ddm.models'],
    py_modules = ['ddm.functions', 'ddm.model', 'ddm.parameters', 'ddm.analytic', 'ddm.plot'],
    requires = ['numpy', 'scipy', 'matplotlib']
)
