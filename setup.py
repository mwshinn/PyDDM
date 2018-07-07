from setuptools import setup

from ddm._version import __version__

setup(
    name = 'pyddm',
    version = __version__,
    description = 'Extensible drift diffusion modeling for Python',
    author = 'Max Shinn, Norman Lam',
    maintainer = 'Max Shinn',
    maintainer_email = 'maxwell.shinn@yale.edu',
    license = 'MIT',
    python_requires='>=3.5',
    packages = ['ddm'],
    install_requires = ['numpy', 'scipy', 'matplotlib', 'paranoid-scientist'],
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Education :: Testing',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Bio-Informatics']
)
