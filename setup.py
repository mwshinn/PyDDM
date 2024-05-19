# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

from setuptools import setup, Extension
import numpy as np

with open("pyddm/_version.py", "r") as f:
    exec(f.read())

with open("README.md", "r") as f:
    long_desc = f.read()


setup(
    name = 'pyddm',
    version = __version__,
    description = 'Generalized drift diffusion modeling for Python',
    long_description = long_desc,
    long_description_content_type='text/markdown',
    author = 'Max Shinn, Norman Lam',
    author_email = 'm.shinn@ucl.ac.uk',
    maintainer = 'Max Shinn',
    maintainer_email = 'm.shinn@ucl.ac.uk',
    license = 'MIT',
    python_requires='>=3.6',
    url='https://github.com/mwshinn/PyDDM',
    packages = ['pyddm', 'pyddm.models', 'ddm'],
    ext_modules = [Extension('pyddm.csolve',
                             sources=['pyddm/csolve.c'],
                             include_dirs=[np.get_include()],
                           )],
    install_requires = ['numpy >= 1.9.2', 'scipy >= 0.16', 'matplotlib', 'paranoid-scientist >= 0.2.1'],
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Education :: Testing',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
)

