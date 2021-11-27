# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

from setuptools import setup, Extension
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
import scipy.__config__ as spc

with open("ddm/_version.py", "r") as f:
    exec(f.read())

with open("README.md", "r") as f:
    long_desc = f.read()

# # Since the PyDDM c extensions use lapack, we need to have a lapack library
# # available somewhere.  Possibilities for lapack.  Try them in this order,
# # first mkl and then openblas.  We can't use atlas because it doesn't support
# # the one routine we need.
# library_locations = []
# if "extra_dll_dir" in spc.__dict__:
#     library_locations.append(spc.extra_dll_dir)
# if "library_dirs" in spc.lapack_mkl_info:
#     library_locations.extend(spc.lapack_mkl_info['library_dirs'])
# if "library_dirs" in spc.lapack_opt_info:
#     library_locations.extend(spc.lapack_opt_info['library_dirs'])
# if "openblas_lapack_info" in spc.__dict__:
#     if "library_dirs" in spc.openblas_lapack_info:
#         library_locations.extend(spc.openblas_lapack_info['library_dirs'])

# # Try mkl first because it is the fastest but usually unavailable.  Then try
# # the others.  "openblas" is lapack+blas.  Openblasp is often distributed with
# # scipy.  Do lapack last because it sometimes links to atlas, which doesn't
# # include the function we need.
# lapack_names_to_try = ['mkl_rt', 'lapacke', 'openblas', 'openblasp', 'lapack', None]

# print(library_locations)

setup(
    name = 'pyddm',
    version = __version__,
    description = 'Extensible drift diffusion modeling for Python',
    long_description = long_desc,
    long_description_content_type='text/markdown',
    author = 'Max Shinn, Norman Lam',
    author_email = 'maxwell.shinn@yale.edu',
    maintainer = 'Max Shinn',
    maintainer_email = 'maxwell.shinn@yale.edu',
    license = 'MIT',
    python_requires='>=3.5',
    url='https://github.com/mwshinn/PyDDM',
    packages = ['ddm', 'ddm.models'],
    ext_modules = [Extension('ddm.csolve',
                              sources=['ddm/csolve.c'],
                              # libraries=library_locations,
                              extra_link_args=["-llapacke"],
                              # extra_link_args=["-l"+c_library_name])] if (c_library_name is not None) else [],
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

#try:
# for name in lapack_names_to_try:
#except (CCompilerError, DistutilsExecError, DistutilsPlatformError) as e:
#    print("WARNING: Compile error.  PyDDM will be slow.")
#    run_setup(False)
