#!/bin/bash -e
PYTHON=python3

# Ensure that the C extension is built before testing: python setup.py build_ext --inplace

$PYTHON -m paranoid paranoid_tests.py
$PYTHON -m pytest unit_tests.py
$PYTHON -m pytest integration_tests.py
