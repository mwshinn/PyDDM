#!/bin/bash -e
PYTHON=python3

$PYTHON -m paranoid paranoid_tests.py
$PYTHON -m pytest unit_tests.py
$PYTHON -m pytest integration_tests.py
