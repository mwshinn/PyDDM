rm -rf build/ dist/
python3 setup.py sdist
python3 setup.py bdist_wheel
