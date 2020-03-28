#!/bin/bash

set -eu

mypy *.py
pycodestyle *.py
pylint *.py

coverage run --module unittest discover -p '*_test.py'
coverage report --show-missing --include='./*'
