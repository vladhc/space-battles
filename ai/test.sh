#!/bin/bash

set -eu

mypy *.py
pycodestyle *.py
pylint *.py

export CUDA_VISIBLE_DEVICES="-1"
coverage run --module unittest discover -p '*_test.py'
coverage report --show-missing --include='./*'
