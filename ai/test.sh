#!/bin/bash

set -eu

mypy *.py
pycodestyle *.py
pylint *.py

coverage report --show-missing --include='./*'
