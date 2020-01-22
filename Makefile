#--------------------------------------------------------------------------#
#
# AUTHOR:   Andreas Toftegaard Kristensen
#
# PURPOSE:  The Makefile is used for exporting the conda environment, fixing
# 			errors/warnings before commits and generating the documentation
#
# EXAMPLE:
#       Generate the environment files
#       > make conda_export
#
#--------------------------------------------------------------------------#

SHELL := /bin/bash

#----------------------------- VARIABLES ----------------------------------#

ENV_NAME =

# Linting rules described at https://lintlyci.github.io/Flake8Rules/
# We currently use the ignored set to contol the linting as the select one is so large
PEP8_IGNORE = "E124,E203,E221,E251,E261,E402,E501,E731"
PEP8_SELECT = "W191,W291,W292,W293,W391,E101,E111,E113,E126,E128,E201,E202,E211.E222,E223,E224,E225,E226,E227,E228,E231,E241,E252,E301,E302,E303,E304,E305,E401,E502,E703,E711,E712,E714,E721"

#----------------------------- TARGETS ------------------------------------#

# Exports your environment
conda_export:
	conda list -e > $(ENV_NAME).txt
	conda env export > $(ENV_NAME).yml

PY_SOURCES := $(shell find ./$(SOURCEDIR) -name '*.py')
PY_SOURCE =

# Print out the warnings/errors in the code
py_lint:
	flake8 --ignore=E121,E124,E127,E203,E221,E251,E261,E402,E501 --count $(PY_SOURCES)

# Fixes your code according to the pep8 rules
py_lint_fix:
	autopep8 --in-place --recursive --aggressive --max-line-length=100 --ignore=$(PEP8_IGNORE) $(PY_SOURCES)

# Use black for aggressive fixing
# Only use for single files!
black_fix:
	black $(PY_SOURCE)

# Use yapf for aggressive fixing
# Only use for single files!
yapf_fix:
	yapf --style "pep8" -i $(PY_SOURCE)

.PHONY: docs
# Generate sphinx documentation
docs:
	$(MAKE) -C docs html

# Open the generated sphinx documentation
docs_view:
	sensible-browser ./docs/build/html/index.html &
