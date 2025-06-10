#!/bin/bash

# Find all Jupyter notebooks in the repository and clear their outputs inplace.
# This helps keep Git diffs small by excluding runtime execution results.

set -e

find . -name '*.ipynb' -print0 | xargs -0 jupyter nbconvert --clear-output --inplace
