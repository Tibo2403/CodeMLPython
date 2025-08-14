#!/bin/bash

# Find all tracked Jupyter notebooks and clear their outputs in place.
# This helps keep Git diffs small by excluding runtime execution results.

set -e

# Use git to list notebooks, avoiding untracked directories
# The -z and -0 options allow filenames with spaces to be handled safely
git ls-files -z '*.ipynb' | xargs -0 -r jupyter nbconvert --clear-output --inplace
