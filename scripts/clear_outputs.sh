#!/bin/bash

# Find all tracked Jupyter notebooks and clear their outputs in place.
# This helps keep Git diffs small by excluding runtime execution results.

set -e

# Use git to list notebooks, avoiding untracked directories
# The -z flag handles filenames with spaces by delimiting with NUL bytes
git ls-files -z '*.ipynb' | xargs -0 -r jupyter nbconvert --clear-output --inplace
