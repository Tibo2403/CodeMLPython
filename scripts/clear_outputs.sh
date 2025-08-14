#!/bin/bash

# Find all tracked Jupyter notebooks and clear their outputs in place.
# This helps keep Git diffs small by excluding runtime execution results.

set -e

# Use git to list notebooks, avoiding untracked directories
git ls-files '*.ipynb' | xargs -r jupyter nbconvert --clear-output --inplace
