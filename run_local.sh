#!/usr/bin/env bash
# get ftlib's environment
cd ftlib
source environment.sh
cd ..
# pass through args to Python
python3 -m py_autotweaker "$@"