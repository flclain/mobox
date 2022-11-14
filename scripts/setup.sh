#!/bin/bash

ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$PYTHONPATH:${ROOT}
