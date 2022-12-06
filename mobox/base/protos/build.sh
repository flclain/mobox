#!/bin/bash

OUT_DIR=./

python3 -m grpc_tools.protoc \
             -I=./ \
             --python_out=$OUT_DIR \
             ./mobox/base/protos/map.proto

python3 -m grpc_tools.protoc \
             -I=./ \
             --python_out=$OUT_DIR \
             ./mobox/base/protos/scenario.proto
