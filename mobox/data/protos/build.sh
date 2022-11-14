#!/bin/bash

# OUT_DIR=./mobox/data/protos
#
# python3 -m grpc_tools.protoc \
#              -I=./ \
#              --python_out=$OUT_DIR \
#              --proto_path=./mobox/data/protos \
#              map.proto
#
# python3 -m grpc_tools.protoc \
#              -I=./ \
#              --python_out=$OUT_DIR \
#              --proto_path=./mobox/data/protos \
#              scenario.proto

OUT_DIR=./

python3 -m grpc_tools.protoc \
             -I=./ \
             --python_out=$OUT_DIR \
             ./mobox/data/protos/map.proto

python3 -m grpc_tools.protoc \
             -I=./ \
             --python_out=$OUT_DIR \
             ./mobox/data/protos/scenario.proto
