#!/bin/bash

_target=${1:-'all'}

rm -rf build
mkdir build

cd build

cmake .. \
  -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
  -DCMAKE_BUILD_TYPE:STRING=Debug \
  -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ \
  -G Ninja

# ninja
cmake \
  --build . \
  --config Debug --target ${_target}
