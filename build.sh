#!/bin/bash

../config_cmake.sh

cmake .. -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++

cmake --build . --parallel 32

echo 'export PYTHONPATH="$TVM_HOME/python:$TVM_HOME/.local/python"' >> ~/.zshrc
echo 'export TVM_LIBRARY_PATH="$TVM_HOME/build/lib"' >> ~/.zshrc