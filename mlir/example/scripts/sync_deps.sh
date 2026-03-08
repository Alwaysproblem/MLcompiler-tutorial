#!/bin/bash

mkdir -p third_party

git clone -b llvmorg-22.1.0 --depth 1 https://github.com/llvm/llvm-project.git third_party/llvm-project
