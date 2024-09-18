#!/bin/bash

mkdir -p third_party

git clone -b release/19.x --depth 1 https://github.com/llvm/llvm-project.git third_party/llvm-project
