#!/bin/bash

mkdir -p third_party

git clone --filter=blob:none --no-checkout https://github.com/llvm/llvm-project.git third_party/llvm-project
cd third_party/llvm-project

git fetch --depth=1 origin cfbb4cc31215d615f605466aef0bcfb42aa9faa5
git checkout --detach cfbb4cc31215d615f605466aef0bcfb42aa9faa5

cd -

git clone https://github.com/Alwaysproblem/cuda-tile third_party/cuda-tile
