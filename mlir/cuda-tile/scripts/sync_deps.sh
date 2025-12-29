#!/bin/bash

mkdir -p third_party

git clone -b release/19.x --depth 1 https://github.com/llvm/llvm-project.git third_party/llvm-project
cd third_party/llvm-project

git switch -c cfbb4cc31215d615f605466aef0bcfb42aa9faa5

git clone https://github.com/Alwaysproblem/cuda-tile third_party/cuda-tile

