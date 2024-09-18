#!/bin/bash

WORKSPACE=`pwd`

_llvm_branch=${1:-"release/19.x"}

_dirs="Ch1 Ch2 Ch3 Ch4 Ch5 Ch6 Ch7"

_mlir_example_dir="third_party/llvm-project/mlir/examples/toy"

[[ -d "third_party/llvm-project" ]] || git clone -b $_llvm_branch https://github.com/llvm/llvm-project.git

for dir in $_dirs; do

  pushd "$WORKSPACE/$dir"
    rm -rf $(find ./ -name "*.cpp")
    rm -rf $(find ./ -name "*.h")
    rm -rf $(find ./ -name "*.td")
  popd

  pushd "$WORKSPACE/third_party/llvm-project/mlir/examples/toy/$dir"

    for cpps in $(find ./ -name "*.cpp"); do
      cp ${cpps} "$WORKSPACE/$dir/${cpps}"
    done

    for hs in $(find ./ -name "*.h"); do
      cp ${hs} "$WORKSPACE/$dir/${hs}"
    done

    for tds in $(find ./ -name "*.td"); do
      cp ${tds} "$WORKSPACE/$dir/${tds}"
    done

  popd

done
