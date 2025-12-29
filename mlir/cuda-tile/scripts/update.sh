#!/bin/bash

WORKSPACE=`pwd`

_llvm_branch=${1:-"release/19.x"}

_dirs="Ch1 Ch2 Ch3 Ch4 Ch5 Ch6 Ch7"
_transform_dirs="Ch2 Ch3 Ch4"

_example_in_llvm_project="third_party/llvm-project/mlir/examples"

_mlir_example_dir="${_example_in_llvm_project}/toy"
_mlir_transform_dir="${_example_in_llvm_project}/transform"

[[ -d "third_party/llvm-project" ]] || git clone -b $_llvm_branch https://github.com/llvm/llvm-project.git third_party/llvm-project

# update the mlir Toy examples

for dir in $_dirs; do

  pushd "$WORKSPACE/$dir"
    rm -rf $(find ./ -name "*.cpp")
    rm -rf $(find ./ -name "*.h")
    rm -rf $(find ./ -name "*.td")
  popd

  pushd "$WORKSPACE/${_mlir_example_dir}/$dir"

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

# update the mlir transform examples

for tdir in $_transform_dirs; do

  pushd "$WORKSPACE/transform_$tdir"
    rm -rf $(find ./ -name "*.cpp")
    rm -rf $(find ./ -name "*.h")
    rm -rf $(find ./ -name "*.td")
  popd

  pushd "$WORKSPACE/${_mlir_transform_dir}/$tdir"

    for cpps in $(find ./ -name "*.cpp"); do
      cp ${cpps} "$WORKSPACE/transform_$tdir/${cpps}"
      # echo "cp ${cpps} $WORKSPACE/transform_$tdir/${cpps}"
    done

    for hs in $(find ./ -name "*.h"); do
      cp ${hs} "$WORKSPACE/transform_$tdir/${hs}"
      # echo "cp ${hs} $WORKSPACE/transform_$tdir/${hs}"
    done

    for tds in $(find ./ -name "*.td"); do
      cp ${tds} "$WORKSPACE/transform_$tdir/${tds}"
      # echo "cp ${tds} $WORKSPACE/transform_$tdir/${tds}"
    done

  popd

done
