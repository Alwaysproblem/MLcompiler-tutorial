#!/bin/bash

rm -rf Ch8
cp -R Ch7 Ch8
cd Ch8
git apply ../scripts/patch/matmul.patch
