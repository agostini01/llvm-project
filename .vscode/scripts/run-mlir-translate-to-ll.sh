#!/bin/bash
set -e
set -x
# Extract the workspace folder and filename from the arguments
workspace_folder="$1"
filename="$2"

# Define the prefix to strip
prefix="${workspace_folder}/standalone/test/"

# Strip the prefix from the filename
modified_filename=$(echo "$filename" | sed "s|^$prefix||")

# Compile and jit execute the command with the modified filename
# ${workspace_folder}/builds/llvm-project/build/bin/mlir-opt ${filename} -split-input-file -pass-pipeline="builtin.module(func.func(convert-math-to-llvm))" | 
${workspace_folder}/builds/llvm-project/build/bin/mlir-translate  --mlir-to-llvmir ${filename}