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

${workspace_folder}/builds/llvm-project/build/bin/mlir-opt '-pass-pipeline=builtin.module(func.func(convert-linalg-to-loops,lower-affine,convert-scf-to-cf,convert-arith-to-llvm),convert-vector-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' ${workspace_folder}/mlir/test/CompilersLectureFall2024/jit-e2e-sgemm-naive-codegen.mlir | ${workspace_folder}/builds/llvm-project/build/bin/mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=${workspace_folder}/builds/llvm-project/build/lib/libmlir_c_runner_utils.so