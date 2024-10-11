#/bin/bash

/workspaces/llvm-project-class/builds/llvm-project/build/bin/mlir-opt '-pass-pipeline=builtin.module(func.func(convert-linalg-to-loops,lower-affine,convert-scf-to-cf,convert-arith-to-llvm),convert-vector-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' /workspaces/llvm-project-class/mlir/test/mlir-cpu-runner/sgemm-naive-codegen.mlir
| /workspaces/llvm-project-class/builds/llvm-project/build/bin/mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=/workspaces/llvm-project-class/builds/llvm-project/build/lib/libmlir_c_runner_utils.so