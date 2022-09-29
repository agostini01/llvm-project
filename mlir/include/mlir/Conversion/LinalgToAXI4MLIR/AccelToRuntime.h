//===- AccelToRuntime.h - Convert Linalg to AXI4MLIR calls ----*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LINALGTOAXI4MLIR_ACCELTORUNTIME_H_
#define MLIR_CONVERSION_LINALGTOAXI4MLIR_ACCELTORUNTIME_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;
class ModuleOp;
template <typename T>
class OperationPass;

struct AccelToRuntimeOptions {
  /// Accelerator Tile Size information
  unsigned tileSize = 1;

  /// DMA Information
  unsigned dmaAddress = 0;
  unsigned dmaInputAddress = 0;
  unsigned dmaInputBufferSize = 100000;
  unsigned dmaOutputAddress = 100000;
  unsigned dmaOutputBufferSize = 100000;

  /// Flow information
  bool flowCpuAcc = false;
  unsigned numberOfCaches = false;
  ArrayRef<unsigned> cacheSizes;
  ArrayRef<unsigned> tileSizes;
  unsigned elementSize = false;
};

/// Collect a set of patterns to convert from the Linagl dialect to AXI4MLIR
/// calls
void populateAccelToRuntimeConversionPatterns(
    RewritePatternSet &patterns,
    const AccelToRuntimeOptions &options = AccelToRuntimeOptions());

// /// Create a pass to convert a linalg.matmul to axi4mlir calls
std::unique_ptr<OperationPass<ModuleOp>> createConvertAccelToRuntimePass();

std::unique_ptr<OperationPass<ModuleOp>> createConvertAccelToRuntimePass(
    const AccelToRuntimeOptions &options);

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOAXI4MLIR_ACCELTORUNTIME_H_
