//===- LinalgGenericToAccel.h - Convert linalg to AXI4MLIR calls ----*- C++
//-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LINALGTOAXI4MLIR_LINALGGENERICTOACCEL_H_
#define MLIR_CONVERSION_LINALGTOAXI4MLIR_LINALGGENERICTOACCEL_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;
class ModuleOp;
template <typename T>
class OperationPass;

struct LinalgGenericToAccelOptions {
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

  /// Anchor
  std::string anchorFuncName;
  std::string anchorOpName;
  
  /// Opcode information
  std::string opcodeMap;
  std::string opcodeFlow;

public:
  /// Utility to print members of the struct
  void dump() const;
};

/// Populate the list with patterns that convert from LinalgOps to AccelOps
void populateLinalgGenericToAccelConversionPatternsWithOptions(
    RewritePatternSet &patterns,
    const LinalgGenericToAccelOptions &options = LinalgGenericToAccelOptions());

/// Create the pass to convert from LinalgOps to AccelOps
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLinalgGenericToAccelPass();

std::unique_ptr<OperationPass<ModuleOp>> createConvertLinalgGenericToAccelPass(
    const LinalgGenericToAccelOptions &options);

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOAXI4MLIR_LINALGGENERICTOACCEL_H_
