//===- Utils.h - Function and method used by axi4mlir passes ----*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LINALGTOAXI4MLIR_UTILS_H_
#define MLIR_CONVERSION_LINALGTOAXI4MLIR_UTILS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;
class PatternRewriter;
class ModuleOp;

struct AccelTransformationOptions {
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
  ArrayRef<unsigned> loopPermutation;

  /// Anchor
  std::string anchorFuncName;
  std::string anchorOpName;
  
  /// Opcode information
  std::string opcodeMap;
  std::string initFlow;
  std::string opcodeFlow;

public:
  /// Utility to print members of the struct
  void dump() const;
};

/// Apply tiling patterns to matmul operations with the correct attribute
void applyPatterns(FuncOp funcOp, const AccelTransformationOptions &options);

void addTilingPatternToSet(RewritePatternSet &patterns, MLIRContext *ctx,
                           const StringRef &srcAttrName,
                           const StringRef &dstAttrName, const unsigned &tsd0,
                           const unsigned &tsd1, const unsigned &tsd2);

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOAXI4MLIR_UTILS_H_
