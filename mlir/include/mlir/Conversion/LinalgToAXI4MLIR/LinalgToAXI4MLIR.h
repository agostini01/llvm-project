//===- LinalgToAXI4MLIR.h - Convert Linalg to AXI4MLIR calls ----*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LINALGTOAXI4MLIR_LINALGTOAXI4MLIR_H_
#define MLIR_CONVERSION_LINALGTOAXI4MLIR_LINALGTOAXI4MLIR_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;

struct LinalgToAXI4MLIROptions {
  /// Tile Size information
  unsigned tileSize = 1;
  LinalgToAXI4MLIROptions &setTileSize(unsigned t) {
    tileSize = t;
    return *this;
  }
};

/// Collect a set of patterns to convert from the Linagl dialect to AXI4MLIR
/// calls
void populateLinalgToAXI4MLIRConversionPatterns(
    RewritePatternSet &patterns,
    const LinalgToAXI4MLIROptions &options = LinalgToAXI4MLIROptions());

/// Create a pass to convert a subset of vector ops to SCF.
std::unique_ptr<Pass> createConvertLinalgToAXI4MLIRPass(
    const LinalgToAXI4MLIROptions &options = LinalgToAXI4MLIROptions());

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOAXI4MLIR_LINALGTOAXI4MLIR_H_
