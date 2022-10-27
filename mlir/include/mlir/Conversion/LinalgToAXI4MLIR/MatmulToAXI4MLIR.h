//===- MatmulToAXI4MLIR.h - Convert Linalg to AXI4MLIR calls ----*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LINALGTOAXI4MLIR_MATMULTOAXI4MLIR_H_
#define MLIR_CONVERSION_LINALGTOAXI4MLIR_MATMULTOAXI4MLIR_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Conversion/LinalgToAXI4MLIR/AXI4MLIRUtils.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;
class ModuleOp;
template <typename T>
class OperationPass;

/// Collect a set of patterns to convert from the Linagl dialect to AXI4MLIR
/// calls
void populateMatmulToAXI4MLIRConversionPatterns(
    RewritePatternSet &patterns,
    const AccelTransformationOptions &options = AccelTransformationOptions());

// /// Create a pass to convert a linalg.matmul to axi4mlir calls
std::unique_ptr<OperationPass<ModuleOp>> createConvertMatmulToAXI4MLIRPass();

std::unique_ptr<OperationPass<ModuleOp>> createConvertMatmulToAXI4MLIRPass(
    const AccelTransformationOptions &options);

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOAXI4MLIR_MATMULTOAXI4MLIR_H_
