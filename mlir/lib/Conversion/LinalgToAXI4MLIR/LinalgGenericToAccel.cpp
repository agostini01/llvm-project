//===- LinalgGenericToAccel.cpp - Generic to accel conversions --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements from linalg generic to accel calls
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToAXI4MLIR/LinalgGenericToAccel.h"

#include "../PassDetail.h"

#include "mlir/Dialect/Accel/IR/Accel.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

const StringLiteral kLinalgTransformMarker = "__internal_linalg_transform__";
const StringLiteral kAccelTransformMarker = "__accel_transform__";
const StringLiteral kAccel_dmaAddress = "accel_dmaAddress";
const StringLiteral kAccel_dmaInputAddress = "accel_dmaInputAddress";
const StringLiteral kAccel_dmaInputBufferSize = "accel_dmaInputBufferSize";
const StringLiteral kAccel_dmaOuputAddress = "accel_dmaOutputAddress";
const StringLiteral kAccel_dmaOuputBufferSize = "accel_dmaOutputBufferSize";

class LinalgGenericToAccel : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {

    auto module = SymbolTable::getNearestSymbolTable(op);
    Location loc = op->getLoc();

    Value cteZero = rewriter.create<arith::ConstantOp>(
        loc, IntegerAttr::get(rewriter.getI32Type(), 0));
    Value initialOffset = cteZero;

    for (Value operand : op.inputs()) {
      initialOffset = rewriter.create<accel::SendOp>(loc, rewriter.getI32Type(),
                                                     operand, initialOffset);
    }

    initialOffset = cteZero;
    for (Value operand : op.outputs()) {
      // TODO: If accumulate on the CPU, must create a temporary buffer
      bool optionsAccOnCPU = true;
      if (optionsAccOnCPU) {
        MemRefType mrType = operand.getType().cast<MemRefType>();
        // MemRefType::get({options.tileSize, options.tileSize}, myType);
        Value tMr = rewriter.create<memref::AllocaOp>(loc, mrType);
        rewriter.create<accel::RecvOp>(
            loc, rewriter.getI32Type(), tMr,
            initialOffset); // TODO: Initial offset? Multiple outputs?

        // Create affine maps and attributes for CPU accumulation
        MemRefType tmpMrType = tMr.getType().cast<MemRefType>();
        unsigned rank = tmpMrType.getRank();
        SmallVector<AffineMap, 3> indexingMaps(
            /*1 inputs, 1 (inplace) output*/ 2,
            rewriter.getMultiDimIdentityMap(rank));
        auto loopsAttr =
            SmallVector<StringRef>(rank, getParallelIteratorTypeName());

        // Create the linalg generic op
        rewriter.create<linalg::GenericOp>(
            loc,
            /*resultTypes=*/TypeRange(),
            /*inputs=*/tMr,
            /*outputs=*/operand,
            /*indexingMaps=*/indexingMaps,
            /*iteratorTypes=*/loopsAttr,
            /*bodyBuilder=*/
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
              Value added =
                  nestedBuilder.create<arith::AddIOp>(loc, args[0], args[1]);
              nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
            });
      } else {
        initialOffset = rewriter.create<accel::RecvOp>(
            loc, rewriter.getI32Type(), operand, initialOffset);
      }
    }
    rewriter.eraseOp(op);

    return success();
  }
};

void mlir::populateLinalgGenericToAccelConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LinalgGenericToAccel>(patterns.getContext());
}

namespace {
struct ConvertLinalgGenericToAccelPass
    : public ConvertLinalgGenericToAccelBase<ConvertLinalgGenericToAccelPass> {
  ConvertLinalgGenericToAccelPass() = default;

  /// Constructor to build this pass using user defined options
  ConvertLinalgGenericToAccelPass(const LinalgGenericToAccelOptions &options) {
    this->tileSize = options.tileSize;
    this->dmaAddress = options.dmaAddress;
    this->dmaInputAddress = options.dmaInputAddress;
    this->dmaInputBufferSize = options.dmaInputBufferSize;
    this->dmaOutputAddress = options.dmaOutputAddress;
    this->dmaOutputBufferSize = options.dmaOutputAddress;
    this->flowCpuAcc = options.flowCpuAcc;
    this->numberOfCaches = options.numberOfCaches;
    this->cacheSizes = options.cacheSizes;
    this->tileSizes = options.tileSizes;
    this->elementSize = options.elementSize;
    this->anchorFuncName = options.anchorFuncName;
    this->anchorOpName = options.anchorOpName;
  }

  void runOnOperation() override;

  void loadOptions(LinalgGenericToAccelOptions &options) {
    options.tileSize = this->tileSize;
    options.dmaAddress = this->dmaAddress;
    options.dmaInputAddress = this->dmaInputAddress;
    options.dmaInputBufferSize = this->dmaInputBufferSize;
    options.dmaOutputAddress = this->dmaOutputAddress;
    options.dmaOutputBufferSize = this->dmaOutputBufferSize;
    options.flowCpuAcc = this->flowCpuAcc;
    options.numberOfCaches = this->numberOfCaches;
    options.cacheSizes = this->cacheSizes;
    options.tileSizes = this->tileSizes;
    options.elementSize = this->elementSize;
    options.anchorFuncName = this->anchorFuncName;
    options.anchorOpName = this->anchorOpName;
  };
};
} // namespace

void ConvertLinalgGenericToAccelPass::runOnOperation() {

  LinalgGenericToAccelOptions options;
  loadOptions(options);

  auto module = getOperation();
  MLIRContext *ctx = &getContext();

  // Mark linalg operations based on anchor-op
  module.walk([&](FuncOp functionOp) {
    if (!anchorFuncName.empty() && anchorFuncName != functionOp.getName())
      return;

    functionOp.walk([&](linalg::LinalgOp op) {
      if (anchorOpName.empty() || anchorOpName != op->getName().getStringRef())
        return;
      if (!op->getAttr(kAccelTransformMarker)) {
        op->setAttr(kLinalgTransformMarker,
                    StringAttr::get(&getContext(), "generalize"));
      }
    });
  });

  // Generalize anchor-ops with a nested pass manager
  PassManager pm(module.getContext());
  linalg::LinalgTransformationFilter f(StringAttr::get(ctx, "generalize"),
                                       StringAttr::get(ctx, "ACCEL"));
  pm.addNestedPass<FuncOp>(
      mlir::createLinalgStrategyGeneralizePass(anchorOpName, f));
  if (failed(pm.run(module)))
    signalPassFailure();

  RewritePatternSet patterns(&getContext());
  populateLinalgGenericToAccelConversionPatterns(patterns);

  ConversionTarget target(getContext());
  // clang-format off
  target.addLegalDialect<linalg::LinalgDialect,
                         scf::SCFDialect,
                         memref::MemRefDialect, 
                         accel::AccelDialect, 
                         arith::ArithmeticDialect, 
                         BuiltinDialect,
                         StandardOpsDialect>();
  // clang-format on
  target.addDynamicallyLegalOp<linalg::GenericOp>(
      [&](linalg::GenericOp op) -> bool {
        auto marker = StringAttr::get(&getContext(), "ACCEL");
        return !((op->getAttr(kAccelTransformMarker) == marker) ||
                 (op->getAttr(kLinalgTransformMarker) == marker));
      });
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgGenericToAccelPass() {
  return std::make_unique<ConvertLinalgGenericToAccelPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgGenericToAccelPass(
    const LinalgGenericToAccelOptions &options) {
  return std::make_unique<ConvertLinalgGenericToAccelPass>(options);
}
