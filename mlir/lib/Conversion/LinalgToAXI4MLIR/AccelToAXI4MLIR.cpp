//===- AccelToAXI4MLIR.cpp - Convert Accel to AXI4MLIR calls --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of Accel to AXI4MLIR calls
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToAXI4MLIR/AccelToAXI4MLIR.h"

#include "../PassDetail.h"

#include "mlir/Dialect/Accel/IR/Accel.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

//===----------------------------------------------------------------------===//
// AXI4MLIR Runtime C API declaration.
//===----------------------------------------------------------------------===//
static constexpr const char *kDmaInit = "dma_init";
static constexpr const char *kDmaFree = "dma_free";
static constexpr const char *kCopyToInbufferF32 = "copy_to_inbuffer_f32";
static constexpr const char *kCopyFromOutbufferF32 = "copy_from_outbuffer_f32";
static constexpr const char *kCopyToInbufferI32 = "copy_to_inbuffer_i32";
static constexpr const char *kCopyFromOutbufferI32 = "copy_from_outbuffer_i32";
static constexpr const char *kDmaStartSend = "dma_start_send";
static constexpr const char *kDmaWaitSend = "dma_wait_send";
static constexpr const char *kDmaStartRecv = "dma_start_recv";
static constexpr const char *kDmaWaitRecv = "dma_wait_recv";

using namespace mlir;

class InitDMAToAXI4MLIRCall : public OpRewritePattern<accel::InitDMAOp> {
public:
  using OpRewritePattern<accel::InitDMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(accel::InitDMAOp op,
                                PatternRewriter &rewriter) const override {

    // // Lambda to declare a function
    // auto addFuncDecl = [&](StringRef name, FunctionType type) {
    //   if (module.lookupSymbol<FuncOp>(name))
    //     return;
    //   rewriter.create<FuncOp>(name, type).setPrivate();
    //   assert(
    //       isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module,
    //       name)));
    // };

    auto module = SymbolTable::getNearestSymbolTable(op);

    auto name = kDmaInit;
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, name));
    // Forward declare function if it hasn't already been
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());

      Type intTy = rewriter.getI32Type();
      auto opFunctionTy = FunctionType::get(
          rewriter.getContext(), {intTy, intTy, intTy, intTy, intTy}, {});
      rewriter.create<FuncOp>(rewriter.getUnknownLoc(), name, opFunctionTy)
          .setPrivate();
    }
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

    rewriter.replaceOpWithNewOp<CallOp>(op, name, /*TODO no type?*/ TypeRange(),
                                              op->getOperands());
    module->emitWarning();
    return success();
  }
};

void mlir::populateAccelToAXI4MLIRConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<InitDMAToAXI4MLIRCall>(patterns.getContext());
  // patterns.add<SendToAXI4MLIRCall>(patterns.getContext());
  // patterns.add<RecvDMAToAXI4MLIRCall>(patterns.getContext());
}

namespace {
struct ConvertAccelToAXI4MLIRPass
    : public ConvertAccelToAXI4MLIRBase<ConvertAccelToAXI4MLIRPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertAccelToAXI4MLIRPass::runOnOperation() {
  auto module = getOperation();

  RewritePatternSet patterns(&getContext());
  populateAccelToAXI4MLIRConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithmeticDialect, BuiltinDialect,
                         StandardOpsDialect>();
  target.addIllegalDialect<accel::AccelDialect>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertAccelToAXI4MLIRPass() {
  return std::make_unique<ConvertAccelToAXI4MLIRPass>();
}
