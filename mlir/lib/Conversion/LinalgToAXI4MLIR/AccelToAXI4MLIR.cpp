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

    auto module = SymbolTable::getNearestSymbolTable(op);

    auto name = kDmaInit;
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, name));
    // Forward declare function if it hasn't already been
    if (!opFunc) { // TODO: Check dma_free
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());

      MLIRContext *ctx = rewriter.getContext();
      Location uLoc = rewriter.getUnknownLoc();
      Type intTy = rewriter.getI32Type();
      FunctionType fType;

      fType = FunctionType::get(ctx, {intTy, intTy, intTy, intTy, intTy}, {});
      rewriter.create<FuncOp>(uLoc, name, fType).setPrivate();

      fType = FunctionType::get(ctx, {}, {});
      rewriter.create<FuncOp>(uLoc, kDmaFree, fType).setPrivate();
    }
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

    rewriter.replaceOpWithNewOp<CallOp>(op, name, /*TODO no type?*/ TypeRange(),
                                        op->getOperands());
    // TODO: this may create several DMA frees, but only one is needed
    rewriter.setInsertionPoint(op->getBlock()->getTerminator());
    rewriter.create<CallOp>(rewriter.getUnknownLoc(), kDmaFree,
                            /*TODO no type?*/ TypeRange(), ValueRange());

    return success();
  }
};

class SendToAXI4MLIRCall : public OpRewritePattern<accel::SendOp> {
public:
  using OpRewritePattern<accel::SendOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(accel::SendOp op,
                                PatternRewriter &rewriter) const override {

    auto module = SymbolTable::getNearestSymbolTable(op);
    Location loc = op->getLoc();

    // TODO: Name has to match memref type
    // TODO: This is giving the i32 name but memref may be f32
    auto name = kCopyToInbufferI32;
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, name));

    Type intTy = rewriter.getI32Type();
    Value input = op.getInput();
    auto inputType = input.getType().dyn_cast_or_null<MemRefType>();
    if (!inputType)
      return failure();
    auto myType = inputType.getElementType();
    Type mrTy = UnrankedMemRefType::get(myType, 0);

    Value casted = rewriter.create<memref::CastOp>(loc, mrTy, input);

    // Forward declare functions if it hasn't already been
    if (!opFunc) { // TODO: check for the other function names
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());

      MLIRContext *ctx = rewriter.getContext();
      Location uLoc = rewriter.getUnknownLoc();
      FunctionType fType;

      fType = FunctionType::get(ctx, {mrTy, intTy}, {intTy});
      rewriter.create<FuncOp>(uLoc, name, fType).setPrivate();

      fType = FunctionType::get(ctx, {intTy, intTy}, {intTy});
      rewriter.create<FuncOp>(uLoc, kDmaStartSend, fType).setPrivate();

      fType = FunctionType::get(ctx, {}, {});
      rewriter.create<FuncOp>(uLoc, kDmaWaitSend, fType).setPrivate();
    }
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

    auto initOffset = op.getOffsetValue();
    if (!initOffset) {
      initOffset =
          rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(intTy, 0));
    }

    // Send flow: copy, start, wait
    rewriter.create<CallOp>(loc, name, intTy,
                            SmallVector<Value, 2>({casted, initOffset}));

    int numElements = inputType.getNumElements();
    int bitWidth = inputType.getElementTypeBitWidth();
    int bytes = numElements * bitWidth / 8;

    Value nElements = rewriter.create<arith::ConstantOp>(
        loc, IntegerAttr::get(intTy, numElements));
    rewriter.create<CallOp>(loc, kDmaStartSend, intTy,
                            SmallVector<Value, 2>({nElements, initOffset}));
    rewriter.create<CallOp>(loc, kDmaWaitSend, TypeRange());

    Value resultOffset =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(intTy, bytes));
    rewriter.replaceOp(op, {resultOffset});

    return success();
  }
};

// Rewrite SendLiteral to a call of kCopyToInbufferI32.
// This could be optimized to transfer the literal directly to the
// DMA buffer instead of going through a temporary memref.
class SendLiteralToAXI4MLIRCall
    : public OpRewritePattern<accel::SendLiteralOp> {
public:
  using OpRewritePattern<accel::SendLiteralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(accel::SendLiteralOp op,
                                PatternRewriter &rewriter) const override {

    auto module = SymbolTable::getNearestSymbolTable(op);
    Location loc = op->getLoc();

    // TODO: Name has to match memref type
    auto name = kCopyToInbufferI32;
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, name));

    Type intTy = rewriter.getI32Type();
    Value opcode = op.getOpcode();
    // Create a memref and store the opcode in it
    auto tmpMrTy = MemRefType::get(/*shape*/ {}, rewriter.getIntegerType(32));

    auto input = rewriter.create<memref::AllocOp>(loc, tmpMrTy);
    rewriter.create<memref::StoreOp>(loc, opcode, input, ValueRange());

    auto inputType = input.getType().dyn_cast_or_null<MemRefType>();
    if (!inputType)
      return failure();
    auto myType = inputType.getElementType();
    Type mrTy = UnrankedMemRefType::get(myType, 0);

    Value casted = rewriter.create<memref::CastOp>(loc, mrTy, input);

    // Forward declare functions if it hasn't already been
    if (!opFunc) { // TODO: check for the other function names
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());

      MLIRContext *ctx = rewriter.getContext();
      Location uLoc = rewriter.getUnknownLoc();
      FunctionType fType;

      fType = FunctionType::get(ctx, {mrTy, intTy}, {intTy});
      rewriter.create<FuncOp>(uLoc, name, fType).setPrivate();

      fType = FunctionType::get(ctx, {intTy, intTy}, {intTy});
      rewriter.create<FuncOp>(uLoc, kDmaStartSend, fType).setPrivate();

      fType = FunctionType::get(ctx, {}, {});
      rewriter.create<FuncOp>(uLoc, kDmaWaitSend, fType).setPrivate();
    }
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

    auto initOffset = op.getOffsetValue();
    if (!initOffset) {
      initOffset =
          rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(intTy, 0));
    }

    // Send flow: copy, start, wait
    rewriter.create<CallOp>(loc, name, intTy,
                            SmallVector<Value, 2>({casted, initOffset}));

    int numElements = inputType.getNumElements();
    int bitWidth = inputType.getElementTypeBitWidth();
    int bytes = numElements * bitWidth / 8;

    Value nElements = rewriter.create<arith::ConstantOp>(
        loc, IntegerAttr::get(intTy, numElements));
    rewriter.create<CallOp>(loc, kDmaStartSend, intTy,
                            SmallVector<Value, 2>({nElements, initOffset}));
    rewriter.create<CallOp>(loc, kDmaWaitSend, TypeRange());

    // Free the temporary memref
    rewriter.create<memref::DeallocOp>(loc, input);

    Value resultOffset =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(intTy, bytes));
    rewriter.replaceOp(op, {resultOffset});

    return success();
  }
};

class RecvToAXI4MLIRCall : public OpRewritePattern<accel::RecvOp> {
public:
  using OpRewritePattern<accel::RecvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(accel::RecvOp op,
                                PatternRewriter &rewriter) const override {

    auto module = SymbolTable::getNearestSymbolTable(op);
    Location loc = op->getLoc();

    // TODO: Name has to match memref type
    auto name = kCopyFromOutbufferI32;
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, name));

    Type intTy = rewriter.getI32Type();
    Value dst = op.getDst();
    auto inputType = dst.getType().dyn_cast_or_null<MemRefType>();
    if (!inputType)
      return failure();
    auto myType = inputType.getElementType();
    Type mrTy = UnrankedMemRefType::get(myType, 0);

    Value casted = rewriter.create<memref::CastOp>(loc, mrTy, dst);

    // Forward declare functions if it hasn't already been
    if (!opFunc) { // TODO: check for the other function names
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());

      MLIRContext *ctx = rewriter.getContext();
      Location uLoc = rewriter.getUnknownLoc();
      FunctionType fType;

      fType = FunctionType::get(ctx, {mrTy, intTy}, {intTy});
      rewriter.create<FuncOp>(uLoc, name, fType).setPrivate();

      fType = FunctionType::get(ctx, {intTy, intTy}, {intTy});
      rewriter.create<FuncOp>(uLoc, kDmaStartRecv, fType).setPrivate();

      fType = FunctionType::get(ctx, {}, {});
      rewriter.create<FuncOp>(uLoc, kDmaWaitRecv, fType).setPrivate();
    }
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

    auto initOffset = op.getOffsetValue();
    if (!initOffset) {
      initOffset =
          rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(intTy, 0));
    }

    int numElements = inputType.getNumElements();
    int bitWidth = inputType.getElementTypeBitWidth();
    int bytes = numElements * bitWidth / 8;

    Value nElements = rewriter.create<arith::ConstantOp>(
        loc, IntegerAttr::get(intTy, numElements));

    // Recv flow: start, wait, copy
    rewriter.create<CallOp>(loc, kDmaStartRecv, intTy,
                            SmallVector<Value, 2>({nElements, initOffset}));
    rewriter.create<CallOp>(loc, kDmaWaitRecv, TypeRange());
    rewriter.create<CallOp>(loc, name, intTy,
                            SmallVector<Value, 2>({casted, initOffset}));

    Value resultOffset =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(intTy, bytes));
    rewriter.replaceOp(op, {resultOffset});

    return success();
  }
};

void mlir::populateAccelToAXI4MLIRConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<InitDMAToAXI4MLIRCall>(patterns.getContext());
  patterns.add<SendToAXI4MLIRCall>(patterns.getContext());
  patterns.add<SendLiteralToAXI4MLIRCall>(patterns.getContext());
  patterns.add<RecvToAXI4MLIRCall>(patterns.getContext());
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
  // clang-format off
  target.addLegalDialect<scf::SCFDialect,
                         memref::MemRefDialect, 
                         arith::ArithmeticDialect, 
                         BuiltinDialect,
                         StandardOpsDialect>();
  // clang-format on
  target.addIllegalDialect<accel::AccelDialect>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertAccelToAXI4MLIRPass() {
  return std::make_unique<ConvertAccelToAXI4MLIRPass>();
}
