//===- LinalgToAXI4MLIR.cpp - Convert Linalg to AXI4MLIR calls --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of Linalg to AXI4MLIR calls
//
//===----------------------------------------------------------------------===//

// #include <type_traits>

#include "mlir/Conversion/LinalgToAXI4MLIR/LinalgToAXI4MLIR.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/IR/FunctionInterfaces.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AXI4MLIR Runtime C API declaration.
//===----------------------------------------------------------------------===//
static constexpr const char *kDmaInit = "dma_init";
static constexpr const char *kDmaFree = "dma_free";
static constexpr const char *kCopyToInbufferF32 = "copy_to_inbuffer_f32";
static constexpr const char *kCopyFromOutbufferF32 = "copy_from_outbuffer_f32";
static constexpr const char *kDmaStartSend = "dma_start_send";
static constexpr const char *kDmaWaitSend = "dma_wait_send";
static constexpr const char *kDmaStartRecv = "dma_start_recv";
static constexpr const char *kDmaWaitRecv = "dma_wait_recv";

// namespace {

// /// Attribute name used for labeling transfer ops during progressive
// lowering. static const char kPassLabel[] = "__vector_to_scf_lowering__";

// /// Patterns that inherit from this struct have access to
// /// LinalgToAXI4MLIROptions.
// template <typename OpTy>
// struct LinalgToAXI4MLIRPattern : public OpRewritePattern<OpTy> {
//   explicit LinalgToAXI4MLIRPattern(MLIRContext *context,
//                               LinalgToAXI4MLIROptions opt)
//       : OpRewritePattern<OpTy>(context), options(opt) {}

//   LinalgToAXI4MLIROptions options;
// };

static void addAXI4MLIRRuntimeApiDeclarations(ModuleOp module) {

  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  MLIRContext *ctx = module.getContext();

  // Types
  // TODO, for now hardcoded to floats
  Type myType = builder.getF32Type();
  Type intTy = builder.getI64Type();
  Type indexTy = builder.getIndexType();
  Type unrankedType = UnrankedMemRefType::get(myType, 0);

  auto addFuncDecl = [&](StringRef name, FunctionType type) {
    if (module.lookupSymbol<FuncOp>(name))
      return;
    builder.create<FuncOp>(name, type).setPrivate();
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));
  };

  addFuncDecl(kDmaInit,
              FunctionType::get(
                  ctx, {indexTy, indexTy, indexTy, indexTy, indexTy}, {}));
  addFuncDecl(kDmaFree, FunctionType::get(ctx, {}, {}));
  addFuncDecl(kCopyToInbufferF32,
              FunctionType::get(ctx, {unrankedType, intTy}, {intTy}));
  addFuncDecl(kCopyFromOutbufferF32,
              FunctionType::get(ctx, {unrankedType, intTy}, {intTy}));
  addFuncDecl(kDmaStartSend, FunctionType::get(ctx, {intTy, intTy}, {intTy}));
  addFuncDecl(kDmaWaitSend, FunctionType::get(ctx, {}, {}));
  addFuncDecl(kDmaStartRecv, FunctionType::get(ctx, {intTy, intTy}, {intTy}));
  addFuncDecl(kDmaWaitRecv, FunctionType::get(ctx, {}, {}));
}

namespace {

struct ConvertLinalgToAXI4MLIRPass
    : public ConvertLinalgToAXI4MLIRBase<ConvertLinalgToAXI4MLIRPass> {
  ConvertLinalgToAXI4MLIRPass() = default;
  ConvertLinalgToAXI4MLIRPass(const LinalgToAXI4MLIROptions &options) {
    this->tileSize = options.tileSize;
  }

  void runOnOperation() override {
    LinalgToAXI4MLIROptions options;

    ModuleOp module = getOperation();

    addAXI4MLIRRuntimeApiDeclarations(module);

    // Example on how to use options
    // if (lowerPermutationMaps) {
    //   RewritePatternSet lowerTransferPatterns(getOperation().getContext());
    //   mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(
    //       lowerTransferPatterns);
    //   (void)applyPatternsAndFoldGreedily(getOperation(),
    //                                      std::move(lowerTransferPatterns));
    // }

    // RewritePatternSet patterns(getOperation().getContext());
    // populateLinalgToAXI4MLIRConversionPatterns(patterns, options);
    // (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertLinalgToAXI4MLIRPass(
    const LinalgToAXI4MLIROptions &options) {
  return std::make_unique<ConvertLinalgToAXI4MLIRPass>(options);
}
