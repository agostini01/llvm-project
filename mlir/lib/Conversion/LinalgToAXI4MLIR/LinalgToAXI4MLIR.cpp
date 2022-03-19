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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::linalg;

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

// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral kLinalgTransformMarker = "__internal_linalg_transform__";

static void addAXI4MLIRRuntimeApiDeclarations(ModuleOp module) {

  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  MLIRContext *ctx = module.getContext();

  // Types
  // TODO, for now hardcoded to floats
  Type myType = builder.getF32Type();
  // Type intTy = builder.getI64Type();
  Type intTy = builder.getI32Type();
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
                  ctx, {intTy, intTy, intTy, intTy, intTy}, {}));
                  // ctx, {indexTy, indexTy, indexTy, indexTy, indexTy}, {}));
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

/// Apply tiling patterns to matmul operations with the correct attribute
static void applyPatterns(FuncOp funcOp,
                          const LinalgToAXI4MLIROptions &options) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);

  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions().setTileSizes({2000, 3000, 4000}),
      LinalgTransformationFilter(StringAttr::get(ctx, "MEM"),
                                 StringAttr::get(ctx, "L3")));
  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions().setTileSizes({200, 300, 400}),
      LinalgTransformationFilter(StringAttr::get(ctx, "L3"),
                                 StringAttr::get(ctx, "L2")));
  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions().setTileSizes({20, 30, 40}),
      LinalgTransformationFilter(StringAttr::get(ctx, "L2"),
                                 StringAttr::get(ctx, "L1")));

  if (options.tileSize > 1) {
    patterns.add<LinalgTilingPattern>(
        MatmulOp::getOperationName(), ctx,
        LinalgTilingOptions().setTileSizes(
            {options.tileSize, options.tileSize, options.tileSize}),
        LinalgTransformationFilter(StringAttr::get(ctx, "L1"),
                                   StringAttr::get(ctx, "ACCEL")));

  } else {
    patterns.add<LinalgTilingPattern>(
        MatmulOp::getOperationName(), ctx,
        LinalgTilingOptions().setTileSizes({4, 4, 4}),
        LinalgTransformationFilter(StringAttr::get(ctx, "L1"),
                                   StringAttr::get(ctx, "ACCEL")));
  }

  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

  // // Drop the marker.
  // funcOp.walk([](LinalgOp op) {
  //   op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  // });
}

static void addDMAInitCalls(FuncOp funcOp,
                            const LinalgToAXI4MLIROptions &options) {
  auto b = ImplicitLocOpBuilder::atBlockBegin(funcOp.getLoc(),
                                              &(funcOp.body().front()));

  // Type indexTy = b.getIndexType();
  Type indexTy = b.getI32Type();

  SmallVector<Value, 5> dmaInitValues;
  dmaInitValues.push_back(b.create<arith::ConstantOp>(
      IntegerAttr::get(indexTy, options.dmaAddress)));
  dmaInitValues.push_back(b.create<arith::ConstantOp>(
      IntegerAttr::get(indexTy, options.dmaInputAddress)));
  dmaInitValues.push_back(b.create<arith::ConstantOp>(
      IntegerAttr::get(indexTy, options.dmaInputBufferSize)));
  dmaInitValues.push_back(b.create<arith::ConstantOp>(
      IntegerAttr::get(indexTy, options.dmaOutputAddress)));
  dmaInitValues.push_back(b.create<arith::ConstantOp>(
      IntegerAttr::get(indexTy, options.dmaOutputBufferSize)));

  b.create<CallOp>(kDmaInit, TypeRange(), dmaInitValues);

  Operation *terminator = &funcOp.body().front().back();
  b.setInsertionPoint(terminator);
  b.create<CallOp>(kDmaFree, TypeRange());
}

static void castSubViews(linalg::MatmulOp op,
                         const LinalgToAXI4MLIROptions &options) {
  auto b = ImplicitLocOpBuilder(op.getLoc(), op);
  Type myType = b.getF32Type();
  // Type intTy = b.getI64Type();
  Type intTy = b.getI32Type();
  Type unrankedType = UnrankedMemRefType::get(myType, 0);

  SmallVector<Value, 3> casted;
  SmallVector<Value, 6> dims;

  for (Value operand : op->getOperands()) {
    auto v = operand.getDefiningOp<memref::SubViewOp>();

    casted.push_back(b.create<memref::CastOp>(unrankedType, operand));

    for (Value s : v.sizes()) {
      dims.push_back(s);
    }
  }

  SmallVector<Value, 2> tmpMrAndCast;
  if (options.flowCpuAcc) {

    // Create temp memref - same as accelerator tile size
    auto tmpMrType =
        MemRefType::get({options.tileSize, options.tileSize}, myType);
    auto tMr = b.create<memref::AllocaOp>(tmpMrType);
    tmpMrAndCast.push_back(tMr);

    // Cast to the unranked - needed by dma library
    auto tCast = b.create<memref::CastOp>(unrankedType, tMr);
    tmpMrAndCast.push_back(tCast);
  }

  // Calculate transfer sizes and offset
  auto m = b.create<arith::IndexCastOp>(intTy, dims[0]);
  auto k = b.create<arith::IndexCastOp>(intTy, dims[1]);
  auto n = b.create<arith::IndexCastOp>(intTy, dims[3]);
  auto aLen = b.create<arith::MulIOp>(m, k);
  auto bLen = b.create<arith::MulIOp>(k, n);
  auto totalLen = b.create<arith::AddIOp>(aLen, bLen);
  auto oLen = b.create<arith::MulIOp>(m, n);
  // TODO this may depend on the flow order
  auto aOffset = b.create<arith::ConstantOp>(IntegerAttr::get(intTy, 0));
  auto bOffset = aLen;
  auto oOffset = b.create<arith::ConstantOp>(IntegerAttr::get(intTy, 0));

  b.create<CallOp>(kCopyToInbufferF32, intTy,
                   SmallVector<Value, 2>({casted[0], aOffset}));
  b.create<CallOp>(kCopyToInbufferF32, intTy,
                   SmallVector<Value, 2>({casted[1], bOffset}));

  b.create<CallOp>(kDmaStartSend, intTy,
                   SmallVector<Value, 2>({totalLen, aOffset}));
  b.create<CallOp>(kDmaStartRecv, intTy,
                   SmallVector<Value, 2>({oLen, oOffset}));

  b.create<CallOp>(kDmaWaitSend, TypeRange());
  b.create<CallOp>(kDmaWaitRecv, TypeRange());

  if (!options.flowCpuAcc) {
    // Results were accumulated on the accelerator (output stationary)
    b.create<CallOp>(kCopyFromOutbufferF32, intTy,
                     SmallVector<Value, 2>({casted[2], oOffset}));
  } else {
    // Accumulate accelerator results back on output memref/subview

    auto tMr = tmpMrAndCast[0];
    auto tCast = tmpMrAndCast[1];
    auto outSubview = op.outputs()[0]; // This must be updated
    MemRefType tmpMrType = tMr.getType().cast<MemRefType>();
    unsigned rank = tmpMrType.getRank();

    // Copy to the temporary buffer
    b.create<CallOp>(kCopyFromOutbufferF32, intTy,
                     SmallVector<Value, 2>({tCast, oOffset}));

    // Create affine maps and attributes
    SmallVector<AffineMap, 3> indexingMaps(
        /*1 inputs, 1 (inplace) output*/ 2, b.getMultiDimIdentityMap(rank));
    auto loopsAttr =
        SmallVector<StringRef>(rank, getParallelIteratorTypeName());

    // Create the linalg generic op
    Location loc = b.getLoc();
    b.create<linalg::GenericOp>(
        /*resultTypes=*/TypeRange(),
        /*inputs=*/tMr,
        /*outputs=*/outSubview,
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/loopsAttr,
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value added =
              nestedBuilder.create<arith::AddFOp>(loc, args[0], args[1]);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
        });
  }

  op.erase();
}

namespace {

struct ConvertLinalgToAXI4MLIRPass
    : public ConvertLinalgToAXI4MLIRBase<ConvertLinalgToAXI4MLIRPass> {
  ConvertLinalgToAXI4MLIRPass() = default;

  /// Constructor to build this pass using user defined options
  ///
  /// Must manually set the LinalgToAXI4MLIROptions options
  ConvertLinalgToAXI4MLIRPass(const LinalgToAXI4MLIROptions &options) {
    this->tileSize = options.tileSize;
    this->dmaAddress = options.dmaAddress;
    this->dmaInputAddress = options.dmaInputAddress;
    this->dmaInputBufferSize = options.dmaInputBufferSize;
    this->dmaOutputAddress = options.dmaOutputAddress;
    this->dmaOutputBufferSize = options.dmaOutputAddress;
    this->flowCpuAcc = options.flowCpuAcc;
  }

  void runOnOperation() override {
    LinalgToAXI4MLIROptions options;
    options.tileSize = tileSize;
    options.dmaAddress = dmaAddress;
    options.dmaInputAddress = dmaInputAddress;
    options.dmaInputBufferSize = dmaInputBufferSize;
    options.dmaOutputAddress = dmaOutputAddress;
    options.dmaOutputBufferSize = dmaOutputBufferSize;
    options.flowCpuAcc = flowCpuAcc;

    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    addAXI4MLIRRuntimeApiDeclarations(module);

    // Mark any unmarked linalg.matmul for tile generation
    module.walk([&](linalg::MatmulOp op) {
      if (!op->getAttr(kLinalgTransformMarker))
        op->setAttr(kLinalgTransformMarker, StringAttr::get(ctx, "MEM"));
    });

    // Tile matmul operations with MEM attribute
    module.walk([&](FuncOp funcOp) { applyPatterns(funcOp, options); });

    // Replace inner-matmul with ACCEL attribute by accelerator driver logic
    module.walk([&](linalg::MatmulOp op) {
      if (op->getAttr(kLinalgTransformMarker) == StringAttr::get(ctx, "ACCEL"))
        addDMAInitCalls(op->getParentOfType<FuncOp>(), options);
    });

    module.walk([&](linalg::MatmulOp op) {
      if (op->getAttr(kLinalgTransformMarker) == StringAttr::get(ctx, "ACCEL"))
        castSubViews(op, options);
    });

    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgToAXI4MLIRPass() {
  return std::make_unique<ConvertLinalgToAXI4MLIRPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgToAXI4MLIRPass(
    const LinalgToAXI4MLIROptions &options) {
  return std::make_unique<ConvertLinalgToAXI4MLIRPass>(options);
}
