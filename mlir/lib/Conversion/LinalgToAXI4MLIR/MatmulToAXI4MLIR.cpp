//===- MatmulToAXI4MLIR.cpp - Convert Matmul to AXI4MLIR calls --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of Matmul to AXI4MLIR calls
//
//===----------------------------------------------------------------------===//

// #include <type_traits>

#include "mlir/Conversion/LinalgToAXI4MLIR/MatmulToAXI4MLIR.h"
#include "mlir/Conversion/LinalgToAXI4MLIR/AXI4MLIRUtils.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"

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
static constexpr const char *kCopyToInbufferI32 = "copy_to_inbuffer_i32";
static constexpr const char *kCopyFromOutbufferI32 = "copy_from_outbuffer_i32";
static constexpr const char *kDmaStartSend = "dma_start_send";
static constexpr const char *kDmaWaitSend = "dma_wait_send";
static constexpr const char *kDmaStartRecv = "dma_start_recv";
static constexpr const char *kDmaWaitRecv = "dma_wait_recv";

// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral kLinalgTransformMarker = "__internal_linalg_transform__";

static void addAXI4MLIRRuntimeApiDeclarations(ModuleOp module) {

  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  MLIRContext *ctx = module.getContext();

  // Types
  // TODO, for now hardcoded
  // Type myType = builder.getF32Type();
  Type myType = builder.getI32Type();
  // Type intTy = builder.getI64Type();
  Type intTy = builder.getI32Type();
  Type unrankedType = UnrankedMemRefType::get(myType, 0);

  auto addFuncDecl = [&](StringRef name, FunctionType type) {
    if (module.lookupSymbol<FuncOp>(name))
      return;
    builder.create<FuncOp>(name, type).setPrivate();
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));
  };

  addFuncDecl(kDmaInit,
              FunctionType::get(ctx, {intTy, intTy, intTy, intTy, intTy}, {}));
  addFuncDecl(kDmaFree, FunctionType::get(ctx, {}, {}));
  addFuncDecl(kCopyToInbufferF32,
              FunctionType::get(ctx, {unrankedType, intTy}, {intTy}));
  addFuncDecl(kCopyFromOutbufferF32,
              FunctionType::get(ctx, {unrankedType, intTy}, {intTy}));
  addFuncDecl(kCopyToInbufferI32,
              FunctionType::get(ctx, {unrankedType, intTy}, {intTy}));
  addFuncDecl(kCopyFromOutbufferI32,
              FunctionType::get(ctx, {unrankedType, intTy}, {intTy}));
  addFuncDecl(kDmaStartSend, FunctionType::get(ctx, {intTy, intTy}, {intTy}));
  addFuncDecl(kDmaWaitSend, FunctionType::get(ctx, {}, {}));
  addFuncDecl(kDmaStartRecv, FunctionType::get(ctx, {intTy, intTy}, {intTy}));
  addFuncDecl(kDmaWaitRecv, FunctionType::get(ctx, {}, {}));
}

static void addDMAInitCalls(FuncOp funcOp,
                            const AccelTransformationOptions &options) {
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

/// Helper to emit a call.
/// Usage example:
///      Type int64Ty = b.getI64Type();
///      int64_t counter = 0;
///      auto value = b.create<arith::ConstantOp>(IntegerAttr::get(int64Ty,
///                   counter));
///      auto printer =
///          LLVM::lookupOrCreatePrintI64Fn(op->getParentOfType<ModuleOp>());
///
///      emitCall(b,
///        LLVM::lookupOrCreatePrintOpenFn(op->getParentOfType<ModuleOp>()));
///
///      value = b.create<arith::ConstantOp>(IntegerAttr::get(int64Ty,
///              counter++));
///      emitCall(b, printer, {value}); // 0
///
///      emitCall(b,
///               LLVM::lookupOrCreatePrintCloseFn(op->getParentOfType<ModuleOp>()));
static void emitCall(ImplicitLocOpBuilder &builder, Operation *ref,
                     ValueRange params = ValueRange()) {
  builder.create<LLVM::CallOp>(TypeRange(), SymbolRefAttr::get(ref), params);
}

/// This should only be used on MatmulOps that have been generalized.
/// It has Matmul attributes in mind such as support for only 3 loops.
static void generateRuntime(linalg::GenericOp op,
                            const AccelTransformationOptions &options) {
  auto b = ImplicitLocOpBuilder(op.getLoc(), op);
  Type myType = b.getI32Type();
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

  b.create<CallOp>(kCopyToInbufferI32, intTy,
                   SmallVector<Value, 2>({casted[0], aOffset}));
  b.create<CallOp>(kCopyToInbufferI32, intTy,
                   SmallVector<Value, 2>({casted[1], bOffset}));

  b.create<CallOp>(kDmaStartSend, intTy,
                   SmallVector<Value, 2>({totalLen, aOffset}));
  b.create<CallOp>(kDmaStartRecv, intTy,
                   SmallVector<Value, 2>({oLen, oOffset}));

  b.create<CallOp>(kDmaWaitSend, TypeRange());
  b.create<CallOp>(kDmaWaitRecv, TypeRange());

  if (!options.flowCpuAcc) {
    // Results were accumulated on the accelerator (output stationary)
    b.create<CallOp>(kCopyFromOutbufferI32, intTy,
                     SmallVector<Value, 2>({casted[2], oOffset}));
  } else {
    // Accumulate accelerator results back on output memref/subview

    auto tMr = tmpMrAndCast[0];
    auto tCast = tmpMrAndCast[1];
    auto outSubview = op.outputs()[0]; // This must be updated
    MemRefType tmpMrType = tMr.getType().cast<MemRefType>();
    unsigned rank = tmpMrType.getRank();

    // Copy to the temporary buffer
    b.create<CallOp>(kCopyFromOutbufferI32, intTy,
                     SmallVector<Value, 2>({tCast, oOffset}));

    // Create affine maps and attributes
    SmallVector<AffineMap, 3> indexingMaps(
        /*1 inputs, 1 (inplace) output*/ 2, b.getMultiDimIdentityMap(rank));
    auto loopsAttr =
        SmallVector<StringRef>(rank, getParallelIteratorTypeName());

    // // Create print function
    // b.create<CallOp>("print_memref_f32", intTy,
    //               SmallVector<Value, 2>({tCast, oOffset}));

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
              nestedBuilder.create<arith::AddIOp>(loc, args[0], args[1]);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
        });
  }

  op.erase();
}

namespace {

struct ConvertMatmulToAXI4MLIRPass
    : public ConvertMatmulToAXI4MLIRBase<ConvertMatmulToAXI4MLIRPass> {
  ConvertMatmulToAXI4MLIRPass() = default;

  /// Constructor to build this pass using user defined options
  ///
  /// Must manually set the AccelTransformationOptions options
  ConvertMatmulToAXI4MLIRPass(const AccelTransformationOptions &options) {
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
    this->loopPermutation = options.loopPermutation;
    // this->anchorFuncName = options.anchorFuncName;
    // this->anchorOpName = options.anchorOpName;
    // this->opcodeMap = options.opcodeMap;
    // this->initFlow = options.initFlow;
    // this->opcodeFlow = options.opcodeFlow;
  }

  bool areBothSizeOptionsSet() {
    return (this->cacheSizes.size() > 0 && this->tileSizes.size()) ? true
                                                                   : false;
  }

  void setOptions(AccelTransformationOptions &options) {
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
    options.loopPermutation = this->loopPermutation;
    // options.anchorFuncName = this->anchorFuncName;
    // options.anchorOpName = this->anchorOpName;
    // options.opcodeMap = this->opcodeMap;
    // options.initFlow = this->initFlow;
    // options.opcodeFlow = this->opcodeFlow;
  }

  void runOnOperation() override {
    AccelTransformationOptions options;
    setOptions(options);

    assert(options.numberOfCaches < 4 &&
           "There is no support for number-of-caches > 3");

    assert(!areBothSizeOptionsSet() &&
           "Options cache-sizes and tile-sizes cannot be set at the same time");

    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    addAXI4MLIRRuntimeApiDeclarations(module);

    // Mark any unmarked linalg.matmul for transformation
    module.walk([&](linalg::MatmulOp op) {
      if (!op->getAttr(kLinalgTransformMarker))
        op->setAttr(kLinalgTransformMarker, StringAttr::get(ctx, "GENERALIZE"));
    });

    // Tile matmul operations with MEM attribute
    module.walk([&](FuncOp funcOp) { applyPatterns(funcOp, options); });

    // Replace inner-matmul with ACCEL attribute by accelerator driver logic
    module.walk([&](linalg::GenericOp op) {
      if (op->getAttr(kLinalgTransformMarker) == StringAttr::get(ctx, "ACCEL"))
        addDMAInitCalls(op->getParentOfType<FuncOp>(), options);
    });

    module.walk([&](linalg::GenericOp op) {
      if (op->getAttr(kLinalgTransformMarker) == StringAttr::get(ctx, "ACCEL"))
        generateRuntime(op, options);
    });

    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertMatmulToAXI4MLIRPass() {
  return std::make_unique<ConvertMatmulToAXI4MLIRPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertMatmulToAXI4MLIRPass(
    const AccelTransformationOptions &options) {
  return std::make_unique<ConvertMatmulToAXI4MLIRPass>(options);
}
