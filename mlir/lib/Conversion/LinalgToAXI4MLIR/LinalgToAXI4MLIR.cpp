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

static void createFunctionPrototype(const char *name, SmallVector<Type> &inputs,
                                    SmallVector<Type> &outputs, ModuleOp module,
                                    OpBuilder &builder) {

  auto func = module.lookupSymbol<FuncOp>(name);
  if (!func) {

    FunctionType funcTy = builder.getFunctionType(inputs, outputs);

    // Add a private function the top of parent module
    builder.setInsertionPointToStart(module.getBody());
    FuncOp func = builder.create<FuncOp>(builder.getUnknownLoc(), name, funcTy);
    func.setPrivate();

    // builder.setInsertionPoint(op);
  }

  assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));
}

static void declareApi(ModuleOp module, OpBuilder &builder) {

  OpBuilder::InsertionGuard guard(builder);
  // TODO, for now hardcoded to floats
  Type myType = builder.getF32Type();

  Type intTy = builder.getI64Type();
  Type indexTy = builder.getIndexType();
  Type noneTy = builder.getNoneType();
  Type unrankedType = UnrankedMemRefType::get(myType, 0);

  // func private @dma_wait_recv() -> ()
  {
    auto newName = "dma_wait_recv";
    SmallVector<Type> inputs;
    SmallVector<Type> outputs;
    createFunctionPrototype(newName, inputs, outputs, module, builder);
  }

  // func private @dma_start_recv(i64, i64) -> (i64)
  {
    auto newName = "dma_start_recv";
    SmallVector<Type> inputs;
    inputs.reserve(2);
    inputs.push_back(intTy);
    inputs.push_back(intTy);

    SmallVector<Type> outputs;
    outputs.push_back(intTy);

    createFunctionPrototype(newName, inputs, outputs, module, builder);
  }

  // func private @dma_wait_send() -> ()
  {
    auto newName = "dma_wait_send";
    SmallVector<Type> inputs;
    SmallVector<Type> outputs;
    createFunctionPrototype(newName, inputs, outputs, module, builder);
  }

  // func private @dma_start_send(i64, i64) -> (i64)
  {
    auto newName = "dma_start_send";
    SmallVector<Type> inputs;
    inputs.reserve(2);
    inputs.push_back(intTy);
    inputs.push_back(intTy);

    SmallVector<Type> outputs;
    outputs.push_back(intTy);

    createFunctionPrototype(newName, inputs, outputs, module, builder);
  }

  // func private @copy_from_outbuffer_f32(memref<*xf32>, i64) -> (i64)
  {
    auto newName = "copy_from_outbuffer_f32";
    SmallVector<Type> inputs;
    inputs.reserve(2);
    inputs.push_back(unrankedType);
    inputs.push_back(intTy);

    SmallVector<Type> outputs;
    outputs.push_back(intTy);

    createFunctionPrototype(newName, inputs, outputs, module, builder);
  }

  // func private @copy_to_inbuffer_f32(memref<*xf32>, i64) -> (i64)
  {
    auto newName = "copy_to_inbuffer_f32";
    SmallVector<Type> inputs;
    inputs.reserve(2);
    inputs.push_back(unrankedType);
    inputs.push_back(intTy);

    SmallVector<Type> outputs;
    outputs.push_back(intTy);

    createFunctionPrototype(newName, inputs, outputs, module, builder);
  }

  // func private @mlir_dma_copy_from_outbuffer(memref<*xf32>, i64, i64) ->
  // (i64)
  {
    auto newName = "mlir_dma_copy_from_outbuffer";
    SmallVector<Type> inputs;
    inputs.reserve(3);
    inputs.push_back(unrankedType);
    inputs.push_back(intTy);
    inputs.push_back(intTy);

    SmallVector<Type> outputs;
    outputs.push_back(intTy);

    createFunctionPrototype(newName, inputs, outputs, module, builder);
  }

  // func private @mlir_dma_copy_to_inbuffer(memref<*xf32>, i64, i64) -> (i64)
  {
    auto newName = "mlir_dma_copy_to_inbuffer";
    SmallVector<Type> inputs;
    inputs.reserve(3);
    inputs.push_back(unrankedType);
    inputs.push_back(intTy);
    inputs.push_back(intTy);

    SmallVector<Type> outputs;
    outputs.push_back(intTy);

    createFunctionPrototype(newName, inputs, outputs, module, builder);
  }

  // func private @dma_free() -> ()
  {
    auto newName = "dma_free";
    SmallVector<Type> inputs;
    SmallVector<Type> outputs;
    createFunctionPrototype(newName, inputs, outputs, module, builder);
  }

  // func private @dma_init(index, index, index, index, index) -> ()
  {
    auto newName = "dma_init";
    SmallVector<Type> inputs;
    inputs.reserve(5);
    inputs.push_back(indexTy); // dma_address
    inputs.push_back(indexTy); // dma_input_address
    inputs.push_back(indexTy); // dma_input_buffer_size
    inputs.push_back(indexTy); // dma_output_address
    inputs.push_back(indexTy); // dma_output_buffer_size

    SmallVector<Type> outputs;
    outputs.push_back(noneTy);

    createFunctionPrototype(newName, inputs, outputs, module, builder);
  }
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

    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    OpBuilder builder(context);

    declareApi(module, builder);

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
