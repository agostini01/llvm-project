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
#include "mlir/Parser.h"
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
const StringLiteral kAccel_acc_on_cpu = "accel_acc_on_cpu";
const StringLiteral kAccel_opcode_map = "accel_opcode_map";
const StringLiteral kAccel_opcode_map_str = "accel_opcode_map_str";
const StringLiteral kAccel_opcode_flow = "accel_opcode_flow";
const StringLiteral kAccel_opcode_flow_str = "accel_opcode_flow_str";
const StringLiteral kAccel_loop_permutation = "accel_loop_permutation";
const StringLiteral kAccel_accel_tile_size = "accel_accel_tile_size";
const StringLiteral kAccel_tile_sizes = "accel_tile_sizes";
const StringLiteral kAccel_init_flow = "accel_init_flow";
const StringLiteral kAccel_init_flow_str = "accel_init_flow_str";

IntegerAttr getU32IntegerAttr(PatternRewriter &rewriter, unsigned value) {
  return rewriter.getIntegerAttr(rewriter.getIntegerType(32, false), value);
}

/// Remove quotes from string to prevent parser from treating it as string.
static StringRef prepStringOption(std::string &s, const char delim = '\"') {
  // NOTE: There is an inconsistent bug with
  //       StringRef::drop_front(),drop_back(),consume_front(),consume_back()
  //       It likely does not update the size every time.
  // NOTE: Input &s must be live after this function call. Passing by copy
  //       also does not work.
  // return StringRef(s).consume_front(delim).consume_back(delim);

  if (s[s.length() - 1] == delim)
    s.erase(s.end() - ((s.length() > 0) ? 1 : 0), s.end());
  if (s[0] == delim)
    s.erase(s.begin());

  return StringRef(s);
}

/// Sets operation Attrs used in generic to accel conversion
class GenericAttrAnnotation : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  /// Construct a generic pattern applied to all GenericOp that verify `filter`.
  GenericAttrAnnotation(
      MLIRContext *context,
      // LinalgTransformationFilter f = LinalgTransformationFilter(),
      AccelTransformationOptions options = AccelTransformationOptions(),
      PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::GenericOp>(context, benefit),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);

    // DMA Attributes
    op->setAttr(kAccel_dmaAddress,
                rewriter.getI32IntegerAttr(options.dmaAddress));
    op->setAttr(kAccel_dmaInputAddress,
                rewriter.getI32IntegerAttr(options.dmaInputAddress));
    op->setAttr(kAccel_dmaInputBufferSize,
                rewriter.getI32IntegerAttr(options.dmaInputBufferSize));
    op->setAttr(kAccel_dmaOuputAddress,
                rewriter.getI32IntegerAttr(options.dmaOutputAddress));
    op->setAttr(kAccel_dmaOuputBufferSize,
                rewriter.getI32IntegerAttr(options.dmaOutputBufferSize));
    op->setAttr(kAccel_acc_on_cpu, rewriter.getBoolAttr(options.flowCpuAcc));

    // OpcodeMap Attribute
    // as string
    std::string s0 = options.opcodeMap;
    StringRef opcodeMapStr = prepStringOption(s0);
    if (opcodeMapStr == "") {
      rewriter.finalizeRootUpdate(op);
      return success();
    }
    op->setAttr(kAccel_opcode_map_str, rewriter.getStringAttr(opcodeMapStr));
    // as attribute
    OpcodeMapAttr opcodeMapAttr =
        parseAttribute(opcodeMapStr, rewriter.getContext())
            .dyn_cast<OpcodeMapAttr>();
    op->setAttr(kAccel_opcode_map, opcodeMapAttr);

    // OpcodeFlow Attribute
    // as string
    std::string s1 = options.opcodeFlow;
    StringRef opcodeFlowStr = prepStringOption(s1);
    op->setAttr(kAccel_opcode_flow_str, rewriter.getStringAttr(opcodeFlowStr));
    // as attribute
    // TODO: handle kAccel_opcode_flow, parse string to validate identifiers

    // InitFlow Attribute
    // as string
    std::string s2 = options.initFlow;
    StringRef initFlowStr = prepStringOption(s2);
    op->setAttr(kAccel_init_flow_str, rewriter.getStringAttr(initFlowStr));
    // as attribute
    // TODO: handle kAccel_init_flow, parse string to validate identifiers

    // Create a lambda function for ArrayRef<unsigned> options
    auto getArrayAttr = [&](const ArrayRef<unsigned> &inArray) -> ArrayAttr {
      SmallVector<Attribute> tmpArray;
      for (auto v : inArray)
        tmpArray.push_back(rewriter.getI32IntegerAttr(v));
      return rewriter.getArrayAttr(tmpArray);
    };

    // LoopPermutation Attribute
    op->setAttr(kAccel_loop_permutation, getArrayAttr(options.loopPermutation));

    // LoopTiling Attribute
    op->setAttr(kAccel_tile_sizes, getArrayAttr(options.tileSizes));

    // Accelerator Tile Size Attribute
    op->setAttr(kAccel_accel_tile_size, getArrayAttr(options.tileSize));

    rewriter.finalizeRootUpdate(op);
    return success();
  }

private:
  AccelTransformationOptions options;
};

/// Function to materialize DMA attributes as constants
static void materializeDMAConstants(PatternRewriter &rewriter, Operation *op,
                                    Location loc,
                                    SmallVector<Value, 5> &values) {
  Type intTy = rewriter.getI32Type();
  values.push_back(rewriter.create<arith::ConstantOp>(
      loc, op->getAttrOfType<IntegerAttr>(kAccel_dmaAddress)));
  values.push_back(rewriter.create<arith::ConstantOp>(
      loc, op->getAttrOfType<IntegerAttr>(kAccel_dmaInputAddress)));
  values.push_back(rewriter.create<arith::ConstantOp>(
      loc, op->getAttrOfType<IntegerAttr>(kAccel_dmaInputBufferSize)));
  values.push_back(rewriter.create<arith::ConstantOp>(
      loc, op->getAttrOfType<IntegerAttr>(kAccel_dmaOuputAddress)));
  values.push_back(rewriter.create<arith::ConstantOp>(
      loc, op->getAttrOfType<IntegerAttr>(kAccel_dmaOuputBufferSize)));
}

/// Rewrites GenericOp as a series of of accel.<operations>
/// Expects the correct attributes to be already set as it
/// does not use options flags and instead, reads the op attributes.
/// TODO: Let this be the case for accelerators with no OPCODES
class LinalgGenericToAccel : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {

    auto module = SymbolTable::getNearestSymbolTable(op);
    Location loc = op->getLoc();

    // Get location before first operation inside funcOp
    FuncOp funcOp = op->getParentOfType<FuncOp>();
    // Location funcFrontLoc = funcOp.front().front().getLoc();

    rewriter.setInsertionPointToStart(&funcOp.front());
    Location funcFrontLoc = rewriter.getInsertionPoint()->getLoc();

    SmallVector<Value, 5> valuesForInitDMA;
    materializeDMAConstants(rewriter, op, funcFrontLoc, valuesForInitDMA);

    // TODO check if such operation already exists for the same DMA address
    // Create the accel.init_dma operation
    rewriter.create<accel::InitDMAOp>(funcFrontLoc, valuesForInitDMA[0],
                                      valuesForInitDMA[1], valuesForInitDMA[2],
                                      valuesForInitDMA[3], valuesForInitDMA[4]);

    rewriter.setInsertionPoint(op);

    Value cteZero = rewriter.create<arith::ConstantOp>(
        loc, IntegerAttr::get(rewriter.getI32Type(), 0));
    Value initialOffset = cteZero;

    for (Value operand : op.inputs()) {
      initialOffset = rewriter.create<accel::SendOp>(loc, rewriter.getI32Type(),
                                                     operand, initialOffset);
    }

    initialOffset = cteZero;
    for (Value operand : op.outputs()) {
      if (op->getAttrOfType<BoolAttr>(kAccel_acc_on_cpu).getValue()) {
        MemRefType mrType = operand.getType().cast<MemRefType>();
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

void mlir::populateLinalgGenericToAccelConversionPatternsWithOptions(
    RewritePatternSet &patterns, const AccelTransformationOptions &options) {
  patterns.add<GenericAttrAnnotation>(patterns.getContext(), options, 2);
  patterns.add<LinalgGenericToAccel>(patterns.getContext());
}

namespace {
struct ConvertLinalgGenericToAccelPass
    : public ConvertLinalgGenericToAccelBase<ConvertLinalgGenericToAccelPass> {
  ConvertLinalgGenericToAccelPass() = default;

  /// Constructor to build this pass using user defined options
  /// Not used when the pass is created from commandline, helpful for creating
  /// this pass in code
  ConvertLinalgGenericToAccelPass(const AccelTransformationOptions &options) {
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
    this->anchorFuncName = options.anchorFuncName;
    this->anchorOpName = options.anchorOpName;
    this->opcodeMap = options.opcodeMap;
    this->initFlow = options.initFlow;
    this->opcodeFlow = options.opcodeFlow;
  }

  void runOnOperation() override;

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
    options.anchorFuncName = this->anchorFuncName;
    options.anchorOpName = this->anchorOpName;
    options.opcodeMap = this->opcodeMap;
    options.initFlow = this->initFlow;
    options.opcodeFlow = this->opcodeFlow;
  }
};
} // namespace

/// The conversion takes the following steps:
///   1. Marks anchor ops with the "generalize" attribute
///   2. Generalizes the marked ops, marking the Ops with the "ACCEL" attribute
///   3. Annotate attributes to the marked ops
///   4. Convert the marked ops to the accel dialect
void ConvertLinalgGenericToAccelPass::runOnOperation() {

  AccelTransformationOptions options;
  setOptions(options);

  auto module = getOperation();
  MLIRContext *ctx = &getContext();

  // 1. Marks anchor ops with the "generalize" attribute
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

  // 2. Generalizes the marked ops, marking the Ops with the "ACCEL" attribute
  // Uses a nested pass manager
  PassManager pm(module.getContext());
  linalg::LinalgTransformationFilter f(StringAttr::get(ctx, "generalize"),
                                       StringAttr::get(ctx, "ACCEL"));
  pm.addNestedPass<FuncOp>(
      mlir::createLinalgStrategyGeneralizePass(anchorOpName, f));
  if (failed(pm.run(module)))
    signalPassFailure();

  // Using rewrite patterns
  // 3. Annotate attributes to the marked ops
  // 4. Convert the marked ops to the accel dialect
  RewritePatternSet patterns(&getContext());
  populateLinalgGenericToAccelConversionPatternsWithOptions(patterns, options);

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

// std::unique_ptr<OperationPass<ModuleOp>>
// mlir::createConvertLinalgGenericToAccelPass(
//     const AccelTransformationOptions &options) {
//   return std::make_unique<ConvertLinalgGenericToAccelPass>(options);
// }
