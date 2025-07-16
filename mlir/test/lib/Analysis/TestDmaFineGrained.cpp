#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Analysis/CustomDMAAttribute.h"
#include "mlir/IR/IRMapping.h"

#define MVIN 2
#define MVIN2 1
#define MVIN3 14
#define MVOUT 3

using namespace mlir;

int getConstantIntValue(Value val) {
  int val_int;
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    Attribute constantAttr = constOp.getValue();
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constantAttr)) {
      val_int = intAttr.getInt();
    }
  }
  return val_int;
}

namespace {

struct DmaFineGrained : public PassWrapper<DmaFineGrained, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DmaFineGrained)
  DmaFineGrained() = default;
  DmaFineGrained(const DmaFineGrained &other) : PassWrapper(other) {
    systolicSize = other.systolicSize;
  }
  StringRef getArgument() const final { return "dma-fine-grained"; }
  StringRef getDescription() const final {
    return "DMA fine-grained";
  }
  Option<int> systolicSize{*this, "systolic-array-size",
                          llvm::cl::desc("Systolic array size (KxK)"),
                          llvm::cl::init(128)};
  void runOnOperation() override;
  bool traverseOperands(Value op_val, Value input);
  AffineMap buildSramAffineMap(OpBuilder &builder, memref::DmaStartOp op);
  AffineMap buildDramAffineMap(OpBuilder &builder, memref::DmaStartOp op);
  FailureOr<SmallVector<Value>> buildSubtileLoop(memref::DmaStartOp dmaOp, OpBuilder &builder, ArrayRef<int64_t> loopOrder, AffineMap& tagMap);
  FailureOr<std::pair<SmallVector<Value>, memref::DmaStartOp>> createSubtileDMA(memref::DmaStartOp dmaOp, ArrayRef<int64_t> loopOrder, OpBuilder &builder);
  FailureOr<SmallVector<Value>> fuseSubtileLoops(
    OpBuilder &builder, SmallVector<Value> lhsLoopVars, SmallVector<Value> rhsLoopVars,
    ArrayRef<Block *> loopBodies, std::vector<std::vector<Value>> &loopGroups);
};

} // namespace

void DmaFineGrained::runOnOperation() {
  auto func = getOperation();
  OpBuilder builder(func.getContext());

  // -------------------------------
  // Extract informations
  // -------------------------------
  linalg::MatmulOp matmulOp;
  func.walk([&](linalg::MatmulOp op) {
    if (!matmulOp)
      matmulOp = op;
  });
  if (!matmulOp) // only apply to functions with matmul
    return;

  // Extract inputs and outputs
  Value matmulResult = matmulOp.getOutputs()[0];
  Value matmulInput = matmulOp.getInputs()[0];
  Value matmulWeight = matmulOp.getInputs()[1];

  // inner loop fine-grained dma
  SmallVector<memref::DmaStartOp, 2> dmaOps;
  func.walk([&](memref::DmaStartOp dmaStartOp) {
    dmaOps.push_back(dmaStartOp);
  });

  // check Bias is moved to Output buffer
  bool is_bias = false;
  memref::DmaStartOp mvin_bias, mvin_input, mvin_weight;
  for(auto dmaOp : dmaOps) {
    int dmaType = getConstantIntValue(dmaOp.getNumElements());
    if (dmaType == MVOUT)
      continue;

    if (traverseOperands(matmulInput, dmaOp.getDstMemRef())) {
      mvin_input = dmaOp;
    } else if (traverseOperands(matmulWeight, dmaOp.getDstMemRef())) {
      mvin_weight = dmaOp;
    } else if (traverseOperands(matmulResult, dmaOp.getDstMemRef())) {
      if (getSubtileSize(dmaOp).size() > 1) {
        mvin_bias = dmaOp;
        is_bias = true;
      }
    }
  }
  if (!getAsyncValue(mvin_input) & !getAsyncValue(mvin_weight))
    return; // no async dma

  // -------------------------------
  // Code Generation
  // -------------------------------
  if (is_bias) {
    // BIAS MVIN
    unsigned rank = llvm::dyn_cast<MemRefType>(mvin_bias.getDstMemRef().getType()).getRank();
    llvm::SmallVector<int64_t> loopOrder;
    // FIXME. loopOrder is hardcoded for 2D and 4D bias subtile
    if (rank == 2) {
      loopOrder = {0, 1};
    } else if (rank == 4) {
      loopOrder = {2, 3, 0, 1};
    } else {
      mvin_bias.emitError("Unsupported memref rank: expected 2 or 4.");
      return;
    }
    (void)createSubtileDMA(mvin_bias, loopOrder, builder);
  }

  unsigned rank;
  llvm::SmallVector<int64_t> inputLoopOrder, weightLoopOrder;
  rank = llvm::dyn_cast<MemRefType>(mvin_input.getDstMemRef().getType()).getRank();
  // FIXME. loopOrder is hardcoded for 2D and 4D bias subtile
  // Note that this order was determined empirically
  if (rank == 2) {
    inputLoopOrder = {0, 1};
    weightLoopOrder = {1, 0};
  } else if (rank == 3) {
    inputLoopOrder = {0, 1, 2};
    weightLoopOrder = {0, 2, 1};
  } else if (rank == 4) {
    inputLoopOrder = {0, 1, 2, 3};
    weightLoopOrder = {0, 1, 3, 2};
  }
  auto [inputLoopVarOuts, inputDmaOp] = createSubtileDMA(mvin_input, inputLoopOrder, builder).value();
  auto [weightLoopVarOuts, weightDmaOp] = createSubtileDMA(mvin_weight, weightLoopOrder, builder).value();

  SmallVector<Block *> loopBodies = {inputDmaOp.getOperation()->getBlock(), weightDmaOp.getOperation()->getBlock()};
  std::vector<std::vector<Value>> loopGroups;
  auto outermostInputLoop = llvm::dyn_cast<affine::AffineForOp>(mlir::isa<BlockArgument>(inputLoopVarOuts.front())
      ? llvm::dyn_cast<BlockArgument>(inputLoopVarOuts.front()).getOwner()->getParent()->getParentOp()
      : nullptr);
  auto outermostWeightLoop = llvm::dyn_cast<affine::AffineForOp>(mlir::isa<BlockArgument>(weightLoopVarOuts.front())
      ? llvm::dyn_cast<BlockArgument>(weightLoopVarOuts.front()).getOwner()->getParent()->getParentOp()
      : nullptr);

  /* Set fused loop order */
  builder.setInsertionPointAfter(outermostWeightLoop);
  if (rank == 2) {
    loopGroups = {
        {inputLoopVarOuts[0]},
        {inputLoopVarOuts[1], weightLoopVarOuts[0]}, // K dim will be fused
        {weightLoopVarOuts[1]}
    };
  } else if (rank == 3) {
    loopGroups = {
        {inputLoopVarOuts[0], weightLoopVarOuts[0]}, // Batch dim will be fused
        {inputLoopVarOuts[1]},
        {inputLoopVarOuts[2], weightLoopVarOuts[1]}, // K dim will be fused
        {weightLoopVarOuts[2]}
    };
  } else if (rank == 4) {
    loopGroups = {
        {inputLoopVarOuts[0]},
        {inputLoopVarOuts[1]},
        {weightLoopVarOuts[0]},
        {weightLoopVarOuts[1]},
        {inputLoopVarOuts[2]},
        {inputLoopVarOuts[3], weightLoopVarOuts[2]}, // K dim wil be fused
        {weightLoopVarOuts[3]}
    };
  }

  // Fuse the subtile loops
  (void)fuseSubtileLoops(builder, inputLoopVarOuts, weightLoopVarOuts, loopBodies, loopGroups);

  // Erase subtile loops
  outermostInputLoop.erase();
  outermostWeightLoop.erase();
}

bool DmaFineGrained::traverseOperands(Value op_val, Value input) {
  bool found = false;
  if (op_val == input) {
    return true;
  }
  auto operation = op_val.getDefiningOp();
  if (operation) {
    for (auto operand : operation->getOperands()) {
      found = found | traverseOperands(operand, input);
      if (operand == input) {
        return true;
      }
    }
  }
  return found;
}

AffineMap DmaFineGrained::buildSramAffineMap(OpBuilder &builder, memref::DmaStartOp op) {
  int64_t vectorlane = systolicSize;
  SmallVector<int64_t, 4> tile_shape = getTileShape(op);
  SmallVector<Attribute, 4> tile_stride = getSramStride(op);
  int64_t vlane_split_axis = getVlaneSplitAxis(op);
  int64_t vlane_stride = getVlaneStride(op);
  int64_t target_stride;
  int64_t nr_outerloop;
  int64_t old_size, new_size;

  target_stride = llvm::dyn_cast<mlir::IntegerAttr>(tile_stride[vlane_split_axis]).getInt();
  old_size = tile_shape[vlane_split_axis];
  nr_outerloop = (old_size + vectorlane*vlane_stride - 1) / (vectorlane*vlane_stride);
  new_size = nr_outerloop * vlane_stride;

  // Dynamically compute scaled strides based on the vector size
  SmallVector<AffineExpr, 4> exprs;
  for (size_t i = 0; i < tile_stride.size(); ++i) {
    int64_t stride = llvm::dyn_cast<mlir::IntegerAttr>(tile_stride[i]).getInt();
    if (stride > target_stride) {
      stride = stride / old_size * new_size;
    }
    if (i != static_cast<size_t>(vlane_split_axis))
      exprs.push_back(builder.getAffineDimExpr(i) * stride);
    else
      exprs.push_back(builder.getAffineDimExpr(i).floorDiv(vectorlane) * stride);
  }
  AffineExpr combinedExpr = exprs[0];
  for (size_t i = 1; i < exprs.size(); ++i) {
    combinedExpr = combinedExpr + exprs[i];
  }
  return AffineMap::get(tile_stride.size(), 0, combinedExpr);
}

AffineMap DmaFineGrained::buildDramAffineMap(OpBuilder &builder, memref::DmaStartOp op) {
  SmallVector<Attribute, 4> dramStride = getDramStride(op);
  unsigned rank = dramStride.size();
  MLIRContext *ctx = builder.getContext();

  AffineExpr expr = builder.getAffineConstantExpr(0);
  for (unsigned i = 0; i < rank; ++i) {
    int64_t stride = llvm::dyn_cast<IntegerAttr>(dramStride[i]).getInt();
    expr = expr + getAffineDimExpr(i, ctx) * builder.getAffineConstantExpr(stride);
  }

  return AffineMap::get(rank, 0, expr);
}

FailureOr<SmallVector<Value>> DmaFineGrained::buildSubtileLoop(
    memref::DmaStartOp dmaOp, OpBuilder &builder, ArrayRef<int64_t> loopOrder, AffineMap& tagMap) {
  MLIRContext *ctx = builder.getContext();
  SmallVector<Value> subLoopVarsOut;
  SmallVector<int64_t> tileSizes = getTileShape(dmaOp);
  auto loc = dmaOp.getLoc();
  unsigned rank = tileSizes.size();
  llvm::SmallVector<int64_t> subTileSizes;
  for (auto attr : getSubtileSize(dmaOp)) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
      subTileSizes.push_back(intAttr.getInt());
    } else {
      dmaOp->emitError("Non-integer subtile_size element.");
      return failure();
    }
  }
  if (subTileSizes.size() != tileSizes.size()) {
    dmaOp->emitError("Mismatch between tile_shape and subtile_size.");
    return failure();
  }
  if (loopOrder.size() != rank) {
    dmaOp->emitError("loop_order size must match rank.");
    return failure();
  }

  // Validate loopOrder: permutation of 0..rank-1
  llvm::SmallDenseSet<unsigned> seen;
  for (unsigned idx : loopOrder) {
    if (idx >= rank || !seen.insert(idx).second) {
      dmaOp->emitError("Invalid or duplicate index in loop_order.");
      return failure();
    }
  }

  // Begin constructing nested affine.for loops
  subLoopVarsOut.resize(rank, nullptr);
  builder.setInsertionPoint(dmaOp);  // Safe initial insertion point

  for (unsigned orderIdx = 0; orderIdx <rank; ++orderIdx) {
    unsigned dim = loopOrder[orderIdx];
    int64_t tileSize = tileSizes[dim];
    int64_t subTileSize = subTileSizes[dim];

    auto loop = builder.create<affine::AffineForOp>(loc, 0, tileSize, subTileSize);
    loop->setAttr("inner_loop", builder.getBoolAttr(true));
    subLoopVarsOut[dim] = loop.getInductionVar();
    builder.setInsertionPointToStart(loop.getBody());
  }

  // Now create a unique tag
  SmallVector<int64_t> strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    unsigned curr = loopOrder[i];
    unsigned next = loopOrder[i + 1];
    strides[curr] = strides[next] * ((tileSizes[next]+subTileSizes[next]-1)/subTileSizes[next]);
  }

  AffineExpr tagExpr = builder.getAffineConstantExpr(0);
  for (unsigned i = 0; i < rank; ++i) {
    auto dimExpr = getAffineDimExpr(i, ctx);
    auto strideExpr = builder.getAffineConstantExpr(strides[i]);
    tagExpr = tagExpr + dimExpr * strideExpr;
  }
  tagMap = AffineMap::get(rank, 0, tagExpr, ctx);

  return subLoopVarsOut;
}

FailureOr<std::pair<SmallVector<Value>, memref::DmaStartOp>> DmaFineGrained::createSubtileDMA(memref::DmaStartOp dmaOp,
                                                    ArrayRef<int64_t> loopOrder,
                                                    OpBuilder &builder) {
  Location loc = dmaOp.getLoc();
  builder.setInsertionPoint(dmaOp);
  MLIRContext *ctx = builder.getContext();
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);

  AffineMap tagMap;
  auto subLoopVarsOut = buildSubtileLoop(dmaOp, builder, loopOrder, tagMap);
  if (failed(subLoopVarsOut)) {
    dmaOp.emitError("Failed to build subtile loop for DMA.");
    return failure();
  }

  // Affine maps for DRAM/SRAM offset
  AffineMap subTileDramMap = buildDramAffineMap(builder, dmaOp);
  AffineMap subTileSramMap = buildSramAffineMap(builder, dmaOp);

  Value subTileDramOffsetApply = builder.create<affine::AffineApplyOp>(
      loc, subTileDramMap, subLoopVarsOut.value());
  Value subTileSramApply = builder.create<affine::AffineApplyOp>(
      loc, subTileSramMap, subLoopVarsOut.value());

  // Setting DRAM index and tag index
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  AffineExpr d1 = getAffineDimExpr(1, ctx);
  AffineMap sumMap = AffineMap::get(2, 0, d0 + d1);
  Value subTileDramApply = builder.create<affine::AffineApplyOp>(
      loc, sumMap, ValueRange{subTileDramOffsetApply, dmaOp.getSrcIndices()[0]});
  Value tagApply = builder.create<affine::AffineApplyOp>(loc, tagMap, subLoopVarsOut.value());

  // Setting DMA operand
  SmallVector<Value> dramIndices = {subTileDramApply};
  SmallVector<Value> sramIndices(subLoopVarsOut.value().size(), zeroIndex);
  sramIndices.back() = subTileSramApply;
  SmallVector<Value> tagIndices = {tagApply};
  NamedAttrList dmaAttr = getDmaAttrs(dmaOp, builder);
  auto newDma = builder.create<memref::DmaStartOp>(
      loc,
      dmaOp.getSrcMemRef(), dramIndices,
      dmaOp.getDstMemRef(), sramIndices,
      dmaOp.getNumElements(),
      dmaOp.getTagMemRef(), tagIndices,
      dmaOp.getStride(), dmaOp.getNumElementsPerStride(),
      dmaAttr);
  dmaOp.erase();
  return std::make_pair(subLoopVarsOut.value(), newDma);
}

FailureOr<SmallVector<Value>> DmaFineGrained::fuseSubtileLoops(
    OpBuilder &builder,
    SmallVector<Value> lhsLoopVars,
    SmallVector<Value> rhsLoopVars,
    ArrayRef<Block *> loopBodies,
    std::vector<std::vector<Value>> &loopGroups) {

  SmallVector<Value> newLoopVars;
  IRMapping replacementMap;
  SmallVector<affine::AffineForOp> loopsToErase;

  affine::AffineForOp lastLoop = nullptr;
  for (const auto &group : loopGroups) {
    Value refIV = group.front();
    auto refLoop = llvm::dyn_cast<affine::AffineForOp>(mlir::isa<BlockArgument>(refIV)
          ? llvm::dyn_cast<BlockArgument>(refIV).getOwner()->getParent()->getParentOp()
          : nullptr);
    if (!refLoop)
      return failure();

    loopsToErase.push_back(refLoop);
    auto newLoop = builder.create<affine::AffineForOp>(
        refLoop.getLoc(),
        refLoop.getLowerBoundOperands(), refLoop.getLowerBoundMap(),
        refLoop.getUpperBoundOperands(), refLoop.getUpperBoundMap(),
        refLoop.getStep().getSExtValue());
    newLoop->setAttr("inner_loop", builder.getBoolAttr(true));
    Value newIV = newLoop.getInductionVar();
    newLoopVars.push_back(newIV);
    lastLoop = newLoop;

    for (Value oldIV : group)
      replacementMap.map(oldIV, newIV);

    builder.setInsertionPointToStart(newLoop.getBody());
  }

  if (!lastLoop)
    return failure();

  Block *dstBody = lastLoop.getBody();
  builder.setInsertionPointToStart(dstBody);
  for (Block* srcBody : loopBodies) {
    for (Operation &op : srcBody->without_terminator()) {
      builder.clone(op, replacementMap);
    }
  }

  return newLoopVars;
}

namespace mlir {
namespace test {
void registerDmaFineGrainedPass() { PassRegistration<DmaFineGrained>(); }
} // namespace test
} // namespace mlir