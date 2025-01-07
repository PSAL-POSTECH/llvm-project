#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

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
    tileSize = other.tileSize;
  }
  StringRef getArgument() const final { return "dma-fine-grained"; }
  StringRef getDescription() const final {
    return "DMA fine-grained";
  }
  Option<int> systolicSize{*this, "systolic-array-size",
                          llvm::cl::desc("Systolic array size (KxK)"),
                          llvm::cl::init(128)};
  // option for tileSizeM, tileSizeN, tileSizeK
  ListOption<int> tileSize{*this, "tile-size",
                          llvm::cl::desc("Tile size for M, N, K"),
                          llvm::cl::ZeroOrMore};
  void runOnOperation() override;
  llvm::SmallVector<mlir::Attribute, 2> getSubtileSize(mlir::Operation *operation);
  int getAsyncValue(mlir::Operation *operation);
  int is_transpose(mlir::Operation *operation);
};

} // namespace

void DmaFineGrained::runOnOperation() {
  auto func = getOperation();
  OpBuilder builder(func.getContext());
  bool hasMatmul = false;
  Value matmulResult, matmulInput, matmulWeight;
  func.walk([&](linalg::MatmulOp matmulOp) {
    hasMatmul = true;
    matmulResult = matmulOp.getOutputs()[0];
    matmulInput = matmulOp.getInputs()[0];
    matmulWeight = matmulOp.getInputs()[1];
  });
  if (!hasMatmul) // only apply to functions with matmul
    return;
  int64_t tileSizeM = tileSize[0];
  int64_t tileSizeN = tileSize[1];
  int64_t tileSizeK = tileSize[2];
  int64_t vectorlane = systolicSize;
  builder.setInsertionPointToStart(&func.front());
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(func.getLoc(), 0);
  auto c_set = builder.create<arith::ConstantIndexOp>(func.getLoc(), 2147483650);
  auto tagMemRefType = MemRefType::get({tileSizeN / vectorlane, 1, tileSizeM / vectorlane}, builder.getIntegerType(32));
  auto BtagMemRefType = MemRefType::get({tileSizeN / vectorlane, tileSizeM / vectorlane}, builder.getIntegerType(32));
  auto XtagMemRef = builder.create<memref::AllocOp>(func.getLoc(), tagMemRefType);
  auto WtagMemRef = builder.create<memref::AllocOp>(func.getLoc(), tagMemRefType);
  auto BtagMemRef = builder.create<memref::AllocOp>(func.getLoc(), BtagMemRefType);

  // outer loop step modify
  int loopDepth = 0;
  Value b;
  func.walk([&](affine::AffineForOp loop) {
    // Adjust the step size based on loop depth
    if (loopDepth == 0) {
      loop.setStep(tileSizeK); // First loop (e.g., %t_m) uses tileSizeM
    } else if (loopDepth == 1) {
      loop.setStep(tileSizeN); // Second loop (e.g., %t_n) uses tileSizeN
    } else if (loopDepth == 2) {
      loop.setStep(tileSizeM); // Third loop (e.g., %t_k) uses tileSizeK
    } else if (loopDepth == 3) {
      b = loop.getInductionVar(); // Fourth loop (e.g., %b) is the batch loop
    }
    loopDepth++;
  });
  bool is_bmm = false;
  if (loopDepth == 4) { // bmm has 4 loops (b, m, n, k)
    is_bmm = true;
  }

  // inner loop fine-grained dma
  SmallVector<affine::AffineDmaStartOp, 2> dmaOps;
  func.walk([&](affine::AffineDmaStartOp dmaStartOp) {
    dmaOps.push_back(dmaStartOp);
  });

  // check Bias is moved to Output buffer
  bool is_bias = false;
  affine::AffineDmaStartOp mvin_bias;
  affine::AffineDmaStartOp mvin_input;
  affine::AffineDmaStartOp mvin_weight;

  for(auto dmaOp : dmaOps) {
    Value numElements = dmaOp.getNumElements();
    int dmaType = getConstantIntValue(numElements);
    if (dmaType != MVOUT) {
      if (dmaOp.getDstMemRef() == matmulInput) {
        mvin_input = dmaOp;
      } else if (dmaOp.getDstMemRef() == matmulWeight) {
        mvin_weight = dmaOp;
      } else if (dmaOp.getDstMemRef() == matmulResult) {
        mvin_bias = dmaOp;
        is_bias = true;
      }
    }
  }

  Value i, j, k;
  SmallVector<Value, 2> src_indices;
  SmallVector<Value, 2> dst_indices;
  SmallVector<Value, 3> tag_indices;
  AffineMap new_map;
  SmallVector<Value> new_map_indices;
  ValueRange srcIndices;
  // sum_map = affine_map<(d0, d1) -> (d0 + d1)>
  auto sum_map = AffineMap::get(2, 0, builder.getAffineDimExpr(0) + builder.getAffineDimExpr(1));
  AffineMap tag_idx_map = AffineMap::get(1, 0, builder.getAffineDimExpr(0).floorDiv(vectorlane));
  llvm::SmallVector<mlir::Attribute, 2> dma1Subtile = getSubtileSize(mvin_input);
  llvm::SmallVector<mlir::Attribute, 2> dma2Subtile = getSubtileSize(mvin_weight);
  int dma1Async = getAsyncValue(mvin_input);
  int dma2Async = getAsyncValue(mvin_weight);

  NamedAttrList dma1Attr;
  NamedAttrList dma2Attr;

  int64_t subTileSizeM;
  int64_t subTileSizeN;
  int64_t subTileSizeK;

  // Sanity check
  if (dma1Subtile.size() == 2 && dma2Subtile.size() == 2) {
    if (auto intAttr1 = llvm::dyn_cast<mlir::IntegerAttr>(dma1Subtile[1])) {
      if (auto intAttr2 = llvm::dyn_cast<mlir::IntegerAttr>(dma2Subtile[0]))
        if (intAttr1.getInt() != intAttr2.getInt()) {
          mvin_weight.emitError() << " Not matched: "
                          << "dma1Subtile[1] = " << intAttr1.getInt()
                          << ", dma2Subtile[0] = " << intAttr2.getInt()
                          << "\n";
        } else {
          subTileSizeK = intAttr1.getInt();
        }
      else
        mvin_weight.emitError() << "dma2Subtile[0] is not an IntegerAttr.\n";
    } else
      mvin_input.emitError() << "dma1Subtile[1] is not an IntegerAttr.\n";
  } else {
    mvin_input.emitError() << "subtile_size attribute required for matmul.\n";
  }
  subTileSizeM = llvm::dyn_cast<mlir::IntegerAttr>(dma1Subtile[0]).getInt();
  subTileSizeN = llvm::dyn_cast<mlir::IntegerAttr>(dma2Subtile[1]).getInt();

  if (dma1Subtile.size())
    dma1Attr.set("subtile_size", builder.getArrayAttr(dma1Subtile));
  if (dma2Subtile.size())
    dma2Attr.set("subtile_size", builder.getArrayAttr(dma2Subtile));
  dma1Attr.set("async", builder.getIntegerAttr(builder.getI1Type(), dma1Async));
  dma2Attr.set("async", builder.getIntegerAttr(builder.getI1Type(), dma2Async));
  dma1Attr.set("transpose", builder.getIntegerAttr(builder.getI1Type(), is_transpose(mvin_input)));
  dma2Attr.set("transpose", builder.getIntegerAttr(builder.getI1Type(), is_transpose(mvin_weight)));

  if (is_bias) {
    // BIAS MVIN
    // reset builder location
    llvm::SmallVector<mlir::Attribute, 2> dmaSubtile = getSubtileSize(mvin_bias);
    int dmaAsync = getAsyncValue(mvin_bias);
    NamedAttrList dmaAttr;
    if (dmaSubtile.size()) {
      dmaAttr.set("subtile_size", builder.getArrayAttr(dmaSubtile));
    }
    dmaAttr.set("async", builder.getBoolAttr(dmaAsync));
    dmaAttr.set("transpose", builder.getBoolAttr(is_transpose(mvin_bias)));

    auto loc = mvin_bias.getLoc();
    builder.setInsertionPoint(mvin_bias);
    auto loopN = builder.create<affine::AffineForOp>(loc, 0, tileSizeN, subTileSizeM);
    loopN->setAttr("inner_loop", builder.getBoolAttr(true));
    builder.setInsertionPointToStart(loopN.getBody());
    j = loopN.getInductionVar();
    auto loopM = builder.create<affine::AffineForOp>(loc, 0, tileSizeM, subTileSizeN);
    loopM->setAttr("inner_loop", builder.getBoolAttr(true));
    builder.setInsertionPointToStart(loopM.getBody());
    i = loopM.getInductionVar();
    srcIndices = mvin_bias.getSrcIndices();
    for (auto index : srcIndices) {
      if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
        new_map = applyOp.getAffineMap();
      }
    }
    new_map_indices = {i, j}; // bmm has no bias
    Value new_idx;
    if (new_map) {
      new_idx = builder.create<affine::AffineApplyOp>(loc, new_map, new_map_indices);
    } else {
      new_idx = j;
    }
    new_idx = builder.create<affine::AffineApplyOp>(loc, sum_map, ValueRange{new_idx, srcIndices[0]});
    src_indices.push_back(new_idx);
    auto num_elt_per_stride = mvin_bias.getNumElementsPerStride();
    uint64_t chunk_size = getConstantIntValue(num_elt_per_stride);
    auto new_set = builder.create<arith::ConstantIndexOp>(loc, 2147483648 + chunk_size);
    AffineMap spad_map_m = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * (tileSizeM / vectorlane) + builder.getAffineDimExpr(1));
    auto dst_idx = builder.create<affine::AffineApplyOp>(loc, spad_map_m, ValueRange{j, i});

    dst_indices.push_back(zeroIndex);
    dst_indices.push_back(dst_idx);
    tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, tag_idx_map, j));
    tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, tag_idx_map, i));
    auto src_map = builder.getMultiDimIdentityMap(src_indices.size());
    auto dst_map = builder.getMultiDimIdentityMap(dst_indices.size());
    auto tag_map = builder.getMultiDimIdentityMap(tag_indices.size());

    // Insert the first dma_start operation
    builder.create<affine::AffineDmaStartOp>(
        loc, mvin_bias.getSrcMemRef(), src_map, src_indices,
        mvin_bias.getDstMemRef(), dst_map, dst_indices,
        BtagMemRef, tag_map, tag_indices,
        mvin_bias.getNumElements(), mvin_bias.getStride(), new_set, dmaAttr);
    mvin_bias.erase();
    src_indices.clear();
    dst_indices.clear();
    tag_indices.clear();
  }

  // Get insertion point for new loops
  auto loc = mvin_input.getLoc();
  builder.setInsertionPoint(mvin_input);

  // Create three nested affine.for loops
  auto loopN = builder.create<affine::AffineForOp>(loc, 0, tileSizeN, subTileSizeN);
  loopN->setAttr("inner_loop", builder.getBoolAttr(true));
  builder.setInsertionPointToStart(loopN.getBody());
  j = loopN.getInductionVar();
  auto loopK = builder.create<affine::AffineForOp>(loc, 0, tileSizeK, subTileSizeK);
  loopK->setAttr("inner_loop", builder.getBoolAttr(true));
  builder.setInsertionPointToStart(loopK.getBody());
  k = loopK.getInductionVar();
  auto loopM = builder.create<affine::AffineForOp>(loc, 0, tileSizeM, subTileSizeM);
  loopM->setAttr("inner_loop", builder.getBoolAttr(true));
  builder.setInsertionPointToStart(loopM.getBody());
  i = loopM.getInductionVar();

  srcIndices = mvin_input.getSrcIndices();
  for (auto index : srcIndices) {
    if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
      new_map = applyOp.getAffineMap();
    }
  }
  if (is_bmm) {
    new_map_indices = {zeroIndex, i, k}; // other approach is make sub map using only i, k
  } else {
    new_map_indices = {i, k};
  }
  auto new_idx = builder.create<affine::AffineApplyOp>(loc, new_map, new_map_indices);
  new_idx = builder.create<affine::AffineApplyOp>(loc, sum_map, ValueRange{new_idx, srcIndices[0]});
  // update srcIndices
  src_indices.push_back(new_idx);
  auto num_elt_per_stride = mvin_input.getNumElementsPerStride();
  uint64_t chunk_size = getConstantIntValue(num_elt_per_stride);
  auto new_set = builder.create<arith::ConstantIndexOp>(loc, 2147483648 + chunk_size);
  // affine_map<(d0, d1) -> (d0 * (tileSizeM / vectorlane) + d1)>
  AffineMap spad_map_m = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * (tileSizeM / vectorlane) + builder.getAffineDimExpr(1));
  auto dst_idx = builder.create<affine::AffineApplyOp>(loc, spad_map_m, ValueRange{k, i});
  dst_indices.push_back(zeroIndex);
  dst_indices.push_back(dst_idx);
  tag_indices.push_back(zeroIndex);
  tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, tag_idx_map, k));
  tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, tag_idx_map, i));
  auto src_map = builder.getMultiDimIdentityMap(src_indices.size());
  auto dst_map = builder.getMultiDimIdentityMap(dst_indices.size());
  auto tag_map = builder.getMultiDimIdentityMap(tag_indices.size());

  // Insert the first dma_start operation
  builder.create<affine::AffineDmaStartOp>(
      loc, mvin_input.getSrcMemRef(), src_map, src_indices,
      mvin_input.getDstMemRef(), dst_map, dst_indices,
      XtagMemRef, tag_map, tag_indices,
      mvin_input.getNumElements(), mvin_input.getStride(), new_set, dma1Attr);

  // Insert the second dma_start operation
  srcIndices = mvin_weight.getSrcIndices();
  for (auto index : srcIndices) {
    if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
      new_map = applyOp.getAffineMap();
    }
  }
  if (is_bmm) {
    new_map_indices = {zeroIndex, k, j};
  } else {
    new_map_indices = {k, j};
  }
  new_idx = builder.create<affine::AffineApplyOp>(loc, new_map, new_map_indices);
  new_idx = builder.create<affine::AffineApplyOp>(loc, sum_map, ValueRange{new_idx, srcIndices[0]});
  src_indices.clear();
  src_indices.push_back(new_idx);

  num_elt_per_stride = mvin_weight.getNumElementsPerStride();
  chunk_size = getConstantIntValue(num_elt_per_stride);
  new_set = builder.create<arith::ConstantIndexOp>(loc, 2147483648 + chunk_size);
  AffineMap spad_map_k = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * (tileSizeK / vectorlane) + builder.getAffineDimExpr(1));

  dst_idx = builder.create<affine::AffineApplyOp>(loc, spad_map_k, ValueRange{j, k});
  dst_indices.clear();
  dst_indices.push_back(zeroIndex);
  dst_indices.push_back(dst_idx);
  tag_indices.clear();
  tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, tag_idx_map, j));
  tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, tag_idx_map, k));
  tag_indices.push_back(zeroIndex);
  src_map = builder.getMultiDimIdentityMap(src_indices.size());
  dst_map = builder.getMultiDimIdentityMap(dst_indices.size());
  tag_map = builder.getMultiDimIdentityMap(tag_indices.size());
  builder.create<affine::AffineDmaStartOp>(
      loc, mvin_weight.getSrcMemRef(), src_map, src_indices,
      mvin_weight.getDstMemRef(), dst_map, dst_indices,
      WtagMemRef, tag_map, tag_indices,
      mvin_weight.getNumElements(), mvin_weight.getStride(), new_set, dma2Attr);

  // Erase the original dma_start operations
  mvin_input.erase();
  mvin_weight.erase();
}

llvm::SmallVector<mlir::Attribute, 2> DmaFineGrained::getSubtileSize(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Attribute, 2> subtileSizes;
  auto attr = operation->getAttr("subtile_size");
  if (!attr) {
    return subtileSizes; // Return empty SmallVector
  }

  if (auto arrayAttr = llvm::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (auto element : arrayAttr) {
      // Assume the elements are integers
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(element)) {
        subtileSizes.push_back(intAttr);
      } else {
        llvm::errs() << "Unsupported element type in 'subtile_size'.\n";
      }
    }
  }
  return subtileSizes;
}

int DmaFineGrained::getAsyncValue(mlir::Operation *operation) {
  auto attr = operation->getAttr("async");
  if (!attr)
    return 0;

  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getInt(); // Treat non-zero as true
  else
    return 1;
}

int DmaFineGrained::is_transpose(mlir::Operation *operation) {
  auto attr = operation->getAttr("transpose");
  if (!attr)
    return 0;

  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getInt(); // Treat non-zero as true
  else
    return 1;
}

namespace mlir {
namespace test {
void registerDmaFineGrainedPass() { PassRegistration<DmaFineGrained>(); }
} // namespace test
} // namespace mlir