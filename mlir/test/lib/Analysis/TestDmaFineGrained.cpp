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
};

} // namespace

void DmaFineGrained::runOnOperation() {
  auto func = getOperation();
  OpBuilder builder(func.getContext());
  bool hasMatmul = false;
  func.walk([&](linalg::MatmulOp matmulOp) {
    hasMatmul = true;
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
  bool is_bias = false;
  if (dmaOps.size() == 4) {
    is_bias = true;
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

  if (is_bias) {
    // BIAS MVIN
    // reset builder location
    affine::AffineDmaStartOp dma = dmaOps[0];
    llvm::SmallVector<mlir::Attribute, 2> dmaSubtile = getSubtileSize(dma);
    int dmaAsync = getAsyncValue(dma);
    NamedAttrList dmaAttr;
    if (dmaSubtile.size()) {
      dmaAttr.set("subtile_size", builder.getArrayAttr(dmaSubtile));
    }
    dmaAttr.set("async", builder.getBoolAttr(dmaAsync));

    auto loc = dma.getLoc();
    builder.setInsertionPoint(dma);
    auto loopN = builder.create<affine::AffineForOp>(loc, 0, tileSizeN, vectorlane);
    loopN->setAttr("inner_loop", builder.getBoolAttr(true));
    builder.setInsertionPointToStart(loopN.getBody());
    j = loopN.getInductionVar();
    auto loopM = builder.create<affine::AffineForOp>(loc, 0, tileSizeM, vectorlane);
    loopM->setAttr("inner_loop", builder.getBoolAttr(true));
    builder.setInsertionPointToStart(loopM.getBody());
    i = loopM.getInductionVar();
    srcIndices = dma.getSrcIndices();
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
    auto num_elt_per_stride = dma.getNumElementsPerStride();
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
        loc, dma.getSrcMemRef(), src_map, src_indices,
        dma.getDstMemRef(), dst_map, dst_indices,
        BtagMemRef, tag_map, tag_indices,
        dma.getNumElements(), dma.getStride(), new_set, dmaAttr);
    dma.erase();
    src_indices.clear();
    dst_indices.clear();
    tag_indices.clear();
  }


  affine::AffineDmaStartOp dma1 = dmaOps[0 + is_bias];
  affine::AffineDmaStartOp dma2 = dmaOps[1 + is_bias];
  affine::AffineDmaStartOp dma3 = dmaOps[2 + is_bias];
  llvm::SmallVector<mlir::Attribute, 2> dma1Subtile = getSubtileSize(dma1);
  llvm::SmallVector<mlir::Attribute, 2> dma2Subtile = getSubtileSize(dma2);
  llvm::SmallVector<mlir::Attribute, 2> dma3Subtile = getSubtileSize(dma3);
  int dma1Async = getAsyncValue(dma1);
  int dma2Async = getAsyncValue(dma2);
  int dma3Async = getAsyncValue(dma3);
  NamedAttrList dma1Attr;
  NamedAttrList dma2Attr;
  NamedAttrList dma3Attr;

  if (dma1Subtile.size())
    dma1Attr.set("subtile_size", builder.getArrayAttr(dma1Subtile));
  if (dma2Subtile.size())
    dma2Attr.set("subtile_size", builder.getArrayAttr(dma2Subtile));
  if (dma3Subtile.size())
    dma3Attr.set("subtile_size", builder.getArrayAttr(dma3Subtile));
  dma1Attr.set("async", builder.getIntegerAttr(builder.getI1Type(), dma1Async));
  dma2Attr.set("async", builder.getIntegerAttr(builder.getI1Type(), dma2Async));
  dma3Attr.set("async", builder.getIntegerAttr(builder.getI1Type(), dma3Async));

  // Get insertion point for new loops
  auto loc = dma1.getLoc();
  builder.setInsertionPoint(dma1);

  // Create three nested affine.for loops
  auto loopN = builder.create<affine::AffineForOp>(loc, 0, tileSizeN, vectorlane);
  loopN->setAttr("inner_loop", builder.getBoolAttr(true));
  builder.setInsertionPointToStart(loopN.getBody());
  j = loopN.getInductionVar();
  auto loopK = builder.create<affine::AffineForOp>(loc, 0, tileSizeK, tileSizeK);
  loopK->setAttr("inner_loop", builder.getBoolAttr(true));
  builder.setInsertionPointToStart(loopK.getBody());
  k = loopK.getInductionVar();
  auto loopM = builder.create<affine::AffineForOp>(loc, 0, tileSizeM, vectorlane);
  loopM->setAttr("inner_loop", builder.getBoolAttr(true));
  builder.setInsertionPointToStart(loopM.getBody());
  i = loopM.getInductionVar();

  srcIndices = dma1.getSrcIndices();
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
  auto num_elt_per_stride = dma1.getNumElementsPerStride();
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
      loc, dma1.getSrcMemRef(), src_map, src_indices,
      dma1.getDstMemRef(), dst_map, dst_indices,
      XtagMemRef, tag_map, tag_indices,
      dma1.getNumElements(), dma1.getStride(), new_set, dma1Attr);

  // Insert the second dma_start operation
  srcIndices = dma2.getSrcIndices();
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

  num_elt_per_stride = dma2.getNumElementsPerStride();
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
      loc, dma2.getSrcMemRef(), src_map, src_indices,
      dma2.getDstMemRef(), dst_map, dst_indices,
      WtagMemRef, tag_map, tag_indices,
      dma2.getNumElements(), dma2.getStride(), new_set, dma2Attr);

  // Erase the original dma_start operations
  dma1.erase();
  dma2.erase();

  //MVOUT
  // reset builder location
  loc = dma3.getLoc();
  builder.setInsertionPoint(dma3);
  // Create two nested affine.for loops
  auto loopN2 = builder.create<affine::AffineForOp>(loc, 0, tileSizeN, vectorlane);
  loopN2->setAttr("inner_loop", builder.getBoolAttr(true));
  builder.setInsertionPointToStart(loopN2.getBody());
  j = loopN2.getInductionVar();
  auto loopM2 = builder.create<affine::AffineForOp>(loc, 0, tileSizeM, vectorlane);
  loopM2->setAttr("inner_loop", builder.getBoolAttr(true));
  builder.setInsertionPointToStart(loopM2.getBody());
  i = loopM2.getInductionVar();


  auto dstIndices = dma3.getDstIndices();
  for (auto index : dstIndices) {
    if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
      new_map = applyOp.getAffineMap();
    }
  }
  if (is_bmm) {
    new_map_indices = {zeroIndex, i, j};
  } else {
    new_map_indices = {i, j};
  }
  new_idx = builder.create<affine::AffineApplyOp>(loc, new_map, new_map_indices);
  new_idx = builder.create<affine::AffineApplyOp>(loc, sum_map, ValueRange{new_idx, dstIndices[0]});
  dst_indices.clear();
  dst_indices.push_back(new_idx);
  auto src_idx = builder.create<affine::AffineApplyOp>(loc, spad_map_m, ValueRange{j, i});
  src_indices.clear();
  src_indices.push_back(zeroIndex);
  src_indices.push_back(src_idx);
  tag_indices.clear();
  tag_indices.push_back(zeroIndex);

  src_map = builder.getMultiDimIdentityMap(src_indices.size());
  dst_map = builder.getMultiDimIdentityMap(dst_indices.size());
  tag_map = builder.getMultiDimIdentityMap(tag_indices.size());
  builder.create<affine::AffineDmaStartOp>(
      loc, dma3.getSrcMemRef(), src_map, src_indices,
      dma3.getDstMemRef(), dst_map, dst_indices,
      dma3.getTagMemRef(), tag_map, tag_indices,
      dma3.getNumElements(), dma3.getStride(), c_set, dma3Attr);
  dma3.erase();
}

llvm::SmallVector<mlir::Attribute, 2> DmaFineGrained::getSubtileSize(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Attribute, 2> subtileSizes;
  auto attr = operation->getAttr("subtile_size");
  if (!attr) {
    llvm::errs() << "'subtile_size' attribute is not set.\n";
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

namespace mlir {
namespace test {
void registerDmaFineGrainedPass() { PassRegistration<DmaFineGrained>(); }
} // namespace test
} // namespace mlir