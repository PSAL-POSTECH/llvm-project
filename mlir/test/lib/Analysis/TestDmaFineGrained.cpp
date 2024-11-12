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
  auto c_set = builder.create<arith::ConstantIndexOp>(func.getLoc(), 2147483650); // FIXME: transpose is not supported
  auto tagMemRefType = MemRefType::get({tileSizeM / vectorlane, tileSizeN / vectorlane, tileSizeK / vectorlane}, builder.getIntegerType(32));
  auto XtagMemRef = builder.create<memref::AllocOp>(func.getLoc(), tagMemRefType);
  auto WtagMemRef = builder.create<memref::AllocOp>(func.getLoc(), tagMemRefType);

  // outer loop step modify
  int loopDepth = 0;
  func.walk([&](affine::AffineForOp loop) {
    // Adjust the step size based on loop depth
    if (loopDepth == 0) {
      loop.setStep(tileSizeK); // First loop (e.g., %t_m) uses tileSizeM
    } else if (loopDepth == 1) {
      loop.setStep(tileSizeN); // Second loop (e.g., %t_n) uses tileSizeN
    } else if (loopDepth == 2) {
      loop.setStep(tileSizeM); // Third loop (e.g., %t_k) uses tileSizeK
    }
    loopDepth++;
  });

  // inner loop fine-grained dma
  SmallVector<affine::AffineDmaStartOp, 2> dmaOps;
  func.walk([&](affine::AffineDmaStartOp dmaStartOp) {
    dmaOps.push_back(dmaStartOp);
  });
  bool is_bias = false;
  if (dmaOps.size() == 4) {
    is_bias = true;
  }
  affine::AffineDmaStartOp dma1 = dmaOps[0 + is_bias];
  affine::AffineDmaStartOp dma2 = dmaOps[1 + is_bias];
  // Get insertion point for new loops
  auto loc = dma1.getLoc();
  builder.setInsertionPoint(dma1);

  // Create three nested affine.for loops
  Value i, j, k;
  auto loopN = builder.create<affine::AffineForOp>(loc, 0, tileSizeN, vectorlane);
  loopN->setAttr("inner_loop", builder.getBoolAttr(true));
  builder.setInsertionPointToStart(loopN.getBody());
  j = loopN.getInductionVar();
  auto loopK = builder.create<affine::AffineForOp>(loc, 0, tileSizeK, vectorlane);
  loopK->setAttr("inner_loop", builder.getBoolAttr(true));
  builder.setInsertionPointToStart(loopK.getBody());
  k = loopK.getInductionVar();
  auto loopM = builder.create<affine::AffineForOp>(loc, 0, tileSizeM, vectorlane);
  loopM->setAttr("inner_loop", builder.getBoolAttr(true));
  builder.setInsertionPointToStart(loopM.getBody());
  i = loopM.getInductionVar();

  SmallVector<Value, 2> src_indices;
  SmallVector<Value, 2> dst_indices;
  SmallVector<Value, 3> tag_indices;
  auto sum_map = AffineMap::get(2, 0, builder.getAffineDimExpr(0) + builder.getAffineDimExpr(1));
  ValueRange srcIndices = dma1.getSrcIndices();
  AffineMap new_map;
  for (auto index : srcIndices) {
    if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
      new_map = applyOp.getAffineMap();
    }
  }
  auto new_idx = builder.create<affine::AffineApplyOp>(loc, new_map, ValueRange{i, k});
  // sum_map = affine_map<(d0, d1) -> (d0 + d1)>
  new_idx = builder.create<affine::AffineApplyOp>(loc, sum_map, ValueRange{new_idx, srcIndices[0]});
  // update srcIndices
  src_indices.push_back(new_idx);
  // affine_map<(d0, d1) -> (d0 * (tileSizeK / vectorlane) + d1)>
  auto num_elt_per_stride = dma1.getNumElementsPerStride();
  uint64_t chunk_size = getConstantIntValue(num_elt_per_stride);
  bool is_transposed = chunk_size > 2; // use x_chunk
  auto new_set = builder.create<arith::ConstantIndexOp>(loc, 2147483648 + chunk_size);
  AffineMap spad_map_m;
  if (is_transposed) {
    spad_map_m = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * (tileSizeK / vectorlane) + builder.getAffineDimExpr(1));
  } else {
    spad_map_m = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * (tileSizeM / vectorlane) + builder.getAffineDimExpr(1));
  }
  auto dst_idx = builder.create<affine::AffineApplyOp>(loc, spad_map_m, ValueRange{k, i});
  dst_indices.push_back(zeroIndex);
  dst_indices.push_back(dst_idx);
  tag_indices.push_back(zeroIndex);
  tag_indices.push_back(k);
  tag_indices.push_back(i);
  auto src_map = builder.getMultiDimIdentityMap(src_indices.size());
  auto dst_map = builder.getMultiDimIdentityMap(dst_indices.size());
  auto tag_map = builder.getMultiDimIdentityMap(tag_indices.size());

  // Insert the first dma_start operation
  builder.create<affine::AffineDmaStartOp>(
      loc, dma1.getSrcMemRef(), src_map, src_indices,
      dma1.getDstMemRef(), dst_map, dst_indices,
      XtagMemRef, tag_map, tag_indices,
      dma1.getNumElements(), dma1.getStride(), new_set);

  // Insert the second dma_start operation
  srcIndices = dma2.getSrcIndices();
  for (auto index : srcIndices) {
    if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
      new_map = applyOp.getAffineMap();
    }
  }
  new_idx = builder.create<affine::AffineApplyOp>(loc, new_map, ValueRange{k, j});
  new_idx = builder.create<affine::AffineApplyOp>(loc, sum_map, ValueRange{new_idx, srcIndices[0]});
  src_indices.clear();
  src_indices.push_back(new_idx);

  num_elt_per_stride = dma2.getNumElementsPerStride();
  chunk_size = getConstantIntValue(num_elt_per_stride);
  new_set = builder.create<arith::ConstantIndexOp>(loc, 2147483648 + chunk_size);
  is_transposed = chunk_size > 2; // use w_chunk
  AffineMap spad_map_k;
  if (is_transposed) {
    spad_map_k = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * (tileSizeN / vectorlane) + builder.getAffineDimExpr(1));
  } else {
    spad_map_k = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * (tileSizeK / vectorlane) + builder.getAffineDimExpr(1));
  }

  dst_idx = builder.create<affine::AffineApplyOp>(loc, spad_map_k, ValueRange{j, k});
  dst_indices.clear();
  dst_indices.push_back(zeroIndex);
  dst_indices.push_back(dst_idx);
  tag_indices.clear();
  tag_indices.push_back(j);
  tag_indices.push_back(k);
  tag_indices.push_back(zeroIndex);
  src_map = builder.getMultiDimIdentityMap(src_indices.size());
  dst_map = builder.getMultiDimIdentityMap(dst_indices.size());
  tag_map = builder.getMultiDimIdentityMap(tag_indices.size());
  builder.create<affine::AffineDmaStartOp>(
      loc, dma2.getSrcMemRef(), src_map, src_indices,
      dma2.getDstMemRef(), dst_map, dst_indices,
      WtagMemRef, tag_map, tag_indices,
      dma2.getNumElements(), dma2.getStride(), new_set);

  // Erase the original dma_start operations
  dma1.erase();
  dma2.erase();

  //MVOUT
  // reset builder location
  affine::AffineDmaStartOp dma3 = dmaOps[2 + is_bias];
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
  new_idx = builder.create<affine::AffineApplyOp>(loc, new_map, ValueRange{i, j});
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
      dma3.getNumElements(), dma3.getStride(), c_set);
  dma3.erase();
}

namespace mlir {
namespace test {
void registerDmaFineGrainedPass() { PassRegistration<DmaFineGrained>(); }
} // namespace test
} // namespace mlir