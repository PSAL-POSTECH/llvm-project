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
  llvm::SmallVector<mlir::Attribute> getSubtileSize(mlir::Operation *operation);
  llvm::SmallVector<mlir::Attribute> getSramStride(mlir::Operation *operation);
  int getAsyncValue(mlir::Operation *operation);
  bool traverseOperands(Value op_val, Value input);
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
  int64_t tileSizeK, tileSizeN, tileSizeM, tileSizeH;
  int64_t vectorlane = systolicSize;
  builder.setInsertionPointToStart(&func.front());
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(func.getLoc(), 0);

  // outer loop step modify
  int loopDepth = 0;
  func.walk([&](affine::AffineForOp loop) {
    // Adjust the step size based on loop depth
    if (loopDepth == 0) {
      tileSizeK = loop.getStepAsInt(); // Third loop (e.g., %t_k) uses tileSizeK
    } else if (loopDepth == 1) {
      tileSizeN = loop.getStepAsInt(); // Second loop (e.g., %t_n) uses tileSizeN
    } else if (loopDepth == 2) {
      tileSizeM = loop.getStepAsInt(); // First loop (e.g., %t_m) uses tileSizeM (tileSizeW for CONV2D)
    } else if (loopDepth == 3) {
      tileSizeH = loop.getStepAsInt(); // Fourth loop (e.g., %t_h) uses tileSizeH for CONV2D
    }
    loopDepth++;
  });
  bool is_bmm = false, is_conv2d = false;
  if (loopDepth == 4) { // bmm has 4 loops (b, m, n, k)
    is_bmm = true;
  } else if (loopDepth > 5) { // conv2d has 7 loops (b, kh, kw, oh, ow, oc, ic)
    is_conv2d = true; // batch is not implemented yet
  }

  // inner loop fine-grained dma
  SmallVector<memref::DmaStartOp, 2> dmaOps;
  func.walk([&](memref::DmaStartOp dmaStartOp) {
    dmaOps.push_back(dmaStartOp);
  });

  // check Bias is moved to Output buffer
  bool is_bias = false;
  memref::DmaStartOp mvin_bias;
  memref::DmaStartOp mvin_input;
  memref::DmaStartOp mvin_weight;

  for(auto dmaOp : dmaOps) {
    Value numElements = dmaOp.getNumElements();
    int dmaType = getConstantIntValue(numElements);
    if (dmaType != MVOUT) {
      if (traverseOperands(matmulInput, dmaOp.getDstMemRef())) {
        mvin_input = dmaOp;
      } else if (traverseOperands(matmulWeight, dmaOp.getDstMemRef())) {
        mvin_weight = dmaOp;
      } else if (traverseOperands(matmulResult, dmaOp.getDstMemRef())) {
        mvin_bias = dmaOp;
        is_bias = true;
      }
    }
  }

  Value h, i, j, k;
  SmallVector<Value> src_indices;
  SmallVector<Value> dst_indices;
  SmallVector<Value> tag_indices;
  AffineMap new_map;
  SmallVector<Value> new_map_indices, new_dst_indices;
  ValueRange srcIndices;
  // sum_map = affine_map<(d0, d1) -> (d0 + d1)>
  auto sum_map = AffineMap::get(2, 0, builder.getAffineDimExpr(0) + builder.getAffineDimExpr(1));
  AffineMap tag_idx_map = AffineMap::get(1, 0, builder.getAffineDimExpr(0).floorDiv(vectorlane));
  llvm::SmallVector<mlir::Attribute> dma1Subtile = getSubtileSize(mvin_input);
  llvm::SmallVector<mlir::Attribute> dma2Subtile = getSubtileSize(mvin_weight);
  llvm::SmallVector<mlir::Attribute> dma1SramStrides = getSramStride(mvin_input);
  llvm::SmallVector<mlir::Attribute> dma2SramStrides = getSramStride(mvin_weight);
  int dma1Async = getAsyncValue(mvin_input);
  int dma2Async = getAsyncValue(mvin_weight);

  NamedAttrList dma1Attr;
  NamedAttrList dma2Attr;

  int64_t subTileSizeH, subTileSizeM, subTileSizeN, subTileSizeK;

  // Sanity check
  if (dma1Subtile.size() > 0 && dma2Subtile.size() > 0) {
    if (auto intAttr1 = llvm::dyn_cast<mlir::IntegerAttr>(dma1Subtile.back())) {
      if (auto intAttr2 = llvm::dyn_cast<mlir::IntegerAttr>(dma2Subtile[dma2Subtile.size() - 2]))
        if (intAttr1.getInt() != intAttr2.getInt()) {
          mvin_weight.emitError() << " Not matched: "
                          << "dma1Subtile[-1] = " << intAttr1.getInt()
                          << ", dma2Subtile[-2] = " << intAttr2.getInt()
                          << "\n";
        } else {
          subTileSizeK = intAttr1.getInt();
        }
      else
        mvin_weight.emitError() << "dma2Subtile[-1] is not an IntegerAttr.\n";
    } else
      mvin_input.emitError() << "dma1Subtile[-2] is not an IntegerAttr.\n";
  } else {
    mvin_input.emitError() << "subtile_size attribute required for matmul.\n";
  }
  if (dma1Subtile.size() == 3) {
    subTileSizeH = llvm::dyn_cast<mlir::IntegerAttr>(dma1Subtile[0]).getInt();
  } else if (dma1Subtile.size() > 3 || dma2Subtile.size() > 3) {
    mvin_input.emitError() << "4D subtile_size attribute is not supported.\n";
  }
  if (dma2Subtile.size() == 3) {
    mvin_weight.emitError() << "3D weight subtile size is not supported\n";
  }
  subTileSizeM = llvm::dyn_cast<mlir::IntegerAttr>(dma1Subtile[dma1Subtile.size() - 2]).getInt();
  subTileSizeN = llvm::dyn_cast<mlir::IntegerAttr>(dma2Subtile[1]).getInt();

  if (dma1Subtile.size())
    dma1Attr.set("subtile_size", builder.getArrayAttr(dma1Subtile));
  if (dma2Subtile.size())
    dma2Attr.set("subtile_size", builder.getArrayAttr(dma2Subtile));
  if (dma1SramStrides.size())
    dma1Attr.set("sram_stride", builder.getArrayAttr(dma1SramStrides));
  if (dma2SramStrides.size())
    dma2Attr.set("sram_stride", builder.getArrayAttr(dma2SramStrides));
  dma1Attr.set("async", builder.getIntegerAttr(builder.getI1Type(), dma1Async));
  dma2Attr.set("async", builder.getIntegerAttr(builder.getI1Type(), dma2Async));
  dma1Attr.set("fine_grained", builder.getIntegerAttr(builder.getI1Type(), 1));
  dma2Attr.set("fine_grained", builder.getIntegerAttr(builder.getI1Type(), 1));

  if (is_bias) {
    // BIAS MVIN
    // reset builder location
    llvm::SmallVector<mlir::Attribute> dmaSubtile = getSubtileSize(mvin_bias);
    llvm::SmallVector<mlir::Attribute> dmaSramStrides = getSramStride(mvin_bias);
    int dmaAsync = getAsyncValue(mvin_bias);
    NamedAttrList dmaAttr;
    if (dmaSubtile.size()) {
      dmaAttr.set("subtile_size", builder.getArrayAttr(dmaSubtile));
    }
    if (dmaSramStrides.size()) {
      dmaAttr.set("sram_stride", builder.getArrayAttr(dmaSramStrides));
    }
    dmaAttr.set("async", builder.getBoolAttr(dmaAsync));
    dmaAttr.set("fine_grained", builder.getBoolAttr(true));

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
    AffineMap spad_map_m = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * (tileSizeM / vectorlane) + builder.getAffineDimExpr(1));
    AffineMap new_tag_map = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * (tileSizeM / subTileSizeM) + builder.getAffineDimExpr(1));
    new_map_indices = {builder.create<affine::AffineApplyOp>(loc, tag_idx_map, j), builder.create<affine::AffineApplyOp>(loc, tag_idx_map, i)};
    auto dst_idx = builder.create<affine::AffineApplyOp>(loc, spad_map_m, ValueRange{j, i});

    dst_indices.push_back(zeroIndex);
    dst_indices.push_back(dst_idx);
    tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, new_tag_map, new_map_indices));
    auto src_map = builder.getMultiDimIdentityMap(src_indices.size());
    auto dst_map = builder.getMultiDimIdentityMap(dst_indices.size());
    auto tag_map = builder.getMultiDimIdentityMap(tag_indices.size());
    auto maybeExpandedSrcMap = affine::expandAffineMap(builder, loc, src_map, src_indices);
    auto maybeExpandedDstMap = affine::expandAffineMap(builder, loc, dst_map, dst_indices);
    auto maybeExpandedTagMap = affine::expandAffineMap(builder, loc, tag_map, tag_indices);
    // Insert the first dma_start operation
    builder.create<memref::DmaStartOp>(
        loc, mvin_bias.getSrcMemRef(), *maybeExpandedSrcMap, mvin_bias.getDstMemRef(),
        *maybeExpandedDstMap, mvin_bias.getNumElements(), mvin_bias.getTagMemRef(),
        *maybeExpandedTagMap, mvin_bias.getStride(), mvin_bias.getNumElementsPerStride(),
        dmaAttr);
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
  if (is_conv2d) {
    auto loopH = builder.create<affine::AffineForOp>(loc, 0, tileSizeH, subTileSizeH);
    loopH->setAttr("inner_loop", builder.getBoolAttr(true));
    builder.setInsertionPointToStart(loopH.getBody());
    h = loopH.getInductionVar();
  }

  srcIndices = mvin_input.getSrcIndices();
  for (auto index : srcIndices) {
    if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
      new_map = applyOp.getAffineMap();
    }
  }
  if (is_bmm) {
    new_map_indices = {zeroIndex, i, k}; // other approach is make sub map using only i, k
  } else if (is_conv2d) {
    new_map_indices = {h, i, k};
  } else {
    new_map_indices = {i, k};
  }
  auto new_idx = builder.create<affine::AffineApplyOp>(loc, new_map, new_map_indices);
  new_idx = builder.create<affine::AffineApplyOp>(loc, sum_map, ValueRange{new_idx, srcIndices[0]});
  src_indices.push_back(new_idx);
  AffineMap spad_map_m;
  AffineMap new_tag_map;
  int64_t tag_k_stride = (tileSizeM / subTileSizeM);
  if (is_conv2d) {
    int64_t tag_i_stride = (tileSizeH / subTileSizeH);
    spad_map_m = AffineMap::get(3, 0, builder.getAffineDimExpr(0) + builder.getAffineDimExpr(1) * (tileSizeM / vectorlane) + builder.getAffineDimExpr(2));
    new_dst_indices = {h, k, i}; // FIXME: check the order of indices
    tag_k_stride *= tag_i_stride;
    new_tag_map = AffineMap::get(3, 0 , builder.getAffineDimExpr(0) * tag_k_stride + builder.getAffineDimExpr(1) * tag_i_stride + builder.getAffineDimExpr(2));
    new_map_indices = {builder.create<affine::AffineApplyOp>(loc, tag_idx_map, k), builder.create<affine::AffineApplyOp>(loc, tag_idx_map, i), builder.create<affine::AffineApplyOp>(loc, tag_idx_map, h)};
    dst_indices.push_back(zeroIndex);
  } else {
    // affine_map<(d0, d1) -> (d0 * (tileSizeM / vectorlane) + d1)>
    spad_map_m = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * (tileSizeM / vectorlane) + builder.getAffineDimExpr(1));
    new_dst_indices = {k, i};
    new_tag_map = AffineMap::get(2, 0 , builder.getAffineDimExpr(0) * tag_k_stride + builder.getAffineDimExpr(1));
    new_map_indices = {builder.create<affine::AffineApplyOp>(loc, tag_idx_map, k), builder.create<affine::AffineApplyOp>(loc, tag_idx_map, i)};
  }
  auto dst_idx = builder.create<affine::AffineApplyOp>(loc, spad_map_m, new_dst_indices);
  dst_indices.push_back(zeroIndex);
  dst_indices.push_back(dst_idx);
  tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, new_tag_map, new_map_indices));
  auto src_map = builder.getMultiDimIdentityMap(src_indices.size());
  auto dst_map = builder.getMultiDimIdentityMap(dst_indices.size());
  auto tag_map = builder.getMultiDimIdentityMap(tag_indices.size());
  auto maybeExpandedSrcMap = affine::expandAffineMap(builder, loc, src_map, src_indices);
  auto maybeExpandedDstMap = affine::expandAffineMap(builder, loc, dst_map, dst_indices);
  auto maybeExpandedTagMap = affine::expandAffineMap(builder, loc, tag_map, tag_indices);

  // Insert the first dma_start operation
  builder.create<memref::DmaStartOp>(
      loc, mvin_input.getSrcMemRef(), *maybeExpandedSrcMap, mvin_input.getDstMemRef(),
      *maybeExpandedDstMap, mvin_input.getNumElements(), mvin_input.getTagMemRef(),
      *maybeExpandedTagMap, mvin_input.getStride(), mvin_input.getNumElementsPerStride(),
      dma1Attr);

  // Insert the second dma_start operation
  srcIndices = mvin_weight.getSrcIndices();
  for (auto index : srcIndices) {
    if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
      new_map = applyOp.getAffineMap();
    }
  }
  if (is_bmm) {
    new_map_indices = {zeroIndex, k, j};
  } else if (is_conv2d) {
    new_map_indices = {zeroIndex, k, j};
  } else {
    new_map_indices = {k, j};
  }
  new_idx = builder.create<affine::AffineApplyOp>(loc, new_map, new_map_indices);
  new_idx = builder.create<affine::AffineApplyOp>(loc, sum_map, ValueRange{new_idx, srcIndices[0]});
  src_indices.clear();
  src_indices.push_back(new_idx);

  AffineMap spad_map_k = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * (tileSizeK / vectorlane) + builder.getAffineDimExpr(1));

  dst_idx = builder.create<affine::AffineApplyOp>(loc, spad_map_k, ValueRange{j, k});
  dst_indices.clear();
  dst_indices.push_back(zeroIndex);
  dst_indices.push_back(dst_idx);
  tag_indices.clear();
  new_tag_map = AffineMap::get(2, 0 , builder.getAffineDimExpr(0) * (tileSizeK / subTileSizeK) + builder.getAffineDimExpr(1));
  new_map_indices = {builder.create<affine::AffineApplyOp>(loc, tag_idx_map, j), builder.create<affine::AffineApplyOp>(loc, tag_idx_map, k)};
  tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, new_tag_map, new_map_indices));
  src_map = builder.getMultiDimIdentityMap(src_indices.size());
  dst_map = builder.getMultiDimIdentityMap(dst_indices.size());
  tag_map = builder.getMultiDimIdentityMap(tag_indices.size());
  maybeExpandedSrcMap = affine::expandAffineMap(builder, loc, src_map, src_indices);
  maybeExpandedDstMap = affine::expandAffineMap(builder, loc, dst_map, dst_indices);
  maybeExpandedTagMap = affine::expandAffineMap(builder, loc, tag_map, tag_indices);
  builder.create<memref::DmaStartOp>(
      loc, mvin_weight.getSrcMemRef(), *maybeExpandedSrcMap, mvin_weight.getDstMemRef(),
      *maybeExpandedDstMap, mvin_weight.getNumElements(), mvin_weight.getTagMemRef(),
      *maybeExpandedTagMap, mvin_weight.getStride(), mvin_weight.getNumElementsPerStride(),
      dma2Attr);

  // Erase the original dma_start operations
  mvin_input.erase();
  mvin_weight.erase();
}

bool DmaFineGrained::traverseOperands(Value op_val, Value input) {
  bool found = false;
  if (op_val == input) {
    return true;
  }
  auto operation = op_val.getDefiningOp();
  for (auto operand : operation->getOperands()) {
    found = found | traverseOperands(operand, input);
    if (operand == input) {
      return true;
    }
  }
  return found;
}

llvm::SmallVector<mlir::Attribute> DmaFineGrained::getSubtileSize(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Attribute> subtileSizes;
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

llvm::SmallVector<mlir::Attribute> DmaFineGrained::getSramStride(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Attribute> sram_stride;
  auto attr = operation->getAttr("sram_stride");
  if (!attr) {
    return sram_stride; // Return empty SmallVector
  }

  if (auto arrayAttr = llvm::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (auto element : arrayAttr) {
      // Assume the elements are integers
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(element)) {
        sram_stride.push_back(intAttr);
      } else {
        llvm::errs() << "Unsupported element type in 'sram_stride'.\n";
      }
    }
  }
  return sram_stride;
}

namespace mlir {
namespace test {
void registerDmaFineGrainedPass() { PassRegistration<DmaFineGrained>(); }
} // namespace test
} // namespace mlir