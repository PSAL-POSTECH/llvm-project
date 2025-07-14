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
  void buildDmaOp(OpBuilder &builder, Location loc, memref::DmaStartOp op, SmallVector<Value> src_indices,
                  SmallVector<Value> dst_indices, SmallVector<Value> tag_indices, NamedAttrList attr);
  AffineExpr buildAffineDimExpr(OpBuilder &builder, int idx, int64_t tileSize);
  AffineMap build4DSpadMap(OpBuilder &builder, int64_t tileSize1, int64_t tileSize2, int64_t tileSize3);
};

} // namespace

void DmaFineGrained::runOnOperation() {
  auto func = getOperation();
  OpBuilder builder(func.getContext());
  builder.setInsertionPointToStart(&func.front());
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(func.getLoc(), 0);

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

  // outer loop step modify
  affine::AffineForOp affineLoopM, affineLoopN,affineLoopK;
  int loopDepth = 0;
  func.walk([&](affine::AffineForOp loop) {
    // Adjust the step size based on loop depth
    if (auto attr = loop->getAttrOfType<BoolAttr>("accumulation_loop")) {
      loopDepth++;
    }
    if (auto attr = loop->getAttrOfType<BoolAttr>("outer_loop")) {
      loopDepth++;
    }
    if (auto attr = loop->getAttrOfType<BoolAttr>("inner_loop")) {
      if (attr.getValue())
        loopDepth++;
    }
    // Handle subtile_loop attributes and map them to the correct affine loops
    if (auto subtile_attr = loop->getAttrOfType<StringAttr>("subtile_loop")) {
      std::string subtile_loop_str = subtile_attr.getValue().str();
      if (subtile_loop_str == "m") {
        affineLoopM = loop;
      } else if (subtile_loop_str == "n") {
        affineLoopN = loop;
      } else if (subtile_loop_str == "k") {
        affineLoopK = loop;
      }
    }
  });

  if (!affineLoopM || !affineLoopN || !affineLoopK) {
    func.emitError("Failed to find subtile_loop attribute...");
    return;
  }

  llvm::ArrayRef<int64_t> input_tile_shape, weight_tile_shape, output_tile_shape;
  int64_t tileSizeK = 0, tileSizeN = 0, tileSizeM = 0;
  Value loop_M, loop_N, loop_K;
  int64_t tileSizeK_H = 0, tileSizeK_W = 0, tileSizeH = 0, tileSizeW = 0;
  int64_t tileSizeO_H = 0, tileSizeO_W = 0;

  // Retrieve tile info
  tileSizeK = affineLoopK.getStepAsInt();
  tileSizeN = affineLoopN.getStepAsInt();
  tileSizeM = affineLoopM.getStepAsInt();

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
  // -------------------------------
  // Pattern Matching
  // -------------------------------
  bool is_bmm = false, is_conv2d = false;
  if (loopDepth == 4) { // bmm has 4 loops (b, m, n, k)
    is_bmm = true;
  } else if (loopDepth > 9) { // conv2d has 7 loops (b, kh, kw, oh, ow, oc, ic)
    is_conv2d = true;
  } else if (loopDepth != 3) {
    func.emitError() << "Unsupported loop depth: " << loopDepth;
    return;
  }
  if (!getAsyncValue(mvin_input) & !getAsyncValue(mvin_weight))
    return; // no async dma

  // -------------------------------
  // Code Generation
  // -------------------------------
  Value subLoopVarM, subLoopVarN, subLoopVarK, subLoopVarKW, subLoopVarKH, subLoopVarW, subLoopVarH; // subtile loops indices
  SmallVector<Value> src_indices;
  SmallVector<Value> dst_indices;
  SmallVector<Value> tag_indices;
  AffineMap dram_map, new_spad_map, new_tag_map;
  SmallVector<Value> new_src_indices, new_dst_indices;
  ValueRange srcIndices;
  // sum_map = affine_map<(d0, d1) -> (d0 + d1)>
  auto sum_map = AffineMap::get(2, 0, builder.getAffineDimExpr(0) + builder.getAffineDimExpr(1));
  llvm::SmallVector<mlir::Attribute> dma1Subtile = getSubtileSize(mvin_input);
  llvm::SmallVector<mlir::Attribute> dma2Subtile = getSubtileSize(mvin_weight);
  llvm::SmallVector<mlir::Attribute> dma1SramStrides = getSramStride(mvin_input);
  llvm::SmallVector<mlir::Attribute> dma2SramStrides = getSramStride(mvin_weight);
  llvm::SmallVector<mlir::Attribute> dma1DramStrides = getDramStride(mvin_input);
  llvm::SmallVector<mlir::Attribute> dma2DramStrides = getDramStride(mvin_weight);
  int dma1Async = getAsyncValue(mvin_input);
  int dma2Async = getAsyncValue(mvin_weight);
  NamedAttrList dma1Attr;
  NamedAttrList dma2Attr;

  int64_t subTileSizeM, subTileSizeN, subTileSizeK, subTileSizeK_H, subTileSizeK_W, subTileSizeH, subTileSizeW, subTileSizeO_H, subTileSizeO_W;
  bool is_weight_4d_subtile = dma2Subtile.size() == 4;
  bool is_input_4d_subtile = dma1Subtile.size() == 4;

  // Sanity check
  if (dma1Subtile.size() > 1 && dma2Subtile.size() > 1) {
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
  if (is_input_4d_subtile) {
    subTileSizeH = llvm::dyn_cast<mlir::IntegerAttr>(dma1Subtile[0]).getInt();
    subTileSizeW = llvm::dyn_cast<mlir::IntegerAttr>(dma1Subtile[1]).getInt();
    auto dstMemRefType = cast<MemRefType>(mvin_input.getDstMemRef().getType());
    input_tile_shape = dstMemRefType.getShape();
    tileSizeH = input_tile_shape[0];
    tileSizeW = input_tile_shape[1];
    tileSizeM = input_tile_shape[2];
    tileSizeK = input_tile_shape[3];
  } else if (dma1Subtile.size() > 4) {
    mvin_input.emitError() << "more than 4D input attribute is not supported.\n";
  }
  if (is_weight_4d_subtile) {
    subTileSizeK_H = llvm::dyn_cast<mlir::IntegerAttr>(dma2Subtile[0]).getInt();
    subTileSizeK_W = llvm::dyn_cast<mlir::IntegerAttr>(dma2Subtile[1]).getInt();
    auto dstMemRefType = cast<MemRefType>(mvin_weight.getDstMemRef().getType());
    weight_tile_shape = dstMemRefType.getShape();
    tileSizeK_H = weight_tile_shape[0];
    tileSizeK_W = weight_tile_shape[1];
    tileSizeK = weight_tile_shape[2];
    tileSizeN = weight_tile_shape[3];
  } else if (dma2Subtile.size() > 4) {
    mvin_weight.emitError() << "more than 4D weight attribute is not supported.\n";
  }
  if (is_conv2d && (weight_tile_shape[2] != input_tile_shape[3])) {
    mvin_weight.emitError() << "Weight K dimension must match Input K dimension.\n";
  }
  subTileSizeM = llvm::dyn_cast<mlir::IntegerAttr>(dma1Subtile[dma1Subtile.size() - 2]).getInt();
  subTileSizeN = llvm::dyn_cast<mlir::IntegerAttr>(dma2Subtile.back()).getInt();

  if (dma1Subtile.size())
    dma1Attr.set("subtile_size", builder.getArrayAttr(dma1Subtile));
  if (dma2Subtile.size())
    dma2Attr.set("subtile_size", builder.getArrayAttr(dma2Subtile));
  if (dma1SramStrides.size())
    dma1Attr.set("sram_stride", builder.getArrayAttr(dma1SramStrides));
  if (dma2SramStrides.size())
    dma2Attr.set("sram_stride", builder.getArrayAttr(dma2SramStrides));
  if (dma1DramStrides.size())
    dma1Attr.set("dram_stride", builder.getArrayAttr(dma1DramStrides));
  if (dma2DramStrides.size())
    dma2Attr.set("dram_stride", builder.getArrayAttr(dma2DramStrides));
  dma1Attr.set("async", builder.getIntegerAttr(builder.getI1Type(), dma1Async));
  dma2Attr.set("async", builder.getIntegerAttr(builder.getI1Type(), dma2Async));
  dma1Attr.set("fine_grained", builder.getIntegerAttr(builder.getI1Type(), 1));
  dma2Attr.set("fine_grained", builder.getIntegerAttr(builder.getI1Type(), 1));

  if (is_bias) {
    // BIAS MVIN
    llvm::SmallVector<mlir::Attribute> dmaSubtile = getSubtileSize(mvin_bias);
    llvm::SmallVector<mlir::Attribute> dmaSramStrides = getSramStride(mvin_bias);
    llvm::SmallVector<mlir::Attribute> dmaDramStrides = getDramStride(mvin_bias);
    int dmaAsync = getAsyncValue(mvin_bias);
    NamedAttrList dmaAttr;
    if (dmaSubtile.size()) {
      dmaAttr.set("subtile_size", builder.getArrayAttr(dmaSubtile));
    }
    if (dmaSramStrides.size()) {
      dmaAttr.set("sram_stride", builder.getArrayAttr(dmaSramStrides));
    }
    if (dmaDramStrides.size()) {
      dmaAttr.set("dram_stride", builder.getArrayAttr(dmaDramStrides));
    }
    dmaAttr.set("async", builder.getBoolAttr(dmaAsync));
    dmaAttr.set("fine_grained", builder.getBoolAttr(true));

    auto loc = mvin_bias.getLoc();
    builder.setInsertionPoint(mvin_bias);
    auto dstMemRefType = cast<MemRefType>(mvin_bias.getDstMemRef().getType());
    output_tile_shape = dstMemRefType.getShape();
    if (is_conv2d) {
      tileSizeO_H = output_tile_shape[2];
      tileSizeO_W = output_tile_shape[3];
      subTileSizeO_H = llvm::dyn_cast<mlir::IntegerAttr>(dmaSubtile[2]).getInt();
      subTileSizeO_W = llvm::dyn_cast<mlir::IntegerAttr>(dmaSubtile[3]).getInt();
      // Create 2 nested affine.for loops for O_H x O_W Outputs
      auto subLoopH = builder.create<affine::AffineForOp>(loc, 0, tileSizeO_H, subTileSizeO_H);
      builder.setInsertionPointToStart(subLoopH.getBody());
      auto subLoopW = builder.create<affine::AffineForOp>(loc, 0, tileSizeO_W, subTileSizeO_W);
      builder.setInsertionPointToStart(subLoopW.getBody());
      subLoopH->setAttr("inner_loop", builder.getBoolAttr(true));
      subLoopW->setAttr("inner_loop", builder.getBoolAttr(true));
      subLoopVarH = subLoopH.getInductionVar();
      subLoopVarW = subLoopW.getInductionVar();
    }
    auto subLoopN = builder.create<affine::AffineForOp>(loc, 0, tileSizeN, subTileSizeN);
    builder.setInsertionPointToStart(subLoopN.getBody());
    auto subLoopM = builder.create<affine::AffineForOp>(loc, 0, tileSizeM, subTileSizeM);
    builder.setInsertionPointToStart(subLoopM.getBody());
    subLoopN->setAttr("inner_loop", builder.getBoolAttr(true));
    subLoopM->setAttr("inner_loop", builder.getBoolAttr(true));
    subLoopVarN = subLoopN.getInductionVar();
    subLoopVarM = subLoopM.getInductionVar();
    srcIndices = mvin_bias.getSrcIndices();
    for (auto index : srcIndices) {
      if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
        dram_map = applyOp.getAffineMap();
      }
    }
    new_src_indices = {subLoopVarM, subLoopVarN}; // bmm has no bias
    Value dram_idx;
    if (dram_map) {
      // FiXME. This is not a good solution but works for now.
      if (dram_map && dram_map.getNumDims() == 4) {
        new_src_indices.push_back(zeroIndex);
        new_src_indices.push_back(zeroIndex);
      }
      dram_idx = builder.create<affine::AffineApplyOp>(loc, dram_map, new_src_indices);
    } else if (srcIndices[0] == affineLoopN.getInductionVar()) {
      dram_idx = subLoopVarN;
    } else if (srcIndices[0] == affineLoopM.getInductionVar()) {
      dram_idx = subLoopVarM;
    } else {
      mvin_bias.emitError("Failed to match bias index");
      return;
    }
    dram_idx = builder.create<affine::AffineApplyOp>(loc, sum_map, ValueRange{dram_idx, srcIndices[0]});
    src_indices.push_back(dram_idx);
    int64_t tag_j_stride = ((tileSizeM+subTileSizeM-1) / subTileSizeM);
    if (is_conv2d) {
      dst_indices.push_back(zeroIndex);
      dst_indices.push_back(zeroIndex);
      new_spad_map = build4DSpadMap(builder, tileSizeO_W, tileSizeM, tileSizeN);
      new_dst_indices = {subLoopVarH, subLoopVarW, subLoopVarN, subLoopVarM};
      int64_t tag_w_stride = tag_j_stride * ((tileSizeN+subTileSizeN-1) / subTileSizeN);
      int64_t tag_h_stride = tag_w_stride * ((tileSizeO_W+subTileSizeO_W-1) / subTileSizeO_W);
      new_tag_map = AffineMap::get(4, 0, builder.getAffineDimExpr(0) * tag_h_stride + builder.getAffineDimExpr(1) * tag_w_stride + builder.getAffineDimExpr(2) * tag_j_stride + builder.getAffineDimExpr(3));
    } else {
      new_spad_map = AffineMap::get(2, 0, buildAffineDimExpr(builder, 0, tileSizeM) + builder.getAffineDimExpr(1));
      new_tag_map = AffineMap::get(2, 0, builder.getAffineDimExpr(0) * tag_j_stride + builder.getAffineDimExpr(1));
      new_dst_indices = {subLoopVarN, subLoopVarM};
    }
    auto dst_idx = builder.create<affine::AffineApplyOp>(loc, new_spad_map, new_dst_indices);

    dst_indices.push_back(zeroIndex);
    dst_indices.push_back(dst_idx);
    tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, new_tag_map, new_dst_indices));
    buildDmaOp(builder, loc, mvin_bias, src_indices, dst_indices, tag_indices, dmaAttr);

    mvin_bias.erase();
    src_indices.clear();
    dst_indices.clear();
    tag_indices.clear();
  }

  // Get insertion point for new loops
  auto loc = mvin_input.getLoc();
  builder.setInsertionPointAfter(mvin_weight);

  if (is_weight_4d_subtile && is_input_4d_subtile && is_conv2d) {
    // Create 4 nested affine.for loops for KxK CONV kernels & HxW Outputs
    auto loopK_H = builder.create<affine::AffineForOp>(loc, 0, tileSizeK_H, subTileSizeK_H);
    builder.setInsertionPointToStart(loopK_H.getBody());
    auto loopK_W = builder.create<affine::AffineForOp>(loc, 0, tileSizeK_W, subTileSizeK_W);
    builder.setInsertionPointToStart(loopK_W.getBody());
    auto loopH = builder.create<affine::AffineForOp>(loc, 0, tileSizeH, subTileSizeH);
    builder.setInsertionPointToStart(loopH.getBody());
    auto loopW = builder.create<affine::AffineForOp>(loc, 0, tileSizeW, subTileSizeW);
    builder.setInsertionPointToStart(loopW.getBody());
    loopK_H->setAttr("inner_loop", builder.getBoolAttr(true));
    loopK_W->setAttr("inner_loop", builder.getBoolAttr(true));
    loopH->setAttr("inner_loop", builder.getBoolAttr(true));
    loopW->setAttr("inner_loop", builder.getBoolAttr(true));
    subLoopVarKH = loopK_H.getInductionVar();
    subLoopVarKW = loopK_W.getInductionVar();
    subLoopVarH = loopH.getInductionVar();
    subLoopVarW = loopW.getInductionVar();
  }
  // Create three nested affine.for loops
  auto subLoopM = builder.create<affine::AffineForOp>(loc, 0, tileSizeM, subTileSizeM);
  builder.setInsertionPointToStart(subLoopM.getBody());
  auto subLoopK = builder.create<affine::AffineForOp>(loc, 0, tileSizeK, subTileSizeK);
  builder.setInsertionPointToStart(subLoopK.getBody());
  auto subLoopN = builder.create<affine::AffineForOp>(loc, 0, tileSizeN, subTileSizeN);
  builder.setInsertionPointToStart(subLoopN.getBody());
  subLoopM->setAttr("inner_loop", builder.getBoolAttr(true));
  subLoopK->setAttr("inner_loop", builder.getBoolAttr(true));
  subLoopN->setAttr("inner_loop", builder.getBoolAttr(true));
  subLoopVarM = subLoopM.getInductionVar();
  subLoopVarK = subLoopK.getInductionVar();
  subLoopVarN = subLoopN.getInductionVar();

  // src_indices = dram index, dst_indices = spad index
  // calculate the dram address
  srcIndices = mvin_input.getSrcIndices();
  for (auto index : srcIndices) {
    if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
      dram_map = applyOp.getAffineMap();
    }
  }
  if (is_bmm) {
    new_src_indices = {zeroIndex, subLoopVarM, subLoopVarK}; // other approach is make sub map using only i, k
  } else if (is_input_4d_subtile && is_conv2d) {
    new_src_indices = {subLoopVarH, subLoopVarW, subLoopVarM, subLoopVarK};
  } else if (is_conv2d) {
    new_src_indices = {zeroIndex, zeroIndex, subLoopVarM, subLoopVarK};
  } else {
    new_src_indices = {subLoopVarM, subLoopVarK};
  }
  auto dram_idx = builder.create<affine::AffineApplyOp>(loc, dram_map, new_src_indices);
  // Total dram idx = Big Tile idx(srcIndices[0]) + Sub Tile idx(dram_idx)
  dram_idx = builder.create<affine::AffineApplyOp>(loc, sum_map, ValueRange{dram_idx, srcIndices[0]});
  src_indices.push_back(dram_idx);

  // calculate the spad address & tag idx
  int64_t tag_k_stride = ((tileSizeM+subTileSizeM-1) / subTileSizeM);
  if (is_input_4d_subtile) {
    new_spad_map = build4DSpadMap(builder, tileSizeW, tileSizeM, tileSizeK);
    new_dst_indices = {subLoopVarH, subLoopVarW, subLoopVarK, subLoopVarM};
    int64_t tag_w_stride = tag_k_stride * ((tileSizeK+subTileSizeK-1) / subTileSizeK);
    int64_t tag_h_stride = tag_w_stride * ((tileSizeW+subTileSizeW-1) / subTileSizeW);
    new_tag_map = AffineMap::get(4, 0, builder.getAffineDimExpr(0) * tag_h_stride + builder.getAffineDimExpr(1) * tag_w_stride + builder.getAffineDimExpr(2) * tag_k_stride + builder.getAffineDimExpr(3));
  } else {
    new_spad_map = AffineMap::get(2, 0, buildAffineDimExpr(builder, 0, tileSizeM) + builder.getAffineDimExpr(1));
    new_dst_indices = {subLoopVarK, subLoopVarM};
    new_tag_map = AffineMap::get(2, 0 , builder.getAffineDimExpr(0) * tag_k_stride + builder.getAffineDimExpr(1));
  }
  auto dst_idx = builder.create<affine::AffineApplyOp>(loc, new_spad_map, new_dst_indices);
  for (int i=0; i<dma1Subtile.size()-1;i++)
    dst_indices.push_back(zeroIndex);
  dst_indices.push_back(dst_idx);
  tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, new_tag_map, new_dst_indices));
  buildDmaOp(builder, loc, mvin_input, src_indices, dst_indices, tag_indices, dma1Attr);
  src_indices.clear();
  dst_indices.clear();
  tag_indices.clear();

  // Insert the second dma_start operation
  srcIndices = mvin_weight.getSrcIndices();
  for (auto index : srcIndices) {
    if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
      dram_map = applyOp.getAffineMap();
    }
  }
  if (is_bmm) {
    new_src_indices = {zeroIndex, subLoopVarK, subLoopVarN};
  } else if (is_weight_4d_subtile && is_conv2d) {
    new_src_indices = {subLoopVarKH, subLoopVarKW, subLoopVarK, subLoopVarN};
  } else if (is_conv2d) {
    new_src_indices = {zeroIndex, subLoopVarK, subLoopVarN};
  } else {
    new_src_indices = {subLoopVarK, subLoopVarN};
  }
  dram_idx = builder.create<affine::AffineApplyOp>(loc, dram_map, new_src_indices);
  dram_idx = builder.create<affine::AffineApplyOp>(loc, sum_map, ValueRange{dram_idx, srcIndices[0]});
  src_indices.push_back(dram_idx);;

  int64_t tag_j_stride = ((tileSizeK+subTileSizeK-1) / subTileSizeK);
  if (is_weight_4d_subtile) {
    new_dst_indices = {subLoopVarKH, subLoopVarKW, subLoopVarN, subLoopVarK};
    new_spad_map = build4DSpadMap(builder, tileSizeK_W, tileSizeK, tileSizeN);
    int64_t tag_w_stride = tag_j_stride * ((tileSizeN+subTileSizeN-1) / subTileSizeN);
    int64_t tag_h_stride = tag_w_stride * ((tileSizeK_W+subTileSizeK_W-1) / subTileSizeK_W);
    new_tag_map = AffineMap::get(4, 0, builder.getAffineDimExpr(0) * tag_h_stride + builder.getAffineDimExpr(1) * tag_w_stride + builder.getAffineDimExpr(2) * tag_j_stride + builder.getAffineDimExpr(3));
  } else {
    new_dst_indices = {subLoopVarN, subLoopVarK};
    new_spad_map = AffineMap::get(2, 0, buildAffineDimExpr(builder, 0, tileSizeK) + builder.getAffineDimExpr(1));
    new_tag_map = AffineMap::get(2, 0 , builder.getAffineDimExpr(0) * tag_j_stride + builder.getAffineDimExpr(1));
  }

  dst_idx = builder.create<affine::AffineApplyOp>(loc, new_spad_map, new_dst_indices);
  for (int i=0; i<dma2Subtile.size()-1;i++)
    dst_indices.push_back(zeroIndex);
  dst_indices.push_back(dst_idx);
  tag_indices.push_back(builder.create<affine::AffineApplyOp>(loc, new_tag_map, new_dst_indices));
  buildDmaOp(builder, loc, mvin_weight, src_indices, dst_indices, tag_indices, dma2Attr);

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

void DmaFineGrained::buildDmaOp(OpBuilder &builder, Location loc, memref::DmaStartOp op, SmallVector<Value> src_indices,
                                SmallVector<Value> dst_indices, SmallVector<Value> tag_indices, NamedAttrList attr) {
  auto src_map = builder.getMultiDimIdentityMap(src_indices.size());
  auto dst_map = builder.getMultiDimIdentityMap(dst_indices.size());
  auto tag_map = builder.getMultiDimIdentityMap(tag_indices.size());
  auto maybeExpandedSrcMap = affine::expandAffineMap(builder, loc, src_map, src_indices);
  auto maybeExpandedDstMap = affine::expandAffineMap(builder, loc, dst_map, dst_indices);
  auto maybeExpandedTagMap = affine::expandAffineMap(builder, loc, tag_map, tag_indices);
  builder.create<memref::DmaStartOp>(
      loc, op.getSrcMemRef(), *maybeExpandedSrcMap, op.getDstMemRef(),
      *maybeExpandedDstMap, op.getNumElements(), op.getTagMemRef(),
      *maybeExpandedTagMap, op.getStride(), op.getNumElementsPerStride(),
      attr);
}

AffineExpr DmaFineGrained::buildAffineDimExpr(OpBuilder &builder, int idx, int64_t tileSize) {
  if (tileSize / systolicSize == 0)
    return builder.getAffineDimExpr(idx).floorDiv(systolicSize / tileSize);
  else
    return builder.getAffineDimExpr(idx) * (tileSize / systolicSize);
}

AffineMap DmaFineGrained::build4DSpadMap(OpBuilder &builder, int64_t tileSize1, int64_t tileSize2, int64_t tileSize3) {
  int64_t vectorlane = systolicSize;
  int64_t spad_1_stride = tileSize2 * ((tileSize3 + vectorlane - 1) / vectorlane);
  int64_t spad_0_stride = spad_1_stride * tileSize1;
  AffineMap spad_map = AffineMap::get(4, 0, builder.getAffineDimExpr(0) * spad_0_stride + builder.getAffineDimExpr(1) * spad_1_stride + buildAffineDimExpr(builder, 2, tileSize2) + builder.getAffineDimExpr(3));
  return spad_map;
}

namespace mlir {
namespace test {
void registerDmaFineGrainedPass() { PassRegistration<DmaFineGrained>(); }
} // namespace test
} // namespace mlir