//===- TestPyTorchSimToVCIXConversion.cpp - Test conversion to gemmini ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/VCIXDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace {

int SYSTOLIC_SIZE = 128;
int VLEN = 128;

int64_t getLoopUpperBound(mlir::affine::AffineForOp forOp) {
  if (auto constantOp = forOp.getUpperBoundMap().getSingleConstantResult()) {
    return constantOp;
  }
  return -1;  // This is an example, handle dynamic cases as needed
}

std::pair<Value, bool> getDramMemRef(mlir::memref::DmaStartOp dmaOp) {
  auto dst_space = dmaOp.getDstMemorySpace();
  auto src_space = dmaOp.getSrcMemorySpace();
  Value dram_memref;
  bool is_write;

  if (dst_space == 0 && src_space == 1) {
    dram_memref = dmaOp.getDstMemRef();
    is_write = true;
  } else if (dst_space == 1 && src_space == 0) {
    dram_memref = dmaOp.getSrcMemRef();
    is_write = false;
  } else {
    dmaOp.emitError() << "Unexpected memory space, src: " << src_space << ", dst: " << dst_space << "\n";
  }
  return std::make_pair(dram_memref, is_write);
}

std::pair<Value, bool> getSramMemRef(mlir::memref::DmaStartOp dmaOp) {
  auto dst_space = dmaOp.getDstMemorySpace();
  auto src_space = dmaOp.getSrcMemorySpace();
  Value sram_memref;
  bool is_write;

  if (dst_space == 0 && src_space == 1) {
    sram_memref = dmaOp.getSrcMemRef();
    is_write = true;
  } else if (dst_space == 1 && src_space == 0) {
    sram_memref = dmaOp.getDstMemRef();
    is_write = false;
  } else {
    dmaOp.emitError() << "Unexpected memory space, src: " << src_space << ", dst: " << dst_space << "\n";
  }
  return std::make_pair(sram_memref, is_write);
}

static std::pair<unsigned, VectorType> legalizeVectorType(const Type &type) {
  VectorType vt = cast<VectorType>(type);
  // To simplify test pass, avoid multi-dimensional vectors.
  if (!vt || vt.getRank() != 1)
    return {0, nullptr};

  Type eltTy = vt.getElementType();
  unsigned sew = 0;
  if (eltTy.isF32())
    sew = 32;
  else if (eltTy.isF64())
    sew = 64;
  else if (auto intTy = dyn_cast<IntegerType>(eltTy))
    sew = intTy.getWidth();
  else
    return {0, nullptr};

  unsigned eltCount = vt.getShape()[0];
  const unsigned lmul = eltCount * sew / 64;

  unsigned n = lmul > 8 ? llvm::Log2_32(lmul) - 2 : 1;
  if (!vt.isScalable()) {
    n = lmul > 32 ? llvm::Log2_32(lmul) - 2 : 1;
    if (n == 1)
      return {n, vt};
    else
      return {n, VectorType::get({VLEN / (sew / 8)}, eltTy)}; // max eltcount = VLEN / element size [byte]
  }
  return {n, VectorType::get({eltCount >> (n - 1)}, eltTy, {true})};
}

bool traverseMMOperands(Value op_val, Value input) {
  bool found = false;
  if (op_val == input) {
    return true;
  }
  auto operation = op_val.getDefiningOp();
  if (operation) {
    for (auto operand : operation->getOperands()) {
      found = found | traverseMMOperands(operand, input);
      if (operand == input) {
        return true;
      }
    }
  }
  return found;
}

struct MatmulOpLowering : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
    // Get the operands
    Value A = op.getInputs()[0];
    Value B = op.getInputs()[1];
    Value C = op.getOutputs()[0];

    Location loc = op.getLoc();

    int vlen = VLEN; //FIXME
    int elen = 0;
    int nr_element;

    // Get input matrix's shape and type
    mlir::MemRefType memRefTypeA, memRefTypeB, memRefTypeC;
    mlir::Type elementTypeA, elementTypeB, elementTypeC;

    memRefTypeA = mlir::dyn_cast<mlir::MemRefType>(A.getType());
    memRefTypeB = mlir::dyn_cast<mlir::MemRefType>(B.getType());
    memRefTypeC = mlir::dyn_cast<mlir::MemRefType>(C.getType());
    if (!memRefTypeA || !memRefTypeB || !memRefTypeC) {
      op.emitError () << "expected MemRefType inputs";
      return failure(true);
    }
    elementTypeA = memRefTypeA.getElementType();
    elementTypeB = memRefTypeB.getElementType();
    elementTypeC = memRefTypeC.getElementType();
    if (elementTypeA != elementTypeB || elementTypeA != elementTypeC || elementTypeB != elementTypeC) {
      op.emitError () << "expected same type";
      return failure(true);
    }

    // Get the dimensions of the input matrices
    int M = memRefTypeA.getShape()[0];
    int K = memRefTypeA.getShape()[1];
    int N = memRefTypeB.getShape()[1];

    // Ensure the dimensions are multiples of SYSTOLIC_SIZE
    if ((M > SYSTOLIC_SIZE && (M % SYSTOLIC_SIZE != 0))) {
      op.emitError() << "M must be multiples of SYSTOLIC_SIZE";
      return failure();
    }
    if ((N > SYSTOLIC_SIZE && (N % SYSTOLIC_SIZE != 0))) {
      op.emitError() << "N must be multiples of SYSTOLIC_SIZE";
      return failure();
    }
    if ((K > SYSTOLIC_SIZE && (K % SYSTOLIC_SIZE != 0))) {
      op.emitError() << "K must be multiples of SYSTOLIC_SIZE";
      return failure();
    }

    if (memRefTypeB.getShape()[0] != K) {
      op.emitError() << "K dimension mismatch: A(" << K << ") != B(" << memRefTypeB.getShape()[0] << ")";
      return failure();
    }


    // Get element size
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementTypeA)) {
      elen = intType.getWidth();
    } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementTypeA)) {
      elen = floatType.getWidth();
    } else {
      return failure();
    }

    nr_element = vlen / elen;

    // Constants
    Value c0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value rvl = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(nr_element));
    Attribute compute_cycle = rewriter.getI64IntegerAttr(4); // FIXME: 5 bits bound & hardcoded

    // Opcode attribute
    Attribute zeroImmAttr = rewriter.getI64IntegerAttr(0);
    Attribute vipush_opcode =  rewriter.getI64IntegerAttr(0b000000);
    Attribute vwpush_opcode =  rewriter.getI64IntegerAttr(0b000001);
    Attribute compute_opcode = rewriter.getI64IntegerAttr(0b000001);
    Attribute vpop_opcode = rewriter.getI64IntegerAttr(0b000010);
    Attribute sew = rewriter.getI64IntegerAttr(elen);
    Attribute lmul = rewriter.getI64IntegerAttr(0); // 0: m1, 1: m2, 2: m4, 3: m8, 5: mf8, 6: mf4, 7: mf2
    auto vectorType = VectorType::get({nr_element}, elementTypeA);
    int nr_m_element = std::max(std::min(M, nr_element), 2); // required 2 elements for vector load/store
    auto vectorMType = VectorType::get({nr_m_element}, elementTypeA);
    Value n_idx;
    Value k_idx;
    Value m_idx;

    Value M_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), M);
    Value K_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), K);
    Value N_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), N);
    Value SYSTOLIC_SIZE_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), SYSTOLIC_SIZE);

    auto spadIdxMap = AffineMap::get(
      /*dimCount=*/3, /*symbolCount=*/2,
      rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(0) +
      rewriter.getAffineDimExpr(1) * rewriter.getAffineSymbolExpr(1) +
      rewriter.getAffineDimExpr(2), // This represents `n_idx * K + k_idx * SYSTOLIC_SIZE + i`
      rewriter.getContext()
    );
    auto spadXIdxMap = AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/1,
      rewriter.getAffineDimExpr(0).floorDiv(rewriter.getAffineSymbolExpr(0)),
      rewriter.getContext()
    );
    auto spadYIdxMap = AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/1,
      rewriter.getAffineDimExpr(0) % rewriter.getAffineSymbolExpr(0),
      rewriter.getContext()
    );
    auto spadIdxMapAttr = mlir::AffineMapAttr::get(spadIdxMap);
    auto spadXIdxMapAttr = mlir::AffineMapAttr::get(spadXIdxMap);
    auto spadYIdxMapAttr = mlir::AffineMapAttr::get(spadYIdxMap);

    // Put dma wait operation
    mlir::Value ADmaTag;
    mlir::Value BDmaTag;
    mlir::Value BiasDmaTag;
    ValueRange BiasDMAIndices;
    mlir::Value numElements = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value OuterKLoopVar;
    int KStep = SYSTOLIC_SIZE;
    int MStep = SYSTOLIC_SIZE;
    int NStep = SYSTOLIC_SIZE;
    std::vector<affine::AffineForOp> accumulationLoops;
    std::vector<affine::AffineForOp> outerLoops;
    std::vector<affine::AffineForOp> innerLoops;

    // Find accumulation loops and set last outerloop
    auto affineForOp = llvm::dyn_cast_or_null<affine::AffineForOp>(op->getParentRegion()->getParentOp());
    while (affineForOp) {
      if (auto attr = affineForOp->getAttrOfType<BoolAttr>("accumulation_loop")) {
        accumulationLoops.insert(accumulationLoops.begin(), affineForOp);
      }
      if (auto attr = affineForOp->getAttrOfType<BoolAttr>("outer_loop")) {
        outerLoops.insert(outerLoops.begin(), affineForOp);
      }
      if (auto attr = affineForOp->getAttrOfType<BoolAttr>("inner_loop")) {
        innerLoops.insert(innerLoops.begin(), affineForOp);
      }
      affineForOp = llvm::dyn_cast_or_null<affine::AffineForOp>(affineForOp->getParentOp());
    }
    assert(accumulationLoops.size()>=1);
    assert(outerLoops.size()>=2);
    // Assume last accumulation loop is K loop
    KStep = accumulationLoops.back().getStep().getZExtValue();
    NStep = outerLoops.at(outerLoops.size()-1).getStep().getZExtValue();
    MStep = outerLoops.at(outerLoops.size()-2).getStep().getZExtValue();

    // Set Last outer loop
    affineForOp = outerLoops.back();
    affineForOp->walk([&](mlir::Operation *nestedOp) {
      if (auto dmaStartOp = llvm::dyn_cast<memref::DmaStartOp>(nestedOp)) { // Replace DMAStartOp with actual `dma_start` op type
        auto result = getDramMemRef(dmaStartOp);
        auto sramRef = getSramMemRef(dmaStartOp);
        bool sramUsedInMatmul = false;

        for (auto operand : op->getOperands()) {
          if (traverseMMOperands(operand, sramRef.first)) { // for CONV2D reshape op, we need to traverse operands
            sramUsedInMatmul = true;
            break;
          }
        }
        /* Only DMA load */
        if (result.second)
          return WalkResult::advance();

        /* Only wait operand dma */
        if (!sramUsedInMatmul)
          return WalkResult::advance();
        auto blockArg = mlir::cast<mlir::BlockArgument>(result.first);
        if (!blockArg)
          return WalkResult::advance();
        if (blockArg.getArgNumber() == 0) {
          ADmaTag = dmaStartOp.getTagMemRef(); // Assuming `getTag()` retrieves the `tag` from `dma_start`.
        } else if (blockArg.getArgNumber() == 1) {
          BDmaTag = dmaStartOp.getTagMemRef(); // Assuming `getTag()` retrieves the `tag` from `dma_start`.
        } else if (blockArg.getArgNumber() == 2) {
          BiasDmaTag = dmaStartOp.getTagMemRef(); // Assuming `getTag()` retrieves the `tag` from `dma_start`.
          BiasDMAIndices = dmaStartOp.getTagIndices();
        }
      }
      return WalkResult::advance();
    });

    if (!ADmaTag || !BDmaTag) {
      op.emitError () << "Failed to locate dma_start for retrieving tag.";
      return failure();
    }

    if (N > SYSTOLIC_SIZE) {
      // N Loop
      auto inner_loop = rewriter.create<affine::AffineForOp>(loc, 0, N/SYSTOLIC_SIZE, 1);
      inner_loop->setAttr("inner_loop", rewriter.getBoolAttr(true));
      rewriter.setInsertionPointToStart(inner_loop.getBody());
      n_idx = inner_loop.getInductionVar();
    } else {
      n_idx = c0;
    }

    Value zero_vector;
    if (K > SYSTOLIC_SIZE) {
      // K Loop
      auto inner_loop = rewriter.create<affine::AffineForOp>(loc, 0, K/SYSTOLIC_SIZE, 1);
      inner_loop->setAttr("inner_loop", rewriter.getBoolAttr(true));
      rewriter.setInsertionPointToStart(inner_loop.getBody());
      k_idx = inner_loop.getInductionVar();
    } else {
      k_idx = c0;
      SmallVector<mlir::Attribute> values(nr_element, rewriter.getFloatAttr(rewriter.getF32Type(), 0.0));
      auto denseAttr = mlir::DenseElementsAttr::get(vectorType, values);
      zero_vector = rewriter.create<arith::ConstantOp>(loc, denseAttr);
    }

    if (M > SYSTOLIC_SIZE) {
      // M Loop
      auto inner_loop = rewriter.create<affine::AffineForOp>(loc, 0, M/SYSTOLIC_SIZE, 1);
      inner_loop->setAttr("inner_loop", rewriter.getBoolAttr(true));
      rewriter.setInsertionPointToStart(inner_loop.getBody());
      m_idx = inner_loop.getInductionVar();
    } else {
      m_idx = c0;
    }

    size_t numAccumulationLoops = accumulationLoops.size();
    // Notice that A, B is dependent to accumlation axis
    mlir::AffineExpr ATagExpr = rewriter.getAffineDimExpr(0) * -1;
    mlir::AffineExpr BTagExpr = rewriter.getAffineDimExpr(0) * -1;
    llvm::SmallVector<mlir::Value, 4> ATagOperands = {accumulationLoops.at(0).getInductionVar()};
    llvm::SmallVector<mlir::Value, 4> BTagOperands = {accumulationLoops.at(0).getInductionVar()};
    for (size_t i = 1; i < numAccumulationLoops; ++i) {
      ATagExpr = ATagExpr + rewriter.getAffineDimExpr(i) * -1;
      BTagExpr = BTagExpr + rewriter.getAffineDimExpr(i) * -1;
      ATagOperands.push_back(accumulationLoops.at(i).getInductionVar());
      BTagOperands.push_back(accumulationLoops.at(i).getInductionVar());
    }
    int ADimOffset = numAccumulationLoops;
    int BDimOffset = numAccumulationLoops;
    if (innerLoops.size()==4) {
      /* FIXME. this is totally heuristic based lowering... */
      ADimOffset = innerLoops.size() + numAccumulationLoops;
      BDimOffset = ADimOffset-2;
      int64_t oW, kW, iW;
      oW = getLoopUpperBound(innerLoops.at(1));
      kW = getLoopUpperBound(innerLoops.at(3));
      iW = oW + kW - 1;
      ATagExpr = ATagExpr + \
        rewriter.getAffineDimExpr(ADimOffset-4)*((K/KStep)*(M/MStep)*iW) + \
        rewriter.getAffineDimExpr(ADimOffset-3)*((K/KStep)*(M/MStep)) + \
        rewriter.getAffineDimExpr(ADimOffset-2)*((K/KStep)*(M/MStep)*iW) + \
        rewriter.getAffineDimExpr(ADimOffset-1)*((K/KStep)*(M/MStep));
      BTagExpr = BTagExpr + \
        rewriter.getAffineDimExpr(BDimOffset-2)*((N/NStep)*(K/KStep)*kW) + \
        rewriter.getAffineDimExpr(BDimOffset-1)*((N/NStep)*(K/KStep));
      /* Add Operands */
      ATagOperands.push_back(innerLoops.at(0).getInductionVar());
      ATagOperands.push_back(innerLoops.at(1).getInductionVar());
      ATagOperands.push_back(innerLoops.at(2).getInductionVar());
      ATagOperands.push_back(innerLoops.at(3).getInductionVar());

      BTagOperands.push_back(innerLoops.at(2).getInductionVar());
      BTagOperands.push_back(innerLoops.at(3).getInductionVar());
    }
    ATagExpr = ATagExpr + rewriter.getAffineDimExpr(ADimOffset)*(M/MStep) + \
      rewriter.getAffineDimExpr(ADimOffset+1).floorDiv((MStep+SYSTOLIC_SIZE-1)/SYSTOLIC_SIZE);
    BTagExpr = BTagExpr + rewriter.getAffineDimExpr(BDimOffset).floorDiv((NStep+SYSTOLIC_SIZE-1)/SYSTOLIC_SIZE)*(K/KStep) + \
      rewriter.getAffineDimExpr(BDimOffset+1)*1;
    ATagExpr.dump();
    BTagExpr.dump();
    auto ATagMap = mlir::AffineMap::get(ADimOffset+2, 0, ATagExpr);
    auto BTagMap = mlir::AffineMap::get(BDimOffset+2, 0, BTagExpr);

    ATagOperands.push_back(c0);    //K_idx, m_idx
    ATagOperands.push_back(m_idx);
    BTagOperands.push_back(n_idx); //N_idx, K_Idx
    BTagOperands.push_back(c0);
    auto ATagIdx = rewriter.create<affine::AffineApplyOp>(loc, ATagMap, ATagOperands);
    auto BTagIdx = rewriter.create<affine::AffineApplyOp>(loc, BTagMap, BTagOperands);
    rewriter.create<memref::DmaWaitOp>(loc, ADmaTag, ValueRange{ATagIdx}, numElements);
    rewriter.create<memref::DmaWaitOp>(loc, BDmaTag, ValueRange{BTagIdx}, numElements);
    if (BiasDmaTag) {
      /* Bias could be 1D or 2D */
      Value first_index = BiasDMAIndices[0].getDefiningOp<mlir::arith::ConstantIndexOp>() ? c0 : n_idx;
      Value third_index = BiasDMAIndices[0].getDefiningOp<mlir::arith::ConstantIndexOp>() ? c0 : m_idx;
      mlir::AffineExpr BiasTagExpr = rewriter.getAffineDimExpr(0).floorDiv((NStep+SYSTOLIC_SIZE-1)/SYSTOLIC_SIZE)*(M/MStep) + rewriter.getAffineDimExpr(1).floorDiv((MStep+SYSTOLIC_SIZE-1)/SYSTOLIC_SIZE); // N, M
      auto BiasTagMap = mlir::AffineMap::get(2, 0, BiasTagExpr);
      auto BiasTagIdx = rewriter.create<affine::AffineApplyOp>(loc, BiasTagMap, ValueRange{first_index, third_index});
      rewriter.create<memref::DmaWaitOp>(loc, BiasDmaTag, ValueRange{BiasTagIdx}, numElements);
    }

    // For vpush weight loop part
    for (int i=0; i<SYSTOLIC_SIZE; i+=nr_element) { // KxN
      Value weight_vector;
      if (i < K) {
        Value i_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), i);
        Value spad_idx = rewriter.create<affine::AffineApplyOp>(loc, spadIdxMapAttr,
                                                        ValueRange{n_idx, k_idx, i_val, K_val, SYSTOLIC_SIZE_val});
        Value w_x_idx = rewriter.create<affine::AffineApplyOp>(loc, spadXIdxMapAttr, ValueRange{spad_idx, N_val});
        Value w_y_idx = rewriter.create<affine::AffineApplyOp>(loc, spadYIdxMapAttr, ValueRange{spad_idx, N_val});
        weight_vector = rewriter.create<vector::TransferReadOp>(
                                            loc, vectorType, B, ValueRange{w_x_idx, w_y_idx});
      } else {
        weight_vector = zero_vector;
      }
      rewriter.create<vcix::BinaryNoDestImmOp>(weight_vector.getLoc(), vwpush_opcode, weight_vector, zeroImmAttr, zeroImmAttr, rvl);
    }

    // For vpush input loop part
    int64_t M_LOOP = M > SYSTOLIC_SIZE ? SYSTOLIC_SIZE : M;
    for (int i=0; i<M_LOOP; i+=nr_element) { // MxK
      Value i_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), i);
      Value spad_idx = rewriter.create<affine::AffineApplyOp>(loc, spadIdxMapAttr,
                                                      ValueRange{k_idx, m_idx, i_val, M_val, SYSTOLIC_SIZE_val});
      Value x_idx = rewriter.create<affine::AffineApplyOp>(loc, spadXIdxMapAttr, ValueRange{spad_idx, K_val});
      Value y_idx = rewriter.create<affine::AffineApplyOp>(loc, spadYIdxMapAttr, ValueRange{spad_idx, K_val});
      auto input_vector = rewriter.create<vector::TransferReadOp>(
                                          loc, vectorMType, A, ValueRange{x_idx, y_idx});
      rewriter.create<vcix::BinaryNoDestImmOp>(input_vector.getLoc(), vipush_opcode, input_vector, zeroImmAttr, zeroImmAttr, rvl);
    }

    // Compute instruction
    rewriter.create<vcix::UnaryNoDestImmOp>(loc, compute_opcode, zeroImmAttr, compute_cycle, zeroImmAttr, sew, lmul, rvl);
    // For vpop loop part
    for (int i=0; i<M_LOOP; i+=nr_element) { // MxN
      Value i_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), i);
      Value spad_idx = rewriter.create<affine::AffineApplyOp>(loc, spadIdxMapAttr,
                                                      ValueRange{n_idx, m_idx, i_val, M_val, SYSTOLIC_SIZE_val});
      Value vpop = rewriter.create<vcix::UnaryImmOp>(loc, vectorMType, vpop_opcode, zeroImmAttr, zeroImmAttr, rvl);
      Value x_idx = rewriter.create<affine::AffineApplyOp>(loc, spadXIdxMapAttr, ValueRange{spad_idx, N_val});
      Value y_idx = rewriter.create<affine::AffineApplyOp>(loc, spadYIdxMapAttr, ValueRange{spad_idx, N_val});
      auto prev_output = rewriter.create<vector::TransferReadOp>(
                                          vpop.getLoc(), vectorMType, C, ValueRange{x_idx, y_idx});
      VectorType vt = cast<VectorType>(prev_output.getType());
      if (vt.getElementType().isInteger()) {
        auto output_vector = rewriter.create<arith::AddIOp>(loc, prev_output, vpop);
        rewriter.create<vector::TransferWriteOp>(output_vector.getLoc(), output_vector, C, ValueRange{x_idx, y_idx});
      }
      else if (vt.getElementType().isIntOrFloat())  {
        auto output_vector = rewriter.create<arith::AddFOp>(loc, prev_output, vpop);
        rewriter.create<vector::TransferWriteOp>(output_vector.getLoc(), output_vector, C, ValueRange{x_idx, y_idx});
      } else {
        op.emitError () << "expected same type";
        return failure();
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct MathExpToVCIX: public OpRewritePattern<math::ExpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(math::ExpOp op, PatternRewriter &rewriter) const override {
    const Type opType = op.getOperand().getType();
    auto [n, legalType] = legalizeVectorType(opType);
    if (!legalType)
      return rewriter.notifyMatchFailure(op, "cannot legalize type for RVV");
    Location loc = op.getLoc();
    Value vec = op.getOperand();
    Attribute zeroImmAttr = rewriter.getI32IntegerAttr(0);
    Attribute opcodeAttr = rewriter.getI64IntegerAttr(0b000011);
    Value rvl = nullptr;
    VectorType vt = cast<VectorType>(opType);
    unsigned totalEltCount = vt.getShape()[0];
    const unsigned eltCount = legalType.getShape()[0];
    if (legalType.isScalable())
      // Use arbitrary runtime vector length when vector type is scalable.
      // Proper conversion pass should take it from the IR.
      rvl = rewriter.create<arith::ConstantOp>(loc,
                                                rewriter.getI64IntegerAttr(9));
    Value res;
    if (n == 1) {
      res = rewriter.create<vcix::BinaryImmOp>(loc, legalType, opcodeAttr, vec,
                                                zeroImmAttr, rvl);
    } else {
      Type eltTy = legalType.getElementType();
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, eltTy, rewriter.getZeroAttr(eltTy));
      res = rewriter.create<vector::BroadcastOp>(loc, opType, zero /*dummy*/);
      if (legalType.isScalable()) {
        for (unsigned i = 0; i < n; ++i) {
          Value extracted = rewriter.create<vector::ScalableExtractOp>(
              loc, legalType, vec, i * eltCount);
          Value v = rewriter.create<vcix::BinaryImmOp>(loc, legalType, opcodeAttr,
                                                        extracted, zeroImmAttr, rvl);
          res = rewriter.create<vector::ScalableInsertOp>(loc, v, res,
                                                          i * eltCount);
        }
      } else { // Fixed-length vector > VLEN
        for (unsigned i = 0; i < totalEltCount/eltCount; i++) {
          Value extracted = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, vec, i * eltCount, eltCount, 1);
          Value v = rewriter.create<vcix::BinaryImmOp>(loc, legalType, opcodeAttr,
                                                        extracted, zeroImmAttr, rvl);
          res = rewriter.create<vector::InsertStridedSliceOp>(loc, v, res,
                                                              i * eltCount, 1);
        }
      }
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};


struct TestPyTorchSimToVCIX
    : PassWrapper<TestPyTorchSimToVCIX, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPyTorchSimToVCIX)
    // Define an integer option with a default value
  StringRef getArgument() const final { return "test-pytorchsim-to-vcix"; }
  StringRef getDescription() const final {
    return "Test lowering patterns that converts linag and spefial function(e.g., exp) to VCIX";
  }
  TestPyTorchSimToVCIX() = default;
  TestPyTorchSimToVCIX(const TestPyTorchSimToVCIX &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect,
                    vcix::VCIXDialect, vector::VectorDialect, affine::AffineDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    SYSTOLIC_SIZE = systolicSize;
    VLEN = vlen;
    patterns.add<MatmulOpLowering, MathExpToVCIX>(ctx);
    ConversionTarget target(getContext());
    target.addIllegalOp<linalg::MatmulOp>();
    target.addIllegalOp<math::ExpOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }


private:
  Option<int> systolicSize{*this, "systolic-array-size",
                          llvm::cl::desc("Systolic array size (KxK)"),
                          llvm::cl::init(128)};
  Option<int> vlen{*this, "vlen",
                   llvm::cl::desc("vector register size(bit)"),
                   llvm::cl::init(128)};
};

} // namespace
namespace test {
void registerTestPyTorchSimToVCIXPass() { PassRegistration<TestPyTorchSimToVCIX>(); }
} // namespace test
} // namespace mlir