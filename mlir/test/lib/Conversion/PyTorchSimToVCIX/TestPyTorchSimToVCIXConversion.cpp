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

std::pair<Value, bool> getDramMemRef(mlir::affine::AffineDmaStartOp dmaOp) {
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

std::pair<Value, bool> getSramMemRef(mlir::affine::AffineDmaStartOp dmaOp) {
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
      return {n, VectorType::get({eltCount >> (n - 2)}, eltTy)};
  }
  return {n, VectorType::get({eltCount >> (n - 1)}, eltTy, {true})};
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

    // Allocate a memref for the reshape shape memref<1xi64>
    Value shapeMemRef = rewriter.create<memref::AllocOp>(loc, MemRefType::get({1}, rewriter.getI64Type()));

    auto reshapedTypeA = MemRefType::get({memRefTypeA.getNumElements()}, memRefTypeA.getElementType(), {}, memRefTypeA.getMemorySpaceAsInt());
    auto reshapedTypeB = MemRefType::get({memRefTypeB.getNumElements()}, memRefTypeB.getElementType(), {}, memRefTypeB.getMemorySpaceAsInt());
    auto reshapedTypeC = MemRefType::get({memRefTypeC.getNumElements()}, memRefTypeC.getElementType(), {}, memRefTypeC.getMemorySpaceAsInt());

    // Reshape A, B and C
    Value A1D = rewriter.create<memref::ReshapeOp>(loc, reshapedTypeA, A, shapeMemRef);
    Value B1D = rewriter.create<memref::ReshapeOp>(loc, reshapedTypeB, B, shapeMemRef);
    Value C1D = rewriter.create<memref::ReshapeOp>(loc, reshapedTypeC, C, shapeMemRef);

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
    if (M % SYSTOLIC_SIZE != 0 || N % SYSTOLIC_SIZE != 0 || K % SYSTOLIC_SIZE != 0) {
      op.emitError() << "M, N, and K must be multiples of SYSTOLIC_SIZE";
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
    Value n_idx;
    Value k_idx;
    Value m_idx;

    Value M_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), M);
    Value K_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), K);
    Value SYSTOLIC_SIZE_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), SYSTOLIC_SIZE);

    auto spadIdxMap = AffineMap::get(
      /*dimCount=*/3, /*symbolCount=*/2,
      rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(0) +
      rewriter.getAffineDimExpr(1) * rewriter.getAffineSymbolExpr(1) +
      rewriter.getAffineDimExpr(2), // This represents `n_idx * K + k_idx * SYSTOLIC_SIZE + i`
      rewriter.getContext()
    );
    auto spadIdxMapAttr = mlir::AffineMapAttr::get(spadIdxMap);

    if (N != SYSTOLIC_SIZE) {
      // N Loop
      auto inner_loop = rewriter.create<affine::AffineForOp>(loc, 0, N/SYSTOLIC_SIZE, 1);
      inner_loop->setAttr("inner_loop", rewriter.getBoolAttr(true));
      rewriter.setInsertionPointToStart(inner_loop.getBody());
      n_idx = inner_loop.getInductionVar();
    } else {
      n_idx = c0;
    }

    if (K != SYSTOLIC_SIZE) {
      // K Loop
      auto inner_loop = rewriter.create<affine::AffineForOp>(loc, 0, K/SYSTOLIC_SIZE, 1);
      inner_loop->setAttr("inner_loop", rewriter.getBoolAttr(true));
      rewriter.setInsertionPointToStart(inner_loop.getBody());
      k_idx = inner_loop.getInductionVar();
    } else {
      k_idx = c0;
    }

    if (M != SYSTOLIC_SIZE) {
      // M Loop
      auto inner_loop = rewriter.create<affine::AffineForOp>(loc, 0, M/SYSTOLIC_SIZE, 1);
      inner_loop->setAttr("inner_loop", rewriter.getBoolAttr(true));
      rewriter.setInsertionPointToStart(inner_loop.getBody());
      m_idx = inner_loop.getInductionVar();
    } else {
      m_idx = c0;
    }

    // Put dma wait operation
    mlir::Value ADmaTag;
    mlir::Value BDmaTag;
    mlir::Value BiasDmaTag;
    ValueRange BiasDMAIndices;
    mlir::Value numElements = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    // Search outer K loop
    op->getParentRegion()->getParentRegion()->walk([&](mlir::Operation *nestedOp) {
      if (auto dmaStartOp = llvm::dyn_cast<affine::AffineDmaStartOp>(nestedOp)) { // Replace DMAStartOp with actual `dma_start` op type
        auto result = getDramMemRef(dmaStartOp);
        auto sramRef = getSramMemRef(dmaStartOp);
        bool sramUsedInMatmul = false;

        for (auto operand : op->getOperands()) {
          if (operand == sramRef.first) {
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
    auto ATagMap = rewriter.getMultiDimIdentityMap(llvm::dyn_cast<MemRefType>(ADmaTag.getType()).getRank());
    auto BTagMap = rewriter.getMultiDimIdentityMap(llvm::dyn_cast<MemRefType>(BDmaTag.getType()).getRank());
    rewriter.create<affine::AffineDmaWaitOp>(loc, ADmaTag, ATagMap, ValueRange{c0, c0, m_idx}, numElements);
    rewriter.create<affine::AffineDmaWaitOp>(loc, BDmaTag, BTagMap, ValueRange{n_idx, c0, c0}, numElements);
    if (BiasDmaTag) {
      /* Bias could be 1D or 2D */
      Value first_index = BiasDMAIndices[0].getDefiningOp<mlir::arith::ConstantIndexOp>() ? c0 : n_idx;
      Value third_index = BiasDMAIndices[0].getDefiningOp<mlir::arith::ConstantIndexOp>() ? c0 : m_idx;
      auto BiasTagMap = rewriter.getMultiDimIdentityMap(llvm::dyn_cast<MemRefType>(BiasDmaTag.getType()).getRank());
      rewriter.create<affine::AffineDmaWaitOp>(loc, BiasDmaTag, BiasTagMap, ValueRange{first_index, third_index}, numElements);
    }

    // For vpush weight loop part
    for (int i=0; i<SYSTOLIC_SIZE; i+=nr_element) { // KxN
      Value i_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), i);
      Value spad_idx = rewriter.create<affine::AffineApplyOp>(loc, spadIdxMapAttr,
                                                      ValueRange{n_idx, k_idx, i_val, K_val, SYSTOLIC_SIZE_val});
      auto weight_vector = rewriter.create<vector::TransferReadOp>(
                                          loc, vectorType, B1D, ValueRange{spad_idx});
      rewriter.create<vcix::BinaryNoDestImmOp>(weight_vector.getLoc(), vwpush_opcode, weight_vector, zeroImmAttr, zeroImmAttr, rvl);
    }

    // For vpush input loop part
    for (int i=0; i<SYSTOLIC_SIZE; i+=nr_element) { // MxK
      Value i_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), i);
      Value spad_idx = rewriter.create<affine::AffineApplyOp>(loc, spadIdxMapAttr,
                                                      ValueRange{k_idx, m_idx, i_val, M_val, SYSTOLIC_SIZE_val});
      auto input_vector = rewriter.create<vector::TransferReadOp>(
                                          loc, vectorType, A1D, ValueRange{spad_idx});
      rewriter.create<vcix::BinaryNoDestImmOp>(input_vector.getLoc(), vipush_opcode, input_vector, zeroImmAttr, zeroImmAttr, rvl);
    }

    // Compute instruction
    rewriter.create<vcix::UnaryNoDestImmOp>(loc, compute_opcode, zeroImmAttr, compute_cycle, zeroImmAttr, sew, lmul, rvl);
    // For vpop loop part
    for (int i=0; i<SYSTOLIC_SIZE; i+=nr_element) { // MxN
      Value i_val = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), i);
      Value spad_idx = rewriter.create<affine::AffineApplyOp>(loc, spadIdxMapAttr,
                                                      ValueRange{n_idx, m_idx, i_val, M_val, SYSTOLIC_SIZE_val});
      Value vpop = rewriter.create<vcix::UnaryImmOp>(loc, vectorType, vpop_opcode, zeroImmAttr, zeroImmAttr, rvl);
      auto prev_output = rewriter.create<vector::TransferReadOp>(
                                          vpop.getLoc(), vectorType, C1D, ValueRange{spad_idx});
      VectorType vt = cast<VectorType>(prev_output.getType());
      if (vt.getElementType().isInteger()) {
        auto output_vector = rewriter.create<arith::AddIOp>(loc, prev_output, vpop);
        rewriter.create<vector::TransferWriteOp>(output_vector.getLoc(), output_vector, C1D, ValueRange{spad_idx});
      }
      else if (vt.getElementType().isIntOrFloat())  {
        auto output_vector = rewriter.create<arith::AddFOp>(loc, prev_output, vpop);
        rewriter.create<vector::TransferWriteOp>(output_vector.getLoc(), output_vector, C1D, ValueRange{spad_idx});
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
      const unsigned eltCount = legalType.getShape()[0];
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
        for (unsigned i = 0; i < n; ++i) {
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
    LLVMConversionTarget target(getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
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