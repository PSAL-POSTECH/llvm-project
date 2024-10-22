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
static std::pair<unsigned, VectorType> legalizeVectorType(const Type &type) {
  VectorType vt = cast<VectorType>(type);
  // To simplify test pass, avoid multi-dimensional vectors.
  if (!vt || vt.getRank() != 1)
    return {0, nullptr};

  if (!vt.isScalable())
    return {1, vt};

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
  return {n, VectorType::get({eltCount >> (n - 1)}, eltTy, {true})};
}

void createUnrolledIndices(OpBuilder *builder, int64_t lower, int64_t upper, int64_t step, SmallVector<Value> &indices) {
  for (int64_t i = lower; i < upper; i += step) {
    Value indexValue = builder->create<arith::ConstantIndexOp>(builder->getUnknownLoc(), i);
    indices.push_back(indexValue);
  }
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
    int nr_k = (K/SYSTOLIC_SIZE);
    int nr_n = (N/SYSTOLIC_SIZE);
    int max_nk = std::max(nr_k, nr_n);
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
    SmallVector<Value> indices;

    // Create indexes
    createUnrolledIndices(&rewriter, 0, M*(max_nk+1), nr_element, indices);

    if (N != SYSTOLIC_SIZE) {
      auto inner_loop = rewriter.create<affine::AffineForOp>(loc, 0, N/SYSTOLIC_SIZE, 1);
      rewriter.setInsertionPointToStart(inner_loop.getBody());
      n_idx = inner_loop.getInductionVar();
    } else {
      n_idx = c0;
    }
    // N Loop
    {
      if (K != SYSTOLIC_SIZE) {
        auto inner_loop = rewriter.create<affine::AffineForOp>(loc, 0, K/SYSTOLIC_SIZE, 1);
        rewriter.setInsertionPointToStart(inner_loop.getBody());
        k_idx = inner_loop.getInductionVar();
      } else {
        k_idx = c0;
      }
      // K Loop
      {
        Value N_offset = rewriter.create<arith::MulIOp>(loc, indices[(N/nr_element)], k_idx);
        Value K_offset = rewriter.create<arith::MulIOp>(loc, indices[(K/nr_element)], k_idx);
        Value M_K_offset = rewriter.create<arith::MulIOp>(loc, indices[(M/nr_element)], k_idx);
        Value M_N_offset = rewriter.create<arith::MulIOp>(loc, indices[(M/nr_element)], n_idx);
        // For vpush weight loop part
        for (int i=0; i<SYSTOLIC_SIZE; i+=nr_element) { // KxN
          Value spad_idx = rewriter.create<arith::AddIOp>(loc, N_offset, indices[i/nr_element]);
          Value new_k_idx = rewriter.create<arith::DivUIOp>(loc, spad_idx, indices[N/nr_element]);
          Value new_m_idx = rewriter.create<arith::RemUIOp>(loc, spad_idx, indices[N/nr_element]);
          auto weight_vector = rewriter.create<vector::TransferReadOp>(
                                               loc, vectorType, B, ValueRange{new_k_idx, new_m_idx});
          rewriter.create<vcix::BinaryNoDestImmOp>(weight_vector.getLoc(), vwpush_opcode, weight_vector, zeroImmAttr, zeroImmAttr, rvl);
        }

        // For vpush input loop part
        for (int i=0; i<M; i+=nr_element) { // MxK
          Value spad_idx = rewriter.create<arith::AddIOp>(loc, M_K_offset, indices[i/nr_element]);
          Value new_m_idx = rewriter.create<arith::DivUIOp>(loc, spad_idx, indices[K/nr_element]);
          Value new_k_idx = rewriter.create<arith::RemUIOp>(loc, spad_idx, indices[K/nr_element]);
          auto input_vector = rewriter.create<vector::TransferReadOp>(
                                               loc, vectorType, A, ValueRange{new_m_idx, new_k_idx});
          rewriter.create<vcix::BinaryNoDestImmOp>(input_vector.getLoc(), vipush_opcode, input_vector, zeroImmAttr, zeroImmAttr, rvl);
        }

        // Compute instruction
        rewriter.create<vcix::UnaryNoDestImmOp>(loc, compute_opcode, zeroImmAttr, compute_cycle, zeroImmAttr, sew, lmul, rvl);
        // For vpop loop part
        for (int i=0; i<M; i+=nr_element) { // MxN
          Value spad_idx = rewriter.create<arith::AddIOp>(loc, M_N_offset, indices[i/nr_element]);
          Value new_m_idx = rewriter.create<arith::DivUIOp>(loc, spad_idx, indices[N/nr_element]);
          Value new_n_idx = rewriter.create<arith::RemUIOp>(loc, spad_idx, indices[N/nr_element]);
          Value vpop = rewriter.create<vcix::UnaryImmOp>(loc, vectorType, vpop_opcode, zeroImmAttr, zeroImmAttr, rvl);
          auto prev_output = rewriter.create<vector::TransferReadOp>(
                                              vpop.getLoc(), vectorType, C, ValueRange{new_m_idx, new_n_idx});
          VectorType vt = cast<VectorType>(prev_output.getType());
          if (vt.getElementType().isInteger()) {
            auto output_vector = rewriter.create<arith::AddIOp>(loc, prev_output, vpop);
            rewriter.create<vector::TransferWriteOp>(output_vector.getLoc(), output_vector, C, ValueRange{new_m_idx, new_n_idx});
          }
          else if (vt.getElementType().isIntOrFloat())  {
            auto output_vector = rewriter.create<arith::AddFOp>(loc, prev_output, vpop);
            rewriter.create<vector::TransferWriteOp>(output_vector.getLoc(), output_vector, C, ValueRange{new_m_idx, new_n_idx});
          } else {
            op.emitError () << "expected same type";
            return failure();
          }
        }
        rewriter.eraseOp(op);
        return success();
      }
    }
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
      for (unsigned i = 0; i < n; ++i) {
        Value extracted = rewriter.create<vector::ScalableExtractOp>(
            loc, legalType, vec, i * eltCount);
        Value v = rewriter.create<vcix::BinaryImmOp>(loc, legalType, opcodeAttr,
                                                      extracted, zeroImmAttr, rvl);
        res = rewriter.create<vector::ScalableInsertOp>(loc, v, res,
                                                        i * eltCount);
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