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


struct MatmulOpLowering : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
    // Get the operands
    Value A = op.getInputs()[0];
    Value B = op.getInputs()[1];
    Value C = op.getOutputs()[0];

    Location loc = op.getLoc();

    int vlen = 128; //FIXME
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

    int A_N = memRefTypeA.getShape()[0];
    int A_K = memRefTypeA.getShape()[1];
    int B_K = memRefTypeB.getShape()[0];
    int B_M = memRefTypeB.getShape()[1];
    if (A_K != B_K) {
      op.emitError () << "K dimension is not same " << A_K << "!=" << B_K;
      return failure(true);
    }
    if (A_K != SYSTOLIC_SIZE) {
      op.emitError () << "A_K dimension (" << A_K << ") is not same to systolic array size (" << SYSTOLIC_SIZE << ")";
      return failure(true);
    }
    if (B_K != SYSTOLIC_SIZE || B_M != SYSTOLIC_SIZE) {
      op.emitError () << "B_K, B_M dimension (" << B_K << ", " << B_M <<
        ") is not same to systolic array size (" << SYSTOLIC_SIZE << ")";
      return failure(true);
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
    Attribute compute_cycle = rewriter.getI64IntegerAttr(4); // FIXME: 5 bits bound & hardcoded

    auto vectorType = VectorType::get({nr_element}, rewriter.getF32Type());

    // Opcode attribute
    Attribute zeroImmAttr = rewriter.getI64IntegerAttr(0);
    Attribute vipush_opcode =  rewriter.getI64IntegerAttr(0b000000);
    Attribute vwpush_opcode =  rewriter.getI64IntegerAttr(0b000001);
    Attribute compute_opcode = rewriter.getI64IntegerAttr(0b000001);
    Attribute vpop_opcode = rewriter.getI64IntegerAttr(0b000010);
    Value rvl = nullptr;

    // For loop part
    auto loop = rewriter.create<affine::AffineForOp>(loc, 0, A_N, nr_element);
    Value index = loop.getInductionVar();
    rewriter.setInsertionPointToStart(loop.getBody());

    auto input_vector = rewriter.create<vector::TransferReadOp>(
                                        loc, vectorType, A, ValueRange{c0, index});
    auto weight_vector = rewriter.create<vector::TransferReadOp>(
                                        loc, vectorType, B, ValueRange{c0, index});
    rvl = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(nr_element));
    rewriter.create<vcix::BinaryNoDestImmOp>(loc, vipush_opcode, input_vector, zeroImmAttr, zeroImmAttr, rvl);
    rewriter.create<vcix::BinaryNoDestImmOp>(loc, vwpush_opcode, weight_vector, zeroImmAttr, zeroImmAttr, rvl);
    Attribute sew = rewriter.getI64IntegerAttr(elen);
    Attribute lmul = rewriter.getI64IntegerAttr(0); // 0: m1, 1: m2, 2: m4, 3: m8, 5: mf8, 6: mf4, 7: mf2
    rewriter.create<vcix::ImmOp>(loc, compute_opcode, zeroImmAttr, compute_cycle, zeroImmAttr, sew, lmul, rvl);
    Value vpop = rewriter.create<vcix::BinaryImmOp>(loc, input_vector.getVectorType(), vpop_opcode, weight_vector, zeroImmAttr, rvl);
    auto output_vector = rewriter.create<vector::TransferWriteOp>(
                                        loc, vpop, C, ValueRange{c0, index});

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

    // Set global var from option
    SYSTOLIC_SIZE = systolicSize;
    patterns.add<MatmulOpLowering, MathExpToVCIX>(ctx);
    LLVMConversionTarget target(getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }


private:
  Option<int> systolicSize{*this, "systolic-array-size",
                          llvm::cl::desc("Systolic array size (KxK)"),
                          llvm::cl::init(128)};
};

} // namespace
namespace test {
void registerTestPyTorchSimToVCIXPass() { PassRegistration<TestPyTorchSimToVCIX>(); }
} // namespace test
} // namespace mlir