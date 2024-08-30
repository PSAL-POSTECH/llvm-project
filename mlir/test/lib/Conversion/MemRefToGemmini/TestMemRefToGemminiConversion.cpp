//===- TestMemRefToLLVMConversion.cpp - Test conversion to gemmini ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace {

char* getAsmString(unsigned func7) {
  switch (func7) {
  case 0x2:
    return ".insn r CUSTOM_1, 0x3, 2, x0, $0, $1";
  case 0x3:
    return ".insn r CUSTOM_1, 0x3, 3, x0, $0, $1";
  default:
    return "";
  }
}

/// Lowering memref.dma_start operation to Gemmini instructions with LLVM Asm.
struct DmaStartOpLowering : public ConvertOpToLLVMPattern<memref::DmaStartOp> {
  using ConvertOpToLLVMPattern<memref::DmaStartOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto asmDialectAttr =
      LLVM::AsmDialectAttr::get(rewriter.getContext(), LLVM::AsmDialect::AD_ATT);
    const char* constraintStr = "r,r,~{dirflag},~{fpsr},~{flags}";
    auto loc = op.getLoc();
    unsigned func7;

    SmallVector<Value> operands;
    for (auto val : adaptor.getOperands())
      operands.push_back(val);

    Value SrcMemref = operands[0];
    auto srcMemRefType = cast<MemRefType>(op.getSrcMemRef().getType());
    ValueRange src_indices = ValueRange({operands.begin() + 1, operands.begin() + 1 + op.getSrcMemRefRank()});
    Value SrcPtr = getStridedElementPtr(loc, srcMemRefType, SrcMemref, src_indices, rewriter);
    Value DstMemref = operands[op.getSrcMemRefRank() + 1];
    auto dstMemRefType = cast<MemRefType>(op.getDstMemRef().getType());
    ValueRange dst_indices = ValueRange({operands.begin() + 1 + op.getSrcMemRefRank() + 1,
                                         operands.begin() + 1 + op.getSrcMemRefRank() + 1 +
                                         op.getDstMemRefRank()});

    Value DstPtr = getStridedElementPtr(loc, dstMemRefType, DstMemref, dst_indices, rewriter);
    Value cols = operands[op.getNumOperands() - 1];
    Value num_elt = operands[1 + op.getSrcMemRefRank() + 1 + op.getDstMemRefRank()];
    Value rows = rewriter.create<LLVM::UDivOp>(loc, rewriter.getI64Type(), num_elt, cols); // rows = num_elements / cols
    unsigned SrcAddressSpace =
        *getTypeConverter()->getMemRefAddressSpace(srcMemRefType);
    unsigned DstAddressSpace =
        *getTypeConverter()->getMemRefAddressSpace(dstMemRefType);
    Value rs1;
    Value spad_addr;
    if (SrcAddressSpace == 0 && DstAddressSpace == 1) { // MVIN
      rs1 = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), SrcPtr);
      spad_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), DstPtr);
      func7 = 0x2;
    } else if (SrcAddressSpace == 1 && DstAddressSpace == 0) { // MVOUT
      rs1 = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), DstPtr);
      spad_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), SrcPtr);
      func7 = 0x3;
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported DMA operation");
    }
    char* asmStr = getAsmString(func7);

    // encoding rs2
    // rs2 = rows << (ADDR_LEN + 16) | (cols << ADDR_LEN) | spad_addr
    Value shift48 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(48));
    Value rs2 = rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rows, shift48);
    Value shift32 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(32));
    rs2 = rewriter.create<LLVM::OrOp>(loc, rs2, rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), cols, shift32));
    rs2 = rewriter.create<LLVM::OrOp>(loc, rs2, spad_addr);

    SmallVector<Value> asmVals;
    asmVals.push_back(rs1);
    asmVals.push_back(rs2);
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(op,
        /*resultTypes=*/TypeRange(),
        /*operands=*/asmVals,
        /*asm_string=*/asmStr,
        /*constraints=*/constraintStr,
        /*has_side_effects=*/true,
        /*is_align_stack=*/false,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());
    return success();
  }
};

struct TestMemRefToGemmini
    : PassWrapper<TestMemRefToGemmini, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMemRefToGemmini)

  StringRef getArgument() const final { return "test-memref-to-gemmini"; }

  StringRef getDescription() const final {
    return "Test lowering patterns that converts memref dialect to LLVM Asm."
           "with custom gemmini instructions.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);

    RewritePatternSet patterns(ctx);
    patterns.add<DmaStartOpLowering>(typeConverter);
    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }

};

} // namespace

namespace test {
void registerTestMemRefToGemminiPass() { PassRegistration<TestMemRefToGemmini>(); }
} // namespace test
} // namespace mlir