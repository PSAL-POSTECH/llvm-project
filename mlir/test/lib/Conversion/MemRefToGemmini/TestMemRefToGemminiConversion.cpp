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

#define MVIN 0
#define MVOUT 1

namespace mlir {
namespace {

int VECTOR_LANE_STRIDE = 4;

int extractConstantIntValue(Value val) {
  int val_int;
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    Attribute constantAttr = constOp.getValue();
    if (auto intAttr = constantAttr.dyn_cast<IntegerAttr>()) {
      val_int = intAttr.getInt();
    }
  }
  return val_int;
}

char* getAsmString(unsigned func7) {
  switch (func7) {
  case 0x0:
    return ".insn r CUSTOM_1, 0x3, 0, x0, $0, $1"; // config_mvin
  case 0x1:
    return ".insn r CUSTOM_1, 0x3, 1, x0, $0, $1"; // config_mvout
  case 0x2:
    return ".insn r CUSTOM_1, 0x3, 2, x0, $0, $1"; // mvin
  case 0x3:
    return ".insn r CUSTOM_1, 0x3, 3, x0, $0, $1"; // mvout
  default:
    return "";
  }
}

/// Lowering memref.dma_start operation to Gemmini instructions with LLVM Asm.
struct DmaStartOpLowering : public ConvertOpToLLVMPattern<memref::DmaStartOp> {
  // using ConvertOpToLLVMPattern<memref::DmaStartOp>::ConvertOpToLLVMPattern;
  int vectorlaneStride;
  DmaStartOpLowering(LLVMTypeConverter &typeConverter, int vectorlaneStride = 4)
      : ConvertOpToLLVMPattern<memref::DmaStartOp>(typeConverter) {
    this->vectorlaneStride = vectorlaneStride;
  }

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
    Value main_mem_stride = op.getStride();
    int main_mem_stride_val = extractConstantIntValue(main_mem_stride);
    Value num_elt = op.getNumElementsPerStride();
    int num_elt_val = extractConstantIntValue(num_elt);
    int is_transpose = 0;
    if (num_elt_val == 1) // num_elt = 1 means transposed
      is_transpose = 1;
    unsigned SrcAddressSpace =
        *getTypeConverter()->getMemRefAddressSpace(srcMemRefType);
    unsigned DstAddressSpace =
        *getTypeConverter()->getMemRefAddressSpace(dstMemRefType);
    Value rs1;
    Value spad_addr;
    llvm::ArrayRef<int64_t> tile_shape;
    bool dmaType = SrcAddressSpace == 0 && DstAddressSpace == 1 ? MVIN : (SrcAddressSpace == 1 && DstAddressSpace == 0 ? MVOUT : -1);
    if (dmaType == MVIN) { // MVIN
      rs1 = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), SrcPtr);
      spad_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), DstPtr);
      func7 = 0x2;
      tile_shape = dstMemRefType.getShape();
    } else if (dmaType == MVOUT) { // MVOUT
      rs1 = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), DstPtr);
      spad_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), SrcPtr);
      func7 = 0x3;
      tile_shape = srcMemRefType.getShape();
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported DMA operation");
    }
    Value rows;
    Value cols;
    if (tile_shape.size() == 2) {
      rows = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(tile_shape[0]));
      cols = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(tile_shape[1]));
    } else if (tile_shape.size() == 1) {
      rows = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
      cols = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(tile_shape[0]));
    }

    char* asmStr = getAsmString(func7);
    // encoding rs2
    // rs2 = rows << (ADDR_LEN + 16) | (cols << ADDR_LEN) | spad_addr
    Value shift48 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(48));
    Value rs2 = rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rows, shift48);
    Value shift32 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(32));
    rs2 = rewriter.create<LLVM::OrOp>(loc, rs2, rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), cols, shift32));
    rs2 = rewriter.create<LLVM::OrOp>(loc, rs2, spad_addr);

    // config_mvin, config_mvout instructions
    func7 = dmaType == MVIN ? 0x0 : 0x1;
    char* configAsmStr = getAsmString(func7);
    auto InnerRegion = op->getParentRegion();
    auto OuterRegion = InnerRegion->getParentRegion();
    if (OuterRegion && !OuterRegion->empty()) {
      auto &outerBlock = OuterRegion->front();
      if (!outerBlock.empty()) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&outerBlock);
        // config_rs1 = main memory stride
        // config_rs2 = is_transpose << 32 | element size
        int elen = 0;
        auto elementTypeA = srcMemRefType.getElementType();
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementTypeA)) {
          elen = intType.getWidth();
        } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementTypeA)) {
          elen = floatType.getWidth();
        } else {
          return failure();
        }
        Value config_rs1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(main_mem_stride_val));
        Value config_shift32 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(32));
        Value config_rs2 = rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(is_transpose)), config_shift32);
        config_rs2 = rewriter.create<LLVM::OrOp>(loc, config_rs2, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(int(elen/8)))); // element size [bytes]
        rewriter.create<LLVM::InlineAsmOp>(
            loc,
            /*resultTypes=*/TypeRange(),
            /*operands=*/ValueRange({config_rs1, config_rs2}),
            /*asm_string=*/configAsmStr,
            /*constraints=*/constraintStr,
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            /*asm_dialect=*/asmDialectAttr,
            /*operand_attrs=*/ArrayAttr());
      }
    }

    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(op,
        /*resultTypes=*/TypeRange(),
        /*operands=*/ValueRange({rs1, rs2}),
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

  Option<int> vectorlaneStride{
      *this, "vectorlane-stride",
      llvm::cl::desc("Vector lane stride for the custom gemmini instructions"),
      llvm::cl::init(4)};

  TestMemRefToGemmini() = default;
  TestMemRefToGemmini(const TestMemRefToGemmini &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);

    VECTOR_LANE_STRIDE = vectorlaneStride;
    RewritePatternSet patterns(ctx);
    // vectorlaneStride is passed to the pattern as an argument
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