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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Analysis/CustomDMAAttribute.h"

#define MAX_TENSOR_DIM 4

namespace mlir {
namespace {

int64_t VECTOR_LANE = 128;

char* getAsmString(unsigned func7) {
  // return ".insn r CUSTOM_1, 0x3, " + std::to_string(func7) + ", x0, $0, $1";
  const char *asmStr = ".insn r CUSTOM_1, 0x3, ";
  const char *commaStr = ", x0, $0, $1";
  char *func7Str = (char*) malloc(10);
  sprintf(func7Str, "%d", func7);
  char *result = (char*) malloc(strlen(asmStr) + strlen(func7Str) + strlen(commaStr) + 1);
  strcpy(result, asmStr);
  strcat(result, func7Str);
  strcat(result, commaStr);
  return result;
}

unsigned int getElementBitWidth(mlir::Type elementType) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    return intType.getWidth();
  } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
    return floatType.getWidth();
  } else {
    mlir::failure();
    return 0;
  }
}

/// Lowering memref.dma_start operation to Gemmini instructions with LLVM Asm.
struct DmaWaitOpLowering : public ConvertOpToLLVMPattern<memref::DmaWaitOp> {
  int vectorlaneStride;
  DmaWaitOpLowering(LLVMTypeConverter &typeConverter, int vectorlaneStride = 4)
      : ConvertOpToLLVMPattern<memref::DmaWaitOp>(typeConverter) {
    this->vectorlaneStride = vectorlaneStride;
  }

  LogicalResult
  matchAndRewrite(memref::DmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering memref.dma_start operation to Gemmini instructions with LLVM Asm.
struct DmaStartOpLowering : public ConvertOpToLLVMPattern<memref::DmaStartOp> {
  // using ConvertOpToLLVMPattern<memref::DmaStartOp>::ConvertOpToLLVMPattern;
  DmaStartOpLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<memref::DmaStartOp>(typeConverter) {
  }

  LogicalResult
  matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();

    // Extract source/destination operands
    SmallVector<Value> operands(adaptor.getOperands().begin(), adaptor.getOperands().end());
    Value SrcMemref = operands[0];
    Value DstMemref = operands[op.getSrcMemRefRank() + 1];
    auto srcMemRefType = cast<MemRefType>(op.getSrcMemRef().getType());
    auto dstMemRefType = cast<MemRefType>(op.getDstMemRef().getType());
    ValueRange src_indices = ValueRange({operands.begin() + 1, operands.begin() + 1 + op.getSrcMemRefRank()});
    ValueRange dst_indices = ValueRange({operands.begin() + 1 + op.getSrcMemRefRank() + 1,
                                         operands.begin() + 1 + op.getSrcMemRefRank() + 1 +
                                         op.getDstMemRefRank()});
    Value SrcPtr = getStridedElementPtr(loc, srcMemRefType, SrcMemref, src_indices, rewriter);
    Value DstPtr = getStridedElementPtr(loc, dstMemRefType, DstMemref, dst_indices, rewriter);

    // Extract other operands
    int elen = getElementBitWidth(srcMemRefType.getElementType());
    Value num_elt_per_stride = op.getNumElementsPerStride();
    uint64_t vlane_split_axis = getVlaneSplitAxis(op);
    uint64_t vlane_stride = getVlaneStride(op);
    uint64_t vlane_stride_byte = vlane_stride * elen / 8;
    int dmaType = getDMAType(op);
    bool is_mvin = dmaType == MVIN || dmaType == MVIN2 || dmaType == MVIN3;
    if (elen < 8) {
      if (vlane_stride_byte < 1) {
        vlane_stride_byte = 1;
      }
      elen = 8;
    }

    // Extract attributes
    SmallVector<mlir::Attribute, 2> dmaSubtile = getSubtileSize(op);
    SmallVector<mlir::Attribute> spad_strides = getSramStride(op);
    SmallVector<mlir::Attribute> dram_strides = getDramStride(op);

    Value dram_addr;
    Value spad_addr;
    SmallVector<int64_t,4> tile_shape;
    ValueRange indices;
    if (is_mvin) { // MVIN
      dram_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), SrcPtr);
      spad_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), DstPtr);
      tile_shape = SmallVector<int64_t,4>(dstMemRefType.getShape());
      indices = op.getSrcIndices();;
    } else { // MVOUT
      dram_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), DstPtr);
      spad_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), SrcPtr);
      tile_shape = SmallVector<int64_t,4>(srcMemRefType.getShape());
      indices = op.getDstIndices();
    }

    AffineMap index_map;
    ValueRange parentIndices;
    uint64_t indirect_access = 0;
    Value indirect_spad_addr = 0;
    uint64_t indirect_element_size = 1;
    uint64_t indirect_stride = 1;

    for (auto index : indices) {
      if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
        index_map = applyOp.getAffineMap();
        indirect_access = index_map.getNumSymbols()!=0;
        parentIndices = applyOp.getOperands();
        if (indirect_access) {
          // FIXME. How to get converted type?
          for (mlir::Value operand : applyOp.getMapOperands()) {
            auto indexCastOp = operand.getDefiningOp<arith::IndexCastOp>();
            if (!indexCastOp)
              continue;
            // Found index cast
            auto loadOp = indexCastOp.getIn().getDefiningOp<mlir::affine::AffineLoadOp>();
            if (!loadOp)
              continue;
            // Found index spad memref
            bool found = false;
            Value indirectMemref = loadOp.getMemRef();
            auto indirectMemRefType = dyn_cast<MemRefType>(indirectMemref.getType());
            for (auto &use : indirectMemref.getUses()) {
              mlir::Operation* userOp = use.getOwner();
              if (auto castOp = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(userOp)) {
                indirectMemref = castOp->getResult(0);
                found = true;
              }
            }
            if (!found) {
              op.emitError("Failed to find converted memref...");
              return failure();
            }
            indirect_spad_addr = getStridedElementPtr(loc, indirectMemRefType, indirectMemref, {}, rewriter);
            indirect_spad_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), indirect_spad_addr);
            indirect_element_size = getElementBitWidth(indirectMemRefType.getElementType()) / 8;
            break;
          }
        }
      }
    }

    /* Use subtile size if it has subtile attribute */
    if (dmaSubtile.size()) {
      tile_shape.clear();
      for (Attribute attr : dmaSubtile) {
        auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr);
        if (!intAttr) {
          llvm::errs() << "Error: Non-integer attribute found in 'dmaSubtile'.\n";
          return failure();
        }
        tile_shape.push_back(intAttr.getInt());
      }
    }

    /* Sanity check for attributes */
    if (tile_shape.size() != dram_strides.size() ||
      tile_shape.size() != spad_strides.size()) {
      llvm::errs() << "Size: " << tile_shape.size() << "\n";
      llvm::errs() << "Size: " << dram_strides.size() << "\n";
      llvm::errs() << "Size: " << spad_strides.size() << "\n";
      llvm::errs() << "Size mismatch: tile_shape, dram_strides, and spad_strides must have the same size.\n";
      return failure();
    }

    /* Expand attributes if needed */
    int64_t expanding_dim = MAX_TENSOR_DIM - tile_shape.size();
    vlane_split_axis += expanding_dim;
    SmallVector<int64_t> tensor_shape_4d(MAX_TENSOR_DIM, 1);
    SmallVector<int64_t> dram_strides_4d(MAX_TENSOR_DIM, 0);
    SmallVector<int64_t> spad_strides_4d(MAX_TENSOR_DIM, 0);

    for (int i = 0; i < static_cast<int>(tile_shape.size()); i++) {
      tensor_shape_4d[expanding_dim + i] = tile_shape[i];
      auto dramAttr = llvm::dyn_cast<mlir::IntegerAttr>(dram_strides[i]);
      auto spadAttr = llvm::dyn_cast<mlir::IntegerAttr>(spad_strides[i]);
      if (!spadAttr || !dramAttr) {
        llvm::errs() << "Error: Non-integer attribute found in 'dmaSubtile'.\n";
        return failure();
      }
      dram_strides_4d[expanding_dim + i] = dramAttr.getInt();
      spad_strides_4d[expanding_dim + i] = spadAttr.getInt();
    }

    /* ============================================== */
    /* =        Inline assembly codegen phase       = */
    /* ============================================== */
    auto asmDialectAttr =
      LLVM::AsmDialectAttr::get(rewriter.getContext(), LLVM::AsmDialect::AD_ATT);
    const char* constraintStr = "r,r,~{dirflag},~{fpsr},~{flags}";
    unsigned func7 = dmaType;
    char* asmStr = getAsmString(func7);
    // constants for shifting
    uint64_t shift14 = 14;
    uint64_t shift16 = 16;
    uint64_t shift17 = 17;
    uint64_t shift32 = 32;
    uint64_t shift48 = 48;

    // config_mvin, config_mvout instructions
    func7 = CONFIG;
    int64_t config_type;
    if (dmaType == MVIN) {
      config_type = CONFIG_MVIN;
    } else if (dmaType == MVIN2) {
      config_type = CONFIG_MVIN2;
    } else if (dmaType == MVIN3) {
      config_type = CONFIG_MVIN3;
    } else if (dmaType == MVOUT) {
      config_type = CONFIG_MVOUT;
    } else{
      return failure();
    }

    // config1
    // config_rs1 = 1st dim << 48 | 2nd dim << 32 | 3rd dim << 16 | 4th dim size
    // config_rs2 = vlane_stride << 32 | config_type << 17  | vlane_split_axis << 14 | element size
    char* configAsmStr = getAsmString(func7);
    Value config_rs1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr((
      ((tensor_shape_4d[0]&0xFFFF)<<shift48) | ((tensor_shape_4d[1]&0xFFFF)<<shift32) | ((tensor_shape_4d[2]&0xFFFF)<<shift16) | ((tensor_shape_4d[3])&0xFFFF))
    ));
    Value config_rs2 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(
      (vlane_stride<<shift32)| ((config_type&0x3)<<shift17) | ((indirect_access&0b1)<<shift16) | ((vlane_split_axis&0x3)<<shift14) | (int(elen/8))
    ));
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

    // config2
    // config_rs1 = 1st dim stride << 32 | 2nd dim stride
    // config_rs2 = 3rd dim stride << 32 | 4th dim stride
    char* config2AsmStr = getAsmString(CONFIG2);
    config_rs1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(
      (dram_strides_4d[0]<<shift32) | (dram_strides_4d[1]&0xFFFFFFFF)
    ));
    config_rs2 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(
      (dram_strides_4d[2]<<shift32) | (dram_strides_4d[3]&0xFFFFFFFF)
    ));
    rewriter.create<LLVM::InlineAsmOp>(
        loc,
        /*resultTypes=*/TypeRange(),
        /*operands=*/ValueRange({config_rs1, config_rs2}),
        /*asm_string=*/config2AsmStr,
        /*constraints=*/constraintStr,
        /*has_side_effects=*/true,
        /*is_align_stack=*/false,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());

    // config3
    // config_rs1 = 1st dim spad_stride << 32 | 2nd dim spad_stride
    // config_rs2 = 3rd dim spad_stride << 32 | 4th dim spad_stride
    char* config3AsmStr = getAsmString(CONFIG3);
    config_rs1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(
      (spad_strides_4d[0]<<shift32) | (spad_strides_4d[1]&0xFFFFFFFF)
    ));
    config_rs2 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(
      (spad_strides_4d[2]<<shift32) | (spad_strides_4d[3]&0xFFFFFFFF)
    ));
    rewriter.create<LLVM::InlineAsmOp>(
        loc,
        /*resultTypes=*/TypeRange(),
        /*operands=*/ValueRange({config_rs1, config_rs2}),
        /*asm_string=*/config3AsmStr,
        /*constraints=*/constraintStr,
        /*has_side_effects=*/true,
        /*is_align_stack=*/false,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());

    // config4
    if (indirect_access) {
      char* config4AsmStr = getAsmString(CONFIG4);
      config_rs2 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(
        ((indirect_element_size & 0xFF) << shift16) | (indirect_stride & 0xFFFF)
      ));
      rewriter.create<LLVM::InlineAsmOp>(
          loc,
          /*resultTypes=*/TypeRange(),
          /*operands=*/ValueRange({indirect_spad_addr, config_rs2}),
          /*asm_string=*/config4AsmStr,
          /*constraints=*/constraintStr,
          /*has_side_effects=*/true,
          /*is_align_stack=*/false,
          /*asm_dialect=*/asmDialectAttr,
          /*operand_attrs=*/ArrayAttr());
    }
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(op,
        /*resultTypes=*/TypeRange(),
        /*operands=*/ValueRange({dram_addr, spad_addr}),
        /*asm_string=*/asmStr,
        /*constraints=*/constraintStr,
        /*has_side_effects=*/true,
        /*is_align_stack=*/false,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());
    return success();
  }
};

/// Lowering memref.dma_start operation to Gemmini instructions with LLVM Asm.
struct TimingDmaStartOpLowering : public ConvertOpToLLVMPattern<memref::DmaStartOp> {
  // using ConvertOpToLLVMPattern<memref::DmaStartOp>::ConvertOpToLLVMPattern;
  TimingDmaStartOpLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<memref::DmaStartOp>(typeConverter) {
  }

  LogicalResult
  matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
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

  Option<int> vectorlane{
      *this, "vectorlane",
      llvm::cl::desc("Vector lane size for gemmini instructions"),
      llvm::cl::init(4)};

  Option<bool> timing_mode{*this, "timing",
                   llvm::cl::desc("tming mode switch"),
                   llvm::cl::init(false)};
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

    VECTOR_LANE = vectorlane;
    RewritePatternSet patterns(ctx);
    // vectorlane is passed to the pattern as an argument
    if (timing_mode)
      patterns.add<TimingDmaStartOpLowering>(typeConverter);
    else
      patterns.add<DmaStartOpLowering>(typeConverter);
    patterns.add<DmaWaitOpLowering>(typeConverter);
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