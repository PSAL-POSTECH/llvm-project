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

#define CONFIG 0
#define CONFIG2 4
#define CONFIG3 5
#define MVIN 2
#define MVIN2 1
#define MVIN3 14
#define MVOUT 3

#define CONFIG_MVIN 0
#define CONFIG_MVIN2 1
#define CONFIG_MVIN3 2
#define CONFIG_MVOUT 3

#define MAX_TENSOR_DIM 4

namespace mlir {
namespace {

int64_t VECTOR_LANE = 128;

int extractConstantIntValue(Value val) {
  int val_int;
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    Attribute constantAttr = constOp.getValue();
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constantAttr)) {
      val_int = intAttr.getInt();
    }
  }
  return val_int;
}

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

llvm::SmallVector<int64_t, 2> getSubtileSize(mlir::Operation *operation) {
  llvm::SmallVector<int64_t, 2> subtileSizes;
  auto attr = operation->getAttr("subtile_size");
  if (!attr) {
    return subtileSizes; // Return empty SmallVector
  }

  if (auto arrayAttr = llvm::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (auto element : arrayAttr) {
      // Assume the elements are integers
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(element)) {
        subtileSizes.push_back(intAttr.getInt());
      } else {
        llvm::errs() << "Unsupported element type in 'subtile_size'.\n";
      }
    }
  }
  return subtileSizes;
}

int getAsyncValue(mlir::Operation *operation) {
  auto attr = operation->getAttr("async");
  if (!attr)
    return 0;

  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getInt(); // Treat non-zero as true
  else
    return 1;
}

int is_fine_grained(mlir::Operation *operation) {
  auto attr = operation->getAttr("fine_grained");
  if (!attr)
    return 0;

  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getInt(); // Treat non-zero as true
  else
    return 1;
}

llvm::SmallVector<int64_t> getSramStride(mlir::Operation *operation) {
  llvm::SmallVector<int64_t> sram_stride;
  auto attr = operation->getAttr("sram_stride");
  if (!attr) {
    return sram_stride; // Return empty SmallVector
  }

  if (auto arrayAttr = llvm::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (auto element : arrayAttr) {
      // Assume the elements are integers
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(element)) {
        sram_stride.push_back(intAttr.getInt());
      } else {
        llvm::errs() << "Unsupported element type in 'sram_stride'.\n";
      }
    }
  }
  return sram_stride;
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
    auto asmDialectAttr =
      LLVM::AsmDialectAttr::get(rewriter.getContext(), LLVM::AsmDialect::AD_ATT);
    const char* constraintStr = "r,r,~{dirflag},~{fpsr},~{flags}";
    auto loc = op.getLoc();
    unsigned func7;
    llvm::SmallVector<int64_t, 2> dmaSubtile = getSubtileSize(op);
    llvm::SmallVector<int64_t> spad_strides = getSramStride(op);

    SmallVector<Value> operands;
    for (auto val : adaptor.getOperands())
      operands.push_back(val);
    Value SrcMemref = operands[0];
    auto srcMemRefType = cast<MemRefType>(op.getSrcMemRef().getType());
    ValueRange src_indices = ValueRange({operands.begin() + 1, operands.begin() + 1 + op.getSrcMemRefRank()});
    ValueRange dst_indices = ValueRange({operands.begin() + 1 + op.getSrcMemRefRank() + 1,
                                         operands.begin() + 1 + op.getSrcMemRefRank() + 1 +
                                         op.getDstMemRefRank()});
    int elen = 0;
    auto elementTypeA = srcMemRefType.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementTypeA)) {
      elen = intType.getWidth();
    } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementTypeA)) {
      elen = floatType.getWidth();
    } else {
      return failure();
    }
    Value SrcPtr = getStridedElementPtr(loc, srcMemRefType, SrcMemref, src_indices, rewriter);
    Value DstMemref = operands[op.getSrcMemRefRank() + 1];
    auto dstMemRefType = cast<MemRefType>(op.getDstMemRef().getType());
    Value DstPtr = getStridedElementPtr(loc, dstMemRefType, DstMemref, dst_indices, rewriter);
    Value vlane_split_axis_val = op.getStride();
    uint64_t vlane_split_axis = extractConstantIntValue(vlane_split_axis_val);
    Value num_elt_per_stride = op.getNumElementsPerStride();
    uint64_t vlane_stride = extractConstantIntValue(num_elt_per_stride);
    vlane_stride = (vlane_stride & 0x7FFF); // mask out the fine-grained bit
    uint64_t vlane_stride_byte = vlane_stride * elen / 8;
    Value numElements = op.getNumElements();
    int dmaType = extractConstantIntValue(numElements);
    Value rs1;
    Value spad_addr;
    llvm::ArrayRef<int64_t> tile_shape;
    llvm::ArrayRef<int64_t> subtile_shape(dmaSubtile);
    bool is_mvin = dmaType == MVIN || dmaType == MVIN2 || dmaType == MVIN3;

    SmallVector<int64_t> mm_strides;
    ValueRange indices;
    int64_t mm_offset;
    if (is_mvin) { // MVIN
      rs1 = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), SrcPtr);
      spad_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), DstPtr);
      tile_shape = dstMemRefType.getShape();
      std::tie(mm_strides, mm_offset) = getStridesAndOffset(srcMemRefType);
      indices = op.getSrcIndices();;
    } else { // MVOUT
      rs1 = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), DstPtr);
      spad_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), SrcPtr);
      tile_shape = srcMemRefType.getShape();
      std::tie(mm_strides, mm_offset) = getStridesAndOffset(dstMemRefType);
      indices = op.getDstIndices();
    }

    AffineMap index_map;
    ValueRange parentIndices;
    for (auto index : indices) {
      if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
        index_map = applyOp.getAffineMap();
        parentIndices = applyOp.getOperands();
      }
    }
    if (is_fine_grained(op)) { // fine-grained case has more than one parent
      index_map = AffineMap();
      for (auto parentIndex : parentIndices) {
        if (auto applyOp = parentIndex.getDefiningOp<affine::AffineApplyOp>()) {
          index_map = applyOp.getAffineMap();
        }
      }
    }

    if (index_map) {
      mm_strides = SmallVector<int64_t>(index_map.getNumDims(), 0);
      // Ensure the AffineMap has at least one result
      if (index_map.getNumResults() != 1) {
        llvm::errs() << "AffineMap should have exactly one result.\n";
      }

      // Extract the single result expression
      mlir::AffineExpr resultExpr = index_map.getResult(0);
      // Traverse the result expression to calculate strides
      resultExpr.walk([&](mlir::AffineExpr subExpr) {
        if (auto dimExpr = subExpr.dyn_cast<mlir::AffineDimExpr>()) {
          // Set stride for the corresponding dimension
          mm_strides[dimExpr.getPosition()] = 1;
        } else if (auto mulExpr = subExpr.dyn_cast<mlir::AffineBinaryOpExpr>()) {
          // Handle multiplications
          if (mulExpr.getKind() == mlir::AffineExprKind::Mul) {
            if (auto dimExpr = mulExpr.getLHS().dyn_cast<mlir::AffineDimExpr>()) {
              if (auto constExpr = mulExpr.getRHS().dyn_cast<mlir::AffineConstantExpr>()) {
                mm_strides[dimExpr.getPosition()] = constExpr.getValue();
              }
            }
          }
        }
      });
    }

    /* Use subtile size if it has subtile attribute */
    if (subtile_shape.size()) {
      tile_shape = subtile_shape;
    }

    func7 = dmaType;

    if (elen < 8) {
      if (vlane_stride_byte < 1) {
        vlane_stride_byte = 1;
      }
      elen = 8;
    }

    char* asmStr = getAsmString(func7);
    // constants for shifting
    uint64_t shift14 = 14;
    uint64_t shift16 = 16;
    uint64_t shift17 = 17;
    uint64_t shift32 = 32;
    uint64_t shift48 = 48;

    // encoding rs2
    // rs2 = spad_addr
    Value rs2 = spad_addr;

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
    char* configAsmStr = getAsmString(func7);

    // expand vlane_split_axis to 4D
    int64_t expanding_dim = MAX_TENSOR_DIM - tile_shape.size();
    int64_t mm_expanding_dim = MAX_TENSOR_DIM - mm_strides.size();
    vlane_split_axis += mm_expanding_dim;
    // config_rs1 = 1st dim << 48 | 2nd dim << 32 | 3rd dim << 16 | 4th dim size
    // config_rs2 = vlane_stride << 32 | config_type << 17  | vlane_split_axis << 14 | element size
    SmallVector<int64_t> sub_tensor_shape(MAX_TENSOR_DIM, 1);
    for (int i = 0; i < static_cast<int>(tile_shape.size()); i++) {
      sub_tensor_shape[expanding_dim + i] = tile_shape[i];
    }
    Value config_rs1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr((
      ((sub_tensor_shape[0]&0xFFFF)<<shift48) | ((sub_tensor_shape[1]&0xFFFF)<<shift32) | ((sub_tensor_shape[2]&0xFFFF)<<shift16) | ((sub_tensor_shape[3])&0xFFFF))
    ));
    Value config_rs2 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(
      (vlane_stride<<shift32)| ((config_type&0x3)<<shift17) | ((vlane_split_axis&0x3)<<shift14) | (int(elen/8))
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
    char* config2AsmStr = getAsmString(CONFIG2);
    // config_rs1 = 1st dim stride << 32 | 2nd dim stride
    // config_rs2 = 3rd dim stride << 32 | 4th dim stride
    SmallVector<int64_t> mm_strides_4d(MAX_TENSOR_DIM, 0);
    for (int i = 0; i < static_cast<int>(mm_strides.size()); i++) {
      mm_strides_4d[mm_expanding_dim + i] = mm_strides[i];
    }
    config_rs1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(
      (mm_strides_4d[0]<<shift32) | (mm_strides_4d[1]&0xFFFFFFFF)
    ));
    config_rs2 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(
      (mm_strides_4d[2]<<shift32) | (mm_strides_4d[3]&0xFFFFFFFF)
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
    char* config3AsmStr = getAsmString(CONFIG3);
    // config_rs1 = 1st dim spad_stride << 32 | 2nd dim spad_stride
    // config_rs2 = 3rd dim spad_stride << 32 | 4th dim spad_stride
    SmallVector<int64_t> spad_strides_4d(MAX_TENSOR_DIM, 0);
    for (int i = 0; i < static_cast<int>(spad_strides.size()); i++) {
      spad_strides_4d[expanding_dim + i] = spad_strides[i];
    }
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