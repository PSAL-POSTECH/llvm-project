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
    int is_fine_grained = (vlane_stride >> 31) & 1; // FIXME: Do we need this? (in sub-tile case)
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
    SmallVector<int64_t> spad_strides;
    ValueRange indices;
    int64_t mm_offset;
    int64_t spad_offset;
    if (is_mvin) { // MVIN
      rs1 = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), SrcPtr);
      spad_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), DstPtr);
      tile_shape = dstMemRefType.getShape();
      std::tie(mm_strides, mm_offset) = getStridesAndOffset(srcMemRefType);
      std::tie(spad_strides, spad_offset) = getStridesAndOffset(dstMemRefType);
      indices = op.getSrcIndices();;
    } else { // MVOUT
      rs1 = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), DstPtr);
      spad_addr = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), SrcPtr);
      tile_shape = srcMemRefType.getShape();
      std::tie(mm_strides, mm_offset) = getStridesAndOffset(dstMemRefType);
      std::tie(spad_strides, spad_offset) = getStridesAndOffset(srcMemRefType);
      indices = op.getDstIndices();
    }
    AffineMap index_map;
    for (auto index : indices) {
      if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
        index_map = applyOp.getAffineMap();
      }
    }
    if (index_map) {
      // SmallVector<int64_t> strides(index_map.getNumDims(), 0);
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
    uint64_t spad_stride = tile_shape[0] * elen / 8;
    bool lane_split_axis = vlane_stride < subtile_shape[1] ? 0 : 1;
    if (subtile_shape.size()) {
      spad_stride = tile_shape[lane_split_axis] * elen / 8;
      tile_shape = subtile_shape;
    }

    func7 = dmaType;
    int col_factor = 1;
    uint64_t tile_row;
    uint64_t tile_col;

    if (elen < 8) {
      if (vlane_stride_byte < 1) {
        vlane_stride_byte = 1;
      }
      col_factor = 8 / elen;
      elen = 8;
    }

    if (tile_shape.size() == 2) {
      tile_row = tile_shape[0];
      tile_col = tile_shape[1];
    } else if (tile_shape.size() == 1) {
      if (is_fine_grained) {
        tile_col = std::min((tile_shape[0] / col_factor), VECTOR_LANE);
      } else {
        tile_col = tile_shape[0];
      }
    }

    char* asmStr = getAsmString(func7);
    // constants for shifting
    Value shift14 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(14));
    Value shift17 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(17));
    Value shift16 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(16));
    Value shift32 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(32));
    Value shift48 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(48));

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

    int64_t max_tensor_dim = 4;
    // config1
    char* configAsmStr = getAsmString(func7);

    // expand vlane_split_axis to 4D
    int64_t expanding_dim = max_tensor_dim - tile_shape.size();
    vlane_split_axis += expanding_dim;
    // config_rs1 = 1st dim << 48 | 2nd dim << 32 | 3rd dim << 16 | 4th dim size
    // config_rs2 = vlane_stride << 32 | config_type << 17  | vlane_split_axis << 14 | element size
    SmallVector<int64_t> sub_tensor_shape(max_tensor_dim, 1);
    for (int i = 0; i < tile_shape.size(); i++) {
      sub_tensor_shape[expanding_dim + i] = tile_shape[i];
    }
    Value config_rs1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(sub_tensor_shape[0]));
    config_rs1 = rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), config_rs1, shift48);
    config_rs1 = rewriter.create<LLVM::OrOp>(loc, config_rs1, rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(sub_tensor_shape[1])), shift32));
    config_rs1 = rewriter.create<LLVM::OrOp>(loc, config_rs1, rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(sub_tensor_shape[2])), shift16));
    config_rs1 = rewriter.create<LLVM::OrOp>(loc, config_rs1, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(sub_tensor_shape[3])));
    Value config_rs2 = rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(vlane_stride)), shift32);
    config_rs2 = rewriter.create<LLVM::OrOp>(loc, config_rs2, rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(config_type)), shift17));
    config_rs2 = rewriter.create<LLVM::OrOp>(loc, config_rs2, rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(vlane_split_axis)), shift14));
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

    // config2
    char* config2AsmStr = getAsmString(CONFIG2);
    // config_rs1 = 1st dim stride << 32 | 2nd dim stride
    // config_rs2 = 3rd dim stride << 32 | 4th dim stride
    SmallVector<int64_t> mm_strides_4d(max_tensor_dim, 1);
    for (int i = 0; i < mm_strides.size(); i++) {
      mm_strides_4d[expanding_dim + i] = mm_strides[i];
    }
    config_rs1 = rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(mm_strides_4d[0])), shift32);
    config_rs1 = rewriter.create<LLVM::OrOp>(loc, config_rs1, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(mm_strides_4d[1])));
    config_rs2 = rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(mm_strides_4d[2])), shift32);
    config_rs2 = rewriter.create<LLVM::OrOp>(loc, config_rs2, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(mm_strides_4d[3])));
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
    SmallVector<int64_t> spad_strides_4d(max_tensor_dim, 1);
    for (int i = 0; i < spad_strides.size(); i++) {
      spad_strides_4d[expanding_dim + i] = spad_strides[i];
    }
    config_rs1 = rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(spad_strides_4d[0])), shift32);
    config_rs1 = rewriter.create<LLVM::OrOp>(loc, config_rs1, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(spad_strides_4d[1])));
    config_rs2 = rewriter.create<LLVM::ShlOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(spad_strides_4d[2])), shift32);
    config_rs2 = rewriter.create<LLVM::OrOp>(loc, config_rs2, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(spad_strides_4d[3])));
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