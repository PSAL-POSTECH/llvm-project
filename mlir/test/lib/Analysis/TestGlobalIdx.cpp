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

using namespace mlir;

namespace {

struct GloablIdx : public PassWrapper<GloablIdx, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GloablIdx)
  GloablIdx() = default;
  GloablIdx(const GloablIdx &other) : PassWrapper(other) {
    vlen = other.vlen;
  }
  StringRef getArgument() const final { return "global-idx"; }
  StringRef getDescription() const final {
    return "global idx";
  }
  Option<int> vlen{*this, "vlen",
                   llvm::cl::desc("vector register size(bit)"),
                   llvm::cl::init(128)};
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect,
                    vcix::VCIXDialect, vector::VectorDialect, affine::AffineDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() override;
  affine::AffineParallelOp findParallelOp(Value input);
  int is_global_idx(mlir::Operation *operation);
  Value make_sf_vc_v_i(Location loc, OpBuilder &builder, const Type opType, uint64_t opcode);
  std::pair<unsigned, VectorType> legalizeVectorType(const Type &type);
};

} // namespace

void GloablIdx::runOnOperation() {
  auto func = getOperation();
  OpBuilder builder(func.getContext());
  func.walk([&](affine::AffineApplyOp op) {
    AffineExpr expr = op.getAffineMap().getResults()[0];
    SmallVector<Value, 8> operands = op.getMapOperands();
    if (expr.isFunctionOfDim(operands.size()-1) && is_global_idx(op)) {
      int offset = 0;
      Value operand = operands[operands.size()-1]; // FIXME: only for vlane split axis is last index
      affine::AffineParallelOp parallelOp = findParallelOp(operand);
      auto upper_bound = parallelOp.getUpperBoundMap(parallelOp.getNumDims()-1);
      if (upper_bound.isSingleConstant()) {
        offset = static_cast<int>(upper_bound.getSingleConstantResult());
      } else {
        AffineExpr expr = upper_bound.getResult(0);
        expr.walk([&](AffineExpr subExpr) {
          if (auto constExpr = llvm::dyn_cast<AffineConstantExpr>(subExpr)) {
            offset = constExpr.getValue();
          }
        });
      }
      Value result = op.getResult();
      // find vector.broadcast operation using the result as an operand
      for (auto &use : result.getUses()) {
        if (auto broadcastOp = dyn_cast<vector::BroadcastOp>(use.getOwner())) {
          // find arith.index_cast operation using the result as an operand
          Value broad_result = broadcastOp.getResult();
          for (auto &nesteduse : broad_result.getUses()) {
            if (auto castOp = dyn_cast<arith::IndexCastOp>(nesteduse.getOwner())) {
              const Type opType = castOp.getResult().getType();
              auto [n, legalType] = legalizeVectorType(opType);
              if (!legalType) {
                op.emitError() << "cannot legalize type for RVV";
                return;
              }
              Location loc = castOp.getLoc();
              Value vec = castOp.getResult();
              uint64_t opcode = 0b000000;
              builder.setInsertionPointAfter(castOp);
              Value res = make_sf_vc_v_i(loc, builder, opType, opcode);
              // make offset vector with size vec
              VectorType vt = cast<VectorType>(opType);
              auto vectorAttr = DenseElementsAttr::get(vt, builder.getI64IntegerAttr(offset));
              Value offset_vec = builder.create<arith::ConstantOp>(loc, vt, vectorAttr);
              Value offset_add = builder.create<arith::MulIOp>(loc, res, offset_vec);
              Value res_add = builder.create<arith::AddIOp>(loc, vec, offset_add);
              vec.replaceUsesWithIf(res_add, [&](OpOperand &operand) {
                return operand.getOwner() != res_add.getDefiningOp();
              });
            }
          }
        }
      }
    }
  });
}

affine::AffineParallelOp GloablIdx::findParallelOp(Value input) {
  auto operation = input.getDefiningOp();
  if (operation) {
    if (auto parallelOp = dyn_cast<affine::AffineParallelOp>(operation)) {
      return parallelOp;
    }
    for (auto operand : operation->getOperands()) {
      affine::AffineParallelOp ret = findParallelOp(operand);
      if (ret) {
        return ret;
      }
    }
  } else if (auto blockArg = dyn_cast<BlockArgument>(input)) {
    if (auto parallelOp = llvm::dyn_cast<affine::AffineParallelOp>(blockArg.getOwner()->getParentOp())) {
      if (llvm::isa<affine::AffineParallelOp>(parallelOp)) {
        return parallelOp;
      }
    }
  } else {
    llvm::errs() << "Can't find parallel op for global idx\n";
  }
  return nullptr;
}

int GloablIdx::is_global_idx(mlir::Operation *operation) {
  auto attr = operation->getAttr("global_idx");
  if (!attr)
    return 0;

  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getInt(); // Treat non-zero as true
  else
    return 1;
}

Value GloablIdx::make_sf_vc_v_i(Location loc, OpBuilder &builder, const Type opType, uint64_t opcode) {
  Attribute zeroImmAttr = builder.getI32IntegerAttr(0);
  Attribute opcodeAttr = builder.getI64IntegerAttr(opcode);
  VectorType vt = cast<VectorType>(opType);
  int nr_element = vt.getShape()[0];
  Value rvl = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(nr_element));
  return builder.create<vcix::UnaryImmOp>(loc, vt, opcodeAttr, zeroImmAttr, zeroImmAttr, rvl);
}

std::pair<unsigned, VectorType> GloablIdx::legalizeVectorType(const Type &type) {
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
  else if (auto intTy = dyn_cast<IndexType>(eltTy))
    sew = 64;
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
      return {n, VectorType::get({vlen / (sew / 8)}, eltTy)}; // max eltcount = vlen / element size [byte]
  }
  return {n, VectorType::get({eltCount >> (n - 1)}, eltTy, {true})};
}

namespace mlir {
namespace test {
void registerGloablIdxPass() { PassRegistration<GloablIdx>(); }
} // namespace test
} // namespace mlir