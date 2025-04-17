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
  GloablIdx(const GloablIdx &other) : PassWrapper(other) {}
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
  int get_vlane_offset(mlir::Operation *operation);
  Value make_sf_vc_v_i(Location loc, OpBuilder &builder, const Type opType, uint64_t opcode);
  std::pair<unsigned, VectorType> legalizeVectorType(const Type &type);
};

} // namespace

void GloablIdx::runOnOperation() {
  auto func = getOperation();
  OpBuilder builder(func.getContext());
  func.walk([&](arith::AddIOp op) {
    if (get_vlane_offset(op)!=-1) {
      int offset = get_vlane_offset(op);

      Location loc = op.getLoc();
      Value vec = op.getResult();
      const Type opType = vec.getType();
      auto [n, legalType] = legalizeVectorType(opType);
      if (!legalType) {
        op.emitError() << "cannot legalize type for RVV";
        return;
      }
      builder.setInsertionPointAfter(op);
      uint64_t opcode = 0b000000;
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
  });
}

int GloablIdx::get_vlane_offset(mlir::Operation *operation) {
  auto attr = operation->getAttr("vlane_offset");
  if (!attr)
    return -1;

  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getInt(); // Treat non-zero as true
  else
    return -1;
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