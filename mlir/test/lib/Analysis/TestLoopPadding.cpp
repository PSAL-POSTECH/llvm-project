//===- TestLoopPadding.cpp - Test CFG loop info analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing the TileOperationGraph analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// Helper function to get the upper bound of an affine.for loop
int64_t getLoopUpperBound(mlir::affine::AffineForOp forOp) {
  if (auto constantOp = forOp.getUpperBoundMap().getSingleConstantResult()) {
    return constantOp;
  }
  // Handle other cases if needed
  return -1;  // This is an example, handle dynamic cases as needed
}

// Helper function to set the upper bound of an affine.for loop
void setLoopUpperBound(mlir::affine::AffineForOp forOp, int64_t newUpperBound) {
  OpBuilder builder(forOp);
  auto newBoundMap = builder.getConstantAffineMap(newUpperBound);
  forOp.setUpperBoundMap(newBoundMap);
}

std::pair<Value, bool> getDramMemRef(mlir::memref::DmaStartOp dmaOp) {
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

class MapDimInfo {
public:
  enum MapDimType {
    APPLY = 0,
    LOOP_VAR = 1
  };
private:
  MapDimType type;
  mlir::AffineMap affineMap;
  std::vector<MapDimInfo *> argMapDimList;
  std::tuple<int64_t, int64_t, int64_t> loopRange;
public:
  std::vector<MapDimInfo *>& getArgMapDimInfo() { return argMapDimList; }
  void addForOp(affine::AffineForOp affineForOp) {
    type = MapDimType::LOOP_VAR;
    int64_t lowerBound = 0;
    int64_t upperBound = getLoopUpperBound(affineForOp);
    int64_t stepSize = 1;
    loopRange = std::make_tuple(lowerBound, upperBound, stepSize);
  }
  void addConstantIndexOp(arith::ConstantIndexOp op) {
    type = MapDimType::LOOP_VAR;
    int64_t lowerBound = op.value();
    int64_t upperBound = op.value() + 1;
    int64_t stepSize = 1;
    loopRange = std::make_tuple(lowerBound, upperBound, stepSize);
  }
  const mlir::AffineMap& getMap() { return affineMap; }
  void addApplyOp(affine::AffineApplyOp applyOp) {
    type = MapDimType::APPLY;
    affineMap = applyOp.getAffineMap();
    for (mlir::Value operand : applyOp.getMapOperands()) {
      // Walk up the block to find the loop defining this operand as an induction variable
      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
        if (auto owner = blockArg.getOwner()) {
          auto operation = owner->getParentOp();
          if (auto affineForOp = llvm::dyn_cast<affine::AffineForOp>(operation)) {
            MapDimInfo *subDimInfo = new MapDimInfo();
            subDimInfo->addForOp(affineForOp);
            argMapDimList.push_back(subDimInfo);
          }
        }
      } else if (auto nestedApplyOp = operand.getDefiningOp<mlir::affine::AffineApplyOp>()) {
        MapDimInfo *subDimInfo = new MapDimInfo();
        subDimInfo->addApplyOp(nestedApplyOp);
        argMapDimList.push_back(subDimInfo);
      } else if (auto constandOp = operand.getDefiningOp<arith::ConstantIndexOp>()) {
        MapDimInfo *subDimInfo = new MapDimInfo();
        subDimInfo->addConstantIndexOp(constandOp);
        argMapDimList.push_back(subDimInfo);
      } else {
        applyOp.emitError("Unexpected apply format...");
      }
    }
  }
  void print(std::string indent) const {
    llvm::errs() << indent <<"Log for MapDimInfo:\n";
    if (type == MapDimType::APPLY) {
      llvm::errs() << indent << "  AffineMap: " << affineMap << "\n";
      std::string new_indent = indent + "  ";
      for (auto *argDim : argMapDimList) {
        argDim->print(new_indent);
      }
    } else if (type == MapDimType::LOOP_VAR) {
      llvm::errs() << indent << "  loopRange(" << \
                      std::get<0>(loopRange) << ", " << \
                      std::get<1>(loopRange) << ", " << \
                      std::get<2>(loopRange) << ")\n";
    }
  }
  std::pair<mlir::Value, mlir::Value> createLoops(mlir::OpBuilder& builder, MapDimInfo& pair) {
    if (type == MapDimType::APPLY) {
      mlir::SmallVector<mlir::Value> mapOperands;
      mlir::SmallVector<mlir::Value> pairMapOperands;
      for (size_t i=0; i<argMapDimList.size();i++) {
        auto* argDim = argMapDimList.at(i);
        auto* pairArgDim = pair.getArgMapDimInfo().at(i);
        mlir::Value ret, pairRet;
        std::tie(ret, pairRet) = argDim->createLoops(builder, *pairArgDim);
        mapOperands.push_back(ret);
        pairMapOperands.push_back(pairRet);
      }
      auto applyOp = builder.create<affine::AffineApplyOp>(builder.getUnknownLoc(), getMap(), mapOperands);
      auto pairApplyOp = builder.create<affine::AffineApplyOp>(builder.getUnknownLoc(), pair.getMap(), pairMapOperands);
      return std::make_pair(applyOp.getResult(), pairApplyOp.getResult());
    } else if (type == MapDimType::LOOP_VAR) {
      int64_t lowerBound = std::get<0>(loopRange);
      int64_t upperBound = std::get<1>(loopRange);
      int64_t stepSize = std::get<2>(loopRange);

      auto last = builder.create<affine::AffineForOp>(builder.getUnknownLoc(), lowerBound, upperBound, stepSize);
      builder.setInsertionPointToStart(last.getBody());
      return std::make_pair(last.getInductionVar(), last.getInductionVar());
    }
  }
  bool isEqual(const MapDimInfo &other) const {
    if (type==MapDimType::LOOP_VAR) {
      return loopRange != other.loopRange;
    }

    if (affineMap != other.affineMap)
      return false;
    if (argMapDimList.size() != other.argMapDimList.size())
      return false;

    for (size_t i = 0; i < argMapDimList.size(); i++) {
      if (!argMapDimList[i]->isEqual(*other.argMapDimList[i])) {
        return false;
      }
    }
    return true;
  }
};

class MemRefAffineMapForOps {
private:
  mlir::Value memRef;                            // Single memref
  std::vector<mlir::AffineMap> affineMap;        // Single affine map
  std::vector<std::tuple<int64_t, int64_t, int64_t>> loopRange;   // Multiple AffineForOp loops
  std::vector<MapDimInfo> info;
  bool is_write;
  int padding_type;
public:
  MemRefAffineMapForOps(mlir::Value memRef, bool is_write, int padding_type)
  : memRef(memRef), is_write(is_write), padding_type(padding_type) {}
  MemRefAffineMapForOps(mlir::Value memRef, affine::AffineApplyOp applyOp, bool is_write, int padding_type)
  : memRef(memRef), is_write(is_write), padding_type(padding_type) {
    affineMap.push_back(applyOp.getAffineMap());
    addLoopsFromApplyOp(applyOp);
  }
  MemRefAffineMapForOps(mlir::Value memRef, affine::AffineForOp affineForOp, bool is_write, int padding_type)
  : memRef(memRef), is_write(is_write), padding_type(padding_type) {
    mlir::MLIRContext *context = memRef.getContext();
    mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, context); // Represents d0
    mlir::AffineMap identityMap = mlir::AffineMap::get(1, 0, d0);
    affineMap.push_back(identityMap);
    addLoopFromAffineFor(affineForOp);
  }
  mlir::Value getMemRef() const { return memRef; }
  const std::vector<mlir::AffineMap>& getAffineMap() const { return affineMap; }
  void addAffineMap(mlir::AffineMap map) { affineMap.push_back(map); }
  void addLoopsFromApplyOp(affine::AffineApplyOp applyOp);
  void addLoopFromAffineFor(affine::AffineForOp affineForOp);
  bool getIsWrite() { return is_write; }
  int getPaddingType() { return padding_type; }
  std::vector<MapDimInfo>& getDimInfo() { return info; }
  bool areLoopRangesEqual(MemRefAffineMapForOps &other) const {
    if (info.size() != other.getDimInfo().size())
      return false;
    for (uint64_t i=0; i<info.size(); i++) {
      auto operand1 = info.at(i);
      auto operand2 = other.getDimInfo().at(i);
      if (!operand1.isEqual(operand2))
        return false;
    }
    return true;
  }
  void printLog() const {
    llvm::errs() << "Log for MemRefAffineMapForOps:\n";
    if (memRef)
      llvm::errs() << "  MemRef: " << memRef << "\n";
    else
      llvm::errs() << "  MemRef: Not set\n";

    std::string indent = "  ";
    for (auto& dimInfo : info) {
      dimInfo.print(indent);
    }
  }
};

/// A testing pass that applies the TileOperationGraph analysis on a region and prints
/// the information it collected to llvm::errs().
struct TestLoopPadding
    : public PassWrapper<TestLoopPadding, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLoopPadding)
  StringRef getArgument() const final { return "test-loop-padding"; }
  StringRef getDescription() const final {
    return "Test loop padding";
  }
  TestLoopPadding() = default;
  TestLoopPadding(const TestLoopPadding &) {}

  void updateFunctionSignatureWithMemRef(func::FuncOp function, Value dramMemRef);
  void runOnOperation() override;
  void getAffineForBounds(affine::AffineForOp &op, int &start, int &end, int &step);
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect, linalg::LinalgDialect>();
  }
  int64_t roundUpToMultiple(int64_t value, int64_t multiple);
  void modifyMemrefWithPadding(Value memref, int64_t stepSize, int64_t paddedUpperBound);
  int findLoopVariableIndex(mlir::Value loopVar);
  int getArgumentIndex(mlir::affine::AffineApplyOp applyOp, mlir::Value targetValue);
  SmallVector<std::pair<int64_t, unsigned>, 4> collectCoefficientsFromAffineExpr(mlir::AffineExpr expr);
  int64_t getCoefficientFromDim(mlir::AffineExpr expr, int dim);
  mlir::AffineExpr updateAffineExprWithBounds(mlir::AffineExpr expr,
                                            int updated_position_index,
                                            int64_t upperBound,
                                            int64_t paddedUpperBound,
                                            mlir::MLIRContext *context);
  AffineMap updateAffineMapWithNewExpr(AffineMap& oldMap,
                                       mlir::AffineExpr updatedExpr,
                                       unsigned exprIndex,
                                       mlir::MLIRContext *context,
                                       affine::AffineForOp& forOp);
                                       // Function to check if a loop is already registered
  void analysisDMAStartNode(func::FuncOp function, std::vector<MemRefAffineMapForOps>& targetBuffer);
  void createWrapperFunction(mlir::ModuleOp module, mlir::OpBuilder builder, mlir::FunctionType kernelFuncType, bool timing_mode);
  int getPaddingType(mlir::Operation *operation);
  std::vector<std::tuple<int64_t, int64_t, Value>> loopInfoMap;
  std::vector<MemRefAffineMapForOps> prePaddingInfo;
  std::vector<MemRefAffineMapForOps> postPaddingInfo;

  Option<int> timing_mode{*this, "timing_mode",
                   llvm::cl::desc("timing mode: skip copying from padding"),
                   llvm::cl::init(0)};
};

} // namespace

void MemRefAffineMapForOps::addLoopsFromApplyOp(affine::AffineApplyOp applyOp) {
  MapDimInfo dimInfo;
  dimInfo.addApplyOp(applyOp);
  info.push_back(dimInfo);
}

void MemRefAffineMapForOps::addLoopFromAffineFor(affine::AffineForOp affineForOp) {
  MapDimInfo dimInfo;
  dimInfo.addForOp(affineForOp);
  info.push_back(dimInfo);
}

void TestLoopPadding::updateFunctionSignatureWithMemRef(func::FuncOp function, Value dramMemRef) {
  // Get the current function type
  auto funcType = function.getFunctionType();
  SmallVector<Type, 4> newArgTypes(funcType.getInputs().begin(), funcType.getInputs().end());

  for (auto &arg : function.getBody().getArguments()) {
    if (arg.getType() == dramMemRef.getType()) {
      arg.setType(dramMemRef.getType());
      newArgTypes[arg.getArgNumber()] = dramMemRef.getType();
    }
  }
  auto newFuncType = FunctionType::get(function.getContext(), newArgTypes, funcType.getResults());
  function.setType(newFuncType);
}

void TestLoopPadding::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();
  mlir::ModuleOp module = getOperation()->getParentOfType<mlir::ModuleOp>();
  func::FuncOp prevKernelFunc = module.lookupSymbol<func::FuncOp>("kernel");
  if (!prevKernelFunc) {
    module.emitError() << "Function 'kernel' not found!\n";
    return;
  }
  mlir::FunctionType prevkernelFuncType = prevKernelFunc.getFunctionType();

  // Analysis pre-padding info
  analysisDMAStartNode(function, prePaddingInfo);
  //for (auto i : prePaddingInfo)
  //  i.printLog();

  // Traverse the function to find affine.for loops
  function.walk([&](mlir::affine::AffineForOp forOp) {
    // Step 1: Get loop step size and end condition
    int64_t stepSize = forOp.getStep().getZExtValue();
    int64_t upperBound = getLoopUpperBound(forOp);
    if (upperBound == -1) {
      forOp.emitError() << "Unexpected loop upper size\n";
      return;
    }
    loopInfoMap.push_back(std::make_tuple(upperBound, stepSize, forOp.getInductionVar()));

    // Step 2: Modify the loop bounds to make upperBound a multiple of stepSize
    int64_t paddedUpperBound = roundUpToMultiple(upperBound, stepSize);
    setLoopUpperBound(forOp, paddedUpperBound);
    if (paddedUpperBound == upperBound)
      return;

    // Step 3: traverse dmaOp to store original affineExpr which are used to get orignal stride
    std::unordered_map<const void*, mlir::AffineExpr> dmaOpExpr;
    function.walk([&](mlir::memref::DmaStartOp dmaOp) {
      for (auto operand : dmaOp.getOperands()) {
        auto applyOp = operand.getDefiningOp<mlir::affine::AffineApplyOp>();
        if (!applyOp) {
          continue;
        }
        mlir::AffineMap map = applyOp.getAffineMap();
        mlir::AffineExpr expr = map.getResult(0);
        dmaOpExpr[dmaOp.getAsOpaquePointer()] = expr;
      }
    });
    std::vector<Value> memRefSet;
    //llvm::errs() << "=============updated axis===========\n";
    function.walk([&](mlir::memref::DmaStartOp dmaOp) {
      auto result = getDramMemRef(dmaOp);
      Value dram_memref = result.first;
      Value fifthOperand = dmaOp.getStride();
      auto strideVal = fifthOperand.getDefiningOp<arith::ConstantIndexOp>();
      //auto index_pos = findLoopVariableIndex(forOp.getInductionVar());

      if (!strideVal) {
        dmaOp.emitError() << "Can't found stride value from dmaOp\n";
        dmaOp.dump();
        return;
      }
      // this axis affecti this dmaOp
      bool isAffected = false;
      for (auto operand : dmaOp.getOperands()) {
        if (auto applyOp = operand.getDefiningOp<mlir::affine::AffineApplyOp>()) {
          AffineMap map = applyOp.getAffineMap();
          for (unsigned i = 0; i < map.getNumDims(); i++) {
            Value loopVar = applyOp.getOperand(i);
            if (loopVar == forOp.getInductionVar()) {
              isAffected = true;
              break;
            }
          }
        } else if (operand.getAsOpaquePointer() == forOp.getInductionVar().getAsOpaquePointer()){
          isAffected = true;
          break;
        }
      }
      if (!isAffected)
        return;
      if (std::find(memRefSet.begin(), memRefSet.end(), dram_memref) == memRefSet.end()) {
        // Padding memref and update the function signature to match the new padded memref types
        modifyMemrefWithPadding(dram_memref, upperBound, paddedUpperBound);
        updateFunctionSignatureWithMemRef(function, dram_memref);
        memRefSet.push_back(dram_memref);
      }

      for (auto operand : dmaOp.getOperands()) {
        int expr_idx = 0;
        auto applyOp = operand.getDefiningOp<mlir::affine::AffineApplyOp>();
        if (!applyOp) {
          continue;
        }
        auto index_pos = getArgumentIndex(applyOp, forOp.getInductionVar());
        AffineMap map = applyOp.getAffineMap();
        AffineExpr expr = dmaOpExpr[dmaOp.getAsOpaquePointer()];
        /* Need to check which is updated */

        /* Update affine expression */
        AffineExpr newExpr = updateAffineExprWithBounds(expr, index_pos, upperBound, paddedUpperBound, context);
        map = updateAffineMapWithNewExpr(map, newExpr, expr_idx, context, forOp);
        applyOp.setMap(map);
      }
    });
  });

  // Analysis post-padding info
  analysisDMAStartNode(function, postPaddingInfo);
  //for (auto i : postPaddingInfo)
  //  i.printLog();

  // Create padding wrapper function
  mlir::OpBuilder builder(module.getContext());
  // FIXME. I want to share wrapper. But validiation binary failed to provide functionality when padding is applied.
  createWrapperFunction(module, builder, prevkernelFuncType, timing_mode);
  return;
}

// Helper function to round up a value to the nearest multiple
int64_t TestLoopPadding::roundUpToMultiple(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}


// Helper function to modify a memref type with padding
void TestLoopPadding::modifyMemrefWithPadding(Value memref, int64_t upperBound, int64_t paddedUpperBound) {
  auto memrefType = mlir::dyn_cast<MemRefType>(memref.getType());
  if (!memrefType)
    return;

  // Check if the memref is 1D (one-dimensional)
  if (memrefType.getRank() != 1) {
    memref.getDefiningOp()->emitError() << "Expected 1D memref, but got a memref of rank "
                                        << memrefType.getRank();
    return;
  }

  // Calculate the original flattened size (for 1D memref, it's just the first dimension)
  int64_t flattenedSize = memrefType.getShape().front();  // Only one dimension, so it's the size itself
  int64_t newFlattenedSize = (flattenedSize / upperBound) * paddedUpperBound;
  SmallVector<int64_t, 1> newShape = {newFlattenedSize};  // For 1D, the shape is a single value

  // Recalculate the flattened size based on the padding and set the new memref type
  auto paddedMemrefType = MemRefType::get(newShape, memrefType.getElementType());

  // Replace the original memref type with the padded one
  memref.setType(paddedMemrefType);
}

SmallVector<std::pair<int64_t, unsigned>, 4> TestLoopPadding::collectCoefficientsFromAffineExpr(mlir::AffineExpr expr) {
  SmallVector<std::pair<int64_t, unsigned>, 4> coefficients;

  std::function<void(mlir::AffineExpr)> collectCoefficients = [&](mlir::AffineExpr expr) {
    if (auto binExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
      if (binExpr.getKind() == mlir::AffineExprKind::Mul) {
        if (auto lhs = llvm::dyn_cast<mlir::AffineConstantExpr>(binExpr.getLHS())) {
          if (auto rhs = llvm::dyn_cast<mlir::AffineDimExpr>(binExpr.getRHS())) {
            coefficients.push_back(std::make_pair(lhs.getValue(), rhs.getPosition()));
          }
        } else if (auto rhs = llvm::dyn_cast<mlir::AffineConstantExpr>(binExpr.getRHS())) {
          if (auto lhs = llvm::dyn_cast<mlir::AffineDimExpr>(binExpr.getLHS())) {
            coefficients.push_back(std::make_pair(rhs.getValue(), lhs.getPosition()));
          }
        }
      } else if (binExpr.getKind() == mlir::AffineExprKind::Add) {
        // Add: Recursively collect both sides
        collectCoefficients(binExpr.getLHS());
        collectCoefficients(binExpr.getRHS());
      }
    } else if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      coefficients.push_back(std::make_pair(1, dimExpr.getPosition()));
    }
  };

  collectCoefficients(expr);
 return coefficients;
}

int64_t TestLoopPadding::getCoefficientFromDim(mlir::AffineExpr expr, int dim) {
  std::function<int64_t(mlir::AffineExpr)> collectCoefficients = [&](mlir::AffineExpr expr) {
    if (auto binExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
      if (binExpr.getKind() == mlir::AffineExprKind::Mul) {
        if (auto lhs = llvm::dyn_cast<mlir::AffineConstantExpr>(binExpr.getLHS())) {
          if (auto rhs = llvm::dyn_cast<mlir::AffineDimExpr>(binExpr.getRHS())) {
            if (rhs.isFunctionOfDim(dim)) {
              return lhs.getValue();
            }
          }
        } else if (auto rhs = llvm::dyn_cast<mlir::AffineConstantExpr>(binExpr.getRHS())) {
          if (auto lhs = llvm::dyn_cast<mlir::AffineDimExpr>(binExpr.getLHS())) {
            if (lhs.isFunctionOfDim(dim)) {
              return rhs.getValue();
            }
          }
        }
      } else if (binExpr.getKind() == mlir::AffineExprKind::Add) {
        // Add: Recursively collect both sides
        int64_t result = collectCoefficients(binExpr.getLHS());
        if (result != -1)
          return result;
        result = collectCoefficients(binExpr.getRHS());
        if (result != -1)
          return result;
      }
    } else if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      if (dimExpr.isFunctionOfDim(dim)) {
        return 1L;
      }
    }
    return -1L;
  };
  return collectCoefficients(expr);
}

mlir::AffineExpr TestLoopPadding::updateAffineExprWithBounds(mlir::AffineExpr expr,
                                            int updated_position_index,
                                            int64_t upperBound,
                                            int64_t paddedUpperBound,
                                            mlir::MLIRContext *context) {
  // Collect coefficients from the AffineExpr
  auto coefficients = collectCoefficientsFromAffineExpr(expr);

  // Step 2: Modify the coefficients based on the updated_position_index
  SmallVector<std::tuple<int64_t, int>, 4> modifiedCoefficients;
  int64_t targetCoefficient = getCoefficientFromDim(expr, updated_position_index);
  //llvm::dbgs() << "Target coeff: " << targetCoefficient << "\n";
  //llvm::dbgs() << "Upper: " << upperBound << "\n";
  //llvm::dbgs() << "PaddedUpper: " << paddedUpperBound << "\n";

  for (int i = 0; i < static_cast<int>(coefficients.size()); ++i) {
    int64_t coeff = coefficients[i].first;
    int position = coefficients[i].second;
    if (coeff > targetCoefficient || (coeff == targetCoefficient && position < updated_position_index))
      coeff = (coeff / upperBound) * paddedUpperBound;
    modifiedCoefficients.push_back(std::make_tuple(coeff, position));
  }

  // Updated coeff
  //llvm::dbgs() << "New Coeff: ";
  //for (auto coeff : modifiedCoefficients) {
  //  llvm::dbgs() << std::get<0>(coeff) << ", ";
  //}
  //llvm::dbgs() << "\n";

  std::function<mlir::AffineExpr(mlir::AffineExpr)> rebuildExprWithUpdatedCoefficients =
      [&](mlir::AffineExpr expr) -> mlir::AffineExpr {
    if (auto binExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
      mlir::AffineExpr lhs = rebuildExprWithUpdatedCoefficients(binExpr.getLHS());
      mlir::AffineExpr rhs = rebuildExprWithUpdatedCoefficients(binExpr.getRHS());

      auto result = binExpr.getKind() == mlir::AffineExprKind::Add ? lhs + rhs : lhs * rhs;
      if (binExpr.getKind() == mlir::AffineExprKind::Add)
        auto result =  lhs + rhs;
      else {
        if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs)) {
          return lhs;
        } else {
          return rhs;
        }
      }

      return result;
    } else if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      for (const auto& entry : modifiedCoefficients) {
        int64_t coeff;
        unsigned int position;
        std::tie(coeff, position) = entry;
        if (position == dimExpr.getPosition()) {
          auto expr = mlir::getAffineDimExpr(position, context) * mlir::getAffineConstantExpr(coeff, context);
          return expr;
        }
      }
      llvm::errs() << "Failed to update dimension \"" << dimExpr << "\"";
      return dimExpr;
    } else if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
      return mlir::getAffineConstantExpr(constExpr.getValue(), context);
    }
    return expr; // Return the original expression if no specific case matches
  };

  return rebuildExprWithUpdatedCoefficients(expr);
}

int TestLoopPadding::findLoopVariableIndex(mlir::Value loopVar) {
  int index = 0;
  for (const auto& entry : loopInfoMap) {
    const auto& [upperBound, stepSize, storedValue] = entry;
    if (storedValue == loopVar) {
      return index;
    }
    ++index;
  }
  return -1;
}

int TestLoopPadding::getArgumentIndex(mlir::affine::AffineApplyOp applyOp, mlir::Value targetValue) {
  mlir::OperandRange operands = applyOp.getOperands();
  for (size_t i = 0; i < operands.size(); ++i) {
    if (operands[i] == targetValue) {
      return i;
    }
  }
  return -1;
}

AffineMap TestLoopPadding::updateAffineMapWithNewExpr(AffineMap& oldMap,
                                           mlir::AffineExpr updatedExpr,
                                           unsigned exprIndex,
                                           mlir::MLIRContext *context,
                                           mlir::affine::AffineForOp& forOp) {
  llvm::SmallVector<mlir::AffineExpr, 4> exprs(oldMap.getResults().begin(), oldMap.getResults().end());

  if (exprIndex < exprs.size()) {
    exprs[exprIndex] = updatedExpr;
  } else {
    llvm::errs() << "Failed to update AffineMap\n";
    return oldMap;
  }
  AffineMap newMap = AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), exprs, context);
  return newMap;
}

void TestLoopPadding::analysisDMAStartNode(
  func::FuncOp function, std::vector<MemRefAffineMapForOps>& targetBuffer)
{
  // Analysis pre-padding info
  function.walk([&](mlir::memref::DmaStartOp dmaOp) {
    auto result = getDramMemRef(dmaOp);
    Value dram_memref = result.first;
    bool is_write = result.second;
    int padding_type = getPaddingType(dmaOp);
    auto memRefMap = MemRefAffineMapForOps(dram_memref, is_write, padding_type);
    auto &memRefMapRef = memRefMap;
    bool is_registerd = false;

    for (auto &existingInfo : targetBuffer) {
      if (existingInfo.getMemRef() == dram_memref) {
        memRefMapRef = existingInfo;
        is_registerd = true;
        break;
      }
    }

    for (auto operand : dmaOp.getOperands()) {
      if (auto applyOp = operand.getDefiningOp<mlir::affine::AffineApplyOp>()) {
        memRefMapRef.addLoopsFromApplyOp(applyOp);
        break;
      } else if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(operand)) {
        auto definingOp = blockArg.getOwner()->getParentOp();
        if (auto affineForOp = llvm::dyn_cast<mlir::affine::AffineForOp>(definingOp)) {
          memRefMapRef.addLoopFromAffineFor(affineForOp);
          break;
        }
      }
    }

    if (!is_registerd)
      targetBuffer.push_back(memRefMapRef);
  });
}

void TestLoopPadding::createWrapperFunction(mlir::ModuleOp module, mlir::OpBuilder builder, mlir::FunctionType prevKernelFuncType, bool timing_mode) {
  func::FuncOp kernelFunc = module.lookupSymbol<func::FuncOp>("kernel");
  if (!kernelFunc) {
    module.emitError() << "Function 'kernel' not found!\n";
    return;
  }
  std::vector<std::optional<mlir::memref::AllocOp>> padded_buffer;
  std::set<void*> usedMemRefs;
  padded_buffer.resize(kernelFunc.getNumArguments());
  affine::AffineForOp last;

  // Create wrapper function
  builder.setInsertionPointToEnd(module.getBody());
  auto wrapperFunc = builder.create<func::FuncOp>(
      module.getLoc(), "wrapper_kernel", prevKernelFuncType);
  mlir::Block *entryBlock = wrapperFunc.addEntryBlock();

  // Declare local buffers
  for (size_t i = 0; i < prePaddingInfo.size() && i < postPaddingInfo.size(); ++i) {
    builder.setInsertionPointToStart(entryBlock);
    auto& postInfo = postPaddingInfo[i];
    auto& preInfo = prePaddingInfo[i];

    mlir::Value postMemRef = postInfo.getMemRef();
    auto blockArg = mlir::cast<mlir::BlockArgument>(postMemRef);
    unsigned int argIdx = blockArg.getArgNumber();

    if (preInfo.areLoopRangesEqual(postInfo)) {
      continue;
    }

    Value indexZero = builder.create<arith::ConstantOp>(wrapperFunc.getLoc(), builder.getIndexType(), builder.getIntegerAttr(builder.getIndexType(), 0));
    if (usedMemRefs.find(postMemRef.getAsOpaquePointer()) == usedMemRefs.end()) {
      std::string paddedBufName = std::string("_padding_buffer") + std::to_string(i);
      mlir::MemRefType paddedMemRefType = mlir::dyn_cast<mlir::MemRefType>(postMemRef.getType());
      Type elementType = paddedMemRefType.getElementType();
      Value zeroValue;
      if (elementType.isF32()) {
        zeroValue = builder.create<arith::ConstantOp>(wrapperFunc.getLoc(), elementType, builder.getF32FloatAttr(0.0));
      } else if (elementType.isInteger(32)) {
        zeroValue = builder.create<arith::ConstantOp>(wrapperFunc.getLoc(), elementType, builder.getIntegerAttr(elementType, 0));
      } else {
        wrapperFunc.emitError("Unsupported padding type");
      }
      auto allocOp = builder.create<mlir::memref::AllocOp>(
        builder.getUnknownLoc(), paddedMemRefType);
      auto paddedAllocOp = builder.create<mlir::linalg::FillOp>(wrapperFunc.getLoc(), ValueRange{zeroValue}, ValueRange{allocOp});
      padded_buffer[argIdx] = allocOp;
      usedMemRefs.insert(postMemRef.getAsOpaquePointer());
    }
  }
  builder.setInsertionPointToStart(entryBlock);

  // Create read padding phase
  for (size_t i = 0; i < prePaddingInfo.size() && i < postPaddingInfo.size() && !timing_mode; ++i) {
    auto& preInfo = prePaddingInfo[i];
    auto& postInfo = postPaddingInfo[i];

    mlir::SmallVector<mlir::Value, 2> mapOperands;
    std::vector<AffineMap> preAffineMap = preInfo.getAffineMap();
    std::vector<AffineMap> postAffineMap = postInfo.getAffineMap();
    builder.setInsertionPointToEnd(entryBlock);
    mlir::Value preMemRef = preInfo.getMemRef();
    auto blockArg = mlir::cast<mlir::BlockArgument>(preMemRef);
    unsigned int argIdx = blockArg.getArgNumber();
    mlir::Value argValue = wrapperFunc.getArgument(argIdx);

    if (!padded_buffer[argIdx].has_value())
      continue;
    mlir::memref::AllocOp allocatedBuffer = padded_buffer[argIdx].value();

    for (size_t i=0; i<preInfo.getDimInfo().size(); i++) {
      auto& preDimInfo = preInfo.getDimInfo().at(i);
      auto& postDimInfo = postInfo.getDimInfo().at(i);
      mlir::Value preIndex, postIndex;
      std::tie(preIndex, postIndex) = preDimInfo.createLoops(builder, postDimInfo);
      auto loadedValue = builder.create<affine::AffineLoadOp>(builder.getUnknownLoc(), argValue, preIndex);
      builder.create<affine::AffineStoreOp>(builder.getUnknownLoc(), loadedValue, allocatedBuffer.getMemref(), postIndex);
    }
  }

  // Prepare the arguments of the wrapper to be passed to the 'kernel' function
  llvm::SmallVector<mlir::Value, 4> callArgs;
  callArgs.resize(kernelFunc.getNumArguments());
  builder.setInsertionPointToEnd(entryBlock);
  for (size_t i=0; i<padded_buffer.size(); i++) {
    if (!padded_buffer[i].has_value()) {
      callArgs[i] = wrapperFunc.getArgument(i);
    } else {
      mlir::memref::AllocOp allocatedBuffer = padded_buffer[i].value();
      callArgs[i] = allocatedBuffer.getResult();
    }
  }

  // Create a call to the 'kernel' function inside the wrapper
  builder.create<mlir::func::CallOp>(module.getLoc(), kernelFunc, callArgs);

  // Create write padding phase
  for (size_t i = 0; i < prePaddingInfo.size() && i < postPaddingInfo.size() && !timing_mode; ++i) {
    auto& preInfo = prePaddingInfo[i];
    auto& postInfo = postPaddingInfo[i];

    mlir::SmallVector<mlir::Value, 2> mapOperands;
    std::vector<AffineMap> preAffineMap = preInfo.getAffineMap();
    std::vector<AffineMap> postAffineMap = postInfo.getAffineMap();
    builder.setInsertionPointToEnd(entryBlock);
    mlir::Value preMemRef = preInfo.getMemRef();
    auto blockArg = mlir::cast<mlir::BlockArgument>(preMemRef);
    unsigned int argIdx = blockArg.getArgNumber();
    mlir::Value argValue = wrapperFunc.getArgument(argIdx);

    if (!padded_buffer[argIdx].has_value())
      continue;
    mlir::memref::AllocOp allocatedBuffer = padded_buffer[argIdx].value();

    for (size_t i=0; i<preInfo.getDimInfo().size(); i++) {
      auto& preDimInfo = preInfo.getDimInfo().at(i);
      auto& postDimInfo = postInfo.getDimInfo().at(i);
      mlir::Value preIndex, postIndex;
      std::tie(preIndex, postIndex) = preDimInfo.createLoops(builder, postDimInfo);
      auto loadedValue = builder.create<affine::AffineLoadOp>(builder.getUnknownLoc(), allocatedBuffer.getMemref(), postIndex);
      builder.create<affine::AffineStoreOp>(builder.getUnknownLoc(), loadedValue, argValue, preIndex);
    }
  }

  // Add a return operation to the wrapper function
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<mlir::func::ReturnOp>(module.getLoc());
}

int TestLoopPadding::getPaddingType(mlir::Operation *operation) {
  auto attr = operation->getAttr("padding");
  if (!attr) {
    return 0;
  }
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    return intAttr.getInt();
  }
  operation->emitError() << "Invalid padding type (0: zero, 1: negative)\n";
  return 0;
}

namespace mlir {
namespace test {
void registerTestLoopPaddingPass() { PassRegistration<TestLoopPadding>(); }
} // namespace test
} // namespace mlir
