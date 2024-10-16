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

class MemRefAffineMapForOps {
private:
  mlir::Value memRef;                            // Single memref
  mlir::AffineMap affineMap;                     // Single affine map
  std::vector<std::tuple<int64_t, int64_t, int64_t>> loopRange;   // Multiple AffineForOp loops
  bool is_write;
public:
  MemRefAffineMapForOps(mlir::Value memRef, affine::AffineApplyOp applyOp, bool is_write) : memRef(memRef), is_write(is_write) {
    affineMap = applyOp.getAffineMap();
    addLoopsFromApplyOp(applyOp);
  }
  MemRefAffineMapForOps(mlir::Value memRef, affine::AffineForOp affineForOp, bool is_write) : memRef(memRef), is_write(is_write) {
    mlir::MLIRContext *context = memRef.getContext();
    mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, context); // Represents d0
    mlir::AffineMap identityMap = mlir::AffineMap::get(1, 0, d0);
    affineMap = identityMap;
    addLoopFromAffineFor(affineForOp);
  }
  void addLoopRange(std::tuple<int64_t, int64_t, int64_t>& loop) { loopRange.push_back(loop); }
  mlir::Value getMemRef() const { return memRef; }
  mlir::AffineMap getAffineMap() const { return affineMap; }
  const std::vector<std::tuple<int64_t, int64_t, int64_t>>& getLoopRange() const { return loopRange; }
  void clearLoopRange() { loopRange.clear(); }
  void addLoopsFromApplyOp(affine::AffineApplyOp applyOp);
  void addLoopFromAffineFor(affine::AffineForOp affineForOp);
  bool getIsWrite() { return is_write; }
  void printLog() const {
    llvm::errs() << "Log for MemRefAffineMapForOps:\n";
    if (memRef)
      llvm::errs() << "  MemRef: " << memRef << "\n";
    else
      llvm::errs() << "  MemRef: Not set\n";
    llvm::errs() << "  AffineMap: " << affineMap << "\n";
    llvm::errs() << "  loopRange (" << loopRange.size() << " loops):\n";
    for (auto &iter : loopRange) {
      llvm::errs() << "    Loop Lower " << std::get<0>(iter) << "\n";
      llvm::errs() << "    Loop Upper " << std::get<1>(iter) << "\n";
      llvm::errs() << "    Loop Step " << std::get<2>(iter) << "\n";
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

  void runOnOperation() override;
  void getAffineForBounds(affine::AffineForOp &op, int &start, int &end, int &step);
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect>();
  }
  int64_t roundUpToMultiple(int64_t value, int64_t multiple);
  void modifyMemrefWithPadding(Value memref, int64_t stepSize, int64_t paddedUpperBound);
  SmallVector<int64_t, 4> findCoefficient(AffineExpr expr);
  int findLoopVariableIndex(mlir::Value loopVar);
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
  bool isLoopRegistered(AffineMap& map, affine::AffineForOp& forOp) {
    // Check if the map exists in the loopMap
    auto it = paddedMapInfo.find(map.getAsOpaquePointer());
    if (it != paddedMapInfo.end()) {
      // Check if the forOp is already in the set of loops for this map
      return it->second.find(forOp.getAsOpaquePointer()) != it->second.end();
    }
    return false;  // Map not found, so forOp can't be registered
  }
  void analysisDMAStartNode(func::FuncOp function, std::vector<MemRefAffineMapForOps>& targetBuffer);
  void createWrapperFunction(mlir::ModuleOp module, mlir::OpBuilder builder, mlir::FunctionType kernelFuncType);
  std::vector<std::tuple<int64_t, int64_t, Value>> loopInfoMap;
  std::map<const void*, std::set<const void*>> paddedMapInfo;
  std::vector<MemRefAffineMapForOps> prePaddingInfo;
  std::vector<MemRefAffineMapForOps> postPaddingInfo;
};

} // namespace

void MemRefAffineMapForOps::addLoopsFromApplyOp(affine::AffineApplyOp applyOp) {
  // Extract operands of the affine.apply (which are the loop indices)
  for (mlir::Value operand : applyOp.getMapOperands()) {
    // Walk up the block to find the loop defining this operand as an induction variable
    auto blockArg = mlir::cast<mlir::BlockArgument>(operand);
    auto owner = blockArg.getOwner();
    if (owner) {
      auto operation = owner->getParentOp();
      if (auto affineForOp = llvm::dyn_cast<affine::AffineForOp>(operation))
        addLoopFromAffineFor(affineForOp);
    }
  }
}

void MemRefAffineMapForOps::addLoopFromAffineFor(affine::AffineForOp affineForOp) {
  int64_t lowerBound = 0;
  int64_t upperBound = getLoopUpperBound(affineForOp);
  int64_t stepSize = 1; //affineForOp.getStep().getZExtValue();
  loopRange.push_back(std::make_tuple(lowerBound, upperBound, stepSize));
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
    if (upperBound == -1)
        forOp.emitError() << "Unexpected loop upper size\n";
    loopInfoMap.push_back(std::make_tuple(upperBound, stepSize, forOp.getInductionVar()));

    // Step 2: Modify the loop bounds to make upperBound a multiple of stepSize
    int64_t paddedUpperBound = roundUpToMultiple(upperBound, stepSize);
    setLoopUpperBound(forOp, paddedUpperBound);
    if (paddedUpperBound == upperBound)
      return;

    // Step 3: Find dma_start operations inside the loop
    forOp.getBody()->walk([&](Operation *op) {
      if (auto dmaOp = dyn_cast<mlir::affine::AffineDmaStartOp>(op)) {
        // Step 4: Get the memref being accessed and modify its size
        auto dst_space = dmaOp.getDstMemorySpace();
        auto src_space = dmaOp.getSrcMemorySpace();
        ValueRange dram_indices;
        Value dram_memref;
        if (dst_space == 0 && src_space == 1) {
          dram_memref = dmaOp.getDstMemRef();
          dram_indices = dmaOp.getDstIndices();
        } else if (dst_space == 1 && src_space == 0) {
          dram_memref = dmaOp.getSrcMemRef();
          dram_indices = dmaOp.getSrcIndices();
        } else {
          dmaOp.emitError() << "Unexpected memory space, src: " << src_space << "des: " << dst_space << "\n";
          return;
        }
        // Padding memref
        modifyMemrefWithPadding(dram_memref, upperBound, paddedUpperBound);

        // Update the function signature to match the new padded memref types
        auto funcType = function.getFunctionType();
        SmallVector<Type, 4> newArgTypes(funcType.getInputs().begin(), funcType.getInputs().end());

        for (auto &arg : function.getBody().getArguments()) {
          if (arg.getType() == dram_memref.getType()) {
            arg.setType(dram_memref.getType());
            newArgTypes[arg.getArgNumber()] = dram_memref.getType();
          }
        }

        // Create a new function type with updated argument types
        auto newFuncType = FunctionType::get(function.getContext(), newArgTypes, funcType.getResults());
        function.setType(newFuncType);  // Set the new function type

        for (auto operand : dmaOp.getOperands()) {
          /* check that padding will affect this op */
          if (auto applyOp = operand.getDefiningOp<mlir::affine::AffineApplyOp>()) {
            int64_t mm_stride = -1;
            Value fifthOperand = dmaOp.getStride();
            auto strideVal = fifthOperand.getDefiningOp<arith::ConstantIndexOp>();
            auto index_pos = findLoopVariableIndex(forOp.getInductionVar());
            AffineMap map = applyOp.getAffineMap();
            for (unsigned i = 0; i < map.getNumResults(); ++i) {
              AffineExpr expr = map.getResult(i);

              if (!isLoopRegistered(map, forOp)) {
                AffineExpr newExpr = updateAffineExprWithBounds(expr, index_pos, upperBound, paddedUpperBound, context);
                map = updateAffineMapWithNewExpr(map, newExpr, i, context, forOp);
                applyOp.setMap(map);
                if (strideVal) {
                  /* Update stride info by finding updated coeff */
                  auto oldCoeff = findCoefficient(expr);
                  auto newCoeff = findCoefficient(newExpr);
                  for (size_t i=0; i<oldCoeff.size();i++) {
                    if (oldCoeff[i] == strideVal.value())
                      mm_stride = newCoeff[i];
                  }
                }
              }
              //llvm::errs() << "old_expr: " << expr << " new_expr: " << new_expr << "\n";
              //llvm::errs() << "index_pos: " << index_pos << " upper: " << upperBound << " paddUpper: " << paddedUpperBound << "\n";
            }
            if (strideVal && mm_stride != -1) {
              //llvm::errs() << "update mm_stride: " << strideVal.value() << "  " << mm_stride <<"\n";
              OpBuilder builder(strideVal);
              Value newStrideConstant = builder.create<arith::ConstantIndexOp>(strideVal.getLoc(), mm_stride);
              dmaOp.setOperand(dmaOp.getNumOperands() - 2, newStrideConstant);
              if (strideVal.use_empty()) {
                strideVal.erase();
              }
            }
          }
        }
      }
    });
  });

  // Analysis post-padding info
  analysisDMAStartNode(function, postPaddingInfo);
  //for (auto i : postPaddingInfo)
  //  i.printLog();

  // Create padding wrapper function
  mlir::OpBuilder builder(module.getContext());
  createWrapperFunction(module, builder, prevkernelFuncType);
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

SmallVector<int64_t, 4> TestLoopPadding::findCoefficient(AffineExpr expr) {
  SmallVector<int64_t, 4> coefficients;

  // Lambda to traverse affine expressions
  std::function<void(AffineExpr)> extractCoefficients = [&](AffineExpr expr) {
    if (auto binExpr = llvm::dyn_cast<AffineBinaryOpExpr>(expr)) {
      // For mul expressions, extract the coefficient
      if (binExpr.getKind() == AffineExprKind::Mul) {
        if (auto constExpr = llvm::dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
          coefficients.push_back(constExpr.getValue());
          return;
        } else if (auto constExpr = llvm::dyn_cast<AffineConstantExpr>(binExpr.getLHS())) {
          coefficients.push_back(constExpr.getValue());
          return;
        }
      }
      extractCoefficients(binExpr.getLHS());
      extractCoefficients(binExpr.getRHS());
    } else if (auto constExpr = llvm::dyn_cast<AffineConstantExpr>(expr)) {
      coefficients.push_back(constExpr.getValue());
    } else {
      // If it's not a binary operation, assume the coefficient is 1 (like d1)
      coefficients.push_back(1);
    }
  };
  return coefficients;
}

mlir::AffineExpr TestLoopPadding::updateAffineExprWithBounds(mlir::AffineExpr expr,
                                            int updated_position_index,
                                            int64_t upperBound,
                                            int64_t paddedUpperBound,
                                            mlir::MLIRContext *context) {
  // Step 1: Traverse the AffineExpr to collect coefficients for each index
  SmallVector<std::tuple<int64_t, unsigned>, 4> coefficients;
  std::function<void(mlir::AffineExpr)> collectCoefficients = [&](mlir::AffineExpr expr) {
    if (auto binExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
      if (binExpr.getKind() == mlir::AffineExprKind::Mul) {
        // Mul: Assume binExpr is in the form of constant * dimension (e.g., 47 * index0)
        if (auto lhs = llvm::dyn_cast<mlir::AffineConstantExpr>(binExpr.getLHS())) {
          if (auto rhs = llvm::dyn_cast<mlir::AffineDimExpr>(binExpr.getRHS())) {
            coefficients.push_back(std::make_tuple(lhs.getValue(), rhs.getPosition()));
          }
        } else if (auto rhs = llvm::dyn_cast<mlir::AffineConstantExpr>(binExpr.getRHS())) {
          if (auto lhs = llvm::dyn_cast<mlir::AffineDimExpr>(binExpr.getLHS())) {
            coefficients.push_back(std::make_tuple(rhs.getValue(), lhs.getPosition()));
          }
        }
      } else if (binExpr.getKind() == mlir::AffineExprKind::Add) {
        // Add: Recursively collect both sides
        collectCoefficients(binExpr.getLHS());
        collectCoefficients(binExpr.getRHS());
      }
    } else if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      coefficients.push_back(std::make_tuple(1, dimExpr.getPosition()));
    }
  };

  // Collect coefficients from the AffineExpr
  collectCoefficients(expr);

  // Prev coeff
  //llvm::dbgs() << "Coeff: ";
  //for (auto coeff : coefficients) {
  //  llvm::dbgs() << std::get<0>(coeff) << ", ";
  //}
  //llvm::dbgs() << "\n";

  // Step 2: Modify the coefficients based on the updated_position_index
  SmallVector<std::tuple<int64_t, unsigned>, 4> modifiedCoefficients;
  int64_t targetCoefficient = std::get<0>(coefficients[coefficients.size() -1 - updated_position_index]);

  for (int i = 0; i < static_cast<int>(coefficients.size()); ++i) {
    int64_t coeff = std::get<0>(coefficients[i]);
    unsigned position = std::get<1>(coefficients[i]);
    if (coeff > targetCoefficient)
      coeff = (coeff / upperBound) * paddedUpperBound;
    modifiedCoefficients.push_back(std::make_tuple(coeff, position));
  }

  // Updated coeff
  //llvm::dbgs() << "Coeff: ";
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
      return mlir::getAffineConstantExpr(1, context);
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
  paddedMapInfo[oldMap.getAsOpaquePointer()].insert(forOp.getAsOpaquePointer());
  AffineMap newMap = AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), exprs, context);
  paddedMapInfo[newMap.getAsOpaquePointer()] = paddedMapInfo[oldMap.getAsOpaquePointer()];
  auto it = paddedMapInfo.find(oldMap.getAsOpaquePointer());
  if (it != paddedMapInfo.end()) {
    paddedMapInfo.erase(it);  // Erase the map and its associated loops
  } else {
    llvm::errs() << "AffineMap not found in the map.\n";
  }
  return newMap;
}

void TestLoopPadding::analysisDMAStartNode(
  func::FuncOp function, std::vector<MemRefAffineMapForOps>& targetBuffer)
{
  // Analysis pre-padding info
  function.walk([&](mlir::affine::AffineDmaStartOp dmaOp) {
    auto dst_space = dmaOp.getDstMemorySpace();
    auto src_space = dmaOp.getSrcMemorySpace();
    bool is_write;
    ValueRange dram_indices;
    Value dram_memref;
    if (dst_space == 0 && src_space == 1) {
      dram_memref = dmaOp.getDstMemRef();
      dram_indices = dmaOp.getDstIndices();
      is_write = true;
    } else if (dst_space == 1 && src_space == 0) {
      dram_memref = dmaOp.getSrcMemRef();
      dram_indices = dmaOp.getSrcIndices();
      is_write = false;
    } else {
      dmaOp.emitError() << "Unexpected memory space, src: " << src_space << "des: " << dst_space << "\n";
      return;
    }

    for (const auto &existingInfo : targetBuffer) {
      if (existingInfo.getMemRef() == dram_memref) {
        return;
      }
    }

    for (auto operand : dmaOp.getOperands()) {
      if (auto applyOp = operand.getDefiningOp<mlir::affine::AffineApplyOp>()) {
        auto info = MemRefAffineMapForOps(dram_memref, applyOp, is_write);
        targetBuffer.push_back(info);
        break;
      } else if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(operand)) {
        auto definingOp = blockArg.getOwner()->getParentOp();
        if (auto affineForOp = llvm::dyn_cast<mlir::affine::AffineForOp>(definingOp)) {
          auto info = MemRefAffineMapForOps(dram_memref, affineForOp, is_write);
          targetBuffer.push_back(info);
          break;
        }
      }
    }
  });
}

void TestLoopPadding::createWrapperFunction(mlir::ModuleOp module, mlir::OpBuilder builder, mlir::FunctionType prevKernelFuncType) {
  func::FuncOp kernelFunc = module.lookupSymbol<func::FuncOp>("kernel");
  if (!kernelFunc) {
    module.emitError() << "Function 'kernel' not found!\n";
    return;
  }
  mlir::SmallVector<mlir::memref::GlobalOp, 4> padded_buffer;
  std::set<void*> usedMemRefs;
  padded_buffer.resize(kernelFunc.getNumArguments());
  affine::AffineForOp last;

  // Declare global buffers
  builder.setInsertionPointToEnd(module.getBody());
  for (size_t i = 0; i < prePaddingInfo.size() && i < postPaddingInfo.size(); ++i) {
    auto& postInfo = postPaddingInfo[i];
    mlir::Value postMemRef = postInfo.getMemRef();
    auto blockArg = mlir::cast<mlir::BlockArgument>(postMemRef);
    unsigned int argIdx = blockArg.getArgNumber();

    if (usedMemRefs.find(postMemRef.getAsOpaquePointer()) == usedMemRefs.end()) {
      std::string paddedBufName = std::string("global_buffer") + std::to_string(i);
      mlir::MemRefType paddedMemRefType = mlir::dyn_cast<mlir::MemRefType>(postMemRef.getType());
      auto globalMemRefOp = builder.create<mlir::memref::GlobalOp>(
                              builder.getUnknownLoc(), paddedBufName,
                              builder.getStringAttr("private"),
                              paddedMemRefType, mlir::Attribute(), false,
                              builder.getIntegerAttr(builder.getIntegerType(64), 0x1000));
      padded_buffer[argIdx] = globalMemRefOp;
      usedMemRefs.insert(postMemRef.getAsOpaquePointer());
    }
 }

  // Create wrapper function
  auto wrapperFunc = builder.create<func::FuncOp>(
      module.getLoc(), "wrapper_kernel", prevKernelFuncType);
  mlir::Block *entryBlock = wrapperFunc.addEntryBlock();
  Location loc = wrapperFunc.getLoc();
  builder.setInsertionPointToStart(entryBlock);

  // Create read padding phase
  for (size_t i = 0; i < prePaddingInfo.size() && i < postPaddingInfo.size(); ++i) {
    auto& preInfo = prePaddingInfo[i];
    auto& postInfo = postPaddingInfo[i];

    mlir::SmallVector<mlir::Value, 2> mapOperands;
    AffineMap preAffineMap = preInfo.getAffineMap();
    AffineMap postAffineMap = postInfo.getAffineMap();
    builder.setInsertionPointToEnd(entryBlock);
    mlir::Value preMemRef = preInfo.getMemRef();
    auto blockArg = mlir::cast<mlir::BlockArgument>(preMemRef);
    unsigned int argIdx = blockArg.getArgNumber();
    mlir::Value argValue = wrapperFunc.getArgument(argIdx);

    auto globalMemRefOp = padded_buffer[argIdx];

    for (const auto& loopInfo : preInfo.getLoopRange()) {
      int64_t lowerBound = std::get<0>(loopInfo);
      int64_t upperBound = std::get<1>(loopInfo);
      int64_t stepSize = std::get<2>(loopInfo);

      last = builder.create<affine::AffineForOp>(loc, lowerBound, upperBound, stepSize);

      builder.setInsertionPointToStart(last.getBody());
      mapOperands.push_back(last.getInductionVar());
      loc = last.getLoc();
    }
    if (mapOperands.size()) {
      auto preApplyOp = builder.create<affine::AffineApplyOp>(last.getLoc(), preAffineMap, mapOperands);
      mlir::Value preResultIndex = preApplyOp.getResult();
      auto loadedValue = builder.create<affine::AffineLoadOp>(preApplyOp.getLoc(), argValue, preResultIndex);

      /* Get Global */
      auto loadedMemRef = builder.create<mlir::memref::GetGlobalOp>(
        loadedValue.getLoc(), globalMemRefOp.getType(), globalMemRefOp.getName());
      auto postApplyOp = builder.create<affine::AffineApplyOp>(loadedValue.getLoc(), postAffineMap, mapOperands);
      mlir::Value postResultIndex = postApplyOp.getResult();
      builder.create<affine::AffineStoreOp>(postApplyOp.getLoc(), loadedValue, loadedMemRef, postResultIndex);
    }
  }

  // Prepare the arguments of the wrapper to be passed to the 'kernel' function
  builder.setInsertionPointToEnd(entryBlock);
  llvm::SmallVector<mlir::Value, 4> callArgs;
  for (auto globalMemRefOp : padded_buffer) {
    auto loadedMemRef = builder.create<mlir::memref::GetGlobalOp>(
          builder.getUnknownLoc(), globalMemRefOp.getType(), globalMemRefOp.getName());
    callArgs.push_back(loadedMemRef);
  }

  // Create a call to the 'kernel' function inside the wrapper
  builder.create<mlir::func::CallOp>(module.getLoc(), kernelFunc, callArgs);

  // Create write padding phase
  for (size_t i = 0; i < prePaddingInfo.size() && i < postPaddingInfo.size(); ++i) {
    auto& preInfo = prePaddingInfo[i];
    auto& postInfo = postPaddingInfo[i];

    mlir::SmallVector<mlir::Value, 2> mapOperands;
    AffineMap preAffineMap = preInfo.getAffineMap();
    AffineMap postAffineMap = postInfo.getAffineMap();
    builder.setInsertionPointToEnd(entryBlock);
    mlir::Value preMemRef = preInfo.getMemRef();
    auto blockArg = mlir::cast<mlir::BlockArgument>(preMemRef);
    unsigned int argIdx = blockArg.getArgNumber();
    mlir::Value argValue = wrapperFunc.getArgument(argIdx);

    auto globalMemRefOp = padded_buffer[argIdx];

    for (const auto& loopInfo : preInfo.getLoopRange()) {
      int64_t lowerBound = std::get<0>(loopInfo);
      int64_t upperBound = std::get<1>(loopInfo);
      int64_t stepSize = std::get<2>(loopInfo);

      last = builder.create<affine::AffineForOp>(loc, lowerBound, upperBound, stepSize);

      builder.setInsertionPointToStart(last.getBody());
      mapOperands.push_back(last.getInductionVar());
      loc = last.getLoc();
    }
    if (mapOperands.size()) {
      /* Get Global */
      auto loadedMemRef = builder.create<mlir::memref::GetGlobalOp>(
        loc, globalMemRefOp.getType(), globalMemRefOp.getName());
      auto postApplyOp = builder.create<affine::AffineApplyOp>(loadedMemRef.getLoc(), postAffineMap, mapOperands);
      mlir::Value postResultIndex = postApplyOp.getResult();
      auto loadedValue = builder.create<affine::AffineLoadOp>(postApplyOp.getLoc(), loadedMemRef, postResultIndex);

      auto preApplyOp = builder.create<affine::AffineApplyOp>(loadedValue.getLoc(), preAffineMap, mapOperands);
      mlir::Value preResultIndex = preApplyOp.getResult();
      builder.create<affine::AffineStoreOp>(preApplyOp.getLoc(), loadedValue, argValue, preResultIndex);
   }
  }

  // Add a return operation to the wrapper function
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<mlir::func::ReturnOp>(module.getLoc());
}

namespace mlir {
namespace test {
void registerTestLoopPaddingPass() { PassRegistration<TestLoopPadding>(); }
} // namespace test
} // namespace mlir
