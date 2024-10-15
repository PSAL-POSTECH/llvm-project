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
  int64_t getLoopUpperBound(affine::AffineForOp forOp);
  void setLoopUpperBound(affine::AffineForOp forOp, int64_t newUpperBound);
  void modifyMemrefWithPadding(Value memref, int64_t stepSize, int64_t paddedUpperBound);
  int64_t findSecondSmallestCoefficient(AffineExpr expr);
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
  std::vector<std::tuple<int64_t, int64_t, Value>> loopInfoMap;
  std::map<const void*, std::set<const void*>> paddedMapInfo;
};

} // namespace

void TestLoopPadding::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

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

        int64_t mm_stride = -1;
        int64_t new_mm_stride;
        for (auto operand : dmaOp.getOperands()) {
          if (auto applyOp = operand.getDefiningOp<mlir::affine::AffineApplyOp>()) {
            auto index_pos = findLoopVariableIndex(forOp.getInductionVar());
            AffineMap map = applyOp.getAffineMap();
            for (unsigned i = 0; i < map.getNumResults(); ++i) {
              AffineExpr expr = map.getResult(i);

              // Find old mm_stride
              if (!isLoopRegistered(map, forOp)) {
                expr = updateAffineExprWithBounds(expr, index_pos, upperBound, paddedUpperBound, context);
                map = updateAffineMapWithNewExpr(map, expr, i, context, forOp);
                applyOp.setMap(map);
              }

              auto strideVal = dmaOp.getStride().getDefiningOp<arith::ConstantIndexOp>();
              if (strideVal && strideVal.value() != 1) {
                new_mm_stride = findSecondSmallestCoefficient(expr);
                mm_stride = mm_stride !=1 ? new_mm_stride : stepSize;
              }
              //llvm::errs() << "old_expr: " << expr << " new_expr: " << new_expr << "\n";
              //llvm::errs() << "index_pos: " << index_pos << " upper: " << upperBound << " paddUpper: " << paddedUpperBound << "\n";
            }
          }
        }

        Value fifthOperand = dmaOp.getStride();
        auto constantOp = llvm::dyn_cast<arith::ConstantIndexOp>(fifthOperand.getDefiningOp());
        if (constantOp && mm_stride != -1) {
          //llvm::errs() << constantOp << "  " << mm_stride <<"\n";
          OpBuilder builder(constantOp);
          Value newStrideConstant = builder.create<arith::ConstantIndexOp>(constantOp.getLoc(), mm_stride);
          constantOp.replaceAllUsesWith(newStrideConstant);
          constantOp.erase();
        }
     }
    });
  });
  return;
}

// Helper function to round up a value to the nearest multiple
int64_t TestLoopPadding::roundUpToMultiple(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

// Helper function to get the upper bound of an affine.for loop
int64_t TestLoopPadding::getLoopUpperBound(mlir::affine::AffineForOp forOp) {
  if (auto constantOp = forOp.getUpperBoundMap().getSingleConstantResult()) {
    return constantOp;
  }
  // Handle other cases if needed
  return -1;  // This is an example, handle dynamic cases as needed
}

// Helper function to set the upper bound of an affine.for loop
void TestLoopPadding::setLoopUpperBound(mlir::affine::AffineForOp forOp, int64_t newUpperBound) {
  OpBuilder builder(forOp);
  auto newBoundMap = builder.getConstantAffineMap(newUpperBound);
  forOp.setUpperBoundMap(newBoundMap);
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

int64_t TestLoopPadding::findSecondSmallestCoefficient(AffineExpr expr) {
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

  // Traverse the expression and extract coefficients
  extractCoefficients(expr);

  if (coefficients.size() < 2) {
    return -1;
  }

  std::sort(coefficients.begin(), coefficients.end(), std::less<int64_t>());

  //for (auto coe:coefficients)
  //  llvm::errs() << coe << " ";

  return coefficients[1];
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

namespace mlir {
namespace test {
void registerTestLoopPaddingPass() { PassRegistration<TestLoopPadding>(); }
} // namespace test
} // namespace mlir
