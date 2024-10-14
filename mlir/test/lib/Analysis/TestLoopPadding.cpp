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
  int findLoopVariableIndexInApplyOp(affine::AffineApplyOp applyOp, mlir::Value loopVar);
  mlir::AffineExpr updateAffineExprWithBounds(mlir::AffineExpr expr, 
                                            int updated_position_index, 
                                            int64_t upperBound, 
                                            int64_t paddedUpperBound, 
                                            mlir::MLIRContext *context);
  AffineMap updateAffineMapWithNewExpr(AffineMap oldMap, 
                                       mlir::AffineExpr updatedExpr, 
                                       unsigned exprIndex, 
                                       mlir::MLIRContext *context);
  std::map<const void*, std::tuple<int64_t, int64_t>> loopInfoMap;
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
    loopInfoMap[forOp.getAsOpaquePointer()] = std::make_tuple(upperBound, stepSize);

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
        for (auto operand : dmaOp.getOperands()) {
          if (auto applyOp = operand.getDefiningOp<mlir::affine::AffineApplyOp>()) {
            auto index_pos = findLoopVariableIndexInApplyOp(applyOp, forOp.getInductionVar());
            AffineMap map = applyOp.getAffineMap();
            for (unsigned i = 0; i < map.getNumResults(); ++i) {
              AffineExpr expr = map.getResult(i);

              // Find old mm_stride
              AffineExpr new_expr = updateAffineExprWithBounds(expr, index_pos, upperBound, paddedUpperBound, context);
              map = updateAffineMapWithNewExpr(map, new_expr, i, context);
              applyOp.setMap(map);
              mm_stride = findSecondSmallestCoefficient(new_expr);
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
        } else {
          coefficients.push_back(1);  // No explicit constant means coefficient is 1
        }
      }
      // Recursively handle both sides of the binary expression
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

  std::sort(coefficients.begin(), coefficients.end(), std::greater<int64_t>());

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
      // For binary expressions, handle mul and add differently
      if (binExpr.getKind() == mlir::AffineExprKind::Mul) {
        // Mul: Assume binExpr is in the form of constant * dimension (e.g., 47 * index0)
        if (auto lhs = llvm::dyn_cast<mlir::AffineConstantExpr>(binExpr.getLHS())) {
          if (auto rhs = llvm::dyn_cast<mlir::AffineDimExpr>(binExpr.getRHS())) {
            // Extract coefficient and update the corresponding dimension
            coefficients.push_back(std::make_tuple(lhs.getValue(), rhs.getPosition()));
          }
        } else if (auto rhs = llvm::dyn_cast<mlir::AffineConstantExpr>(binExpr.getRHS())) {
          if (auto lhs = llvm::dyn_cast<mlir::AffineDimExpr>(binExpr.getLHS())) {
            // Extract coefficient and update the corresponding dimension
            coefficients.push_back(std::make_tuple(rhs.getValue(), lhs.getPosition()));
          }
        }
      } else if (binExpr.getKind() == mlir::AffineExprKind::Add) {
        // Add: Recursively collect both sides
        collectCoefficients(binExpr.getLHS());
        collectCoefficients(binExpr.getRHS());
      }
    } else if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      // If it's a dimension expression (index variable), assign coefficient 1
      coefficients.push_back(std::make_tuple(1, dimExpr.getPosition()));
    }
  };

  // Collect coefficients from the AffineExpr
  collectCoefficients(expr);

  // Step 2: Modify the coefficients based on the updated_position_index
  SmallVector<std::tuple<int64_t, unsigned>, 4> modifiedCoefficients;
  int64_t targetCoefficient = std::get<0>(coefficients[updated_position_index]);

  for (int i = 0; i < static_cast<int>(coefficients.size()); ++i) {
    int64_t coeff = std::get<0>(coefficients[i]);
    unsigned position = std::get<1>(coefficients[i]);

    if (coeff > targetCoefficient)
      coeff = (coeff / upperBound) * paddedUpperBound;
    modifiedCoefficients.push_back(std::make_tuple(coeff, position));
  }

  std::function<mlir::AffineExpr(mlir::AffineExpr, int &)> rebuildExprWithUpdatedCoefficients = 
      [&](mlir::AffineExpr expr, int &coefficientIndex) -> mlir::AffineExpr {
    if (auto binExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
      // Recursively rebuild the binary expressions with updated coefficients
      mlir::AffineExpr lhs = rebuildExprWithUpdatedCoefficients(binExpr.getLHS(), coefficientIndex);
      mlir::AffineExpr rhs = rebuildExprWithUpdatedCoefficients(binExpr.getRHS(), coefficientIndex);
      
      // Return either Add or Mul based on the original binary operation
      auto result = binExpr.getKind() == mlir::AffineExprKind::Add ? lhs + rhs : lhs * rhs; 
      //llvm::errs() << "result: " << result << "\n";
      return result;
    } else if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      // For dimension expressions, multiply by the updated coefficient
      int64_t new_coefficient = std::get<0>(modifiedCoefficients[coefficientIndex]);
      unsigned position = std::get<1>(modifiedCoefficients[coefficientIndex]);
  
      // Make sure we update only the right dimension, skip if the dimension doesn't match
      if (dimExpr.getPosition() == position) {
        coefficientIndex++; // Increment the coefficient index
        auto expr = mlir::getAffineDimExpr(position, context) * mlir::getAffineConstantExpr(new_coefficient, context);
        return expr;
      }
    } else if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
      coefficientIndex++;
      return mlir::getAffineConstantExpr(1, context);
    }
    return expr; // Return the original expression if no specific case matches
  };

  int coefficientIndex = 0;
  return rebuildExprWithUpdatedCoefficients(expr, coefficientIndex);
}

int TestLoopPadding::findLoopVariableIndexInApplyOp(mlir::affine::AffineApplyOp applyOp, mlir::Value loopVar) {
  auto operands = applyOp.getOperands();

  for (unsigned i = 0; i < operands.size(); ++i) {
    if (operands[i] == loopVar) {
      return i;
    }
  }
  return -1;
}

AffineMap TestLoopPadding::updateAffineMapWithNewExpr(AffineMap oldMap, 
                                           mlir::AffineExpr updatedExpr, 
                                           unsigned exprIndex, 
                                           mlir::MLIRContext *context) {
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

namespace mlir {
namespace test {
void registerTestLoopPaddingPass() { PassRegistration<TestLoopPadding>(); }
} // namespace test
} // namespace mlir
