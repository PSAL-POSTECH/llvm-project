//===- TestTileOperationGraph.cpp - Test CFG loop info analysis ------------------===//
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

#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// A testing pass that applies the TileOperationGraph analysis on a region and prints
/// the information it collected to llvm::errs().
struct TestTileOperationGraph
    : public PassWrapper<TestTileOperationGraph, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTileOperationGraph)

  StringRef getArgument() const final { return "test-tile-operation-graph"; }
  StringRef getDescription() const final {
    return "Test tile operation graph analysis";
  }

  void runOnOperation() override;
};
} // namespace

void TestTileOperationGraph::runOnOperation() {
  auto func = getOperation();

  // Prints the label of the test.
  llvm::errs() << "Testing : " << func->getName() << "\n";
  func->walk([&](Operation *op) {
      llvm::errs() << "Testing : " << op->getName() << "\n";
      if (op->getNumResults() == 0) {
        return;
      }
  });
}

namespace mlir {
namespace test {
void registerTestTileOperationGraphPass() { PassRegistration<TestTileOperationGraph>(); }
} // namespace test
} // namespace mlir
