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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Analysis/TileOperationGraph.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// A testing pass that applies the TileOperationGraph analysis on a region and prints
/// the information it collected to llvm::errs().
struct TestTileOperationGraph
    : public PassWrapper<TestTileOperationGraph, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTileOperationGraph)
  std::vector<TOGComputeNode *> compute_nodes;
  StringRef getArgument() const final { return "test-tile-operation-graph"; }
  StringRef getDescription() const final {
    return "Test tile operation graph analysis";
  }

  void runOnOperation() override;
  void printOperation(Operation &op, TOGNode *node);
  void getAffineForBounds(affine::AffineForOp &op, int &start, int &end, int &step);
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect>();
  }
};
} // namespace

void TestTileOperationGraph::getAffineForBounds(affine::AffineForOp &op, int &start, int &end, int &step) {
  auto lowerBoundMap = op.getLowerBoundMap();
  if (lowerBoundMap.isSingleConstant()) {
    start = static_cast<int>(lowerBoundMap.getSingleConstantResult());
  } else {
    auto operand = *op.getLowerBoundOperands().begin();
    auto constOp = operand.getDefiningOp<arith::ConstantIndexOp>();
    start = static_cast<int>(constOp.value());
  }

  // Handle upper bound, which can be constant or variable
  auto upperBoundMap = op.getUpperBoundMap();
  if (upperBoundMap.isSingleConstant()) {
    end = static_cast<int>(upperBoundMap.getSingleConstantResult());
  } else {
    auto operand = *op.getUpperBoundOperands().begin();
    auto constOp = operand.getDefiningOp<arith::ConstantIndexOp>();
    end = static_cast<int>(constOp.value());
  }

  // Get the step value, which is always an integer
  step = static_cast<int>(op.getStep().getSExtValue());

  // Modify step size
  op.setStep(end);
}

void TestTileOperationGraph::printOperation(Operation &op, TOGNode *node) {
  StringRef name = op.getName().getStringRef();
  if (name == "affine.yield" || name == "affine.apply" || name == "memref.get_global" || \
      name == "affine.vector_load")
    return;

  // Print the operation itself and some of its properties
  llvm::errs() << "visiting op: '" << op.getName() << "' with "
                << op.getNumOperands() << " operands and "
                << op.getNumResults() << " results\n";

  if (name == "affine.for") {
    auto for_op = dyn_cast<affine::AffineForOp>(op);
    auto outerLoopAttr = for_op->getAttrOfType<BoolAttr>("outer_loop");
    auto outerLoopAttr2 = for_op->getAttrOfType<BoolAttr>("accumulation_loop");
    if ((outerLoopAttr && outerLoopAttr.getValue()) || (outerLoopAttr2 && outerLoopAttr2.getValue())) {
      // Get loop information and create loop node
      int start, end, step;
      getAffineForBounds(for_op, start, end, step);
      std::string loop_type = outerLoopAttr? "outer_loop" : "accumulation_loop";
      TOGLoopNode *tog_loop = new TOGLoopNode("loopNode", "arg", start, end, step, loop_type);
      tog_loop->setOp(&op);

      /* Link child and parent */
      if (node != NULL) {
        node->addChild(tog_loop);
        tog_loop->addParent(node);
      }

      for (Region &region : op.getRegions())
        for (Block &block : region.getBlocks())
          for (Operation &op : block.getOperations())
            printOperation(op, tog_loop);
      return;
    }
  }

  if (name == "affine.dma_start") {
    auto dma_op = dyn_cast<affine::AffineDmaStartOp>(op);
    std::vector<int> tile_size, tile_stride, stride_list;
    std::string address = "arg";
    int element_size = 0;
    bool is_write;
    auto dst_space = dma_op.getDstMemorySpace();
    auto src_space = dma_op.getSrcMemorySpace();
    MemRefType tile_memref_type;
    Value dram_memref;
    if (dst_space == 0 && src_space == 1) {
      is_write = true;
      tile_memref_type = dma_op.getSrcMemRefType();
      dram_memref = dma_op.getDstMemRef();
    } else if (dst_space == 1 && src_space == 0) {
      is_write = false;
      tile_memref_type = dma_op.getDstMemRefType();
      dram_memref = dma_op.getSrcMemRef();
    } else {
      op.emitError() << "Unexpected memory space, src: " << src_space << "des: " << dst_space << "\n";
      return;
    }

    /* Get DRAM argument index */
    if (auto blockArg = dyn_cast<BlockArgument>(dram_memref)) {
      // Get the index of the block argument
      unsigned index = blockArg.getArgNumber();
      address = address + std::to_string(index);
    } else {
      op.emitError() << "Unexpected dram buffer argument: " << dram_memref << "\n";
    }

    auto tile_shape = tile_memref_type.getShape();
    int mm_stride = dma_op.getStride().getDefiningOp<arith::ConstantIndexOp>().value();

    /* Fill stride info */
    tile_stride = {mm_stride, 1};
    stride_list = {mm_stride, 1};

    /* Extract destination tile size */
    for (int64_t iter: tile_shape)
      tile_size.push_back(static_cast<int>(iter));

    /* Extract destination element type */
    mlir::Type element_type = tile_memref_type.getElementType();
    if (auto int_type = dyn_cast<mlir::IntegerType>(element_type)) {
      element_size = int_type.getWidth() / 8; // Convert bits to bytes
    } else if (auto float_type = dyn_cast<mlir::FloatType>(element_type)) {
      element_size = float_type.getWidth() / 8; // Convert bits to bytes
    } else {
      op.emitError() << tile_memref_type << "Unsupported element type\n";
      return;
    }

    TOGDMANode *tog_dma = new TOGDMANode("DMANode", address, stride_list, tile_size, tile_stride,
                                         element_size, is_write);
    tog_dma->setOp(&op);
    /* Link child and parent */
    node->addChild(tog_dma);
    tog_dma->addParent(node);
    return; // Compute node
  }

  /* Skip root */
  if (node == NULL)
    return;

  /* Combine compute node */
  auto type = name == "linalg.matmul" ? TOGComputeNode::MatmulCompute : TOGComputeNode::VectorCompute;
  if (node->getChildren().size()) {
    TOGComputeNode *last_compute_node = dyn_cast<TOGComputeNode>(node->getLastChild());
    if (last_compute_node) {
      last_compute_node->operations.push_back(&op);
      if (type == TOGComputeNode::MatmulCompute)
        last_compute_node->setComputeType(type);
      return;
    }
  }
  /* Create new compute node */
  TOGComputeNode *tog_compute = new TOGComputeNode("ComputeNode", 0, type);
  tog_compute->setOp(&op);
  tog_compute->operations.push_back(&op);
  /* Link child and parent */
  node->addChild(tog_compute);
  tog_compute->addParent(node);

  /* Register compute node to vector */
  compute_nodes.push_back(tog_compute);
  return; // Compute node
}

void TestTileOperationGraph::runOnOperation() {
  func::FuncOp op = getOperation();
  MLIRContext *context = &getContext();
  OpBuilder builder(context);
  llvm::errs() << "Target function : " << op->getName() << "\n";

  // Check kernel function has one region
  llvm::errs() << " " << op->getNumRegions() << " regions:\n";
  if (op->getNumRegions() != 1) {
    op.emitError() << "Expected one region but has " << op->getNumRegions() << "region(s)";
    return;
  }

  Region &region = op->getRegion(0);
  if (region.getBlocks().empty()) {
    op.emitError() << "Expected one block but has 0 block";
    return;
  }
  Block &block = *(region.getBlocks().begin());
  // Find main loops
  for (Operation &op : block.getOperations()) {
    StringRef name = op.getName().getStringRef();
    if (name != "affine.for")
      continue;
    TOGNode *dummy = new TOGNode("root");
    printOperation(op, dummy);
    dummy->bfs();
  }

  /* GEM5 measuring assembly instruction */
  std::string asmBefore = ".insn r CUSTOM_3, 0, 0x40, x0, x0, x0";
  std::string asmAfter = ".insn r CUSTOM_3, 0, 0x41, x0, x0, x0";
  auto asmDialectAttr =
    LLVM::AsmDialectAttr::get(builder.getContext(), LLVM::AsmDialect::AD_ATT);
  for (TOGComputeNode *compute_node : compute_nodes) {
    Location loc = compute_node->operations.front()->getLoc();

    // Insert the inline assembly before the first operation
    builder.setInsertionPoint(compute_node->operations.front());
    OperationState asmBeforeState(loc, "inline_asm");
    builder.create<LLVM::InlineAsmOp>(
      loc,
      /*resultTypes=*/TypeRange(),
      /*operands=*/ValueRange(),
      /*asm_string=*/asmBefore,
      /*constraints=*/"",
      /*has_side_effects=*/true,
      /*is_align_stack=*/false,
      /*asm_dialect=*/asmDialectAttr,
      ArrayAttr()
    );

    // Insert the inline assembly after the last operation
    builder.setInsertionPointAfter(compute_node->operations.back());
    OperationState asmAfterState(loc, "inline_asm");
    builder.create<LLVM::InlineAsmOp>(
      loc,
      /*resultTypes=*/TypeRange(),
      /*operands=*/ValueRange(),
      /*asm_string=*/asmAfter,
      /*constraints=*/"",
      /*has_side_effects=*/true,
      /*is_align_stack=*/false,
      /*asm_dialect=*/asmDialectAttr,
      ArrayAttr()
    );
  }
  return;
}

namespace mlir {
namespace test {
void registerTestTileOperationGraphPass() { PassRegistration<TestTileOperationGraph>(); }
} // namespace test
} // namespace mlir
