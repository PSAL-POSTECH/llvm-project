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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/VCIXDialect.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <iomanip>
#include <sstream>

using namespace mlir;

int64_t VECTOR_LANE = 128;

int extractConstantIntValue(Value val) {
  int val_int;
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    Attribute constantAttr = constOp.getValue();
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constantAttr)) {
      val_int = intAttr.getInt();
    }
  }
  return val_int;
}

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

  Option<int> vectorlane{
      *this, "vectorlane",
      llvm::cl::desc("Vector lane size"),
      llvm::cl::init(4)};

  TestTileOperationGraph() = default;
  TestTileOperationGraph(const TestTileOperationGraph&) {}

  void runOnOperation() override;
  void printOperation(Operation &op, TOGNode *node);
  void getAffineForBounds(affine::AffineForOp &op, int &start, int &end, int &step);
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect,
                    vcix::VCIXDialect, vector::VectorDialect, affine::AffineDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect>();
  }
  void processDramIndices(mlir::Value value,
                        std::map<std::string, int> &loop_index_list,
                        llvm::DenseMap<void *, std::string> &loop_var_name);
  llvm::SmallVector<mlir::Attribute, 2> getSubtileSize(mlir::Operation *operation);
  int getAsyncValue(mlir::Operation *operation);
  std::vector<int> collectCoefficientsFromAffineExpr(mlir::AffineExpr expr);
  int nr_loop = 0;
  llvm::DenseMap<void*, std::string> loop_var_name;
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

std::vector<int> TestTileOperationGraph::collectCoefficientsFromAffineExpr(mlir::AffineExpr expr) {
  std::vector<int> coefficients;

  std::function<void(mlir::AffineExpr)> collectCoefficients = [&](mlir::AffineExpr expr) {
    if (auto binExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
      if (binExpr.getKind() == mlir::AffineExprKind::Mul) {
        if (auto lhs = llvm::dyn_cast<mlir::AffineConstantExpr>(binExpr.getLHS())) {
          if (auto rhs = llvm::dyn_cast<mlir::AffineDimExpr>(binExpr.getRHS())) {
            coefficients.push_back(lhs.getValue());
          }
        } else if (auto rhs = llvm::dyn_cast<mlir::AffineConstantExpr>(binExpr.getRHS())) {
          if (auto lhs = llvm::dyn_cast<mlir::AffineDimExpr>(binExpr.getLHS())) {
            coefficients.push_back(rhs.getValue());
          }
        }
      } else if (binExpr.getKind() == mlir::AffineExprKind::Add) {
        // Add: Recursively collect both sides
        collectCoefficients(binExpr.getLHS());
        collectCoefficients(binExpr.getRHS());
      }
    } else if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      coefficients.push_back(1);
    }
  };

  collectCoefficients(expr);
 return coefficients;
}

void TestTileOperationGraph::printOperation(Operation &op, TOGNode *node) {
  StringRef name = op.getName().getStringRef();
  if (name == "affine.yield" || name == "affine.apply" || name == "memref.get_global" || \
      name == "affine.vector_load")
    return;

  if (name == "affine.for") {
    auto for_op = dyn_cast<affine::AffineForOp>(op);
    auto outerLoopAttr = for_op->getAttrOfType<BoolAttr>("outer_loop");
    auto outerLoopAttr2 = for_op->getAttrOfType<BoolAttr>("accumulation_loop");
    auto innerLoopAttr = for_op->getAttrOfType<BoolAttr>("inner_loop");
    std::string loop_type;
    if (outerLoopAttr && outerLoopAttr.getValue()) {
      loop_type = "outer_loop";
    } else if (outerLoopAttr2 && outerLoopAttr2.getValue()) {
      loop_type = "accumulation_loop";
    } else if (innerLoopAttr && innerLoopAttr.getValue()) {
      loop_type = "inner_loop";
    }

    if ((outerLoopAttr && outerLoopAttr.getValue()) || (outerLoopAttr2 && outerLoopAttr2.getValue()) || (innerLoopAttr && innerLoopAttr.getValue())) {
      // Get loop information and create loop node
      int start, end, step;
      getAffineForBounds(for_op, start, end, step);
      std::ostringstream oss;
      oss << "loop_arg" << std::setw(3) << std::setfill('0') << nr_loop++;
      std::string loop_index = oss.str();
      mlir::Value iter_var = for_op.getInductionVar();
      loop_var_name[iter_var.getAsOpaquePointer()] = loop_index;
      TOGLoopNode *tog_loop = new TOGLoopNode("loopNode", loop_index, start, end, step, loop_type);
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

  if (name == "memref.dma_start") {
    auto dma_op = dyn_cast<memref::DmaStartOp>(op);
    std::vector<int> tile_size, tile_stride, stride_list;
    std::string address = "arg";
    int element_size = 0;
    bool is_write;
    bool dmaAsync = (bool)getAsyncValue(dma_op);
    auto dst_space = dma_op.getDstMemorySpace();
    auto src_space = dma_op.getSrcMemorySpace();
    MemRefType tile_memref_type;
    ValueRange dram_indices;
    Value dram_memref;
    std::vector<std::string> loop_index_list;
    std::vector<std::string> tag_index_list;
    std::vector<int> tag_stride_list;
    std::map<std::string, int> loop_index_map;
    llvm::SmallVector<int64_t, 2> dmaSubtileValues;
    for (auto attr : getSubtileSize(dma_op))
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
        dmaSubtileValues.push_back(intAttr.getInt());

    if (dst_space == 0 && src_space == 1) {
      is_write = true;
      tile_memref_type = llvm::cast<MemRefType>(dma_op.getSrcMemRef().getType());
      dram_memref = dma_op.getDstMemRef();
      dram_indices = dma_op.getDstIndices();
    } else if (dst_space == 1 && src_space == 0) {
      is_write = false;
      tile_memref_type = llvm::cast<MemRefType>(dma_op.getDstMemRef().getType());
      dram_memref = dma_op.getSrcMemRef();
      dram_indices = dma_op.getSrcIndices();
    } else {
      op.emitError() << "Unexpected memory space, src: " << src_space << "des: " << dst_space << "\n";
      return;
    }
    /* Record used loop index names */
    processDramIndices(dram_indices.front(), loop_index_map, loop_var_name);
    for (const auto& pair : loop_index_map) {
        loop_index_list.push_back(pair.first);
        stride_list.push_back(pair.second);
    }

    /* Get DRAM argument index */
    if (auto blockArg = dyn_cast<BlockArgument>(dram_memref)) {
      // Get the index of the block argument
      unsigned index = blockArg.getArgNumber();
      address = address + std::to_string(index);
    } else {
      op.emitError() << "Unexpected dram buffer argument: " << dram_memref << "\n";
    }
    // Select tile or subtile
    auto tile_shape = dmaSubtileValues.size()? dmaSubtileValues : tile_memref_type.getShape();

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

    auto tag_range = dma_op.getTagIndices();
    for (auto tag_idx : tag_range) {
      if (auto constOp = tag_idx.getDefiningOp<arith::ConstantIndexOp>()) {
        auto constant = static_cast<int>(constOp.value());
        tag_index_list.push_back(std::to_string(constant));
      } else if (auto applyOp = tag_idx.getDefiningOp<affine::AffineApplyOp>()) {
        for (auto operand : applyOp.getOperands()) {
          if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
            tag_index_list.push_back(loop_var_name.at(blockArg.getAsOpaquePointer()));
          } else if (auto nestedApplyOp = operand.getDefiningOp<affine::AffineApplyOp>()) {
            for (auto nestedOperand : nestedApplyOp.getOperands()) {
              if (auto blockArg = dyn_cast<BlockArgument>(nestedOperand)) {
                tag_index_list.push_back(loop_var_name.at(blockArg.getAsOpaquePointer()));
              } else if (auto constOp = nestedOperand.getDefiningOp<arith::ConstantIndexOp>()) {
                auto constant = static_cast<int>(constOp.value());
                tag_index_list.push_back(std::to_string(constant));
              } else {
                nestedApplyOp.emitError() << "tag index apply op is not valid!\n";
              }
            }
          } else if (auto constOp = operand.getDefiningOp<arith::ConstantIndexOp>()) {
            auto constant = static_cast<int>(constOp.value());
            tag_index_list.push_back(std::to_string(constant));
          } else {
            applyOp.emitError() << "tag index apply op is not valid!\n";
          }
        }
        mlir::AffineMap map = applyOp.getAffineMap();
        tag_stride_list = collectCoefficientsFromAffineExpr(map.getResult(0));
      }
    }
    if (tag_index_list.size() == 0) {
      tag_index_list.push_back("0");
    }

    if (tag_stride_list.size() == 0) {
      tag_stride_list.push_back(1);
    }

    TOGDMANode *tog_dma = new TOGDMANode("DMANode", address, stride_list, tile_size,
                                         element_size, is_write, dmaAsync, tag_index_list,
                                         tag_stride_list, loop_index_list);
    tog_dma->setOp(&op);
    /* Link child and parent */
    node->addChild(tog_dma);
    tog_dma->addParent(node);
    return; // Compute node
  }

  if (name == "memref.dma_wait") {
    std::string address = "arg";
    std::vector<std::string> tag_index_list;
    std::vector<int> tag_stride_list;
    auto dma_op = dyn_cast<memref::DmaWaitOp>(op);
    auto tag_memref = dma_op.getTagMemRef();
    auto tag_range = dma_op.getTagIndices();

    for (auto tag_idx : tag_range) {
      if (auto applyOp = tag_idx.getDefiningOp<mlir::affine::AffineApplyOp>()) {
        for (mlir::Value operand : applyOp.getMapOperands()) {
          if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(operand)) {
            tag_index_list.push_back(loop_var_name.at(blockArg.getAsOpaquePointer()));
          } else if (auto constOp = operand.getDefiningOp<arith::ConstantIndexOp>()) {
            auto constant = static_cast<int>(constOp.value());
            tag_index_list.push_back(std::to_string(constant));
          } else {
            applyOp.emitError() << "Unexpected tag affine map\n";
          }
        }
        mlir::AffineMap map = applyOp.getAffineMap();
        tag_stride_list = collectCoefficientsFromAffineExpr(map.getResult(0));
      } else {
        dma_op.emitError() << "Unexpected tag indices in the dma.wait\n";
      }
    }
    for (auto &use : tag_memref.getUses()) {
      mlir::Operation *userOp = use.getOwner(); // Get the operation that uses tagMemRef
      if (auto dmaStartOp = llvm::dyn_cast<memref::DmaStartOp>(userOp)) {
        Value dram_memref;
        auto dst_space = dmaStartOp.getDstMemorySpace();
        auto src_space = dmaStartOp.getSrcMemorySpace();

        if (dst_space == 0 && src_space == 1) {
          dram_memref = dmaStartOp.getDstMemRef();
        } else if (dst_space == 1 && src_space == 0) {
          dram_memref = dmaStartOp.getSrcMemRef();
        }
        if (auto blockArg = dyn_cast<BlockArgument>(dram_memref)) {
          // Get the index of the block argument
          unsigned index = blockArg.getArgNumber();
          address = address + std::to_string(index);
        }
      }
    }
    if (tag_stride_list.size() == 0) {
      tag_stride_list.push_back(1);
    }

    TOGDMAWaitNode *tog_dma_wait = new TOGDMAWaitNode("DMAWaitNode", tag_index_list, tag_stride_list, address);
    tog_dma_wait->setOp(&op);
    /* Link child and parent */
    node->addChild(tog_dma_wait);
    tog_dma_wait->addParent(node);
    return;
  }

  /* Skip root */
  if (node == NULL)
    return;

  /* Combine compute node */
  auto type = name == "vcix.i" ? TOGComputeNode::MatmulCompute : TOGComputeNode::VectorCompute;
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
  llvm::StringRef funcName = op.getSymName();
  VECTOR_LANE = vectorlane;
  if (funcName.compare(llvm::StringRef("kernel"))) {
    return;
  }

  // Check kernel function has one region
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
      /*constraints=*/"~{a0}",
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
      /*constraints=*/"~{a0}",
      /*has_side_effects=*/true,
      /*is_align_stack=*/false,
      /*asm_dialect=*/asmDialectAttr,
      ArrayAttr()
    );
  }
  return;
}

int getArgumentIndex(mlir::affine::AffineApplyOp applyOp, mlir::Value targetValue) {
  mlir::OperandRange operands = applyOp.getOperands();
  for (size_t i = 0; i < operands.size(); ++i) {
    if (operands[i] == targetValue) {
      return i;
    }
  }
  return -1;
}

int64_t getCoefficientFromDim(mlir::AffineExpr expr, int dim) {
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

void TestTileOperationGraph::processDramIndices(mlir::Value value,
                        std::map<std::string, int> &loop_index_list,
                        llvm::DenseMap<void *, std::string> &loop_var_name) {
  if (auto applyOp = value.getDefiningOp<mlir::affine::AffineApplyOp>()) {
    mlir::AffineMap map = applyOp.getAffineMap();
    mlir::OperandRange applyOperands = applyOp.getOperands();

    for (unsigned i = 0; i < applyOperands.size(); ++i) {
      auto operand = applyOperands[i];
      if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(operand)) {
        mlir::AffineExpr expr = map.getResult(0);
        auto index_pos = getArgumentIndex(applyOp, blockArg);
        int coeff = getCoefficientFromDim(expr, index_pos);
        loop_index_list[loop_var_name.at(blockArg.getAsOpaquePointer())] = coeff;
      } else {
        // Otherwise, recursively process the operand
        processDramIndices(operand, loop_index_list, loop_var_name);
      }
    }
  } else if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    // If the value itself is a BlockArgument, add it to the list
    loop_index_list[loop_var_name.at(blockArg.getAsOpaquePointer())] = 1;
  } else if (auto constOp = value.getDefiningOp<arith::ConstantIndexOp>()) {
    int constValue = static_cast<int>(constOp.value());
    loop_index_list["c" + std::to_string(constValue)] = constValue;
  }
}

llvm::SmallVector<mlir::Attribute, 2> TestTileOperationGraph::getSubtileSize(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Attribute, 2> subtileSizes;
  auto attr = operation->getAttr("subtile_size");
  if (!attr) {
    return subtileSizes; // Return empty SmallVector
  }

  if (auto arrayAttr = llvm::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (auto element : arrayAttr) {
      // Assume the elements are integers
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(element)) {
        subtileSizes.push_back(intAttr);
      } else {
        llvm::errs() << "Unsupported element type in 'subtile_size'.\n";
      }
    }
  }
  return subtileSizes;
}

int TestTileOperationGraph::getAsyncValue(mlir::Operation *operation) {
  auto attr = operation->getAttr("async");
  if (!attr)
    return 0;

  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getInt(); // Treat non-zero as true
  else
    return 1;
}
namespace mlir {
namespace test {
void registerTestTileOperationGraphPass() { PassRegistration<TestTileOperationGraph>(); }
} // namespace test
} // namespace mlir
