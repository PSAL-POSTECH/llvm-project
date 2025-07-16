#ifndef MLIR_CUSTOM_DMA_ATTRIBUTE
#define MLIR_CUSTOM_DMA_ATTRIBUTE

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#define CONFIG 0
#define CONFIG2 4
#define CONFIG3 5
#define CONFIG4 6
#define MVIN 2
#define MVIN2 1
#define MVIN3 14
#define MVOUT 3

#define CONFIG_MVIN 0
#define CONFIG_MVIN2 1
#define CONFIG_MVIN3 2
#define CONFIG_MVOUT 3

using namespace mlir;
namespace mlir {

static int extractConstantIntValue(Value val) {
  if (!val) {
    llvm::errs() << "Error: Value is null.\n";
    return -1;
  }

  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    Attribute constantAttr = constOp.getValue();
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constantAttr)) {
      return intAttr.getInt();
    } else {
      llvm::errs() << "Error: Constant attribute is not an IntegerAttr.\n";
    }
  } else {
    llvm::errs() << "Error: Value is not defined by an arith::ConstantOp.\n";
  }
  return -1;
}

static uint64_t getVlaneSplitAxis(memref::DmaStartOp op) {
  Value strideValue = op.getStride();
  uint64_t vlane_split_axis = static_cast<uint64_t>(extractConstantIntValue(strideValue));
  if (vlane_split_axis == static_cast<uint64_t>(-1)) {
    llvm::errs() << "Error: Failed to extract vlane_split_axis.\n";
  }
  return vlane_split_axis;
}

static uint64_t getVlaneStride(memref::DmaStartOp op) {
  Value numElementsPerStrideValue = op.getNumElementsPerStride();
  uint64_t vlane_stride = static_cast<uint64_t>(extractConstantIntValue(numElementsPerStrideValue));
  if (vlane_stride == static_cast<uint64_t>(-1)) {
    llvm::errs() << "Error: Failed to extract vlane_stride.\n";
  }
  vlane_stride &= 0x7FFF;
  return vlane_stride;
}

static int getDMAType(memref::DmaStartOp op) {
  return extractConstantIntValue(op.getNumElements());
}

static SmallVector<int64_t, 4> getTileShape(memref::DmaStartOp op) {
  int dmaType = getDMAType(op);
  auto srcMemRefType = cast<MemRefType>(op.getSrcMemRef().getType());
  auto dstMemRefType = cast<MemRefType>(op.getDstMemRef().getType());

  if (dmaType == MVIN || dmaType == MVIN2 || dmaType == MVIN3) {
    return SmallVector<int64_t, 4>(dstMemRefType.getShape());
  } else { // MVOUT or others
    return SmallVector<int64_t, 4>(srcMemRefType.getShape());
  }
}

static llvm::SmallVector<mlir::Attribute, 2> getSubtileSize(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Attribute, 2> subtileSizes;
  auto attr = operation->getAttr("subtile_size");
  if (!attr) {
    return subtileSizes; // Return empty SmallVector
  }

  if (auto arrayAttr = llvm::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (auto element : arrayAttr) {
      // Assume the elements are integers
      if (auto intAttr = llvm::dyn_cast<mlir::Attribute>(element)) {
        subtileSizes.push_back(intAttr);
      } else {
        llvm::errs() << "Unsupported element type in 'subtile_size'.\n";
      }
    }
  }
  return subtileSizes;
}

static int getAsyncValue(mlir::Operation *operation) {
  auto attr = operation->getAttr("async");
  if (!attr)
    return 0;

  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getInt(); // Treat non-zero as true
  else
    return 1;
}

static int is_fine_grained(mlir::Operation *operation) {
  auto attr = operation->getAttr("fine_grained");
  if (!attr)
    return 0;

  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getInt(); // Treat non-zero as true
  else
    return 1;
}

static llvm::SmallVector<mlir::Attribute> getSramStride(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Attribute> sram_stride;
  auto attr = operation->getAttr("sram_stride");
  if (!attr) {
    return sram_stride; // Return empty SmallVector
  }

  if (auto arrayAttr = llvm::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (auto element : arrayAttr) {
      // Assume the elements are integers
      if (auto intAttr = llvm::dyn_cast<mlir::Attribute>(element)) {
        sram_stride.push_back(intAttr);
      } else {
        llvm::errs() << "Unsupported element type in 'sram_stride'.\n";
      }
    }
  }
  return sram_stride;
}

static llvm::SmallVector<mlir::Attribute> getDramStride(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Attribute> sram_stride;
  auto attr = operation->getAttr("dram_stride");
  if (!attr) {
    return sram_stride; // Return empty SmallVector
  }

  if (auto arrayAttr = llvm::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (auto element : arrayAttr) {
      // Assume the elements are integers
      if (auto intAttr = llvm::dyn_cast<mlir::Attribute>(element)) {
        sram_stride.push_back(intAttr);
      } else {
        llvm::errs() << "Unsupported element type in 'sram_stride'.\n";
      }
    }
  }
  return sram_stride;
}

static NamedAttrList getDmaAttrs(memref::DmaStartOp dmaOp, OpBuilder &builder) {
  SmallVector<mlir::Attribute> subtile = getSubtileSize(dmaOp);
  SmallVector<mlir::Attribute> sramStrides = getSramStride(dmaOp);
  SmallVector<mlir::Attribute> dramStrides = getDramStride(dmaOp);
  int async = getAsyncValue(dmaOp);

  NamedAttrList attrs;
  if (!subtile.empty()) {
    attrs.set("subtile_size", builder.getArrayAttr(subtile));
  }
  if (!sramStrides.empty()) {
    attrs.set("sram_stride", builder.getArrayAttr(sramStrides));
  }
  if (!dramStrides.empty()) {
    attrs.set("dram_stride", builder.getArrayAttr(dramStrides));
  }
  attrs.set("async", builder.getBoolAttr(async));
  attrs.set("fine_grained", builder.getBoolAttr(true));

  return attrs;
}
}

#endif 