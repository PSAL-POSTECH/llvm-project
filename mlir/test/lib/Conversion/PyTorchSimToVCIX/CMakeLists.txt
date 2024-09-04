# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestPyTorchSimToVCIX
  TestPyTorchSimToVCIXConversion.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
)
