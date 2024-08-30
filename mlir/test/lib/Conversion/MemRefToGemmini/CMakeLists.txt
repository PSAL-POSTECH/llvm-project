# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestMemRefToGemmini
  TestMemRefToGemminiConversion.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
)
