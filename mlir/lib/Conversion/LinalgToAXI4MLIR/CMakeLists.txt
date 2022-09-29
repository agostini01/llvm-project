add_mlir_conversion_library(MLIRLinalgToAXI4MLIR
  LinalgToAXI4MLIR.cpp
  AccelToRuntime.cpp
  GenericToAccel.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LinalgToAXI4MLIR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRLLVMIR
  MLIRMemRef
  MLIRTransforms
  )
