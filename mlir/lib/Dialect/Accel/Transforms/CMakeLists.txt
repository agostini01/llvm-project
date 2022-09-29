add_mlir_dialect_library(MLIRAccelTransforms
  HoistAccelOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Accel/Transforms

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRStandard
  MLIRTransforms
  )
