add_mlir_dialect_library(MLIRAccel
  AccelOps.cpp
  AccelDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Accel

  DEPENDS
  MLIRAccelOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
)
