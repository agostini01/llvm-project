# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.

set(LLVM_OPTIONAL_SOURCES
  api_v0.cpp
  api_v1.cpp
  )

add_mlir_library(axi_api_v0
  SHARED
  api_v0.cpp

  EXCLUDE_FROM_LIBMLIR
)

add_mlir_library(axi_api_v1
  SHARED
  api_v1.cpp

  EXCLUDE_FROM_LIBMLIR
)

#No additional properties for now
#target_compile_definitions(axi_api_v1 PRIVATE axi_api_EXPORTS)
