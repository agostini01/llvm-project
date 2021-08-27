# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.

set(LLVM_OPTIONAL_SOURCES
  api_v0.cpp
  api_v1.cpp
  api_v1_sysc.cpp
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

add_mlir_library(axi_api_v1_sysc
  SHARED
  api_v1_sysc.cpp

  EXCLUDE_FROM_LIBMLIR
)

# Fixing compiler flags for SytemC --- expects env var "SYSTEMC_HOME" to point to SystemC install
target_include_directories(axi_api_v1_sysc PUBLIC $ENV{SYSTEMC_HOME}/include/)
set_target_properties(axi_api_v1_sysc PROPERTIES COMPILE_FLAGS "")
target_link_libraries(axi_api_v1_sysc PUBLIC $ENV{SYSTEMC_HOME}/lib-linux64/libsystemc.a)

set(tmpcxxflags ${CMAKE_CXX_FLAGS})
string(REPLACE "-Werror=global-constructors" "" FIXED  ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${FIXED}")

#No additional properties for now
#target_compile_definitions(axi_api_v1 PRIVATE axi_api_EXPORTS)