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

set(tmpcxxflags ${CMAKE_CXX_FLAGS})
string(REPLACE "-Werror=global-constructors" "" FIXED  ${CMAKE_CXX_FLAGS})
string(REPLACE "-Wcast-qual" "-Wno-vla-extension" FIXED  ${FIXED})
  

if(AXI_CROSSCOMPILING)
  message (STATUS "Cross-compiling, SystemC api libraries are disabed")
else()
  message (STATUS "Not Cross-compiling, SystemC api libraries are enabled")
  if(DEFINED ENV{SYSTEMC_HOME})
    message (STATUS "SYSTEMC_HOME is set: $ENV{SYSTEMC_HOME} -- building sysc api libs")

      add_mlir_library(axi_api_v1_sysc
        SHARED
        api_v1_sysc.cpp

        EXCLUDE_FROM_LIBMLIR
      )

      target_include_directories(axi_api_v1_sysc PUBLIC $ENV{SYSTEMC_HOME}/include/)
      set_target_properties(axi_api_v1_sysc PROPERTIES COMPILE_FLAGS "")
      target_link_libraries(axi_api_v1_sysc PUBLIC $ENV{SYSTEMC_HOME}/lib-linux64/libsystemc.a)
  endif()
endif()

set(CMAKE_CXX_FLAGS "${FIXED}")

#No additional properties for now
#target_compile_definitions(axi_api_v1 PRIVATE axi_api_EXPORTS)