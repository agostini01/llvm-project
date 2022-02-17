// RUN: mlir-opt %s -test-linalg-to-axi4mlir \
// RUN: | FileCheck %s

// CHECK-LABEL: func @main(
func @main(%A: memref<1584x1584xf32>, %B: memref<1584x1584xf32>, %C: memref<1584x1584xf32>) {
  linalg.matmul {__internal_linalg_transform__="L1"}
   ins(%A, %B: memref<1584x1584xf32>, memref<1584x1584xf32>)
   outs(%C: memref<1584x1584xf32>)

  return
}

func @main2(%A: memref<16x8xf32>, %B: memref<8x32xf32>, %C: memref<16x32xf32>) {
  linalg.matmul {__internal_linalg_transform__="NO"}
   ins(%A, %B: memref<16x8xf32>, memref<8x32xf32>)
   outs(%C: memref<16x32xf32>)

  return
}

// MLIR Runner
// func private @print_memref_f32(memref<*xf32>)

// AXI4MLIR functions
// CHECK-LABEL: func private @dma_init(index, index, index, index, index)
// CHECK-LABEL: func private @dma_free()
// CHECK-LABEL: func private @copy_to_inbuffer_f32(memref<*xf32>, i64) -> i64
// CHECK-LABEL: func private @copy_from_outbuffer_f32(memref<*xf32>, i64) -> i64
// CHECK-LABEL: func private @dma_start_send(i64, i64) -> i64
// CHECK-LABEL: func private @dma_wait_send()
// CHECK-LABEL: func private @dma_start_recv(i64, i64) -> i64
// CHECK-LABEL: func private @dma_wait_recv()
