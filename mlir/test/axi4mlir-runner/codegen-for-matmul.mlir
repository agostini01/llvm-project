// RUN: mlir-opt %s -test-matmul-to-axi4mlir | FileCheck %s
// RUN: mlir-opt %s -test-matmul-to-axi4mlir="flow-cpu-accumulation=true" | FileCheck %s --check-prefix=CPU
// RUN: mlir-opt %s -test-matmul-to-axi4mlir="accel-tile-size=8" | FileCheck %s --check-prefix=TILE
// RUN: mlir-opt %s -test-matmul-to-axi4mlir="dma-address=43 dma-input-buffer-size=56" | FileCheck %s --check-prefix=DMA

// CHECK-LABEL: func @main(
// CPU-LABEL: func @main(
// TILE-LABEL: func @main(
// DMA-LABEL: func @main(
func @main(%A: memref<1584x1584xi32>, %B: memref<1584x1584xi32>, %C: memref<1584x1584xi32>) {
  linalg.matmul {__internal_linalg_transform__="GENERALIZE"}
   ins(%A, %B: memref<1584x1584xi32>, memref<1584x1584xi32>)
   outs(%C: memref<1584x1584xi32>)

  return
}

// DMA:      call @dma_init(%c43_i32, %{{.*}}, %c56_i32, %{{.*}}, %{{.*}})

// CHECK:    call @dma_init
// CHECK:    scf.for
// CHECK:      scf.for
// CHECK:        scf.for
// CHECK:          memref.subview
// CHECK:          memref.subview
// CHECK:          memref.subview
// CHECK:          memref.cast
// CHECK:          memref.cast
// CHECK:          memref.cast
// CHECK:          arith.index_cast
// CHECK:          arith.index_cast
// CHECK:          arith.index_cast
// CHECK:          call @copy_to_inbuffer_i32
// CHECK:          call @copy_to_inbuffer_i32
// CHECK:          call @dma_start_send
// CHECK:          call @dma_start_recv
// CHECK:          @dma_wait_send
// CHECK:          call @dma_wait_recv
// CHECK:          call @copy_from_outbuffer_i32
// CHECK:    call @dma_free

// CPU:           %[[VAL0:.*]] = memref.alloca
// CPU:           %[[VAL1:.*]] = memref.cast
// CPU:           call @copy_from_outbuffer_i32(%[[VAL1]]
// CPU:           linalg.generic {{.*}} ins(%[[VAL0]]

// TILE:    scf.for {{.*}} step %c8
// TILE:    scf.for {{.*}} step %c8
// TILE:    scf.for {{.*}} step %c8

// CHECK-LABEL: func @main2(
func @main2(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {
  linalg.matmul {__internal_linalg_transform__="NO"}
   ins(%A, %B: memref<16x8xi32>, memref<8x32xi32>)
   outs(%C: memref<16x32xi32>)

  return
}
// CHECK-NOT: call

// AXI4MLIR functions
// CHECK-LABEL: func private @dma_init(i32, i32, i32, i32, i32)
// CHECK-LABEL: func private @dma_free()
// CHECK-LABEL: func private @copy_to_inbuffer_i32(memref<*xi32>, i32) -> i32
// CHECK-LABEL: func private @copy_from_outbuffer_i32(memref<*xi32>, i32) -> i32
// CHECK-LABEL: func private @dma_start_send(i32, i32) -> i32
// CHECK-LABEL: func private @dma_wait_send()
// CHECK-LABEL: func private @dma_start_recv(i32, i32) -> i32
// CHECK-LABEL: func private @dma_wait_recv()
