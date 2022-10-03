// RUN: mlir-opt %s --test-accel-to-axi4mlir | FileCheck %s


// CHECK: func private @dma_init
// CHECK-NOT: func private @dma_init

// CHECK-LABEL: test_init_dma
// CHECK:         call @dma_init(%arg0
func @test_init_dma(
  %dmaAddress : i32,
  %dmaInputAddress : i32,
  %dmaInputBufferSize : i32,
  %dmaOutputAddress : i32,
  %dmaOutputBufferSize : i32) {
  accel.init_dma  %dmaAddress,
                  %dmaInputAddress, %dmaInputBufferSize,
                  %dmaOutputAddress, %dmaOutputBufferSize
  : (i32, i32, i32, i32, i32)
  return
}

// CHECK-LABEL: test_init_dma2
// CHECK:         call @dma_init(%arg0
// CHECK-NEXT:    call @dma_init(%arg1
// CHECK-NEXT:    call @dma_init(%arg2
func @test_init_dma2(
  %dmaAddress : i32,
  %dmaAddress1 : i32,
  %dmaAddress2 : i32,
  %dmaInputAddress : i32,
  %dmaInputBufferSize : i32,
  %dmaOutputAddress : i32,
  %dmaOutputBufferSize : i32) {
  accel.init_dma  %dmaAddress,
                  %dmaInputAddress, %dmaInputBufferSize,
                  %dmaOutputAddress, %dmaOutputBufferSize
  : (i32, i32, i32, i32, i32)
  accel.init_dma  %dmaAddress1,
                  %dmaInputAddress, %dmaInputBufferSize,
                  %dmaOutputAddress, %dmaOutputBufferSize
  : (i32, i32, i32, i32, i32)
  accel.init_dma  %dmaAddress2,
                  %dmaInputAddress, %dmaInputBufferSize,
                  %dmaOutputAddress, %dmaOutputBufferSize
  : (i32, i32, i32, i32, i32)
  return
}

// CHECK-LABEL: test_send
// CHECK:   %[[CASTED:.*]] = memref.cast
// CHECK:   %[[C0:.*]] = arith.constant 0
// CHECK:   call @copy_to_inbuffer_i32(%[[CASTED]], %[[C0]]) : (memref<*xf32>, i32) -> i32
func @test_send(%A: memref<60x80xf32>) -> i32 {
  %offset = accel.send %A  : ( memref<60x80xf32> ) -> i32
  return %offset : i32
}

// CHECK-LABEL: test_send_with_offset
// CHECK:   %[[CASTED:.*]] = memref.cast
// CHECK:   call @copy_to_inbuffer_i32(%[[CASTED]], %{{.*}}) : (memref<*xf32>, i32) -> i32
// CHECK:   return %c19200
func @test_send_with_offset(%A: memref<60x80xf32>, %offset0: i32) -> i32 {
  %offset = accel.send %A, %offset0  : (memref<60x80xf32> , i32) -> i32
  return %offset : i32
}

// CHECK-LABEL: test_send_with_subview
// CHECK:   %[[CASTED:.*]] = memref.cast
// CHECK:   call @copy_to_inbuffer_i32(%[[CASTED]], %{{.*}}) : (memref<*xf32>, i32) -> i32
// CHECK:   return %c2048
#map = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
func @test_send_with_subview(%input: memref<4x1024xf32>) -> i32 {
  %cst_2 = arith.constant 2 : index
  %0 = memref.subview %input[%cst_2, 256] [2, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, #map>
  %offset = accel.send %0  : ( memref<2x256xf32, #map> ) -> i32
  return %offset : i32
}

