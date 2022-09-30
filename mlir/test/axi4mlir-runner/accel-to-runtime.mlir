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
