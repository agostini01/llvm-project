// RUN: mlir-opt \
// RUN:  -convert-linalg-to-loops -lower-affine -convert-scf-to-std \
// RUN:  -convert-vector-to-llvm -convert-std-to-llvm %s | \
// RUN: mlir-cpu-runner \
// RUN:  -O0 -e generalize_matmul_buffer -entry-point-result=void \
// RUN:  -shared-libs=%mlir_runner_utils_dir/libmlir_mockaxi_runner_utils%shlibext | \
// RUN: FileCheck %s

// AXI4MLIR types
!void_type = type memref<*xi8>

// Other MLIR functions
func private @print_flops(f64)
func private @rtclock() -> f64

#map0 = affine_map<(d0) -> (2, -d0 + 16)>
#map1 = affine_map<(d0) -> (2, -d0 + 8)>
#map2 = affine_map<(d0, d1)[s0] -> (d0 * 8 + s0 + d1)>
#map3 = affine_map<(d0) -> (2, -d0 + 32)>
#map4 = affine_map<(d0, d1)[s0] -> (d0 * 32 + s0 + d1)>

// AXI4MLIR functions
func private @dma_init(index, index, index, index, index) -> ()
func private @dma_free() -> ()
// func private @dma_get_regaddr() -> i64 attributes { llvm.emit_c_interface }
func private @dma_get_inbuffer() -> (!void_type)
func private @dma_get_outbuffer() -> (!void_type)

func private @dma_start_send(i64, i64) -> (i64)
func private @dma_wait_send() -> ()

func private @dma_start_recv(i64, i64) -> (i64)
func private @dma_wait_recv() -> ()

func @generalize_matmul_buffer(%arg0: memref<16x8xf32>, %arg1: memref<8x32xf32>, %arg2: memref<16x32xf32>) {
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %c8 = constant 8 : index
  %c16 = constant 16 : index
  %c32 = constant 32 : index

  // Prepare tile sizes
  %ts_a1 = constant 2 : i64
  %ts_a2 = constant 2 : i64
  %ts_b1 = constant 2 : i64
  %ts_b2 = constant 2 : i64
  %ts_c1 = constant 2 : i64
  %ts_c2 = constant 2 : i64

  // Initializes the DMA
  %idx = constant 0 : index
  call @dma_init(%idx, %idx, %idx, %idx, %idx) : (index,index,index,index,index ) -> ()

  scf.for %arg3 = %c0 to %c16 step %c2 {
    scf.for %arg4 = %c0 to %c32 step %c2 {
      scf.for %arg5 = %c0 to %c8 step %c2 {
        %0 = affine.min #map0(%arg3)
        %1 = affine.min #map1(%arg5)
        %2 = memref.subview %arg0[%arg3, %arg5] [%0, %1] [1, 1] : memref<16x8xf32> to memref<?x?xf32, #map2>
        %3 = affine.min #map1(%arg5)
        %4 = affine.min #map3(%arg4)
        %5 = memref.subview %arg1[%arg5, %arg4] [%3, %4] [1, 1] : memref<8x32xf32> to memref<?x?xf32, #map4>
        %6 = affine.min #map0(%arg3)
        %7 = affine.min #map3(%arg4)
        %8 = memref.subview %arg2[%arg3, %arg4] [%6, %7] [1, 1] : memref<16x32xf32> to memref<?x?xf32, #map4>
        
        // Call that will be replaced
        // linalg.matmul ins(%2, %5 : memref<?x?xf32, #map2>, memref<?x?xf32, #map4>) outs(%8 : memref<?x?xf32, #map4>)

        // Sizes of in and out buffers
        %inA_lenght = muli %ts_a1, %ts_a2 : i64
        %inB_lenght = muli %ts_b1, %ts_b2 : i64
        %in_lenght = addi %inA_lenght, %inB_lenght : i64
        %out_lenght = muli %ts_c1, %ts_c2 : i64

        %in_offset = constant 0 : i64  // offset on the input buffer
        %out_offset = constant 0 : i64 // offset on the output buffer

        // Get the addresses used for the transfers
        %dma_id = constant 0 : index

        // %in_buf_addr = call @dma_get_inbuffer() : () -> (!void_type)

        // %out_buf_addr = call @dma_get_outbuffer() : () -> (!void_type)

        // Copy data to be transfered and set the transfer size
        // memref.copy() // Copy A tile to input address in_buf_addr
        // memref.copy() // Copy B tile to input address+offset in_buf_addr+A_lenght
        %status1 = call @dma_start_send (%in_lenght, %in_offset) : (i64, i64) -> (i64)

        // // // Send the buffers, and start the accelerator
        call @dma_wait_send () : () -> ()
        // call #accelator_start
        
        // Prepare copy back and receive buffers 
        %status2 =call @dma_start_recv (%out_lenght, %out_offset) : (i64, i64) -> (i64)
        // call @dma_wait_recv () : () -> ()
        // memref.copy() // Copy C tile from output address out_buf_addr
      }
    }
  }
  call @dma_free() : () -> ()
  return
}

//CHECK: dma_init

// This is a repeating pattern. Only check the first 2 iterations.
//CHECK: dma_start_send
//CHECK: dma_wait_send
//CHECK: dma_start_recv
//
//CHECK: dma_start_send
//CHECK: dma_wait_send
//CHECK: dma_start_recv
//
// Many more will happen

//CHECK: dma_free
