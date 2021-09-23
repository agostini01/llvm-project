// RUN: mlir-opt \
// RUN:  -convert-linalg-to-loops -convert-scf-to-std \
// RUN:  -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm %s | \
// RUN: mlir-cpu-runner \
// RUN:  -O0 -e main -entry-point-result=void \
// RUN:  -shared-libs=%mlir_runner_utils_dir/libmlir_mockaxi_runner_utils%shlibext \
// RUN:  -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s

// MLIR Runner
func private @print_memref_f32(memref<*xf32>)

// AXI4MLIR types
!ptr_type = type !llvm.ptr<i8>

// AXI4MLIR functions
func private @dma_init(index, index, index, index, index) -> ()
func private @dma_free() -> ()
// func private @dma_get_regaddr() -> i64 attributes { llvm.emit_c_interface }
func private @dma_get_inbuffer() -> (!ptr_type)
func private @dma_get_outbuffer() -> (!ptr_type)


// func private @memref_to_ptr(memref<*xf32>) -> (!ptr_type)
// func private @ptr_to_memref(!ptr_type) -> (memref<*xf32>)

// DELETE ME
func private @memref_to_ptr() -> (!ptr_type)
func private @ptr_to_memref() -> (memref<*xf32>)
func private @dma_copy_to_inbuffer(!ptr_type, i64, i64) -> (i64)
func private @dma_copy_from_outbuffer(!ptr_type, i64, i64) -> (i64)

func private @mlir_dma_copy_to_inbuffer(memref<*xf32>, i64, i64) -> (i64)
func private @mlir_dma_copy_from_outbuffer(memref<*xf32>, i64, i64) -> (i64)

func private @dma_start_send(i64, i64) -> (i64)
func private @dma_wait_send() -> ()

func private @dma_start_recv(i64, i64) -> (i64)
func private @dma_wait_recv() -> ()

// Performing these C opertaions
// dma1.dma_init(0,0,1000,0,1000);
// dma1.dma_copy_to_inbuffer(reinterpret_cast<unsigned int*>(inputs),rows*depth,0);
// dma1.dma_copy_to_inbuffer(reinterpret_cast<unsigned int*>(weightsT),depth*cols,rows*depth);
// dma1.dma_start_send(dma1.current_input_offset,0);
// dma1.dma_start_recv(rows*cols +1 ,0);
// dma1.dma_wait_send();
// dma1.dma_wait_recv();
// dma1.dma_copy_from_outbuffer(reinterpret_cast<unsigned int*>(accelerated_outputs),cols*rows,0);

func @alloc_2d_filled_f32(%s1 : index, %s2 : index, %f : f32) -> memref<?x?xf32> {
  %buf = memref.alloc(%s1, %s2) : memref<?x?xf32>
  linalg.fill(%f, %buf) : f32, memref<?x?xf32>
  return %buf : memref<?x?xf32>
}

func @main() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c8 = constant 8 : index
  %c16 = constant 16 : index
  %c32 = constant 32 : index
  %c1000 = constant 1000 : index

  // Prepare tile sizes
  %ts_a1 = constant 4 : i64
  %ts_a2 = constant 4 : i64
  %ts_o1 = constant 4 : i64
  %ts_o2 = constant 4 : i64


  %c1_0 = constant 1 : i64
  %cst_1 = constant 1.000000e+00 : f32
  %cst_0 = constant 0.000000e+00 : f32


  %A = call @alloc_2d_filled_f32(%c4, %c4, %cst_1) : (index, index, f32) -> (memref<?x?xf32>)
  %B = call @alloc_2d_filled_f32(%c4, %c4, %cst_1) : (index, index, f32) -> (memref<?x?xf32>)
  %C = call @alloc_2d_filled_f32(%c4, %c4, %cst_0) : (index, index, f32) -> (memref<?x?xf32>)
  
  %arg0 = memref.cast %A: memref<?x?xf32> to memref<4x4xf32>
  %arg1 = memref.cast %B: memref<?x?xf32> to memref<4x4xf32>
  %arg2 = memref.cast %C: memref<?x?xf32> to memref<4x4xf32>

  %in1 = memref.cast %arg0: memref<4x4xf32> to memref<*xf32>
  %in2 = memref.cast %arg1: memref<4x4xf32> to memref<*xf32>
  %out1 = memref.cast %arg2: memref<4x4xf32> to memref<*xf32>


  call @print_memref_f32(%in1) : (memref<*xf32>) -> ()
  call @print_memref_f32(%in2) : (memref<*xf32>) -> ()


  // Initializes the DMA
  call @dma_init(%c0, %c0, %c1000, %c0, %c1000) : (index,index,index,index,index ) -> ()
  
  // Sizes of in and out buffers
  %in1_lenght = muli %ts_a1, %ts_a2 : i64
  %in2_lenght = muli %ts_a1, %ts_a2 : i64
  %total_input_lenght = addi %in1_lenght, %in2_lenght : i64
  %out_lenght = muli %ts_o1, %ts_o2 : i64
  
  // REMOVE 
  // %in_lenght = constant 0 : i64
  // %out_lenght = constant 0 : i64

  %in1_offset = constant 0 : i64  // offset on the input buffer
  %in2_offset = muli %c1_0, %in1_lenght : i64  // offset on the input buffer
  %out_offset = constant 0 : i64 // offset on the output buffer

  // Get the addresses used for the transfers
  %dma_id = constant 0 : index

  // %in1_ptr = call @memref_to_ptr(%in1) : (memref<*xf32>) -> (!ptr_type)
  // %in2_ptr = call @memref_to_ptr(%in2) : (memref<*xf32>) -> (!ptr_type)
 
  // DELETE ME
  // %in1_ptr = call @memref_to_ptr() : () -> (!ptr_type)
  // %in2_ptr = call @memref_to_ptr() : () -> (!ptr_type)
  // call @dma_copy_to_inbuffer (%in1_ptr, %in1_lenght, %in1_offset) : (!ptr_type, i64, i64) -> (i64)
  // call @dma_copy_to_inbuffer (%in2_ptr, %in2_lenght, %in2_offset) : (!ptr_type, i64, i64) -> (i64)
  
  call @mlir_dma_copy_to_inbuffer (%in1, %in1_lenght, %in1_offset) : (memref<*xf32>, i64, i64) -> (i64)
  call @mlir_dma_copy_to_inbuffer (%in2, %in2_lenght, %in2_offset) : (memref<*xf32>, i64, i64) -> (i64)
  // dma1.dma_copy_to_inbuffer(reinterpret_cast<unsigned int*>(inputs),rows*depth,0);
  // dma1.dma_copy_to_inbuffer(reinterpret_cast<unsigned int*>(weightsT),depth*cols,rows*depth);

  // %out_buf_addr = call @dma_get_outbuffer() : () -> (!ptr_type)
  // %out = call @ptr_to_memref(%out_buf_addr) : (!ptr_type) -> (memref<*xf32>)
  // DELETE ME
  // %out = call @ptr_to_memref() : () -> (memref<*xf32>)

  // Copy data to be transfered and set the transfer size
  // memref.copy() // Copy A tile to input address in_buf_addr
  // memref.copy() // Copy B tile to input address+offset in_buf_addr+A_lenght
  call @dma_start_send (%total_input_lenght, %in1_offset) : (i64, i64) -> (i64)
  call @dma_start_recv (%out_lenght, %out_offset) : (i64, i64) -> (i64)

  // Wait for operations to complete
  call @dma_wait_send () : () -> ()
  call @dma_wait_recv () : () -> ()
  
  // memref.copy() // Copy C tile from output address out_buf_addr
  // dma1.dma_copy_from_outbuffer(reinterpret_cast<unsigned int*>(accelerated_outputs),cols*rows,0);
  call @mlir_dma_copy_from_outbuffer (%out1, %in2_lenght, %in2_offset) : (memref<*xf32>, i64, i64) -> (i64)

  call @print_memref_f32(%out1) : (memref<*xf32>) -> ()

  call @dma_free() : () -> ()
  return
}

//CHECK: dma_init
//CHECK: dma_start_send
//CHECK: dma_wait_send
//CHECK: dma_start_recv
//CHECK: dma_wait_recv
//CHECK: dma_free
//CHECK: stop here