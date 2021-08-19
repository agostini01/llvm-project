// RUN: mlir-opt \
// RUN:  -convert-scf-to-std \
// RUN:  -convert-vector-to-llvm -convert-std-to-llvm %s | \
// RUN: mlir-cpu-runner \
// RUN:  -O0 -e in_out -entry-point-result=void \
// RUN:  -shared-libs=%mlir_runner_utils_dir/libmlir_mockaxi_runner_utils%shlibext | \
// RUN: FileCheck %s

// AXI4MLIR types
!void_type = type memref<*xi8>

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

func @in_out(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %c8 = constant 8 : index
  %c16 = constant 16 : index
  %c32 = constant 32 : index

  // Prepare tile sizes
  %ts_a1 = constant 2 : i64
  %ts_a2 = constant 2 : i64
  %ts_o1 = constant 2 : i64
  %ts_o2 = constant 2 : i64

  // Initializes the DMA
  %idx = constant 0 : index
  call @dma_init(%idx, %idx, %idx, %idx, %idx) : (index,index,index,index,index ) -> ()
  
  // Sizes of in and out buffers
  %in_lenght = muli %ts_a1, %ts_a2 : i64
  %out_lenght = muli %ts_o1, %ts_o2 : i64

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

  // Send the buffers, and start the accelerator
  call @dma_wait_send () : () -> ()
  
  // // // call #accelator_start
  
  // Prepare copy back and receive buffers 
  %status2 =call @dma_start_recv (%out_lenght, %out_offset) : (i64, i64) -> (i64)
  call @dma_wait_recv () : () -> ()
  // memref.copy() // Copy C tile from output address out_buf_addr

  call @dma_free() : () -> ()
  return
}

//CHECK: dma_init
//CHECK: dma_start_send
//CHECK: dma_wait_send
//CHECK: dma_start_recv
//CHECK: dma_wait_recv
//CHECK: dma_free
