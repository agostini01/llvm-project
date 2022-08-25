// RUN: mlir-opt \
// RUN:  -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:  -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts %s | \
// RUN: mlir-cpu-runner \
// RUN:  -O0 -e main -entry-point-result=void \
// RUN:  -shared-libs=%mlir_runner_utils_dir/libmlir_mockaxi_runner_utils%shlibext \
// RUN:  -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s


// /working_dir/builds/llvm-project/build/bin/mlir-opt   -convert-linalg-to-loops -lower-affine -convert-scf-to-std   -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts /working_dir/llvm-project/mlir/test/axi4mlir-runner/accelerator_v1_naive.mlir |  /working_dir/builds/llvm-project/build/bin/mlir-cpu-runner   -O0 -e main -entry-point-result=void   -shared-libs=/working_dir/builds/llvm-project/build/lib/libmlir_syscaxi_runner_utils.so   -shared-libs=/working_dir/builds/llvm-project/build/lib/libmlir_runner_utils.so

// Other MLIR functions
func private @print_flops(f64)
func private @rtclock() -> f64

#map0 = affine_map<(d0) -> (4, -d0 + 16)>
#map1 = affine_map<(d0) -> (4, -d0 + 8)>
#map2 = affine_map<(d0, d1)[s0] -> (d0 * 8 + s0 + d1)>
#map3 = affine_map<(d0) -> (4, -d0 + 32)>
#map4 = affine_map<(d0, d1)[s0] -> (d0 * 32 + s0 + d1)>

// MLIR Runner
func private @print_memref_f32(memref<*xf32>)

// AXI4MLIR functions
func private @dma_init(index, index, index, index, index) -> ()
func private @dma_free() -> ()

func private @copy_to_inbuffer_i32(memref<*xi32>, i64) -> i64
func private @copy_to_inbuffer_f32(memref<*xf32>, i64) -> i64
func private @copy_from_outbuffer_f32(memref<*xf32>, i64) -> i64

func private @dma_start_send(i64, i64) -> i64
func private @dma_wait_send()

func private @dma_start_recv(i64, i64) -> i64
func private @dma_wait_recv()

func @alloc_2d_filled_f32(%s1 : index, %s2 : index, %f : f32) -> memref<?x?xf32> {
  %buf = memref.alloc(%s1, %s2) : memref<?x?xf32>
  linalg.fill(%f, %buf) : f32, memref<?x?xf32>
  return %buf : memref<?x?xf32>
}

func @alloc_2d_filled_inc_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+02 : f32
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  linalg.fill(%arg2, %0) : f32, memref<?x?xf32>
  scf.for %arg3 = %c0 to %arg0 step %c1 {
    scf.for %arg4 = %c0 to %arg1 step %c1 {
      %1 = arith.index_cast %arg3 : index to i32
      %2 = arith.index_cast %arg4 : index to i32
      %3 = arith.sitofp %1 : i32 to f32
      %4 = arith.sitofp %2 : i32 to f32
      %5 = arith.mulf %3, %cst : f32
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %0[%arg3, %arg4] : memref<?x?xf32>
    }
  }
  return %0 : memref<?x?xf32>
}

#id_2d = affine_map<(i, j) -> (i, j)>
#pointwise_2d_trait = {
  indexing_maps = [#id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}

func @main() {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1000 = arith.constant 1000 : index

  // Prepare tile sizes
  %ts_a1 = arith.constant 4 : i64
  %ts_a2 = arith.constant 4 : i64
  %ts_b1 = arith.constant 4 : i64
  %ts_b2 = arith.constant 4 : i64
  %ts_c1 = arith.constant 4 : i64
  %ts_c2 = arith.constant 4 : i64

  %c1_0 = arith.constant 1 : i64
  %c0_0 = arith.constant 0 : i64
  %cst_1 = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32

  // Initializes the DMA
  %idx = arith.constant 0 : index

  %A = call @alloc_2d_filled_inc_f32(%c16, %c8, %cst_1) : (index, index, f32) -> (memref<?x?xf32>)
  %B = call @alloc_2d_filled_f32(%c8, %c32, %cst_1) : (index, index, f32) -> (memref<?x?xf32>)
  %C = call @alloc_2d_filled_f32(%c16, %c32, %cst_0) : (index, index, f32) -> (memref<?x?xf32>)
  %Ctmp = call @alloc_2d_filled_f32(%c16, %c32, %cst_0) : (index, index, f32) -> (memref<?x?xf32>)

  %A_typed = memref.cast %A: memref<?x?xf32> to memref<16x8xf32>
  %B_typed = memref.cast %B: memref<?x?xf32> to memref<8x32xf32>
  %C_typed = memref.cast %C: memref<?x?xf32> to memref<16x32xf32>
  %Ctmp_typed = memref.cast %Ctmp: memref<?x?xf32> to memref<16x32xf32>
  
  %in1 = memref.cast %A_typed: memref<16x8xf32> to memref<*xf32>
  %in2 = memref.cast %B_typed: memref<8x32xf32> to memref<*xf32>
  %out1 = memref.cast %C_typed: memref<16x32xf32> to memref<*xf32>
  %outtmp = memref.cast %Ctmp_typed: memref<16x32xf32> to memref<*xf32>

  call @print_memref_f32(%in1) : (memref<*xf32>) -> ()
  call @print_memref_f32(%in2) : (memref<*xf32>) -> ()
  
  %instruction = memref.alloc() : memref<1xi32>

  // OP-Code Stuct
// 000 : 0 = NOP;
// 001 : 1 = read_A;
// 010 : 2 = read_B;
// 011 : 3 = read_A -> read_B;
// 100 : 4 = compute_C;
// 101 : 5 = read_A -> compute_C;
// 110 : 6 = read_B -> compute_C;
// 111 : 7 = read_A -> read_B -> compute_C;

  // Initializes the DMA
  call @dma_init(%c0, %c0, %c1000, %c0, %c1000) : (index,index,index,index,index ) -> ()

    scf.for %arg4 = %c0 to %c32 step %c4 {
      scf.for %arg5 = %c0 to %c8 step %c4 {
        
        // Prepare instruction - send B
        %inst_readB = arith.constant 2 : i32
        memref.store %inst_readB, %instruction[%c0] : memref<1xi32>
        %inst_readB_casted = memref.cast %instruction: memref<1xi32> to memref<*xi32>
        call @copy_to_inbuffer_i32(%inst_readB_casted, %c0_0) : (memref<*xi32>, i64) -> (i64)

        // Slice the memory reference
        %5 = memref.subview %B_typed[%arg5, %arg4] [%c4, %c4] [1, 1] : memref<8x32xf32> to memref<?x?xf32, #map4>
        %in2_tile = memref.cast %5: memref<?x?xf32, #map4> to memref<*xf32>
        
        %inB_lenght = arith.muli %ts_b1, %ts_b2 : i64
        %inB_offset = arith.addi %c0_0, %c1_0 : i64  // offset on the input buffer
        %total_input_lenght_readB = arith.addi %c1_0, %inB_lenght : i64
          
        call @copy_to_inbuffer_f32(%in2_tile, %inB_offset) : (memref<*xf32>, i64) -> (i64)
        call @dma_start_send (%total_input_lenght_readB, %c0_0) : (i64, i64) -> (i64)
        call @dma_wait_send () : () -> ()
  
        scf.for %arg3 = %c0 to %c16 step %c4 {
        
          // Prepare instruction - send A, Compute, retrieve C
          %inst_SCR = arith.constant 5 : i32
          memref.store %inst_SCR, %instruction[%c0] : memref<1xi32>
          %inst_SCR_casted = memref.cast %instruction: memref<1xi32> to memref<*xi32>
          call @copy_to_inbuffer_i32(%inst_SCR_casted, %c0_0) : (memref<*xi32>, i64) -> (i64)

          // Slice the memory references
          %2 = memref.subview %A_typed[%arg3, %arg5] [%c4, %c4] [1, 1] : memref<16x8xf32> to memref<?x?xf32, #map2>
          %8 = memref.subview %C_typed[%arg3, %arg4] [%c4, %c4] [1, 1] : memref<16x32xf32> to memref<?x?xf32, #map4>
          %42 = memref.subview %Ctmp_typed[%arg3, %arg4] [%c4, %c4] [1, 1] : memref<16x32xf32> to memref<?x?xf32, #map4>
          
          // Call that will be replaced
          // linalg.matmul ins(%2, %5 : memref<?x?xf32, #map2>, memref<?x?xf32, #map4>) outs(%8 : memref<?x?xf32, #map4>)

          %in1_tile = memref.cast %2: memref<?x?xf32, #map2> to memref<*xf32>
          %out1_tile = memref.cast %8: memref<?x?xf32, #map4> to memref<*xf32>
          %out1_tile_tmp = memref.cast %42: memref<?x?xf32, #map4> to memref<*xf32>
    
          call @print_memref_f32(%in1_tile) : (memref<*xf32>) -> ()
          call @print_memref_f32(%in2_tile) : (memref<*xf32>) -> ()

          // Sizes of in and out buffers
          %inA_lenght = arith.muli %ts_a1, %ts_a2 : i64
          %total_input_lenght_SCR = arith.addi %inA_lenght, %c1_0 : i64
          %out_lenght = arith.muli %ts_c1, %ts_c2 : i64

          %inA_offset = arith.constant 1 : i64  // offset on the input buffer
          %out_offset = arith.constant 0 : i64 // offset on the output buffer

          // Copy data to be transfered
          call @copy_to_inbuffer_f32(%in1_tile, %inA_offset) : (memref<*xf32>, i64) -> i64
          call @dma_start_send (%total_input_lenght_SCR, %c0_0) : (i64, i64) -> (i64)
          call @dma_start_recv (%out_lenght, %out_offset) : (i64, i64) -> (i64)

          // Wait for operations to complete
          call @dma_wait_send () : () -> ()
          call @dma_wait_recv () : () -> ()
          
          // Copy C tile from DMA output buffer
          call @copy_from_outbuffer_f32 (%out1_tile_tmp, %out_offset) : (memref<*xf32>, i64) -> (i64)
          call @print_memref_f32(%out1_tile_tmp) : (memref<*xf32>) -> ()

          // Accumulate on the output
          linalg.generic #pointwise_2d_trait
            ins(%42: memref<?x?xf32, #map4>)
            outs(%8 : memref<?x?xf32, #map4>) {
          ^bb0(%x: f32, %y: f32):   // no predecessors
            %res = arith.addf %x, %y : f32
            linalg.yield %res : f32
          }
          call @print_memref_f32(%out1_tile) : (memref<*xf32>) -> ()
      }
    }
  }
  call @print_memref_f32(%out1) : (memref<*xf32>) -> ()
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

// Many more will happen

//CHECK: dma_free
