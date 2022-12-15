// RUN: mlir-opt %s \
// RUN: --test-generic-to-accel='anchor-op=linalg.matmul loop-permutation=0,1,2 flow-cpu-accumulation=true opcode-map="opcode_map<s0=[op_send(0)], s1=[op_send(1)], s2=[op_send(2)], r2=[op_recv(2)], reset=[op_send(32)]>" opcode-flow="(sA (sB sC))" init-flow="reset"  number-of-caches=2 tile-sizes=128,128,128,32,32,32 accel-tile-size=8' \
// RUN: | FileCheck %s

#matmul_trait = {
  __accel_transform__="ACCEL",
  accel_loop_permutation = [0,2,1],
  accel_number_of_caches=2,
  accel_tile_sizes = [128,128,128,32,32,32],
  accel_accel_tile_size = 8,
  accel_acc_on_cpu = true,
  accel_opcode_map_str = "opcode_map<s0=[op_send(0)], s1=[op_send(1)], s2=[op_send(2)], r2=[op_recv(2)], s0s1s2r2=[op_send(0), op_send(1), op_send(2), op_recv(2)]>",
  accel_init_flow_str = "reset",
  accel_opcode_flow_str = "(s0 (s1 s2 r2))",

  // Original generic information
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>, // A
    affine_map<(m, n, k) -> (k, n)>, // B
    affine_map<(m, n, k) -> (m, n)>  // C
  ]
}

// CHECK-LABEL: func @test_accel_transform
// CHECK:       accel.init_dma
// CHECK:       accel.send
// CHECK:       accel.send
// CHECK:       accel.recv
// CHECK:       linalg.generic
func @test_accel_transform(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {

  linalg.generic #matmul_trait
    ins (%A, %B : memref<16x8xi32>, memref<8x32xi32>)
    outs(%C : memref<16x32xi32>) {
    ^bb0(%a: i32, %b: i32, %c: i32):
      %0 = arith.muli %a, %b : i32
      %1 = arith.addi %c, %0 : i32
      linalg.yield %1 : i32
  }
  return
}

// CHECK-LABEL: func @test_accel_transform_from_matmul
// CHECK:       accel
func @test_accel_transform_from_matmul(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {

  %0 = arith.constant 0 : i32
  linalg.matmul
    ins (%A, %B : memref<16x8xi32>, memref<8x32xi32>)
    outs(%C : memref<16x32xi32>)
  
  return
}