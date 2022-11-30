// RUN: mlir-opt %s \
// RUN: | FileCheck %s

#matmul_trait = {
  accel_opcode_map = {
    map<s0 -> [send(0)]>,
    map<s1 -> [send(1)]>,
    map<s2 -> [send(1)]>,
    map<r2 -> [recv(2)]>,
    map<s0s1r2 -> [send(0), send(1), send(2), recv(2)]>,
    map<reset -> [send("32")]>,
  },
  // accel_opcode_flow_str = "(s0 (s1 s2 r2))",

  // Original generic information
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>, // A
    affine_map<(m, n, k) -> (k, n)>, // B
    affine_map<(m, n, k) -> (m, n)>  // C
  ]
}

// CHECK-LABEL: func @test_accel_transform
// CHECK: accel_opcode_map = {
// CHECK:   map<s0 -> [send(0)]>,
// CHECK:   map<s1 -> [send(1)]>,
// CHECK:   map<s2 -> [send(1)]>,
// CHECK:   map<r2 -> [recv(2)]>,
// CHECK:   map<s0s1r2 -> [send(0), send(1), send(2), recv(2)]>,
// CHECK:   map<reset -> [send("32")]>,
// CHECK: },
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
