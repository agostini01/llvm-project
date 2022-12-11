// RUN: mlir-opt %s \
// RUN: | FileCheck %s


#map0 = affine_map<(m, n, k) -> (m, k)>

#my_trait = opcode_map< 
  s0 = [op_send(0)],
  s1 = [op_send(1)],
  s2 = [op_send(1)],
  r2 = [op_recv(2)],
  s0s1r2 = [op_send(0), op_send(1), op_send(2), op_recv(2)],
  reset = [op_send(0)],
  s0_dim = [op_send_literal(30), op_send_dim(2)],
  test = [op_send_literal(30), op_send_literal(31)],
  s0_idxs = [op_send_literal(30), op_send_idx(2)]
>

#other_trait = {
  // Original generic information
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>, // A
    affine_map<(m, n, k) -> (k, n)>, // B
    affine_map<(m, n, k) -> (m, n)>  // C
  ]
}

#matmul_trait = {
  accel_opcode_map = opcode_map<
    s0 = [op_send(0)],
    s1 = [op_send(1)],
    s2 = [op_send(1)],
    r2 = [op_recv(2)],
    s0s1r2 = [op_send(0), op_send(1), op_send(2), op_recv(2)],
    reset = [op_send(0)],
    s0_dim = [op_send_literal(30), op_send_dim(2)],
    s0_idxs = [op_send_literal(30), op_send_idx(2)]
  >,

  // accel_opcode_flow_str = "(s0 (s1 s2 r2))",
  // accel_flow_pattern= flow_pattern < (s0 (s0_s1_r2)+) >,
  // accel_flow_pattern= flow_pattern < accel_opcode_map, (s0 (s0_s1_r2)+) >,

  // accel_init_opcodes_str = "(s0 (s1 s2 r2))",
  // accel_init_opcodes = init_opcodes < (reset) >,

  // Original generic information
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>, // A
    affine_map<(m, n, k) -> (k, n)>, // B
    affine_map<(m, n, k) -> (m, n)>  // C
  ]
}

// CHECK-LABEL: func @test_accel_transform
// CHECK:       accel_opcode_map = opcode_map<
// CHECK:           s0 = [op_send(0)],
// CHECK:           s1 = [op_send(1)],
// CHECK:           s2 = [op_send(1)],
// CHECK:           r2 = [op_recv(2)],
// CHECK:           s0s1r2 = [op_send(0), op_send(1), op_send(2), op_recv(2)],
// CHECK:           reset = [op_send(0)],
// CHECK:           s0_dim = [op_send_literal(30), op_send_dim(2)],
// CHECK:           s0_idxs = [op_send_literal(30), op_send_idx(2)]
// CHECK:         >
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
