// RUN: mlir-opt %s \
// RUN: --test-generic-to-accel='anchor-op=linalg.matmul loop-permutation=1,2,0 opcode-map="opcode_map<s0=[op_send_literal(2), op_send(1)], s1c=[op_send_literal(5), op_send(0)], r=[op_recv(2)]>" opcode-flow="(s0 (s1c))" accel-tile-size=4' --cse \
// RUN: | FileCheck %s


// CHECK-LABEL: @main
// CHECK: for
// CHECK: for
// CHECK: for
// CHECK:   for
// CHECK:   for
// CHECK:   for
// CHECK:     for
// CHECK:     for
// CHECK:     for
// CHECK:       accel.send_literal
// CHECK:       accel.send
// CHECK:       accel.send
// CHECK:       accel.recv
// CHECK: FAIL ON PURPOSE
func @main(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {

  linalg.matmul
    ins (%A, %B : memref<16x8xi32>, memref<8x32xi32>)
    outs(%C : memref<16x32xi32>)
  
  return
}