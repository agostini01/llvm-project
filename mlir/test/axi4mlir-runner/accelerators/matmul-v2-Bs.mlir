// RUN: mlir-opt %s \
// RUN: --test-generic-to-accel='anchor-op=linalg.matmul loop-permutation=1,2,0 opcode-map="opcode_map<s1=[op_send_literal(2), op_send(1)], s0c=[op_send_literal(5), op_send(0)], r=[op_recv(2)]>" opcode-flow="(s1 (s0c r))" accel-tile-size=4 acc-on-cpu=2' --cse \
// RUN: | FileCheck %s


// CHECK-LABEL: @main
// CHECK:     for
// CHECK:     for
// CHECK:       accel.sendLiteral
// CHECK:       memref.subview %arg1
// CHECK:       accel.send
// CHECK:     for
// CHECK:       accel.sendLiteral
// CHECK:       accel.send
// CHECK:       accel.recv
func @main(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {

  linalg.matmul
    ins (%A, %B : memref<16x8xi32>, memref<8x32xi32>)
    outs(%C : memref<16x32xi32>)
  
  return
}