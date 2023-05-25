// RUN: mlir-opt %s \
// RUN: --test-generic-to-accel='anchor-op=linalg.matmul loop-permutation=0,1,2 opcode-map="opcode_map<s=[op_send_literal(7),op_send_literal(0), op_send(0), op_send(1)], r=[op_send_literal(8),op_send_literal(1), op_recv(2)]>" opcode-flow="((s) r)",  number-of-caches=2 accel-tile-size=4' --cse \
// RUN: | FileCheck %s


// CHECK-LABEL: @main
// CHECK:     for
// CHECK:     for
// CHECK:     for {{.*}} {
// CHECK:       accel.sendLiteral
// CHECK:       accel.sendLiteral
// CHECK:       accel.send
// CHECK:       accel.send
// CHECK:     }
// CHECK:     accel.sendLiteral
// CHECK:     accel.sendLiteral
// CHECK:     accel.recv
func @main(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {

  linalg.matmul
    ins (%A, %B : memref<16x8xi32>, memref<8x32xi32>)
    outs(%C : memref<16x32xi32>)
  
  return
}