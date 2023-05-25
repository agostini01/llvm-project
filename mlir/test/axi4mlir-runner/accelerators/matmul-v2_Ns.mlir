// RUN: mlir-opt %s \
// RUN: --test-generic-to-accel='anchor-op=linalg.matmul loop-permutation=0,1,2 opcode-map="opcode_map<s=[op_send_literal(7), op_send(0), op_send(1)], r=[op_recv(2)]>" opcode-flow="(s r)",  number-of-caches=2 tile-sizes=128,128,128,32,32,32 accel-tile-size=4  acc-on-cpu=2' --cse \
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
// CHECK:       accel.sendLiteral
// CHECK:       accel.send
// CHECK:       accel.send
// CHECK:       accel.recv
func @main(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {

  linalg.matmul
    ins (%A, %B : memref<16x8xi32>, memref<8x32xi32>)
    outs(%C : memref<16x32xi32>)
  
  return
}