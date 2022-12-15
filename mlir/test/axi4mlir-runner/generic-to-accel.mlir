// RUN: mlir-opt %s --test-generic-to-accel | FileCheck %s
// RUN: mlir-opt %s --test-generic-to-accel='anchor-op=linalg.matmul opcode-map="opcode_map<s0=[op_send(0)]>"' | FileCheck %s --check-prefix=ANCHOR
// RUN: mlir-opt %s --test-generic-to-accel='anchor-op=linalg.matmul flow-cpu-accumulation=true opcode-map="opcode_map<s0=[op_send(0)]>"' | FileCheck %s --check-prefix=CPUACC

#matmul_trait = {
  __accel_transform__="ACCEL",

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

// ---

#double_transpose_trait = {
  __accel_transform__="ACCEL",

  // Original generic information
  iterator_types = ["parallel", "parallel"],
  indexing_maps = [
    affine_map<(m, n) -> (m, n)>, // A
    affine_map<(m, n) -> (m, n)>, // B
    affine_map<(m, n) -> (n, m)>, // C : transpose of A
    affine_map<(m, n) -> (n, m)>  // D : transpose of B
  ]
}

// CHECK-LABEL: func @test_multiple_outputs
// CHECK:       accel.init_dma
// CHECK:       accel.send
// CHECK:       accel.send
// CHECK:       accel.recv
// CHECK:       accel.recv

// CPUACC-LABEL: func @test_multiple_outputs
// CPUACC:       accel.init_dma
// CPUACC:       accel.send
// CPUACC:       accel.send
// CPUACC:       memref.alloc
// CPUACC:       accel.recv
// CPUACC:       linalg.generic
// CPUACC:       memref.alloc
// CPUACC:       accel.recv
// CPUACC:       linalg.generic
func @test_multiple_outputs(%A: memref<16x8xi32>, %B: memref<16x8xi32>, 
                            %C: memref<8x16xi32>, %D: memref<8x16xi32>) {
  linalg.generic #double_transpose_trait
    ins (%A, %B : memref<16x8xi32>, memref<16x8xi32>)
    outs(%C, %D : memref<8x16xi32>, memref<8x16xi32>) {
    ^bb0(%a: i32, %b: i32, %c: i32, %d: i32):
      linalg.yield %a, %b : i32, i32
  }
  return
}

// CHECK-LABEL: func @test_accel_transform_from_matmul
// CHECK:       linalg.fill
// CHECK:       linalg.matmul

// ANCHOR-LABEL: func @test_accel_transform_from_matmul
// ANCHOR:       accel.init_dma
// ANCHOR:       linalg.fill
// ANCHOR:       accel.send
// ANCHOR:       accel.send
// ANCHOR:       accel.recv
func @test_accel_transform_from_matmul(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {

  %0 = arith.constant 0 : i32
  linalg.fill(%0, %C) : i32, memref<16x32xi32>

  linalg.matmul
    ins (%A, %B : memref<16x8xi32>, memref<8x32xi32>)
    outs(%C : memref<16x32xi32>)
  
  return
}