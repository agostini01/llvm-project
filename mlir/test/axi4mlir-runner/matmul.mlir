// RUN: mlir-opt %s -test-linalg-to-axi4mlir \
// RUN: | FileCheck %s

func @alloc_2d_filled_f32(%s1 : index, %s2 : index, %f : f32) -> memref<?x?xf32> {
  %buf = memref.alloc(%s1, %s2) : memref<?x?xf32>
  linalg.fill(%f, %buf) : f32, memref<?x?xf32>
  return %buf : memref<?x?xf32>
}

// CHECK-LABEL: func @matmul(
// OUTER-LABEL: func @matmul(
// GENER-LABEL: func @matmul(
func @main(%A: memref<1584x1584xf32>, %B: memref<1584x1584xf32>, %C: memref<1584x1584xf32>) {
  linalg.matmul
   ins(%A, %B: memref<1584x1584xf32>, memref<1584x1584xf32>)
   outs(%C: memref<1584x1584xf32>)

  // CHECK: Fail
  return
}

func @main2(%A: memref<16x8xf32>, %B: memref<8x32xf32>, %C: memref<16x32xf32>) {
  linalg.matmul
   ins(%A, %B: memref<16x8xf32>, memref<8x32xf32>)
   outs(%C: memref<16x32xf32>)

  // CHECK: FAIL

  return
}
