// RUN: mlir-opt -lower-affine %s | FileCheck %s

// CHECK-LABEL: func @empty() {
func.func @empty() {
  return     // CHECK:  return
}            // CHECK: }

func.func private @body(index) -> ()

// Simple loops are properly converted.
// CHECK-LABEL: func @simple_loop
// CHECK-NEXT:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[c42:.*]] = arith.constant 42 : index
// CHECK-NEXT:   %[[c1_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c1]] to %[[c42]] step %[[c1_0]] {
// CHECK-NEXT:     call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @simple_loop() {
  affine.for %i = 1 to 42 {
    func.call @body(%i) : (index) -> ()
  }
  return
}