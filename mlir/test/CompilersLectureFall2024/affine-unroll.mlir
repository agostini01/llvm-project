// RUN: mlir-opt --affine-loop-unroll -lower-affine %s | FileCheck %s

// CHECK-LABEL: func @empty() {
func.func @empty() {
  return     // CHECK:  return
}            // CHECK: }

func.func private @body(index) -> ()

// Simple loops are properly converted.
// CHECK-LABEL: func @simple_loop
// CHECK:     call @body(%{{.*}}) : (index) -> ()
// CHECK:     call @body(%{{.*}}) : (index) -> ()
// CHECK:     call @body(%{{.*}}) : (index) -> ()
// CHECK:     call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT: }
func.func @simple_loop() {
  affine.for %i = 1 to 42 {
    func.call @body(%i) : (index) -> ()
  }
  return
}