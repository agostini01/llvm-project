; RUN: mlir-translate --import-llvm %s | FileCheck %s
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; CHECK-LABEL: llvm.func @matmul
define void @matmul(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7) {
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, ptr %1, 1
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 %2, 2
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 %3, 3, 0
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 %4, 4, 0
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 0
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %14, 0
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 1
  %17 = getelementptr i8, ptr %16, i64 0
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, ptr %17, 1
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 0, 2
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %7, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 1, 4, 1
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 %5, 3, 0
  %23 = mul i64 1, %7
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 %23, 4, 0
  %25 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %25, 0
  %27 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 1
  %28 = getelementptr i8, ptr %27, i64 0
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, ptr %28, 1
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 0, 2
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %6, 3, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 1, 4, 1
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %7, 3, 0
  %34 = mul i64 1, %6
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %34, 4, 0
  %36 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %36, 0
  %38 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 1
  %39 = getelementptr i8, ptr %38, i64 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, ptr %39, 1
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 0, 2
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %6, 3, 1
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 1, 4, 1
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 %5, 3, 0
  %45 = mul i64 1, %6
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 %45, 4, 0
  br label %47

47:                                               ; preds = %85, %8
  %48 = phi i64 [ %86, %85 ], [ 0, %8 ]
  %49 = icmp slt i64 %48, %5
  br i1 %49, label %50, label %87

50:                                               ; preds = %83, %47
  %51 = phi i64 [ %84, %83 ], [ 0, %47 ]
  %52 = icmp slt i64 %51, %6
  br i1 %52, label %53, label %85

53:                                               ; preds = %56, %50
  %54 = phi i64 [ %82, %56 ], [ 0, %50 ]
  %55 = icmp slt i64 %54, %7
  br i1 %55, label %56, label %83

56:                                               ; preds = %53
  %57 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1
  %58 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 4, 0
  %59 = mul i64 %48, %58
  %60 = add i64 %59, %54
  %61 = getelementptr float, ptr %57, i64 %60
  %62 = load float, ptr %61, align 4
  %63 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %64 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 4, 0
  %65 = mul i64 %54, %64
  %66 = add i64 %65, %51
  %67 = getelementptr float, ptr %63, i64 %66
  %68 = load float, ptr %67, align 4
  %69 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %70 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %71 = mul i64 %48, %70
  %72 = add i64 %71, %51
  %73 = getelementptr float, ptr %69, i64 %72
  %74 = load float, ptr %73, align 4
  %75 = fmul float %62, %68
  %76 = fadd float %74, %75
  %77 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %78 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %79 = mul i64 %48, %78
  %80 = add i64 %79, %51
  %81 = getelementptr float, ptr %77, i64 %80
  store float %76, ptr %81, align 4
  %82 = add i64 %54, 1
  br label %53

83:                                               ; preds = %53
  %84 = add i64 %51, 1
  br label %50

85:                                               ; preds = %50
  %86 = add i64 %48, 1
  br label %47

87:                                               ; preds = %47
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
