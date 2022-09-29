//===- AccelOps.td - Accel op definitions --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ACCEL_OPS
#define ACCEL_OPS

include "mlir/Dialect/Accel/IR/AccelBase.td"
// include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for accel dialect ops. 
class Accel_Op<string mnemonic, list<Trait> traits = []> :
    Op<Accel_Dialect, mnemonic, traits # [NoSideEffect] >;

//===----------------------------------------------------------------------===//
// dmaInitOp
//===----------------------------------------------------------------------===//

def Accel_InitDMAOp : Accel_Op<"init_dma"> {
  let summary = "initializes the DMA";
  let description = [{
    The `abs` operation computes the absolute value. It takes one operand and
    returns one result of the same type. This type may be a float scalar type,
    a vector whose element type is float, or a tensor of floats.

    Example:

    ```mlir
    // Scalar absolute value.
    %a = accel.abs %b : f64

    // SIMD vector element-wise absolute value.
    %f = accel.abs %g : vector<4xf32>

    // Tensor element-wise absolute value.
    %x = accel.abs %y : tensor<4x?xf8>
    ```
  }];
  let arguments = (ins SignlessIntegerLike:$dmaAddress,
                       SignlessIntegerLike:$dmaInputAddress,
                       SignlessIntegerLike:$dmaInputBufferSize,
                       SignlessIntegerLike:$dmaOutputAddress,
                       SignlessIntegerLike:$dmaOutputBufferSize);


 // let results = (outs SignlessIntegerLike:$result);

  let assemblyFormat = [{
  $dmaAddress `,`
  $dmaInputAddress `,`
  $dmaInputBufferSize `,`
  $dmaOutputAddress `,`
  $dmaOutputBufferSize
  attr-dict `:`
  `(`
  type($dmaAddress) `,`
  type($dmaInputAddress) `,`
  type($dmaInputBufferSize) `,`
  type($dmaOutputAddress) `,`
  type($dmaOutputBufferSize) 
  `)`
  }];
}

def Accel_SendOp : Accel_Op<"send"> {
  let summary = "send MemRef to DMA region";
  let description = [{
    TODO
  }];
  let arguments = (ins AnyMemRef:$memref,
                       Optional<I32>:$offset_value);


  let results = (outs I32:$out_offset);

  let assemblyFormat = [{
    $memref (`,` $offset_value^)?  attr-dict `:` 
    `(` type($memref) (`,` type($offset_value)^)? `)` `->` type($out_offset)
  }];
}

#endif // ACCEL_OPS
