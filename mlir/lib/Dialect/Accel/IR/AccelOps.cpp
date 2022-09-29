//===- AccelOps.cpp - MLIR operations for accel implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Accel/IR/Accel.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::accel;

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Accel/IR/AccelOps.cpp.inc"
