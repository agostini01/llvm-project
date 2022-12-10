//===- OpcodeListDetail.h - MLIR Opcode List details Class ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license informetion.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of OpcodeList.
//
//===----------------------------------------------------------------------===//

#ifndef OPCODELISTDETAIL_H_
#define OPCODELISTDETAIL_H_

#include "mlir/IR/OpcodeExpr.h"
#include "mlir/IR/OpcodeList.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
namespace detail {

struct OpcodeListStorage final
    : public StorageUniquer::BaseStorage,
      public llvm::TrailingObjects<OpcodeListStorage, OpcodeExpr> {
  /// The hash key used for uniquing.
  using KeyTy = std::tuple<unsigned, ArrayRef<OpcodeExpr>>;

  unsigned numOpcodes; // TODO:: Using numDims as numOpcodes

  MLIRContext *context;

  /// The opcode expressions for this opcode list.
  ArrayRef<OpcodeExpr> results() const;

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == numOpcodes && std::get<1>(key) == results();
  }

  // Constructs an OpcodeListStorage from a key. The context must be set by the
  // caller.
  static OpcodeListStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto results = std::get<1>(key);
    auto byteSize =
        OpcodeListStorage::totalSizeToAlloc<OpcodeExpr>(results.size());
    auto *rawMem = allocator.allocate(byteSize, alignof(OpcodeListStorage));
    auto *res = new (rawMem) OpcodeListStorage();
    res->numOpcodes = results.size();
    std::uninitialized_copy(results.begin(), results.end(),
                            res->getTrailingObjects<OpcodeExpr>());
    return res;
  }
};

ArrayRef<OpcodeExpr> OpcodeListStorage::results() const {
  return {getTrailingObjects<OpcodeExpr>(), numOpcodes};
}

} // namespace detail
} // namespace mlir

#endif // OPCODELISTDETAIL_H_
