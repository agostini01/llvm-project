//===- OpcodeMapDetail.h - MLIR Opcode Map details Class --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of OpcodeMap.
//
//===----------------------------------------------------------------------===//

#ifndef OPCODEMAPDETAIL_H_
#define OPCODEMAPDETAIL_H_

#include "mlir/IR/OpcodeExpr.h"
#include "mlir/IR/OpcodeMap.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
namespace detail {

struct OpcodeMapStorage final
    : public StorageUniquer::BaseStorage,
      public llvm::TrailingObjects<OpcodeMapStorage, OpcodeExpr> {
  /// The hash key used for uniquing.
  using KeyTy = std::tuple<unsigned, unsigned, ArrayRef<OpcodeExpr>>;

  unsigned numDims;
  unsigned numSymbols;
  unsigned numResults;

  MLIRContext *context;

  /// The affine expressions for this (multi-dimensional) map.
  ArrayRef<OpcodeExpr> results() const {
    return {getTrailingObjects<OpcodeExpr>(), numResults};
  }

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == numDims && std::get<1>(key) == numSymbols &&
           std::get<2>(key) == results();
  }

  // Constructs an OpcodeMapStorage from a key. The context must be set by the
  // caller.
  static OpcodeMapStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto results = std::get<2>(key);
    auto byteSize =
        OpcodeMapStorage::totalSizeToAlloc<OpcodeExpr>(results.size());
    auto *rawMem = allocator.allocate(byteSize, alignof(OpcodeMapStorage));
    auto *res = new (rawMem) OpcodeMapStorage();
    res->numDims = std::get<0>(key);
    res->numSymbols = std::get<1>(key);
    res->numResults = results.size();
    std::uninitialized_copy(results.begin(), results.end(),
                            res->getTrailingObjects<OpcodeExpr>());
    return res;
  }
};

} // namespace detail
} // namespace mlir

#endif // OPCODEMAPDETAIL_H_
