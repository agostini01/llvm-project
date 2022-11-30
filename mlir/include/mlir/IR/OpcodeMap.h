//===- OpcodeMap.h - MLIR Opcode Map Class ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Opcode maps are mathematical functions which map a list of dimension
// identifiers and symbols, to multidimensional affine expressions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPCODEMAP_H
#define MLIR_IR_OPCODEMAP_H

#include "mlir/IR/OpcodeExpr.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace llvm {
class SmallBitVector;
} // namespace llvm

namespace mlir {

namespace detail {
struct OpcodeMapStorage;
} // namespace detail

class Attribute;
struct LogicalResult;
class MLIRContext;

/// A multi-dimensional affine map
/// Opcode map's are immutable like Type's, and they are uniqued.
/// Eg: (d0, d1) -> (d0/128, d0 mod 128, d1)
/// The names used (d0, d1) don't matter - it's the mathematical function that
/// is unique to this affine map.
class OpcodeMap {
public:
  using ImplType = detail::OpcodeMapStorage;

  constexpr OpcodeMap() = default;
  explicit OpcodeMap(ImplType *map) : map(map) {}

  /// Returns a zero result affine map with no dimensions or symbols: () -> ().
  static OpcodeMap get(MLIRContext *context);

  /// Returns a zero result affine map with `dimCount` dimensions and
  /// `symbolCount` symbols, e.g.: `(...) -> ()`.
  static OpcodeMap get(unsigned dimCount, unsigned symbolCount,
                       MLIRContext *context);

  /// Returns an affine map with `dimCount` dimensions and `symbolCount` mapping
  /// to a single output dimension
  static OpcodeMap get(unsigned dimCount, unsigned symbolCount,
                       OpcodeExpr result);

  /// Returns an affine map with `dimCount` dimensions and `symbolCount` mapping
  /// to the given results.
  static OpcodeMap get(unsigned dimCount, unsigned symbolCount,
                       ArrayRef<OpcodeExpr> results, MLIRContext *context);

  /// Returns a single constant result affine map.
  static OpcodeMap getConstantMap(int64_t val, MLIRContext *context);

  /// Returns an OpcodeMap with 'numDims' identity result dim exprs.
  static OpcodeMap getAMultiDimIdentityMap(unsigned numDims,
                                          MLIRContext *context);

  /// Returns an identity affine map (d0, ..., dn) -> (dp, ..., dn) on the most
  /// minor dimensions.
  static OpcodeMap getMinorIdentityMap(unsigned dims, unsigned results,
                                       MLIRContext *context);

  /// Returns an OpcodeMap representing a permutation.
  /// The permutation is expressed as a non-empty vector of integers.
  /// E.g. the permutation `(i,j,k) -> (j,k,i)` will be expressed with
  /// `permutation = [1,2,0]`. All values in `permutation` must be
  /// integers, in the range 0..`permutation.size()-1` without duplications
  /// (i.e. `[1,1,2]` is an invalid permutation).
  static OpcodeMap getPermutationMap(ArrayRef<unsigned> permutation,
                                     MLIRContext *context);

  /// Returns a vector of OpcodeMaps; each with as many results as
  /// `exprs.size()`, as many dims as the largest dim in `exprs` and as many
  /// symbols as the largest symbol in `exprs`.
  static SmallVector<OpcodeMap, 4>
  inferFromExprList(ArrayRef<ArrayRef<OpcodeExpr>> exprsList);
  static SmallVector<OpcodeMap, 4>
  inferFromExprList(ArrayRef<SmallVector<OpcodeExpr, 4>> exprsList);

  MLIRContext *getContext() const;

  explicit operator bool() const { return map != nullptr; }
  bool operator==(OpcodeMap other) const { return other.map == map; }
  bool operator!=(OpcodeMap other) const { return !(other.map == map); }

  /// Returns true if this affine map is an identity affine map.
  /// An identity affine map corresponds to an identity affine function on the
  /// dimensional identifiers.
  bool isIdentity() const;

  /// Returns true if this affine map is a minor identity, i.e. an identity
  /// affine map (d0, ..., dn) -> (dp, ..., dn) on the most minor dimensions.
  bool isMinorIdentity() const;

  /// Returns true if this affine map is a minor identity up to broadcasted
  /// dimensions which are indicated by value 0 in the result. If
  /// `broadcastedDims` is not null, it will be populated with the indices of
  /// the broadcasted dimensions in the result array.
  /// Example: affine_map<(d0, d1, d2, d3, d4) -> (0, d2, 0, d4)>
  ///          (`broadcastedDims` will contain [0, 2])
  bool isMinorIdentityWithBroadcasting(
      SmallVectorImpl<unsigned> *broadcastedDims = nullptr) const;

  /// Return true if this affine map can be converted to a minor identity with
  /// broadcast by doing a permute. Return a permutation (there may be
  /// several) to apply to get to a minor identity with broadcasts.
  /// Ex:
  ///  * (d0, d1, d2) -> (0, d1) maps to minor identity (d1, 0 = d2) with
  ///  perm = [1, 0] and broadcast d2
  ///  * (d0, d1, d2) -> (d0, 0) cannot be mapped to a minor identity by
  ///  permutation + broadcast
  ///  * (d0, d1, d2, d3) -> (0, d1, d3) maps to minor identity (d1, 0 = d2, d3)
  ///  with perm = [1, 0, 2] and broadcast d2
  ///  * (d0, d1) -> (d1, 0, 0, d0) maps to minor identity (d0, d1) with extra
  ///  leading broadcat dimensions. The map returned would be (0, 0, d0, d1)
  ///  with perm = [3, 0, 1, 2]
  bool isPermutationOfMinorIdentityWithBroadcasting(
      SmallVectorImpl<unsigned> &permutedDims) const;

  /// Returns true if this affine map is an empty map, i.e., () -> ().
  bool isEmpty() const;

  /// Returns true if this affine map is a single result constant function.
  bool isSingleConstant() const;

  /// Returns true if this affine map has only constant results.
  bool isConstant() const;

  /// Returns the constant result of this map. This methods asserts that the map
  /// has a single constant result.
  int64_t getSingleConstantResult() const;

  /// Returns the constant results of this map. This method asserts that the map
  /// has all constant results.
  SmallVector<int64_t> getConstantResults() const;

  // Prints affine map to 'os'.
  void print(raw_ostream &os) const;
  void dump() const;

  unsigned getNumDims() const;
  unsigned getNumSymbols() const;
  unsigned getNumResults() const;
  unsigned getNumInputs() const;

  ArrayRef<OpcodeExpr> getResults() const;
  OpcodeExpr getResult(unsigned idx) const;

  /// Extracts the position of the dimensional expression at the given result,
  /// when the caller knows it is safe to do so.
  unsigned getDimPosition(unsigned idx) const;

  /// Extracts the permuted position where given input index resides.
  /// Fails when called on a non-permutation.
  unsigned getPermutedPosition(unsigned input) const;

  /// Return true if any affine expression involves OpcodeDimExpr `position`.
  bool isFunctionOfDim(unsigned position) const {
    return llvm::any_of(getResults(), [&](OpcodeExpr e) {
      return e.isFunctionOfDim(position);
    });
  }

  /// Return true if any affine expression involves OpcodeSymbolExpr `position`.
  bool isFunctionOfSymbol(unsigned position) const {
    return llvm::any_of(getResults(), [&](OpcodeExpr e) {
      return e.isFunctionOfSymbol(position);
    });
  }

  /// Walk all of the OpcodeExpr's in this mapping. Each node in an expression
  /// tree is visited in postorder.
  void walkExprs(llvm::function_ref<void(OpcodeExpr)> callback) const;

  /// This method substitutes any uses of dimensions and symbols (e.g.
  /// dim#0 with dimReplacements[0]) in subexpressions and returns the modified
  /// expression mapping.  Because this can be used to eliminate dims and
  /// symbols, the client needs to specify the number of dims and symbols in
  /// the result.  The returned map always has the same number of results.
  OpcodeMap replaceDimsAndSymbols(ArrayRef<OpcodeExpr> dimReplacements,
                                  ArrayRef<OpcodeExpr> symReplacements,
                                  unsigned numResultDims,
                                  unsigned numResultSyms) const;

  /// Sparse replace method. Apply OpcodeExpr::replace(`expr`, `replacement`) to
  /// each of the results and return a new OpcodeMap with the new results and
  /// with the specified number of dims and symbols.
  OpcodeMap replace(OpcodeExpr expr, OpcodeExpr replacement,
                    unsigned numResultDims, unsigned numResultSyms) const;

  /// Sparse replace method. Apply OpcodeExpr::replace(`map`) to each of the
  /// results and return a new OpcodeMap with the new results and with inferred
  /// number of dims and symbols.
  OpcodeMap replace(const DenseMap<OpcodeExpr, OpcodeExpr> &map) const;

  /// Sparse replace method. Apply OpcodeExpr::replace(`map`) to each of the
  /// results and return a new OpcodeMap with the new results and with the
  /// specified number of dims and symbols.
  OpcodeMap replace(const DenseMap<OpcodeExpr, OpcodeExpr> &map,
                    unsigned numResultDims, unsigned numResultSyms) const;

  /// Replace dims[offset ... numDims)
  /// by dims[offset + shift ... shift + numDims).
  OpcodeMap shiftDims(unsigned shift, unsigned offset = 0) const {
    assert(offset <= getNumDims());
    return OpcodeMap::get(getNumDims() + shift, getNumSymbols(),
                          llvm::to_vector<4>(llvm::map_range(
                              getResults(),
                              [&](OpcodeExpr e) {
                                return e.shiftDims(getNumDims(), shift, offset);
                              })),
                          getContext());
  }

  /// Replace symbols[offset ... numSymbols)
  /// by symbols[offset + shift ... shift + numSymbols).
  OpcodeMap shiftSymbols(unsigned shift, unsigned offset = 0) const {
    return OpcodeMap::get(getNumDims(), getNumSymbols() + shift,
                          llvm::to_vector<4>(llvm::map_range(
                              getResults(),
                              [&](OpcodeExpr e) {
                                return e.shiftSymbols(getNumSymbols(), shift,
                                                      offset);
                              })),
                          getContext());
  }

  /// Folds the results of the application of an affine map on the provided
  /// operands to a constant if possible.
  LogicalResult constantFold(ArrayRef<Attribute> operandConstants,
                             SmallVectorImpl<Attribute> &results) const;

  /// Propagates the constant operands into this affine map. Operands are
  /// allowed to be null, at which point they are treated as non-constant. This
  /// does not change the number of symbols and dimensions. Returns a new map,
  /// which may be equal to the old map if no folding happened. If `results` is
  /// provided and if all expressions in the map were folded to constants,
  /// `results` will contain the values of these constants.
  OpcodeMap
  partialConstantFold(ArrayRef<Attribute> operandConstants,
                      SmallVectorImpl<int64_t> *results = nullptr) const;

  /// Returns the OpcodeMap resulting from composing `this` with `map`.
  /// The resulting OpcodeMap has as many OpcodeDimExpr as `map` and as many
  /// OpcodeSymbolExpr as the concatenation of `this` and `map` (in which case
  /// the symbols of `this` map come first).
  ///
  /// Prerequisites:
  /// The maps are composable, i.e. that the number of OpcodeDimExpr of `this`
  /// matches the number of results of `map`.
  ///
  /// Example:
  ///   map1: `(d0, d1)[s0, s1] -> (d0 + 1 + s1, d1 - 1 - s0)`
  ///   map2: `(d0)[s0] -> (d0 + s0, d0 - s0)`
  ///   map1.compose(map2):
  ///     `(d0)[s0, s1, s2] -> (d0 + s1 + s2 + 1, d0 - s0 - s2 - 1)`
  OpcodeMap compose(OpcodeMap map) const;

  /// Applies composition by the dims of `this` to the integer `values` and
  /// returns the resulting values. `this` must be symbol-less.
  SmallVector<int64_t, 4> compose(ArrayRef<int64_t> values) const;

  /// Returns true if the OpcodeMap represents a subset (i.e. a projection) of a
  /// symbol-less permutation map. `allowZeroInResults` allows projected
  /// permutation maps with constant zero result expressions.
  /// TODO: Remove `allowZeroInResults` when constant zero result expressions
  /// are broadly supported.
  bool isProjectedPermutation(bool allowZeroInResults = false) const;

  /// Returns true if the OpcodeMap represents a symbol-less permutation map.
  bool isPermutation() const;

  /// Returns the map consisting of the `resultPos` subset.
  OpcodeMap getSubMap(ArrayRef<unsigned> resultPos) const;

  /// Returns the map consisting of `length` expressions starting from `start`.
  OpcodeMap getSliceMap(unsigned start, unsigned length) const;

  /// Returns the map consisting of the most major `numResults` results.
  /// Returns the null OpcodeMap if `numResults` == 0.
  /// Returns `*this` if `numResults` >= `this->getNumResults()`.
  OpcodeMap getMajorSubMap(unsigned numResults) const;

  /// Returns the map consisting of the most minor `numResults` results.
  /// Returns the null OpcodeMap if `numResults` == 0.
  /// Returns `*this` if `numResults` >= `this->getNumResults()`.
  OpcodeMap getMinorSubMap(unsigned numResults) const;

  friend ::llvm::hash_code hash_value(OpcodeMap arg);

  /// Methods supporting C API.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(map);
  }
  static OpcodeMap getFromOpaquePointer(const void *pointer) {
    return OpcodeMap(reinterpret_cast<ImplType *>(const_cast<void *>(pointer)));
  }

private:
  ImplType *map{nullptr};

  static OpcodeMap getImpl(unsigned dimCount, unsigned symbolCount,
                           ArrayRef<OpcodeExpr> results, MLIRContext *context);
};

// Make OpcodeExpr hashable.
inline ::llvm::hash_code hash_value(OpcodeMap arg) {
  return ::llvm::hash_value(arg.map);
}

/// A mutable affine map. Its affine expressions are however unique.
struct MutableOpcodeMap {
public:
  MutableOpcodeMap() = default;
  MutableOpcodeMap(OpcodeMap map);

  ArrayRef<OpcodeExpr> getResults() const { return results; }
  OpcodeExpr getResult(unsigned idx) const { return results[idx]; }
  void setResult(unsigned idx, OpcodeExpr result) { results[idx] = result; }
  unsigned getNumResults() const { return results.size(); }
  unsigned getNumDims() const { return numDims; }
  void setNumDims(unsigned d) { numDims = d; }
  unsigned getNumSymbols() const { return numSymbols; }
  void setNumSymbols(unsigned d) { numSymbols = d; }
  MLIRContext *getContext() const { return context; }

  /// Returns true if the idx'th result expression is a multiple of factor.
  bool isMultipleOf(unsigned idx, int64_t factor) const;

  /// Resets this MutableOpcodeMap with 'map'.
  void reset(OpcodeMap map);

  /// Simplify the (result) expressions in this map using analysis (used by
  //-simplify-affine-expr pass).
  void simplify();
  /// Get the OpcodeMap corresponding to this MutableOpcodeMap. Note that an
  /// OpcodeMap will be uniqued and stored in context, while a mutable one
  /// isn't.
  OpcodeMap getOpcodeMap() const;

private:
  // Same meaning as OpcodeMap's fields.
  SmallVector<OpcodeExpr, 8> results;
  unsigned numDims = 0;
  unsigned numSymbols = 0;
  /// A pointer to the IR's context to store all newly created
  /// OpcodeExprStorage's.
  MLIRContext *context = nullptr;
};

/// Simplifies an affine map by simplifying its underlying OpcodeExpr results.
OpcodeMap simplifyOpcodeMap(OpcodeMap map);

/// Drop the dims that are not used.
OpcodeMap compressUnusedDims(OpcodeMap map);

/// Drop the dims that are not used by any of the individual maps in `maps`.
/// Asserts that all maps in `maps` are normalized to the same number of
/// dims and symbols.
SmallVector<OpcodeMap> compressUnusedDims(ArrayRef<OpcodeMap> maps);

/// Drop the dims that are not listed in `unusedDims`.
OpcodeMap compressDims(OpcodeMap map, const llvm::SmallBitVector &unusedDims);

/// Drop the symbols that are not used.
OpcodeMap compressUnusedSymbols(OpcodeMap map);

/// Drop the symbols that are not used by any of the individual maps in `maps`.
/// Asserts that all maps in `maps` are normalized to the same number of
/// dims and symbols.
SmallVector<OpcodeMap> compressUnusedSymbols(ArrayRef<OpcodeMap> maps);

/// Drop the symbols that are not listed in `unusedSymbols`.
OpcodeMap compressSymbols(OpcodeMap map,
                          const llvm::SmallBitVector &unusedSymbols);

/// Returns a map with the same dimension and symbol count as `map`, but whose
/// results are the unique affine expressions of `map`.
OpcodeMap removeDuplicateExprs(OpcodeMap map);

/// Returns a map of codomain to domain dimensions such that the first codomain
/// dimension for a particular domain dimension is selected.
/// Returns an empty map if the input map is empty.
/// Returns null map (not empty map) if `map` is not invertible (i.e. `map` does
/// not contain a subset that is a permutation of full domain rank).
///
/// Prerequisites:
///   1. `map` has no symbols.
///
/// Example 1:
///
/// ```mlir
///    (d0, d1, d2) -> (d1, d1, d0, d2, d1, d2, d1, d0)
///                      0       2   3
/// ```
///
/// returns:
///
/// ```mlir
///    (d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d0, d3)
/// ```
///
/// Example 2:
///
/// ```mlir
///    (d0, d1, d2) -> (d1, d0 + d1, d0, d2, d1, d2, d1, d0)
///                      0            2   3
/// ```
///
/// returns:
///
/// ```mlir
///    (d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d0, d3)
/// ```
OpcodeMap inversePermutation(OpcodeMap map);

/// Return the reverse map of a projected permutation where the projected
/// dimensions are transformed into 0s.
///
/// Prerequisites: `map` must be a projected permuation.
///
/// Example 1:
///
/// ```mlir
///    affine_map<(d0, d1, d2, d3) -> (d2, d0)>
/// ```
///
/// returns:
///
/// ```mlir
///    affine_map<(d0, d1) -> (d1, 0, d0, 0)>
/// ```
///
/// Example 2:
///
/// ```mlir
///    affine_map<(d0, d1, d2, d3) -> (d0, d3)>
/// ```
///
/// returns:
///
/// ```mlir
///    affine_map<(d0, d1) -> (d0, 0, 0, d1)>
/// ```
///
/// Example 3:
///
/// ```mlir
///    affine_map<(d0, d1, d2, d3) -> (d2)>
/// ```
///
/// returns:
///
/// ```mlir
///    affine_map<(d0) -> (0, 0, d0, 0)>
/// ```
/// Example 4:
///
/// ```mlir
///    affine_map<(d0, d1, d2) -> (d0, 0)>
/// ```
///
/// returns:
///
/// ```mlir
///    affine_map<(d0, d1) -> (d0, 0, 0)>
/// ```
OpcodeMap inverseAndBroadcastProjectedPermuation(OpcodeMap map);

/// Concatenates a list of `maps` into a single OpcodeMap, stepping over
/// potentially empty maps. Assumes each of the underlying map has 0 symbols.
/// The resulting map has a number of dims equal to the max of `maps`' dims and
/// the concatenated results as its results.
/// Returns an empty map if all input `maps` are empty.
///
/// Example:
/// When applied to the following list of 3 affine maps,
///
/// ```mlir
///    {
///      (i, j, k) -> (i, k),
///      (i, j, k) -> (k, j),
///      (i, j, k) -> (i, j)
///    }
/// ```
///
/// Returns the map:
///
/// ```mlir
///     (i, j, k) -> (i, k, k, j, i, j)
/// ```
OpcodeMap concatOpcodeMaps(ArrayRef<OpcodeMap> maps);

/// Returns the map that results from projecting out the dimensions specified in
/// `projectedDimensions`. The projected dimensions are set to 0.
///
/// Example:
/// 1) map                  : affine_map<(d0, d1, d2) -> (d0, d1)>
///    projected_dimensions : {2}
///    result               : affine_map<(d0, d1) -> (d0, d1)>
///
/// 2) map                  : affine_map<(d0, d1) -> (d0 + d1)>
///    projected_dimensions : {1}
///    result               : affine_map<(d0) -> (d0)>
///
/// 3) map                  : affine_map<(d0, d1, d2) -> (d0, d1)>
///    projected_dimensions : {1}
///    result               : affine_map<(d0, d1) -> (d0, 0)>
///
/// This function also compresses unused symbols away.
OpcodeMap getProjectedMap(OpcodeMap map,
                          const llvm::SmallBitVector &projectedDimensions);

/// Apply a permutation from `map` to `source` and return the result.
template <typename T>
SmallVector<T> applyPermutationMap(OpcodeMap map, llvm::ArrayRef<T> source) {
  assert(map.isProjectedPermutation());
  assert(map.getNumInputs() == source.size());
  SmallVector<T> result;
  result.reserve(map.getNumResults());
  for (OpcodeExpr expr : map.getResults()) {
    if (auto dimExpr = expr.dyn_cast<OpcodeDimExpr>()) {
      result.push_back(source[dimExpr.getPosition()]);
    } else if (auto constExpr = expr.dyn_cast<OpcodeConstantExpr>()) {
      assert(constExpr.getValue() == 0 &&
             "Unexpected constant in projected permutation map");
      result.push_back(0);
    } else {
      llvm_unreachable("Unexpected result in projected permutation map");
    }
  }
  return result;
}

/// Calculates maxmimum dimension and symbol positions from the expressions
/// in `exprsLists` and stores them in `maxDim` and `maxSym` respectively.
template <typename OpcodeExprContainer>
static void getAMaxDimAndSymbol(ArrayRef<OpcodeExprContainer> exprsList,
                               int64_t &maxDim, int64_t &maxSym) {
  for (const auto &exprs : exprsList) {
    for (auto expr : exprs) {
      expr.walk([&maxDim, &maxSym](OpcodeExpr e) {
        if (auto d = e.dyn_cast<OpcodeDimExpr>())
          maxDim = std::max(maxDim, static_cast<int64_t>(d.getPosition()));
        if (auto s = e.dyn_cast<OpcodeSymbolExpr>())
          maxSym = std::max(maxSym, static_cast<int64_t>(s.getPosition()));
      });
    }
  }
}

inline raw_ostream &operator<<(raw_ostream &os, OpcodeMap map) {
  map.print(os);
  return os;
}
} // namespace mlir

namespace llvm {

// OpcodeExpr hash just like pointers
template <>
struct DenseMapInfo<mlir::OpcodeMap> {
  static mlir::OpcodeMap getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::OpcodeMap(static_cast<mlir::OpcodeMap::ImplType *>(pointer));
  }
  static mlir::OpcodeMap getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::OpcodeMap(static_cast<mlir::OpcodeMap::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::OpcodeMap val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::OpcodeMap LHS, mlir::OpcodeMap RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // MLIR_IR_OPCODEMAP_H
