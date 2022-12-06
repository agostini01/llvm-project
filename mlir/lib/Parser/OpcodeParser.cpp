//===- OpcodeParser.cpp - MLIR Opcode Parser ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a parser for Opcode structures.
//
//===----------------------------------------------------------------------===//

#include "Parser.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpcodeMap.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::detail;

namespace {

/// Lower precedence ops (all at the same precedence level). LNoOp is false in
/// the boolean sense.
enum OpcodeLowPrecOp {
  /// Null value.
  LNoOp,
  Add,
  Sub
};

/// Higher precedence ops - all at the same precedence level. HNoOp is false
/// in the boolean sense.
enum OpcodeHighPrecOp {
  /// Null value.
  HNoOp,
  Mul,
  FloorDiv,
  CeilDiv,
  Mod
};

/// This is a specialized parser for opcode structures (opcode maps, opcode
/// expressions), maintaining the state transient to their bodies.
class OpcodeParser : public Parser {
public:
  OpcodeParser(ParserState &state, bool allowParsingSSAIds = false,
               function_ref<ParseResult(bool)> parseElement = nullptr)
      : Parser(state), allowParsingSSAIds(allowParsingSSAIds),
        parseElement(parseElement), numDimOperands(0), numSymbolOperands(0) {}

  /// opcode_dict  ::= `opcode_map` `<` opcode-entry (`,` opcode-entry)* `>`
  ///
  /// opcode_entry ::= (bare-id | string-literal) `=` opcode_list
  ///
  /// opcode_list  ::= `[` opcode_expr (`,` opcode_expr)* `]
  ///
  /// opcode_expr  ::= op_send(bare-id
  /// .              | send_literal(integer-literal))
  ///                | op_send_dim(bare-id)
  ///                | op_send_idx(bare-id)
  ///                | op_recv(bare-id)
  ParseResult parseOpcodeMapInline(OpcodeMap &map);
  ParseResult parseOpcodeDict(NamedAttrList &attributes);

  OpcodeMap parseOpcodeMapRange(unsigned numDims, unsigned numSymbols);
  ParseResult parseOpcodeMapOrIntegerSetInline(OpcodeMap &map, IntegerSet &set);
  // IntegerSet parseIntegerSetConstraints(unsigned numDims, unsigned
  // numSymbols); // TODO: implement
  ParseResult parseOpcodeMapOfSSAIds(OpcodeMap &map,
                                     OpAsmParser::Delimiter delimiter);
  ParseResult parseOpcodeExprOfSSAIds(OpcodeExpr &expr);
  void getDimsAndSymbolSSAIds(SmallVectorImpl<StringRef> &dimAndSymbolSSAIds,
                              unsigned &numDims);

private:
  // Binary opcode op parsing.
  OpcodeLowPrecOp consumeIfLowPrecOp();
  OpcodeHighPrecOp consumeIfHighPrecOp();

  ParseResult parseSymbol();

  // Identifier lists for polyhedral structures.
  ParseResult parseDimIdList(unsigned &numDims);
  ParseResult parseSymbolIdList(unsigned &numSymbols);
  ParseResult parseDimAndOptionalSymbolIdList(unsigned &numDims,
                                              unsigned &numSymbols);
  ParseResult parseIdentifierDefinition(OpcodeExpr idExpr);

  OpcodeExpr parseOpcodeExpr();
  ParseResult parseSendExpr(Token::Kind expectedToken,
                            function_ref<ParseResult()> parseElementFn);
  OpcodeExpr parseParentheticalExpr();
  OpcodeExpr parseNegateExpression(OpcodeExpr lhs);
  OpcodeExpr parseIntegerExpr();
  OpcodeExpr parseBareIdExpr();
  OpcodeExpr parseSSAIdExpr(bool isSymbol);
  OpcodeExpr parseSymbolSSAIdExpr();

  OpcodeExpr getOpcodeBinaryOpExpr(OpcodeHighPrecOp op, OpcodeExpr lhs,
                                   OpcodeExpr rhs, SMLoc opLoc);
  OpcodeExpr getOpcodeBinaryOpExpr(OpcodeLowPrecOp op, OpcodeExpr lhs,
                                   OpcodeExpr rhs);
  OpcodeExpr parseOpcodeOperandExpr(OpcodeExpr lhs);
  OpcodeExpr parseOpcodeLowPrecOpExpr(OpcodeExpr llhs, OpcodeLowPrecOp llhsOp);
  OpcodeExpr parseOpcodeHighPrecOpExpr(OpcodeExpr llhs, OpcodeHighPrecOp llhsOp,
                                       SMLoc llhsOpLoc);
  OpcodeExpr parseOpcodeConstraint(bool *isEq);

private:
  bool allowParsingSSAIds;
  function_ref<ParseResult(bool)> parseElement;
  unsigned numDimOperands;
  unsigned numSymbolOperands;
  SmallVector<std::pair<StringRef, OpcodeExpr>, 4> dimsAndSymbols;
  SmallVector<std::pair<StringRef, OpcodeExpr>, 4> opcodeAndCommands;
};
} // namespace

/// Create an opcode binary high precedence op expression (mul's, div's, mod).
/// opLoc is the location of the op token to be used to report errors
/// for non-conforming expressions.
OpcodeExpr OpcodeParser::getOpcodeBinaryOpExpr(OpcodeHighPrecOp op,
                                               OpcodeExpr lhs, OpcodeExpr rhs,
                                               SMLoc opLoc) {
  // TODO: make the error location info accurate.
  switch (op) {
  case Mul:
    if (!lhs.isSymbolicOrConstant() && !rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-opcode expression: at least one of the multiply "
                       "operands has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs * rhs;
  case FloorDiv:
    if (!rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-opcode expression: right operand of floordiv "
                       "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.floorDiv(rhs);
  case CeilDiv:
    if (!rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-opcode expression: right operand of ceildiv "
                       "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.ceilDiv(rhs);
  case Mod:
    if (!rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-opcode expression: right operand of mod "
                       "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs % rhs;
  case HNoOp:
    llvm_unreachable("can't create opcode expression for null high prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown OpcodeHighPrecOp");
}

/// Create an opcode binary low precedence op expression (add, sub).
OpcodeExpr OpcodeParser::getOpcodeBinaryOpExpr(OpcodeLowPrecOp op,
                                               OpcodeExpr lhs, OpcodeExpr rhs) {
  switch (op) {
  case OpcodeLowPrecOp::Add:
    return lhs + rhs;
  case OpcodeLowPrecOp::Sub:
    return lhs - rhs;
  case OpcodeLowPrecOp::LNoOp:
    llvm_unreachable("can't create opcode expression for null low prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown OpcodeLowPrecOp");
}

/// Consume this token if it is a lower precedence opcode op (there are only
/// two precedence levels).
OpcodeLowPrecOp OpcodeParser::consumeIfLowPrecOp() {
  switch (getToken().getKind()) {
  case Token::plus:
    consumeToken(Token::plus);
    return OpcodeLowPrecOp::Add;
  case Token::minus:
    consumeToken(Token::minus);
    return OpcodeLowPrecOp::Sub;
  default:
    return OpcodeLowPrecOp::LNoOp;
  }
}

/// Consume this token if it is a higher precedence opcode op (there are only
/// two precedence levels)
OpcodeHighPrecOp OpcodeParser::consumeIfHighPrecOp() {
  switch (getToken().getKind()) {
  case Token::star:
    consumeToken(Token::star);
    return Mul;
  case Token::kw_floordiv:
    consumeToken(Token::kw_floordiv);
    return FloorDiv;
  case Token::kw_ceildiv:
    consumeToken(Token::kw_ceildiv);
    return CeilDiv;
  case Token::kw_mod:
    consumeToken(Token::kw_mod);
    return Mod;
  default:
    return HNoOp;
  }
}

/// Parse a high precedence op expression list: mul, div, and mod are high
/// precedence binary ops, i.e., parse a
///   expr_1 op_1 expr_2 op_2 ... expr_n
/// where op_1, op_2 are all a OpcodeHighPrecOp (mul, div, mod).
/// All opcode binary ops are left associative.
/// Given llhs, returns (llhs llhsOp lhs) op rhs, or (lhs op rhs) if llhs is
/// null. If no rhs can be found, returns (llhs llhsOp lhs) or lhs if llhs is
/// null. llhsOpLoc is the location of the llhsOp token that will be used to
/// report an error for non-conforming expressions.
OpcodeExpr OpcodeParser::parseOpcodeHighPrecOpExpr(OpcodeExpr llhs,
                                                   OpcodeHighPrecOp llhsOp,
                                                   SMLoc llhsOpLoc) {
  OpcodeExpr lhs = parseOpcodeOperandExpr(llhs);
  if (!lhs)
    return nullptr;

  // Found an LHS. Parse the remaining expression.
  auto opLoc = getToken().getLoc();
  if (OpcodeHighPrecOp op = consumeIfHighPrecOp()) {
    if (llhs) {
      OpcodeExpr expr = getOpcodeBinaryOpExpr(llhsOp, llhs, lhs, opLoc);
      if (!expr)
        return nullptr;
      return parseOpcodeHighPrecOpExpr(expr, op, opLoc);
    }
    // No LLHS, get RHS
    return parseOpcodeHighPrecOpExpr(lhs, op, opLoc);
  }

  // This is the last operand in this expression.
  if (llhs)
    return getOpcodeBinaryOpExpr(llhsOp, llhs, lhs, llhsOpLoc);

  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

/// Parse an opcode expression inside parentheses.
///
///   opcode-expr ::= `(` opcode-expr `)`
OpcodeExpr OpcodeParser::parseParentheticalExpr() {
  if (parseToken(Token::l_paren, "expected '('"))
    return nullptr;
  if (getToken().is(Token::r_paren))
    return (emitError("no expression inside parentheses"), nullptr);

  auto expr = parseOpcodeExpr();
  if (!expr)
    return nullptr;
  if (parseToken(Token::r_paren, "expected ')'"))
    return nullptr;

  return expr;
}

/// Parse the negation expression.
///
///   opcode-expr ::= `-` opcode-expr
OpcodeExpr OpcodeParser::parseNegateExpression(OpcodeExpr lhs) {
  if (parseToken(Token::minus, "expected '-'"))
    return nullptr;

  OpcodeExpr operand = parseOpcodeOperandExpr(lhs);
  // Since negation has the highest precedence of all ops (including high
  // precedence ops) but lower than parentheses, we are only going to use
  // parseOpcodeOperandExpr instead of parseOpcodeExpr here.
  if (!operand)
    // Extra error message although parseOpcodeOperandExpr would have
    // complained. Leads to a better diagnostic.
    return (emitError("missing operand of negation"), nullptr);
  return (-1) * operand;
}

/// Parse a bare id that may appear in an opcode expression.
///
///   opcode-expr ::= bare-id
OpcodeExpr OpcodeParser::parseBareIdExpr() {
  if (getToken().isNot(Token::bare_identifier))
    return (emitError("expected bare identifier"), nullptr);

  StringRef sRef = getTokenSpelling();
  for (auto entry : dimsAndSymbols) {
    if (entry.first == sRef) {
      consumeToken(Token::bare_identifier);
      return entry.second;
    }
  }

  return (emitError("use of undeclared identifier"), nullptr);
}

/// Parse an SSA id which may appear in an opcode expression.
OpcodeExpr OpcodeParser::parseSSAIdExpr(bool isSymbol) {
  if (!allowParsingSSAIds)
    return (emitError("unexpected ssa identifier"), nullptr);
  if (getToken().isNot(Token::percent_identifier))
    return (emitError("expected ssa identifier"), nullptr);
  auto name = getTokenSpelling();
  // Check if we already parsed this SSA id.
  for (auto entry : dimsAndSymbols) {
    if (entry.first == name) {
      consumeToken(Token::percent_identifier);
      return entry.second;
    }
  }
  // Parse the SSA id and add an OpcodeDim/SymbolExpr to represent it.
  if (parseElement(isSymbol))
    return (emitError("failed to parse ssa identifier"), nullptr);
  auto idExpr = isSymbol
                    ? getOpcodeSymbolExpr(numSymbolOperands++, getContext())
                    : getOpcodeDimExpr(numDimOperands++, getContext());
  dimsAndSymbols.push_back({name, idExpr});
  return idExpr;
}

OpcodeExpr OpcodeParser::parseSymbolSSAIdExpr() {
  if (parseToken(Token::kw_symbol, "expected symbol keyword") ||
      parseToken(Token::l_paren, "expected '(' at start of SSA symbol"))
    return nullptr;
  OpcodeExpr symbolExpr = parseSSAIdExpr(/*isSymbol=*/true);
  if (!symbolExpr)
    return nullptr;
  if (parseToken(Token::r_paren, "expected ')' at end of SSA symbol"))
    return nullptr;
  return symbolExpr;
}

/// Parse a positive integral constant appearing in an opcode expression.
///
///   opcode-expr ::= integer-literal
OpcodeExpr OpcodeParser::parseIntegerExpr() {
  auto val = getToken().getUInt64IntegerValue();
  if (!val.hasValue() || (int64_t)val.getValue() < 0)
    return (emitError("constant too large for index"), nullptr);

  consumeToken(Token::integer);
  return builder.getOpcodeConstantExpr((int64_t)val.getValue());
}

/// Parses an expression that can be a valid operand of an opcode expression.
/// lhs: if non-null, lhs is an opcode expression that is the lhs of a binary
/// operator, the rhs of which is being parsed. This is used to determine
/// whether an error should be emitted for a missing right operand.
//  Eg: for an expression without parentheses (like i + j + k + l), each
//  of the four identifiers is an operand. For i + j*k + l, j*k is not an
//  operand expression, it's an op expression and will be parsed via
//  parseOpcodeHighPrecOpExpression(). However, for i + (j*k) + -l, (j*k) and
//  -l are valid operands that will be parsed by this function.
OpcodeExpr OpcodeParser::parseOpcodeOperandExpr(OpcodeExpr lhs) {
  switch (getToken().getKind()) {
  case Token::bare_identifier:
    return parseBareIdExpr();
  case Token::kw_symbol:
    return parseSymbolSSAIdExpr();
  case Token::percent_identifier:
    return parseSSAIdExpr(/*isSymbol=*/false);
  case Token::integer:
    return parseIntegerExpr();
  case Token::l_paren:
    return parseParentheticalExpr();
  case Token::minus:
    return parseNegateExpression(lhs);
  case Token::kw_ceildiv:
  case Token::kw_floordiv:
  case Token::kw_mod:
  case Token::plus:
  case Token::star:
    if (lhs)
      emitError("missing right operand of binary operator");
    else
      emitError("missing left operand of binary operator");
    return nullptr;
  default:
    if (lhs)
      emitError("missing right operand of binary operator");
    else
      emitError("expected opcode expression");
    return nullptr;
  }
}

/// Parse opcode expressions that are bare-id's, integer constants,
/// parenthetical opcode expressions, and opcode op expressions that are a
/// composition of those.
///
/// All binary op's associate from left to right.
///
/// {add, sub} have lower precedence than {mul, div, and mod}.
///
/// Add, sub'are themselves at the same precedence level. Mul, floordiv,
/// ceildiv, and mod are at the same higher precedence level. Negation has
/// higher precedence than any binary op.
///
/// llhs: the opcode expression appearing on the left of the one being parsed.
/// This function will return ((llhs llhsOp lhs) op rhs) if llhs is non null,
/// and lhs op rhs otherwise; if there is no rhs, llhs llhsOp lhs is returned
/// if llhs is non-null; otherwise lhs is returned. This is to deal with left
/// associativity.
///
/// Eg: when the expression is e1 + e2*e3 + e4, with e1 as llhs, this function
/// will return the opcode expr equivalent of (e1 + (e2*e3)) + e4, where
/// (e2*e3) will be parsed using parseOpcodeHighPrecOpExpr().
OpcodeExpr OpcodeParser::parseOpcodeLowPrecOpExpr(OpcodeExpr llhs,
                                                  OpcodeLowPrecOp llhsOp) {
  OpcodeExpr lhs;
  if (!(lhs = parseOpcodeOperandExpr(llhs)))
    return nullptr;

  // Found an LHS. Deal with the ops.
  if (OpcodeLowPrecOp lOp = consumeIfLowPrecOp()) {
    if (llhs) {
      OpcodeExpr sum = getOpcodeBinaryOpExpr(llhsOp, llhs, lhs);
      return parseOpcodeLowPrecOpExpr(sum, lOp);
    }
    // No LLHS, get RHS and form the expression.
    return parseOpcodeLowPrecOpExpr(lhs, lOp);
  }
  auto opLoc = getToken().getLoc();
  if (OpcodeHighPrecOp hOp = consumeIfHighPrecOp()) {
    // We have a higher precedence op here. Get the rhs operand for the llhs
    // through parseOpcodeHighPrecOpExpr.
    OpcodeExpr highRes = parseOpcodeHighPrecOpExpr(lhs, hOp, opLoc);
    if (!highRes)
      return nullptr;

    // If llhs is null, the product forms the first operand of the yet to be
    // found expression. If non-null, the op to associate with llhs is llhsOp.
    OpcodeExpr expr =
        llhs ? getOpcodeBinaryOpExpr(llhsOp, llhs, highRes) : highRes;

    // Recurse for subsequent low prec op's after the opcode high prec op
    // expression.
    if (OpcodeLowPrecOp nextOp = consumeIfLowPrecOp())
      return parseOpcodeLowPrecOpExpr(expr, nextOp);
    return expr;
  }
  // Last operand in the expression list.
  if (llhs)
    return getOpcodeBinaryOpExpr(llhsOp, llhs, lhs);
  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

/// Parse an opcode expression.
///  opcode-expr ::= `(` opcode-expr `)`
///                | `-` opcode-expr
///                | opcode-expr `+` opcode-expr
///                | opcode-expr `-` opcode-expr
///                | opcode-expr `*` opcode-expr
///                | opcode-expr `floordiv` opcode-expr
///                | opcode-expr `ceildiv` opcode-expr
///                | opcode-expr `mod` opcode-expr
///                | bare-id
///                | integer-literal
///
/// Additional conditions are checked depending on the production. For eg.,
/// one of the operands for `*` has to be either constant/symbolic; the second
/// operand for floordiv, ceildiv, and mod has to be a positive integer.
OpcodeExpr OpcodeParser::parseOpcodeExpr() {
  return parseOpcodeLowPrecOpExpr(nullptr, OpcodeLowPrecOp::LNoOp);
}

/// Parse a dim or symbol from the lists appearing before the actual
/// expressions of the opcode map. Update our state to store the
/// dimensional/symbolic identifier.
ParseResult OpcodeParser::parseIdentifierDefinition(OpcodeExpr idExpr) {
  if (getToken().isNot(Token::bare_identifier))
    return emitError("expected bare identifier");

  auto name = getTokenSpelling();
  for (auto entry : dimsAndSymbols) {
    if (entry.first == name)
      return emitError("redefinition of identifier '" + name + "'");
  }
  consumeToken(Token::bare_identifier);

  dimsAndSymbols.push_back({name, idExpr});
  return success();
}

/// Parse the list of dimensional identifiers to an opcode map.
ParseResult OpcodeParser::parseDimIdList(unsigned &numDims) {
  auto parseElt = [&]() -> ParseResult {
    auto dimension = getOpcodeDimExpr(numDims++, getContext());
    return parseIdentifierDefinition(dimension);
  };
  return parseCommaSeparatedList(Delimiter::Paren, parseElt,
                                 " in dimensional identifier list");
}

/// Parse the list of symbolic identifiers to an opcode map.
ParseResult OpcodeParser::parseSymbolIdList(unsigned &numSymbols) {
  auto parseElt = [&]() -> ParseResult {
    auto symbol = getOpcodeSymbolExpr(numSymbols++, getContext());
    return parseIdentifierDefinition(symbol);
  };
  return parseCommaSeparatedList(Delimiter::Square, parseElt,
                                 " in symbol list");
}

/// Parse the list of symbolic identifiers to an opcode map.
ParseResult
OpcodeParser::parseDimAndOptionalSymbolIdList(unsigned &numDims,
                                              unsigned &numSymbols) {
  if (parseDimIdList(numDims)) {
    return failure();
  }
  if (!getToken().is(Token::l_square)) {
    numSymbols = 0;
    return success();
  }
  return parseSymbolIdList(numSymbols);
}

/// Parse symbol representing the opcode
ParseResult OpcodeParser::parseSymbol() {
  auto symbol = getOpcodeSymbolExpr(0, getContext()); // only one symbol
  return parseIdentifierDefinition(symbol);
}

/// Parses a comma separated list of opcode entries
/// opcode_dict  ::= `opcode_map` `<` opcode-entry (`,` opcode-entry)* `>`
///
/// OpcodeMaps underlying data structure is a DictionaryAttr.
/// The keys are StringsAttr and the values are ArrayAttr.
///
/// Inside an ArrayAttr we maintain a list of opcode entries types and their
/// values.
ParseResult OpcodeParser::parseOpcodeMapInline(OpcodeMap &map) {

  // NamedAttrList will be transformed into the DictionaryAttr
  NamedAttrList elements;

  // TODO: Final condition for parsing success
  if (parseOpcodeDict(elements)) {
    // llvm::errs()<< "Failure on parseOpcodeMapInline()";
    return failure();
  }

  map = OpcodeMap::get(0, 0, {}, getContext());
  // TODO: Transform NamedAttrList into DictionaryAttr and use it to set the
  // OpcodeMap's underlying data structure.
  // map = OpcodeMap::get(elements.getDictionary(getContext()));

  // llvm::errs()<< "Success on parseOpcodeMapInline()";
  return success();
}

ParseResult
OpcodeParser::parseSendExpr(Token::Kind expectedToken,
                            function_ref<ParseResult()> parseElementFn) {
  consumeToken(expectedToken);
  if (parseToken(Token::l_paren, "expected '('"))
    return failure();
  if (getToken().is(Token::r_paren))
    return (emitError("no identifier inside parentheses"), failure());
  parseElementFn();
  if (parseToken(Token::r_paren, "expected ')'"))
    return failure();
  return success();
}

/// Attribute dictionary.
///
///   attribute-dict ::= `{` `}`
///                    | `{` attribute-entry (`,` attribute-entry)* `}`
///   attribute-entry ::= (bare-id | string-literal) `=` attribute-value
///
ParseResult OpcodeParser::parseOpcodeDict(NamedAttrList &attributes) {
  llvm::SmallDenseSet<StringAttr> seenKeys;

  auto parseKeyValue = [&]() -> ParseResult {
    // The name of an attribute can either be a bare identifier, or a string.
    Optional<StringAttr> nameId;
    if (getToken().is(Token::string))
      nameId = builder.getStringAttr(getToken().getStringValue());
    else if (getToken().isAny(Token::bare_identifier, Token::inttype) ||
             getToken().isKeyword())
      nameId = builder.getStringAttr(getTokenSpelling());
    else {
      return emitError("expected attribute name");
    }
    if (!seenKeys.insert(*nameId).second)
      return emitError("duplicate key '")
             << nameId->getValue() << "' in dictionary attribute";
    consumeToken();

    // Try to parse the '=' for the attribute value.
    if (!consumeIf(Token::equal)) {
      return emitError("expected '=' after '")
             << nameId->getValue() << "' entry in opcode_map'";
    }

    auto parseValue = [&]() -> ParseResult {
      // Function to consume tokens inside parenthesis
      auto fn = [&]() -> ParseResult { return consumeToken(), success(); };
      switch (getToken().getKind()) {
      case Token::kw_op_recv: {
        consumeToken(Token::kw_op_recv);
        if (parseToken(Token::l_paren, "expected '('"))
          return failure();
        if (getToken().is(Token::r_paren))
          return (emitError("no identifier inside parentheses"), failure());
        // TODO: resolve middle
        consumeToken();
        if (parseToken(Token::r_paren, "expected ')'"))
          return failure();
        return success();
      }
      case Token::kw_op_send: {
        return parseSendExpr(Token::kw_op_send, fn);
      }
      case Token::kw_op_send_literal: {
        return parseSendExpr(Token::kw_op_send_literal, fn);
      }
      case Token::kw_op_send_dim: {
        return parseSendExpr(Token::kw_op_send_dim, fn);
      }
      case Token::kw_op_send_idx: {
        return parseSendExpr(Token::kw_op_send_idx, fn);
      }
      default:
        emitError("Warning in parseValue'");
        return failure();
      }
      return success();
    };

    // Parse list of opcode expressions
    // opcode_list ::= `[` opcode_expr (`,` opcode_expr)* `]
    //
    // opcode_expr ::= send(bare-id)
    //               | send_dim(bare-id)
    //               | send_idx(bare-id)
    //               | recv(bare-id)

    if (parseCommaSeparatedList(Delimiter::Square, parseValue,
                                " in opcode_map dictionary"))
      return failure();

    // auto attr = parseAttribute();
    // if (!attr)
    //   return failure();
    // attributes.push_back({*nameId, attr});
    return success();
  };

  if (parseCommaSeparatedList(Delimiter::LessGreater, parseKeyValue,
                              " in opcode_map dictionary"))
    return failure();

  return success();
}

/// Parses an ambiguous opcode map or integer set definition inline.
ParseResult OpcodeParser::parseOpcodeMapOrIntegerSetInline(OpcodeMap &map,
                                                           IntegerSet &set) {
  unsigned numDims = 0, numSymbols = 0;

  // List of dimensional and optional symbol identifiers.
  if (parseDimAndOptionalSymbolIdList(numDims, numSymbols)) {
    return failure();
  }

  // This is needed for parsing attributes as we wouldn't know whether we would
  // be parsing an integer set attribute or an opcode map attribute.
  bool isArrow = getToken().is(Token::arrow);
  bool isColon = getToken().is(Token::colon);
  if (!isArrow && !isColon) {
    return emitError("expected '->' or ':'");
  }
  if (isArrow) {
    parseToken(Token::arrow, "expected '->' or '['");
    map = parseOpcodeMapRange(numDims, numSymbols);
    return map ? success() : failure();
  }
  if (parseToken(Token::colon, "expected ':' or '['"))
    return failure();

  // if ((set = parseIntegerSetConstraints(numDims, numSymbols)))
  //   return success();

  return failure();
}

/// Parse an OpcodeMap where the dim and symbol identifiers are SSA ids.
ParseResult
OpcodeParser::parseOpcodeMapOfSSAIds(OpcodeMap &map,
                                     OpAsmParser::Delimiter delimiter) {

  SmallVector<OpcodeExpr, 4> exprs;
  auto parseElt = [&]() -> ParseResult {
    auto elt = parseOpcodeExpr();
    exprs.push_back(elt);
    return elt ? success() : failure();
  };

  // Parse a multi-dimensional opcode expression (a comma-separated list of
  // 1-d opcode expressions); the list can be empty. Grammar:
  // multi-dim-opcode-expr ::= `(` `)`
  //                         | `(` opcode-expr (`,` opcode-expr)* `)`
  if (parseCommaSeparatedList(delimiter, parseElt, " in opcode map"))
    return failure();

  // Parsed a valid opcode map.
  map = OpcodeMap::get(numDimOperands, dimsAndSymbols.size() - numDimOperands,
                       exprs, getContext());
  return success();
}

/// Parse an OpcodeExpr where the dim and symbol identifiers are SSA ids.
ParseResult OpcodeParser::parseOpcodeExprOfSSAIds(OpcodeExpr &expr) {
  expr = parseOpcodeExpr();
  return success(expr != nullptr);
}

/// Parse the range and sizes opcode map definition inline.
///
///  opcode-map ::= dim-and-symbol-id-lists `->` multi-dim-opcode-expr
///
///  multi-dim-opcode-expr ::= `(` `)`
///  multi-dim-opcode-expr ::= `(` opcode-expr (`,` opcode-expr)* `)`
OpcodeMap OpcodeParser::parseOpcodeMapRange(unsigned numDims,
                                            unsigned numSymbols) {
  SmallVector<OpcodeExpr, 4> exprs;
  auto parseElt = [&]() -> ParseResult {
    auto elt = parseOpcodeExpr();
    ParseResult res = elt ? success() : failure();
    exprs.push_back(elt);
    return res;
  };

  // Parse a multi-dimensional opcode expression (a comma-separated list of
  // 1-d opcode expressions). Grammar:
  // multi-dim-opcode-expr ::= `(` `)`
  //                         | `(` opcode-expr (`,` opcode-expr)* `)`
  if (parseCommaSeparatedList(Delimiter::Paren, parseElt,
                              " in opcode map range"))
    return OpcodeMap();

  // Parsed a valid opcode map.
  return OpcodeMap::get(numDims, numSymbols, exprs, getContext());
}

/// Parse an opcode constraint.
///  opcode-constraint ::= opcode-expr `>=` `0`
///                      | opcode-expr `==` `0`
///
/// isEq is set to true if the parsed constraint is an equality, false if it
/// is an inequality (greater than or equal).
///
OpcodeExpr OpcodeParser::parseOpcodeConstraint(bool *isEq) {
  OpcodeExpr expr = parseOpcodeExpr();
  if (!expr)
    return nullptr;

  if (consumeIf(Token::greater) && consumeIf(Token::equal) &&
      getToken().is(Token::integer)) {
    auto dim = getToken().getUnsignedIntegerValue();
    if (dim.hasValue() && dim.getValue() == 0) {
      consumeToken(Token::integer);
      *isEq = false;
      return expr;
    }
    return (emitError("expected '0' after '>='"), nullptr);
  }

  if (consumeIf(Token::equal) && consumeIf(Token::equal) &&
      getToken().is(Token::integer)) {
    auto dim = getToken().getUnsignedIntegerValue();
    if (dim.hasValue() && dim.getValue() == 0) {
      consumeToken(Token::integer);
      *isEq = true;
      return expr;
    }
    return (emitError("expected '0' after '=='"), nullptr);
  }

  return (emitError("expected '== 0' or '>= 0' at end of opcode constraint"),
          nullptr);
}

/// Parse the constraints that are part of an integer set definition.
///  integer-set-inline
///                ::= dim-and-symbol-id-lists `:`
///                '(' opcode-constraint-conjunction? ')'
///  opcode-constraint-conjunction ::= opcode-constraint (`,`
///                                       opcode-constraint)*
///
// IntegerSet OpcodeParser::parseIntegerSetConstraints(unsigned numDims,
//                                                     unsigned numSymbols) {
//   SmallVector<OpcodeExpr, 4> constraints;
//   SmallVector<bool, 4> isEqs;
//   auto parseElt = [&]() -> ParseResult {
//     bool isEq;
//     auto elt = parseOpcodeConstraint(&isEq);
//     ParseResult res = elt ? success() : failure();
//     if (elt) {
//       constraints.push_back(elt);
//       isEqs.push_back(isEq);
//     }
//     return res;
//   };

//   // Parse a list of opcode constraints (comma-separated).
//   if (parseCommaSeparatedList(Delimiter::Paren, parseElt,
//                               " in integer set constraint list"))
//     return IntegerSet();

//   // If no constraints were parsed, then treat this as a degenerate 'true'
//   case. if (constraints.empty()) {
//     /* 0 == 0 */
//     auto zero = getOpcodeConstantExpr(0, getContext());
//     return IntegerSet::get(numDims, numSymbols, zero, true);
//   }

//   // Parsed a valid integer set.
//   return IntegerSet::get(numDims, numSymbols, constraints, isEqs);
// }

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// Parse an ambiguous reference to either and opcode map or an integer set.
/// TODO: Remove double function call
ParseResult Parser::parseOpcodeMapReference(OpcodeMap &map) {

  if (OpcodeParser(state).parseOpcodeMapInline(map)) {
    // llvm::errs()<< "Failed parseOpcodeMapReference()";
    return failure();
  }

  // TODO: Remove after certifying that errors are capture above.
  // if (!map)
  //   return emitError(getToken().getLoc(), "Something went wrong with opcode
  //   map parsing.");
  // llvm::errs()<< "Success on parseOpcodeMapReference()";
  return success();
}

/// Parse an OpcodeMap of SSA ids. The callback 'parseElement' is used to
/// parse SSA value uses encountered while parsing opcode expressions.
ParseResult
Parser::parseOpcodeMapOfSSAIds(OpcodeMap &map,
                               function_ref<ParseResult(bool)> parseElement,
                               OpAsmParser::Delimiter delimiter) {
  return OpcodeParser(state, /*allowParsingSSAIds=*/true, parseElement)
      .parseOpcodeMapOfSSAIds(map, delimiter);
}

/// Parse an OpcodeExpr of SSA ids. The callback `parseElement` is used to parse
/// SSA value uses encountered while parsing.
ParseResult
Parser::parseOpcodeExprOfSSAIds(OpcodeExpr &expr,
                                function_ref<ParseResult(bool)> parseElement) {
  return OpcodeParser(state, /*allowParsingSSAIds=*/true, parseElement)
      .parseOpcodeExprOfSSAIds(expr);
}

// IntegerSet mlir::parseIntegerSet(StringRef inputStr, MLIRContext *context,
//                                  bool printDiagnosticInfo) {
//   llvm::SourceMgr sourceMgr;
//   auto memBuffer = llvm::MemoryBuffer::getMemBuffer(
//       inputStr, /*BufferName=*/"<mlir_parser_buffer>",
//       /*RequiresNullTerminator=*/false);
//   sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
//   SymbolState symbolState;
//   ParserState state(sourceMgr, context, symbolState, /*asmState=*/nullptr);
//   Parser parser(state);

//   raw_ostream &os = printDiagnosticInfo ? llvm::errs() : llvm::nulls();
//   SourceMgrDiagnosticHandler handler(sourceMgr, context, os);
//   IntegerSet set;
//   if (parser.parseIntegerSetReference(set))
//     return IntegerSet();

//   Token endTok = parser.getToken();
//   if (endTok.isNot(Token::eof)) {
//     parser.emitError(endTok.getLoc(), "encountered unexpected token");
//     return IntegerSet();
//   }

//   return set;
// }
