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
  OpcodeParser(ParserState &state,
               function_ref<ParseResult(bool)> parseElement = nullptr)
      : Parser(state), parseElement(parseElement), numOpcodes(0) {}

  /// opcode_dict  ::= `opcode_map` `<` opcode-entry (`,` opcode-entry)* `>`
  ///
  /// opcode_entry ::= (bare-id | string-literal) `=` opcode_list
  ///
  /// opcode_list  ::= `[` opcode_expr (`,` opcode_expr)* `]
  ///
  /// opcode_expr  ::= op_send(bare-id)
  /// .              | op_send_literal(integer-literal)
  ///                | op_send_dim(bare-id)
  ///                | op_send_idx(bare-id)
  ///                | op_recv(bare-id)
  ParseResult parseOpcodeMapInline(OpcodeMap &map);
  ParseResult parseOpcodeDict();

private:
  ParseResult parseSymbol();

  ParseResult parseIdentifierDefinition(OpcodeExpr idExpr);

  // OpcodeExpr parseOpcodeExpr(); // TODO: Future
  ParseResult parseSendExpr(Token::Kind expectedToken,
                            function_ref<ParseResult()> parseElementFn);
  ParseResult parseRecvExpr(Token::Kind expectedToken,
                            function_ref<ParseResult()> parseElementFn);
  // OpcodeExpr parseParentheticalExpr();
  // OpcodeExpr parseIntegerExpr();
  // OpcodeExpr parseBareIdExpr();

private:
  function_ref<ParseResult(bool)> parseElement;
  unsigned numOpcodes;
  SmallVector<std::tuple<StringRef, OpcodeList>, 4> opcodesAndExprs;
};
} // namespace

/// TODO: Future: Parse an opcode expression inside parentheses.
///
///   opcode-expr ::= `(` opcode-expr `)`
// OpcodeExpr OpcodeParser::parseParentheticalExpr() {
//   if (parseToken(Token::l_paren, "expected '('"))
//     return nullptr;
//   if (getToken().is(Token::r_paren))
//     return (emitError("no expression inside parentheses"), nullptr);

//   auto expr = parseOpcodeExpr();
//   if (!expr)
//     return nullptr;
//   if (parseToken(Token::r_paren, "expected ')'"))
//     return nullptr;

//   return expr;
// }

/// Parses a comma separated list of opcode entries and updates map.
/// opcode_dict  ::= `opcode_map` `<` opcode-entry (`,` opcode-entry)* `>`
///
/// OpcodeMap's underlying data structure is a
///    ArrayRef<std::tuple<StringRef,ArrayRef<OpcodeExpr>>>.
/// The keys are StringsRef and the values are ArrayAttr of opcodeExpr.
///
/// Inside an ArrayAttr we maintain a list of opcode expr types and their
/// values.
ParseResult OpcodeParser::parseOpcodeMapInline(OpcodeMap &map) {

  if (parseOpcodeDict()) {
    return failure();
  }

  map = OpcodeMap::get(numOpcodes, opcodesAndExprs, getContext());
  return success();
}

ParseResult
OpcodeParser::parseSendExpr(Token::Kind expectedToken,
                            function_ref<ParseResult()> parseElementFn) {
  consumeToken(expectedToken);
  if (parseToken(Token::l_paren, "expected '('"))
    return failure();
  if (getToken().is(Token::r_paren))
    return (emitError("no value inside parentheses"), failure());
  parseElementFn();
  if (parseToken(Token::r_paren, "expected ')'"))
    return failure();
  return success();
}

ParseResult
OpcodeParser::parseRecvExpr(Token::Kind expectedToken,
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
ParseResult OpcodeParser::parseOpcodeDict() {
  llvm::SmallDenseSet<StringAttr> seenKeys;

  // Parse one key-value pair.
  auto parseKeyValue = [&]() -> ParseResult {
    // Holds the list of expressions for the attribute value if successful.
    SmallVector<OpcodeExpr, 8> exprs;
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
        auto fn = [&]() -> ParseResult {
          if (getToken().is(Token::integer)) {
            Attribute attr = parseAttribute();
            unsigned id = attr.dyn_cast<IntegerAttr>().getInt();
            exprs.push_back(getOpcodeRecvExpr(id, getContext()));
          } else {
            return emitError("expected unsigned integer literal");
          }
          return success();
        };
        return parseRecvExpr(Token::kw_op_recv, fn);
      }
      case Token::kw_op_send: {
        auto fn = [&]() -> ParseResult {
          if (getToken().is(Token::integer)) {
            Attribute attr = parseAttribute();
            unsigned id = attr.dyn_cast<IntegerAttr>().getInt();
            exprs.push_back(getOpcodeSendExpr(id, getContext()));
          } else {
            return emitError("expected unsigned integer literal");
          }
          return success();
        };
        return parseSendExpr(Token::kw_op_send, fn);
      }
      case Token::kw_op_send_literal: {
        auto fn = [&]() -> ParseResult {
          if (getToken().is(Token::integer)) {
            Attribute attr = parseAttribute();
            int value = attr.dyn_cast<IntegerAttr>().getInt();
            exprs.push_back(getOpcodeSendLiteralExpr(value, getContext()));
          } else {
            return emitError("expected integer literal");
          }
          return success();
        };
        return parseSendExpr(Token::kw_op_send_literal, fn);
      }
      case Token::kw_op_send_dim: {
        auto fn = [&]() -> ParseResult {
          if (getToken().is(Token::integer)) {
            Attribute attr = parseAttribute();
            unsigned id = attr.dyn_cast<IntegerAttr>().getInt();
            // TODO: parse position here.
            exprs.push_back(
                getOpcodeSendDimExpr(id, /*TODO:pos*/ 0, getContext()));
          } else {
            return emitError("expected unsigned integer literal");
          }
          return success();
        };
        return parseSendExpr(Token::kw_op_send_dim, fn);
      }
      case Token::kw_op_send_idx: {
        auto fn = [&]() -> ParseResult {
          if (getToken().is(Token::integer)) {
            Attribute attr = parseAttribute();
            unsigned id = attr.dyn_cast<IntegerAttr>().getInt();
            // TODO: parse position here.
            exprs.push_back(
                getOpcodeSendIdxExpr(id, /*TODO:pos*/ 0, getContext()));
          } else {
            return emitError("expected unsigned integer literal");
          }
          return success();
        };
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

    // At this point we have 1 key and 1 value parsed, saved them.
    opcodesAndExprs.push_back(std::tuple<StringRef, OpcodeList>(
        *nameId, OpcodeList::get(exprs.size(), exprs, getContext())));
    numOpcodes++;

    return success();
  };

  // Parse all key-value pairs.
  // if success, opcodesAndExprs will be populated with the parsed k,v pairs
  if (parseCommaSeparatedList(Delimiter::LessGreater, parseKeyValue,
                              " in opcode_map dictionary"))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// Parse an reference to a opcode map .
/// TODO: Remove double function call
ParseResult Parser::parseOpcodeMapReference(OpcodeMap &map) {

  if (OpcodeParser(state).parseOpcodeMapInline(map)) {
    // llvm::errs()<< "Failed parseOpcodeMapReference()";
    return failure();
  }

  return success();
}
