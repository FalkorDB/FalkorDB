/* vi:set ts=4 sw=4 expandtab:
 *
 * Copyright 2016, Chris Leishman (http://github.com/cleishm)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../../config.h"
#include "operators.h"


// NOTE: higher precedences bind closer

// `l OR r`
static const cypher_operator_t _CYPHER_OP_OR =
    { .precedence = 1, .associativity = LEFT_ASSOC, .str = "OR" };
const cypher_operator_t *CYPHER_OP_OR = &_CYPHER_OP_OR;

// `l XOR r`
static const cypher_operator_t _CYPHER_OP_XOR =
    { .precedence = 2, .associativity = LEFT_ASSOC, .str = "XOR" };
const cypher_operator_t *CYPHER_OP_XOR = &_CYPHER_OP_XOR;

// `l AND r`
static const cypher_operator_t _CYPHER_OP_AND =
    { .precedence = 3, .associativity = LEFT_ASSOC, .str = "AND" };
const cypher_operator_t *CYPHER_OP_AND = &_CYPHER_OP_AND;

// `NOT r`
static const cypher_operator_t _CYPHER_OP_NOT =
    { .precedence = 4, .associativity = UNARY, .str = "NOT" };
const cypher_operator_t *CYPHER_OP_NOT = &_CYPHER_OP_NOT;

// `l = r`
static const cypher_operator_t _CYPHER_OP_EQUAL =
    { .precedence = 5, .associativity = LEFT_ASSOC, .str = "=" };
const cypher_operator_t *CYPHER_OP_EQUAL = &_CYPHER_OP_EQUAL;
// `l <> r`
static const cypher_operator_t _CYPHER_OP_NEQUAL =
    { .precedence = 5, .associativity = LEFT_ASSOC, .str = "<>" };
const cypher_operator_t *CYPHER_OP_NEQUAL = &_CYPHER_OP_NEQUAL;

// `l < r`
static const cypher_operator_t _CYPHER_OP_LT =
    { .precedence = 6, .associativity = LEFT_ASSOC, .str = "<" };
const cypher_operator_t *CYPHER_OP_LT = &_CYPHER_OP_LT;
// `l > r`
static const cypher_operator_t _CYPHER_OP_GT =
    { .precedence = 6, .associativity = LEFT_ASSOC, .str = ">" };
const cypher_operator_t *CYPHER_OP_GT = &_CYPHER_OP_GT;
// `l <= r`
static const cypher_operator_t _CYPHER_OP_LTE =
    { .precedence = 6, .associativity = LEFT_ASSOC, .str = "<=" };
const cypher_operator_t *CYPHER_OP_LTE = &_CYPHER_OP_LTE;
// `l >= r`
static const cypher_operator_t _CYPHER_OP_GTE =
    { .precedence = 6, .associativity = LEFT_ASSOC, .str = ">=" };
const cypher_operator_t *CYPHER_OP_GTE = &_CYPHER_OP_GTE;

// `l + r`
static const cypher_operator_t _CYPHER_OP_PLUS =
    { .precedence = 7, .associativity = LEFT_ASSOC, .str = "+" };
const cypher_operator_t *CYPHER_OP_PLUS = &_CYPHER_OP_PLUS;
// `l - r`
static const cypher_operator_t _CYPHER_OP_MINUS =
    { .precedence = 7, .associativity = LEFT_ASSOC, .str = "-" };
const cypher_operator_t *CYPHER_OP_MINUS = &_CYPHER_OP_MINUS;

// `l * r`
static const cypher_operator_t _CYPHER_OP_MULT =
    { .precedence = 8, .associativity = LEFT_ASSOC, .str = "*" };
const cypher_operator_t *CYPHER_OP_MULT = &_CYPHER_OP_MULT;
// `l / r`
static const cypher_operator_t _CYPHER_OP_DIV =
    { .precedence = 8, .associativity = LEFT_ASSOC, .str = "/" };
const cypher_operator_t *CYPHER_OP_DIV = &_CYPHER_OP_DIV;
// `l % r`
static const cypher_operator_t _CYPHER_OP_MOD =
    { .precedence = 8, .associativity = LEFT_ASSOC, .str = "%" };
const cypher_operator_t *CYPHER_OP_MOD = &_CYPHER_OP_MOD;

// `l ^ r`
static const cypher_operator_t _CYPHER_OP_POW =
    { .precedence = 9, .associativity = RIGHT_ASSOC, .str = "^" };
const cypher_operator_t *CYPHER_OP_POW = &_CYPHER_OP_POW;

// `+r`
static const cypher_operator_t _CYPHER_OP_UNARY_PLUS =
    { .precedence = 10, .associativity = UNARY, .str = "+" };
const cypher_operator_t *CYPHER_OP_UNARY_PLUS = &_CYPHER_OP_UNARY_PLUS;
// `-r`
static const cypher_operator_t _CYPHER_OP_UNARY_MINUS =
    { .precedence = 10, .associativity = UNARY, .str = "-" };
const cypher_operator_t *CYPHER_OP_UNARY_MINUS = &_CYPHER_OP_UNARY_MINUS;

// `l =~ r`
static const cypher_operator_t _CYPHER_OP_REGEX =
    { .precedence = 11, .associativity = LEFT_ASSOC, .str = "=~" };
const cypher_operator_t *CYPHER_OP_REGEX = &_CYPHER_OP_REGEX;
// `l IN r`
static const cypher_operator_t _CYPHER_OP_IN =
    { .precedence = 11, .associativity = LEFT_ASSOC, .str = "IN" };
const cypher_operator_t *CYPHER_OP_IN = &_CYPHER_OP_IN;
// `l STARTS WITH r`
static const cypher_operator_t _CYPHER_OP_STARTS_WITH =
    { .precedence = 11, .associativity = LEFT_ASSOC, .str = "STARTS WITH" };
const cypher_operator_t *CYPHER_OP_STARTS_WITH = &_CYPHER_OP_STARTS_WITH;
// `l ENDS WITH r`
static const cypher_operator_t _CYPHER_OP_ENDS_WITH =
    { .precedence = 11, .associativity = LEFT_ASSOC, .str = "ENDS WITH" };
const cypher_operator_t *CYPHER_OP_ENDS_WITH = &_CYPHER_OP_ENDS_WITH;
// `l CONTAINS r`
static const cypher_operator_t _CYPHER_OP_CONTAINS =
    { .precedence = 11, .associativity = LEFT_ASSOC, .str = "CONTAINS" };
const cypher_operator_t *CYPHER_OP_CONTAINS = &_CYPHER_OP_CONTAINS;
// `l IS NULL`
static const cypher_operator_t _CYPHER_OP_IS_NULL =
    { .precedence = 11, .associativity = UNARY, .str = "IS NULL" };
const cypher_operator_t *CYPHER_OP_IS_NULL = &_CYPHER_OP_IS_NULL;
// `l IS NOT NULL`
static const cypher_operator_t _CYPHER_OP_IS_NOT_NULL =
    { .precedence = 11, .associativity = UNARY, .str = "IS NOT NULL" };
const cypher_operator_t *CYPHER_OP_IS_NOT_NULL = &_CYPHER_OP_IS_NOT_NULL;

// `l{e}`
static const cypher_operator_t _CYPHER_OP_MAP_PROJECTION =
    { .precedence = 12, .associativity = LEFT_ASSOC, .str = "{" };
const cypher_operator_t *CYPHER_OP_MAP_PROJECTION = &_CYPHER_OP_MAP_PROJECTION;
// `l[r]`, `l[s..e]`
static const cypher_operator_t _CYPHER_OP_SUBSCRIPT =
    { .precedence = 12, .associativity = LEFT_ASSOC, .str = "[" };
const cypher_operator_t *CYPHER_OP_SUBSCRIPT = &_CYPHER_OP_SUBSCRIPT;

// `l.r`
static const cypher_operator_t _CYPHER_OP_PROPERTY =
    { .precedence = 13, .associativity = LEFT_ASSOC, .str = "." };
const cypher_operator_t *CYPHER_OP_PROPERTY = &_CYPHER_OP_PROPERTY;
// `l:r`
static const cypher_operator_t _CYPHER_OP_LABEL =
    { .precedence = 13, .associativity = LEFT_ASSOC, .str = ":" };
const cypher_operator_t *CYPHER_OP_LABEL = &_CYPHER_OP_LABEL;
