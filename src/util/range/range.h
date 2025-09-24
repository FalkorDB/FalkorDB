/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../ast/ast_shared.h"

// range expression
// composed of an operator and an expression which supposed to be
// evaluated to a numeric scalar
typedef struct {
	AST_Operator op;   // <, <=, =, >=, >
	AR_ExpNode  *exp;  // expression
} RangeExpression;

// clone a range expression
RangeExpression RangeExpression_Clone
(
	RangeExpression exp
);

void RangeExpression_Free
(
	RangeExpression *exp
);

#include "bitmap_range.h"
#include "string_range.h"
#include "numeric_range.h"
