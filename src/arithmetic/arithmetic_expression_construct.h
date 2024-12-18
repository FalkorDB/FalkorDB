/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../ast/ast.h"
#include "arithmetic_expression.h"

// construct arithmetic expression from AST node
AR_ExpNode *AR_EXP_FromASTNode
(
	const cypher_astnode_t *expr
);

