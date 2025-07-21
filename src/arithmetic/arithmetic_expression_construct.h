/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "ast.h"
#include "arithmetic_expression.h"

// Construct arithmetic expression from AST node
AR_ExpNode *AR_EXP_FromASTNode(const cypher_astnode_t *expr);

