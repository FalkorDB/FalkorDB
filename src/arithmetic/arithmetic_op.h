/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "../ast/ast_shared.h"


/* Reverse an inequality symbol so that optimizations can support
 * inequalities with right-hand variables. */
AST_Operator ArithmeticOp_ReverseOp(AST_Operator op);
