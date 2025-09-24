/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "ast.h"

// Build a temporary AST with one MATCH clause that holds the given path or pattern.
AST *AST_MockMatchClause(AST *master_ast, cypher_astnode_t *node, bool node_is_pattern);

// Free a temporary AST.
void AST_MockFree(AST *ast, bool free_pattern);

