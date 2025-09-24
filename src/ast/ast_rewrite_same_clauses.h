/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "ast.h"

// rewrite result by compressing consecutive clauses of the same type
// to a single clause, returning true if the rewrite has been performed
bool AST_RewriteSameClauses
(
	const cypher_astnode_t *root // root of AST
);

