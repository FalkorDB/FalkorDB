/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "ast.h"

// rewrite WITH/RETURN * clauses in query to project explicit identifiers,
// returning true if a rewrite has been performed
bool AST_RewriteStarProjections
(
    const cypher_astnode_t *root  // root for which to rewrite star projections
);

// rewrite result by compressing consecutive clauses of the same type
// to a single clause, returning true if the rewrite has been performed
bool AST_RewriteSameClauses
(
	const cypher_astnode_t *root // root of AST
);

// if the subquery will result in an eager and returning execution-plan
// rewrites it to contain the projections needed:
// 1. "n"  -> "@n" in the initial WITH clause if exists. Otherwise, creates it.
// 2. "@n" -> "@n" in the intermediate WITH clauses.
// 3. "@n" -> "n" in the final RETURN clause.
// if the subquery will not result in an eager & returning execution-plan, does
// nothing
bool AST_RewriteCallSubquery
(
	const cypher_astnode_t *root // root of AST
);

// AST_RewriteFilters rewrites filters which tries to access a variable which is
// aliased but is no longer available
//
// e.g.
//
// WITH a AS b WHERE a.v = 1
// ->
// WITH a AS b WHERE b.v = 1
//
// from user perspective 'a' still exists, but internally only 'b' is available
//
// in case such as:
// WITH a AS b, c AS a WHERE a.v = 1
// we do not perform the rewrite as 'a' is defined and acceessible
bool AST_RewriteFilters
(
	const cypher_astnode_t *root // root of AST
);

