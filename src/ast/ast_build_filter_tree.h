/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "ast.h"
#include "../graph/query_graph.h"
#include "../filter_tree/filter_tree.h"

/* Convert the given AST predicate node and its children into a filter tree. */
void AST_ConvertFilters(FT_FilterNode **root, const cypher_astnode_t *entity);

/* Collect all appropriate predicate nodes from the given AST
 * and convert them into a filter tree. */
FT_FilterNode *AST_BuildFilterTree(AST *ast);

/* Collect all appropriate predicate nodes from the set of clauses
 * (all of the given type) and convert them into a filter tree. */
FT_FilterNode *AST_BuildFilterTreeFromClauses(const AST *ast,
		const cypher_astnode_t **clauses, uint count);

