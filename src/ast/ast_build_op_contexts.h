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
#include "ast_shared.h"
#include "../graph/query_graph.h"
#include "../graph/graphcontext.h"
#include "../graph/entities/graph_entity.h"

#define DIR_DESC -1
#define DIR_ASC 1

// Unwind operations hold a collection of arithmetic expressions.
typedef struct {
	AR_ExpNode *exp;
} AST_UnwindContext;

typedef struct {
	rax *on_match;                   // rax of updates to make for ON MATCH directives
	rax *on_create;                  // rax of updates to make for ON CREATE directives
	NodeCreateCtx *nodes_to_merge;   // blueprints of nodes in MERGE pattern
	EdgeCreateCtx *edges_to_merge;   // blueprints of edges in MERGE pattern
} AST_MergeContext;

typedef struct {
	NodeCreateCtx *nodes_to_create;
	EdgeCreateCtx *edges_to_create;
} AST_CreateContext;

// convert the properties specified in a node or edge entity
// into a set of string keys and SIValue values
PropertyMap *AST_ConvertPropertiesMap
(
	const cypher_astnode_t *props
);

// extract the necessary information to populate an update operation from a SET clause
rax *AST_PrepareUpdateOp
(
	GraphContext *gc,
	const cypher_astnode_t *set_clause
);

// extract the necessary information to populate a delete operation from a DELETE clause
AR_ExpNode **AST_PrepareDeleteOp
(
	const cypher_astnode_t *delete_clause
);

// determine sort directions (ascending / descending) of multiple sort operations
void AST_PrepareSortOp
(
	const cypher_astnode_t *order_clause,
	int **sort_directions
);

// extract the necessary information to populate a unwind operation from an UNWIND clause
AST_UnwindContext AST_PrepareUnwindOp
(
	const cypher_astnode_t *unwind_clause
);

void AST_PreparePathCreation
(
	const cypher_astnode_t *path,
	const QueryGraph *qg,
	rax *bound_vars,
	NodeCreateCtx **nodes,
	EdgeCreateCtx **edges
);

// extract the necessary information to populate a merge operation from a MERGE clause
AST_MergeContext AST_PrepareMergeOp
(
	const cypher_astnode_t *merge_clause,
	GraphContext *gc,
	QueryGraph *qg,
	rax *bound_vars
);

// extract the necessary information to populate a create operation from all CREATE clauses
AST_CreateContext AST_PrepareCreateOp
(
	QueryGraph *qg,
	rax *bound_vars,
	const cypher_astnode_t *clause
);
