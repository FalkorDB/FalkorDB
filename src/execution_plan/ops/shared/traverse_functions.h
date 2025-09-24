/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../execution_plan.h"
#include "../../../arithmetic/algebraic_expression.h"

// container struct for traversing and populating referenced edges in
// traversal ops like CondTraverse and ExpandInto
typedef struct {
	int *edgeRelationTypes;     // the relation type ids that should be collected
	Edge *edges;                // flexible array of all matching edges for the current endpoints
	int edgeRecIdx;             // the record index for the referenced edge
	GRAPH_EDGE_DIR direction;   // the direction of the referenced edge being traversed
} EdgeTraverseCtx;

// initialize an EdgeTraverseCtx struct to populate edges appropriately
// for traversal operations
EdgeTraverseCtx *EdgeTraverseCtx_New
(
	AlgebraicExpression *ae,
	QGEdge *e,
	int idx
);

// collect all appropriate edges between the given endpoints
void EdgeTraverseCtx_CollectEdges
(
	EdgeTraverseCtx *edge_ctx,
	NodeID src,
	NodeID desti
);

// remove a matching edge from the edges array if one is available
// and set it in the record
bool EdgeTraverseCtx_SetEdge
(
	EdgeTraverseCtx *edge_ctx,
	Record r
);

// return number of edges in context
int EdgeTraverseCtx_EdgeCount
(
	const EdgeTraverseCtx *edge_ctx
);

// reset contained edges within an EdgeTraverseCtx
void EdgeTraverseCtx_Reset
(
	EdgeTraverseCtx *edge_ctx
);

// free an EdgeTraverseCtx struct
void EdgeTraverseCtx_Free
(
	EdgeTraverseCtx *edge_ctx
);

