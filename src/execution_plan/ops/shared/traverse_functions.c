/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */
#include "traverse_functions.h"
#include "../../../query_ctx.h"

// collect edges between the source and destination nodes
static void _Traverse_CollectEdges
(
	EdgeTraverseCtx *edge_ctx,
	NodeID src,
	NodeID dest
) {
	Graph *g = QueryCtx_GetGraph();
	uint count = array_len(edge_ctx->edgeRelationTypes);
	for(uint i = 0; i < count; i++) {
		Graph_GetEdgesConnectingNodes(g,
									  src,
									  dest,
									  edge_ctx->edgeRelationTypes[i],
									  &edge_ctx->edges);
	}
}

// collects traversed edge relations.
// e.g. [e:R0|R1]
// edge_ctx->edgeRelationTypes will hold both R0 and R1 IDs.
// in the case where no relationship types are specified
// edge_ctx->edgeRelationTypes will contain GRAPH_NO_RELATION
static void _Traverse_SetRelationTypes
(
	EdgeTraverseCtx *edge_ctx,
	QGEdge *e
) {
	uint reltype_count = array_len(e->reltypeIDs);
	if(reltype_count > 0) {
		array_clone(edge_ctx->edgeRelationTypes, e->reltypeIDs);
	} else {
		edge_ctx->edgeRelationTypes = array_new(int, 1);
		array_append(edge_ctx->edgeRelationTypes, GRAPH_NO_RELATION);
	}
}

// determine the edge directions we need to collect
static GRAPH_EDGE_DIR _Traverse_SetDirection
(
	AlgebraicExpression *ae,
	const QGEdge *e
) {
	// the default traversal direction is outgoing
	GRAPH_EDGE_DIR dir = GRAPH_EDGE_DIR_OUTGOING;

	// bidirectional traversals should match both incoming and outgoing edges
	if(e->bidirectional) return GRAPH_EDGE_DIR_BOTH;

	/* if this operation traverses a transposed edge, the source and destination
	 * nodes will be swapped in the Record */

	// push down transpose operations to individual operands
	AlgebraicExpression_PushDownTranspose(ae);
	AlgebraicExpression *parent = NULL;
	AlgebraicExpression *operand = NULL;

	// locate operand representing the referenced edge
	bool located = AlgebraicExpression_LocateOperand(ae, &operand, &parent,
			e->src->alias, e->dest->alias, e->alias, NULL);
	ASSERT(located == true);

	// if parent exists and it is a transpose operation, edge is reversed
	if(parent != NULL) {
		ASSERT(parent->type == AL_OPERATION);
		if(parent->operation.op == AL_EXP_TRANSPOSE) {
			dir = GRAPH_EDGE_DIR_INCOMING;
		}
	}

	return dir;
}

EdgeTraverseCtx *EdgeTraverseCtx_New
(
	AlgebraicExpression *ae,
	QGEdge *e,
	int idx
) {
	ASSERT(e != NULL);
	ASSERT(ae != NULL);

	EdgeTraverseCtx *edge_ctx = rm_malloc(sizeof(EdgeTraverseCtx));
	edge_ctx->edges = array_new(Edge, 32); // Instantiate array to collect matching edges.
	_Traverse_SetRelationTypes(edge_ctx, e); // Build the array of relation type IDs.
	edge_ctx->edgeRecIdx = idx;
	edge_ctx->direction = _Traverse_SetDirection(ae, e);
	return edge_ctx;
}

// populate the traverse context's edges array with all edges of the appropriate
// direction connecting the source and destination nodes
void EdgeTraverseCtx_CollectEdges
(
	EdgeTraverseCtx *edge_ctx,
	NodeID src,
	NodeID dest
) {
	ASSERT(edge_ctx != NULL);

	GRAPH_EDGE_DIR dir = src == dest ? GRAPH_EDGE_DIR_OUTGOING : edge_ctx->direction;
	switch(dir) {
		case GRAPH_EDGE_DIR_OUTGOING:
			_Traverse_CollectEdges(edge_ctx, src, dest);
			return;
		case GRAPH_EDGE_DIR_INCOMING:
			// If we're traversing incoming edges, swap the source and destination.
			_Traverse_CollectEdges(edge_ctx, dest, src);
			return;
		case GRAPH_EDGE_DIR_BOTH:
			// If we're traversing in both directions, collect edges in both directions.
			_Traverse_CollectEdges(edge_ctx, src, dest);
			_Traverse_CollectEdges(edge_ctx, dest, src);
			return;
	}
}

bool EdgeTraverseCtx_SetEdge
(
	EdgeTraverseCtx *edge_ctx,
	Record r
) {
	ASSERT(r != NULL);
	ASSERT(edge_ctx != NULL);

	// return false if all edges have been consumed
	if(array_len(edge_ctx->edges) == 0) return false;

	// pop an edge and add it to the Record
	Edge e = array_pop(edge_ctx->edges);
	Record_AddEdge(r, edge_ctx->edgeRecIdx, e);

	return true;
}

int EdgeTraverseCtx_EdgeCount
(
	const EdgeTraverseCtx *edge_ctx
) {
	ASSERT(edge_ctx != NULL);
	return array_len(edge_ctx->edges);
}

void EdgeTraverseCtx_Reset
(
	EdgeTraverseCtx *edge_ctx
) {
	ASSERT(edge_ctx != NULL);

	array_clear(edge_ctx->edges);
}

void EdgeTraverseCtx_Free
(
	EdgeTraverseCtx *edge_ctx
) {
	if(!edge_ctx) return;

	array_free(edge_ctx->edges);
	array_free(edge_ctx->edgeRelationTypes);
	rm_free(edge_ctx);
}

