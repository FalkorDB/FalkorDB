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
	Graph *g = QueryCtx_GetGraph () ;
	uint count = QGEdge_RelationCount (edge_ctx->e) ;
	if (count == 0) {
		Graph_GetEdgesConnectingNodes (g, src, dest, GRAPH_NO_RELATION,
				&edge_ctx->edges) ;
	} else {
		for (uint i = 0; i < count; i++) {
			RelationID rel_id = QGEdge_RelationID (edge_ctx->e, i) ;
			Graph_GetEdgesConnectingNodes (g, src, dest, rel_id,
					&edge_ctx->edges) ;
		}
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
	ASSERT (e  != NULL) ;
	ASSERT (ae != NULL) ;

	EdgeTraverseCtx *edge_ctx = rm_malloc (sizeof (EdgeTraverseCtx)) ;

	// own a deep clone of the QGEdge to decouple our lifetime from the
	// plan's QueryGraph (issue #1823)
	// the source plan's QGEdge can be freed by LRU cache eviction while a
	// cloned ExecutionPlan that borrows the same pointer is still executing
	// on a worker thread
	edge_ctx->e = QGEdge_Clone (e) ;
	edge_ctx->edges = arr_new (Edge, 32) ;   // instantiate array to collect matching edges

	edge_ctx->edgeRecIdx = idx ;
	edge_ctx->direction = _Traverse_SetDirection (ae, e) ;

	return edge_ctx ;
}

// returns the number of relationship types used in this context
uint EdgeTraverseCtx_RelationCount
(
	const EdgeTraverseCtx *edge_ctx  // edge traverse context
) {
	ASSERT (edge_ctx    != NULL) ;
	ASSERT (edge_ctx->e != NULL) ;

	return QGEdge_RelationCount (edge_ctx->e) ;
}

// get the ith relationship type used in this context
RelationID EdgeTraverseCtx_GetRelationIdx
(
	const EdgeTraverseCtx *edge_ctx,  // edge traverse context
	uint idx                          // edge ith rel type
) {
	ASSERT (edge_ctx    != NULL) ;
	ASSERT (edge_ctx->e != NULL) ;
	ASSERT (QGEdge_RelationCount (edge_ctx->e) > idx) ;

	return QGEdge_RelationID (edge_ctx->e, idx) ;
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
	if(arr_len(edge_ctx->edges) == 0) return false;

	// pop an edge and add it to the Record
	Edge e = arr_pop(edge_ctx->edges);
	Record_AddEdge(r, edge_ctx->edgeRecIdx, e);

	return true;
}

int EdgeTraverseCtx_EdgeCount
(
	const EdgeTraverseCtx *edge_ctx
) {
	ASSERT(edge_ctx != NULL);
	return arr_len(edge_ctx->edges);
}

void EdgeTraverseCtx_Reset
(
	EdgeTraverseCtx *edge_ctx
) {
	ASSERT (edge_ctx != NULL) ;

	arr_clear (edge_ctx->edges) ;
	QGEdge_ResolveUnknownRelIDS (edge_ctx->e) ;
}

void EdgeTraverseCtx_Free
(
	EdgeTraverseCtx *edge_ctx
) {
	if (edge_ctx == NULL) {
		return ;
	}

	arr_free (edge_ctx->edges) ;
	QGEdge_Free (edge_ctx->e) ;
	rm_free (edge_ctx) ;
}

