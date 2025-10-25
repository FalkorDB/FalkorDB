/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "./op_delete.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../errors/errors.h"
#include "../../graph/graph_hub.h"
#include "datatypes/path/sipath.h"
#include "../../arithmetic/arithmetic_expression.h"

#include <stdlib.h>

// forward declarations
static Record DeleteConsume(OpBase *opBase);
static OpBase *DeleteClone(const ExecutionPlan *plan, const OpBase *opBase);
static void DeleteFree(OpBase *opBase);

static int entity_cmp
(
	const GraphEntity *a,
	const GraphEntity *b
) {
	return ENTITY_GET_ID(a) - ENTITY_GET_ID(b);
}

static void _DeleteEntities
(
	OpDelete *op
) {
	uint node_count   = array_len(op->deleted_nodes);
	uint edge_count   = array_len(op->deleted_edges);
	uint node_deleted = 0;
	uint edge_deleted = 0;

	// nothing to delete, quickly return
	if((node_count + edge_count) == 0) return;

	Graph        *g  = op->gc->g;
	GraphContext *gc = op->gc;

	//--------------------------------------------------------------------------
	// removing node duplicates
	//--------------------------------------------------------------------------

	// remove node duplicates
	Node *nodes = op->deleted_nodes;
	Node *distinct_nodes = array_new(Node, 1);

	qsort(nodes, node_count, sizeof(Node),
			(int(*)(const void*, const void*))entity_cmp);

	for(uint i = 0; i < node_count; i++) {
		while(i < node_count - 1 &&
			  ENTITY_GET_ID(nodes + i) == ENTITY_GET_ID(nodes + i + 1)) {
			i++;
		}

		Node *n = nodes + i;

		// skip already deleted nodes
		if(Graph_EntityIsDeleted((GraphEntity *)n)) {
			continue;
		}

		array_append(distinct_nodes, *n);

		// mark node's edges for deletion
		Graph_GetNodeEdges(g, n, GRAPH_EDGE_DIR_BOTH, GRAPH_NO_RELATION,
				&op->deleted_edges);
	}

	node_count = array_len(distinct_nodes);
	edge_count = array_len(op->deleted_edges);

	//--------------------------------------------------------------------------
	// remove edge duplicates
	//--------------------------------------------------------------------------

	Edge *edges = op->deleted_edges;
	Edge *distinct_edges = array_new(Edge, 1);

	qsort(edges, edge_count, sizeof(Edge),
			(int(*)(const void*, const void*))entity_cmp);

	for(uint i = 0; i < edge_count; i++) {
		while(i < edge_count - 1 &&
			  ENTITY_GET_ID(edges + i) == ENTITY_GET_ID(edges + i + 1)) {
			i++;
		}

		Edge *e = edges + i;

		// skip already deleted edges
		if(Graph_EntityIsDeleted((GraphEntity *)e)) {
			continue;
		}

		array_append(distinct_edges, *e);
	}

	edge_count = array_len(distinct_edges);

	if((node_count + edge_count) > 0) {
		// lock everything
		QueryCtx_LockForCommit();
		{
			// NOTE: delete edges before nodes
			// required as a deleted node must be detached

			// delete edges
			if(edge_count > 0) {
				GraphHub_DeleteEdges(gc, distinct_edges, edge_count, true);
				edge_deleted = edge_count;
			}

			// delete nodes
			if(node_count > 0) {
				GraphHub_DeleteNodes(gc, distinct_nodes, node_count, true);
				node_deleted = node_count;
			}
		}
	}

	// clean up
	array_free(distinct_nodes);
	array_free(distinct_edges);
}

OpBase *NewDeleteOp
(
	const ExecutionPlan *plan,
	AR_ExpNode **exps
) {
	OpDelete *op = rm_calloc(1, sizeof(OpDelete));

	op->gc            = QueryCtx_GetGraphCtx();
	op->exps          = exps;
	op->exp_count     = array_len(exps);
	op->deleted_nodes = array_new(Node, 32);
	op->deleted_edges = array_new(Edge, 32);

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_DELETE, "Delete", NULL, DeleteConsume,
				NULL, NULL, DeleteClone, DeleteFree, true, plan);

	return (OpBase *)op;
}

// collect nodes and edges to be deleted
static inline void _CollectDeletedEntities
(
	Record r,
	OpBase *opBase
) {
	OpDelete *op = (OpDelete *)opBase;

	// expression should be evaluated to either a node, an edge or a path
	// which will be marked for deletion, if an expression is evaluated
	// to a different value type e.g. Numeric a run-time exception is thrown.
	for(int i = 0; i < op->exp_count; i++) {
		AR_ExpNode *exp = op->exps[i];
		SIValue value = AR_EXP_Evaluate(exp, r);
		SIType type = SI_TYPE(value);
		// enqueue entities for deletion
		if(type & T_NODE) {
			Node *n = (Node *)value.ptrval;
			array_append(op->deleted_nodes, *n);
		} else if(type & T_EDGE) {
			Edge *e = (Edge *)value.ptrval;
			array_append(op->deleted_edges, *e);
		} else if(type & T_PATH) {
			Path *p = (Path *)value.ptrval;
			size_t nodeCount = Path_NodeCount(p);
			size_t edgeCount = Path_EdgeCount(p);

			for(size_t i = 0; i < nodeCount; i++) {
				Node *n = Path_GetNode(p, i);
				array_append(op->deleted_nodes, *n);
			}

			for(size_t i = 0; i < edgeCount; i++) {
				Edge *e = Path_GetEdge(p, i);
				array_append(op->deleted_edges, *e);
			}
		} else if(!(type & T_NULL)) {
			// if evaluating the expression allocated any memory, free it.
			SIValue_Free(value);
			ErrorCtx_RaiseRuntimeException("Delete type mismatch, expecting either Node or Relationship.");
			break;
		}

		// if evaluating the expression allocated any memory, free it.
		SIValue_Free(value);
	}
}

static inline Record _handoff
(
	OpDelete *op
) {
	if(op->rec_idx < array_len(op->records)) {
		return op->records[op->rec_idx++];
	} else {
		return NULL;
	}
}

static Record DeleteConsume
(
	OpBase *opBase
) {
	OpDelete *op = (OpDelete *)opBase;
	Record r;
	ASSERT(op->op.childCount > 0);

	// return mode, all data was consumed
	if(op->records) return _handoff(op);

	// consume mode
	op->records = array_new(Record, 32);

	GraphContext *gc = QueryCtx_GetGraphCtx();
	// pull data until child is depleted
	OpBase *child = op->op.children[0];
	while((r = OpBase_Consume(child))) {
		// save record for later use
		array_append(op->records, r);

		// collect entities to be deleted
		_CollectDeletedEntities(r, opBase);
	}

	// done reading, we're not going to call consume any longer
	if(array_len(op->deleted_nodes) > 0 || array_len(op->deleted_edges) > 0) {
		// delete entities
		// there might be operations e.g. index scan that need to free
		// index R/W lock, as such reset all operation up the chain
		OpBase_PropagateReset(child);
		_DeleteEntities(op);
	}

	// no one consumes our output, return NULL
	if (opBase->parent == NULL) {
		return NULL ;
	}

	// return record
	return _handoff(op);
}

static OpBase *DeleteClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_DELETE);

	OpDelete *op = (OpDelete *)opBase;
	AR_ExpNode **exps;
	array_clone_with_cb(exps, op->exps, AR_EXP_Clone);
	return NewDeleteOp(plan, exps);
}

static void DeleteFree
(
	OpBase *opBase
) {
	OpDelete *op = (OpDelete *)opBase;

	if (op->records) {
		uint rec_count = array_len (op->records) ;
		// records[0..rec_idx-1] had been already emitted, skip them
		for (uint i = op->rec_idx; i < rec_count; i++) {
			OpBase_DeleteRecord (op->records+i) ;
		}
		array_free (op->records) ;
		op->records = NULL ;
	}

	if(op->deleted_nodes) {
		array_free(op->deleted_nodes);
		op->deleted_nodes = NULL;
	}

	if(op->deleted_edges) {
		array_free(op->deleted_edges);
		op->deleted_edges = NULL;
	}

	if(op->exps) {
		for(int i = 0; i < op->exp_count; i++) AR_EXP_Free(op->exps[i]);
		array_free(op->exps);
		op->exps = NULL;
	}
}

