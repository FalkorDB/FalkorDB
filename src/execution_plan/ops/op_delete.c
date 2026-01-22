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
	uint node_count = array_len (op->deleted_nodes) ;
	uint explicit_edge_count = array_len (op->deleted_edges) ;

	ASSERT ((node_count + explicit_edge_count) > 0) ;

	Graph        *g  = op->gc->g ;
	GraphContext *gc = op->gc ;

	Node *nodes = op->deleted_nodes ;
	Edge *edges = op->deleted_edges ;

	//--------------------------------------------------------------------------
	// collect implicit edges
	//--------------------------------------------------------------------------

	Edge *outgoing = NULL ;
	Edge *incoming = NULL ;
	uint64_t outgoing_edge_count = 0 ;
	uint64_t incoming_edge_count = 0 ;

	if (node_count > 0) {
		outgoing = array_new (Edge, 32) ;
		incoming = array_new (Edge, 32) ;
		Graph_CollectInOutEdges (&outgoing, &incoming, g, nodes, node_count) ;

		outgoing_edge_count = array_len (outgoing) ;
		incoming_edge_count = array_len (incoming) ;
	}

	//--------------------------------------------------------------------------
	// remove redundant edges
	//--------------------------------------------------------------------------

	// an edge is considered redundant if one of its endpoints (src/dest)
	// is marked for deletion, as its cheaper to delete an implicit edge
	// than an explicit edge
	if (node_count > 0) {
		for (uint i = 0; i < explicit_edge_count; i++) {
			Edge *e = edges + i ;
			if (roaring64_bitmap_contains (op->node_bitmap, e->src_id)  ||
				roaring64_bitmap_contains (op->node_bitmap, e->dest_id)) {
				array_del_fast (edges, i) ;
				i-- ;
				explicit_edge_count-- ;
			}
		}
	}

	// at least one entity is marked for deletion
	uint64_t total_edge_count =
		explicit_edge_count + outgoing_edge_count + incoming_edge_count ;

	ASSERT ((node_count + total_edge_count) > 0) ;

	// lock everything
	QueryCtx_LockForCommit();

	// NOTE: delete edges before nodes
	// required as a deleted node must be detached

	//--------------------------------------------------------------------------
	// delete edges
	//--------------------------------------------------------------------------

	if (outgoing_edge_count > 0) {
		GraphHub_DeleteEdges (gc, outgoing, outgoing_edge_count, true, true) ;
	}

	if (incoming_edge_count > 0) { 
		GraphHub_DeleteEdges (gc, incoming, incoming_edge_count, true, true) ;
	}

	if (explicit_edge_count > 0) {
		// explicit edge deletions
		GraphHub_DeleteEdges (gc, edges, explicit_edge_count, true, false) ;
	}

	//--------------------------------------------------------------------------
	// delete nodes
	//--------------------------------------------------------------------------

	if (node_count > 0) {
		GraphHub_DeleteNodes (gc, nodes, node_count, true);
	}

	// clean up
	if (outgoing != NULL) {
		array_free (outgoing) ;
	}

	if (incoming != NULL) {
		array_free (incoming) ;
	}
}

OpBase *NewDeleteOp
(
	const ExecutionPlan *plan,
	AR_ExpNode **exps
) {
	OpDelete *op = rm_calloc(1, sizeof(OpDelete));

	op->gc            = QueryCtx_GetGraphCtx () ;
	op->exps          = exps ;
	op->exp_count     = array_len (exps) ;
	op->deleted_nodes = array_new (Node, 32) ;
	op->deleted_edges = array_new (Edge, 32) ;

	op->node_bitmap = roaring64_bitmap_create () ;
	op->edge_bitmap = roaring64_bitmap_create () ;

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_DELETE, "Delete", NULL, DeleteConsume,
				NULL, NULL, DeleteClone, DeleteFree, true, plan);

	return (OpBase *)op;
}

// collect nodes and edges to be deleted
static inline void _CollectDeletedEntities
(
	Record r,
	OpDelete *op
) {
	// expression should be evaluated to either a node, an edge or a path
	// which will be marked for deletion, if an expression is evaluated
	// to a different value type e.g. numeric a run-time exception is thrown

	for (int i = 0; i < op->exp_count; i++) {
		AR_ExpNode *exp = op->exps[i] ;

		SIValue value = AR_EXP_Evaluate (exp, r) ;
		SIType type = SI_TYPE (value) ;

		// enqueue entities for deletion
		if (type & T_NODE) {
			Node *n = (Node *)value.ptrval ;

			// skip duplicated & deleted nodes
			if (!Graph_EntityIsDeleted ((GraphEntity *)n) &&
				roaring64_bitmap_add_checked (op->node_bitmap, ENTITY_GET_ID (n))) {
				array_append (op->deleted_nodes, *n) ;
			}

			continue ;
		}

		else if (type & T_EDGE) {
			Edge *e = (Edge *)value.ptrval ;

			// skip already deleted edges
			if (!Graph_EntityIsDeleted ((GraphEntity *)e)                &&
				!roaring64_bitmap_contains (op->node_bitmap, e->src_id)  &&
				!roaring64_bitmap_contains (op->node_bitmap, e->dest_id) &&
				roaring64_bitmap_add_checked (op->edge_bitmap, ENTITY_GET_ID (e))) {
				array_append (op->deleted_edges, *e) ;
			}

			continue ;
		}

		else if (type & T_PATH) {
			Path *p = (Path *)value.ptrval ;
			size_t nodeCount = Path_NodeCount (p) ;
			size_t edgeCount = Path_EdgeCount (p) ;

			for (size_t j = 0; j < nodeCount; j++) {
				Node *n = Path_GetNode (p, j) ;

				// skip duplicated & deleted nodes
				if (!Graph_EntityIsDeleted ((GraphEntity *)n) &&
					roaring64_bitmap_add_checked (op->node_bitmap, ENTITY_GET_ID (n))) {
					array_append (op->deleted_nodes, *n) ;
				}
			}

			// no need to collect edges, these will be collected implicitly
			// later on
		}
		
		else if (!(type & T_NULL)) {
			// if evaluating the expression allocated any memory, free it
			SIValue_Free (value) ;
			ErrorCtx_RaiseRuntimeException ("Delete type mismatch, expecting either Node or Relationship.") ;
			break ;
		}

		// if evaluating the expression allocated any memory, free it
		SIValue_Free (value) ;
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
	OpDelete *op = (OpDelete *)opBase ;
	ASSERT (op->op.childCount > 0) ;

	Record r ;

	// return mode, all data was consumed
	if (op->records) {
		return _handoff (op) ;
	}

	//--------------------------------------------------------------------------
	// consume mode
	//--------------------------------------------------------------------------

	GraphContext *gc = QueryCtx_GetGraphCtx () ;

	op->records = array_new (Record, 32) ;

	// pull data until child is depleted
	OpBase *child = op->op.children[0] ;
	while ((r = OpBase_Consume (child))) {
		// save record for later use
		array_append (op->records, r) ;

		// collect entities to be deleted
		_CollectDeletedEntities (r, op) ;
	}

	// done reading, we're not going to call consume any longer
	if (array_len (op->deleted_nodes) > 0 || array_len (op->deleted_edges) > 0) {
		// delete entities
		// there might be operations e.g. index scan that need to free
		// index R/W lock, as such reset all operation up the chain
		OpBase_PropagateReset (child) ;
		_DeleteEntities (op) ;
	}

	// no one consumes our output, return NULL
	if (opBase->parent == NULL) {
		return NULL ;
	}

	// return record
	return _handoff (op) ;
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

	if (op->deleted_nodes) {
		array_free (op->deleted_nodes) ;
		op->deleted_nodes = NULL ;
	}

	if (op->deleted_edges) {
		array_free (op->deleted_edges) ;
		op->deleted_edges = NULL ;
	}

	if(op->exps) {
		for(int i = 0; i < op->exp_count; i++) AR_EXP_Free(op->exps[i]);
		array_free(op->exps);
		op->exps = NULL;
	}

	if (op->node_bitmap) {
		roaring64_bitmap_free (op->node_bitmap) ;
		op->node_bitmap = NULL ;
	}

	if (op->edge_bitmap) {
		roaring64_bitmap_free (op->edge_bitmap) ;
		op->edge_bitmap = NULL ;
	}
}

