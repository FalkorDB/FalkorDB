/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_create.h"
#include "RG.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../errors/errors.h"

// forward declarations
static RecordBatch CreateConsume (OpBase *opBase) ;
static OpResult CreateReset(OpBase *opBase);
static OpBase *CreateClone (const ExecutionPlan *plan, const OpBase *opBase) ;
static void CreateFree (OpBase *opBase) ;
static void FreeInternals (OpCreate *op) ;

OpBase *NewCreateOp
(
	const ExecutionPlan *plan,
	NodeCreateCtx *nodes,
	EdgeCreateCtx *edges
) {
	OpCreate *op = rm_calloc(1, sizeof(OpCreate));

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_CREATE, "Create", NULL, CreateConsume,
				CreateReset, NULL, CreateClone, CreateFree, true, plan);

	uint node_blueprint_count = array_len(nodes);
	uint edge_blueprint_count = array_len(edges);

	// construct the array of IDs this operation modifies
	for(uint i = 0; i < node_blueprint_count; i ++) {
		NodeCreateCtx *n = nodes + i;
		n->node_idx = OpBase_Modifies((OpBase *)op, n->alias);
	}

	for(uint i = 0; i < edge_blueprint_count; i ++) {
		EdgeCreateCtx *e = edges + i;
		e->edge_idx = OpBase_Modifies((OpBase *)op, e->alias);
		bool aware;
		UNUSED(aware);
		aware = OpBase_AliasMapping((OpBase *)op, e->src, &e->src_idx);
		ASSERT(aware == true);
		aware = OpBase_AliasMapping((OpBase *)op, e->dest, &e->dest_idx);
		ASSERT(aware == true);
	}

	// prepare all creation variables
	NewPendingCreationsContainer(&op->pending, nodes, edges); 

	return (OpBase *)op;
}

// prepare to create all nodes for the current Record
static void _CreateNodes
(
	OpCreate *op,
	RecordBatch batch,
	GraphContext *gc
) {
	// CREATE (a:A {v:x}), (b:B {z:x+1})
	// `nodes_to_create_count` = 2
	// NodeCreateCtx represent the blueprint for a created node pattern
	// e.g (a:A {v:x})

	uint nodes_to_create_count = array_len (op->pending.nodes.nodes_to_create) ;
	if (nodes_to_create_count == 0) {
		return ;
	}

	Graph *g = GraphContext_GetGraph (gc) ;
	size_t batch_size = RecordBatch_Size (batch) ;
	EntityID ids[batch_size] ;

	for (uint i = 0 ; i < nodes_to_create_count ; i++) {
		// get specified node to create
		NodeCreateCtx *ctx = op->pending.nodes.nodes_to_create + i ;

		// convert query-level properties
		PropertyMap *map = ctx->properties;
		AttributeSet sets[batch_size] ;
		memset (sets, 0, sizeof (AttributeSet) * batch_size) ;

		if (map != NULL) {
			// ConvertPropertyMap (gc, &converted_attr, r, map, false);
			ConvertPropertyMaps (sets, batch, map) ;
		}

		// create a new node
		Graph_ReserveNodeIDs (ids, g, batch_size) ;

		for (uint j = 0 ; j < batch_size ; j++) {
			Record r = batch[j] ;

			// add new node to Record and save a reference to it
			Node *node = Record_GetSetNode (r, ctx->node_idx) ;
			node->id = ids[j] ;

			// save node for later insertion
			array_append (op->pending.nodes.created_nodes, node) ;

			// save attributes to insert with node
			array_append (op->pending.nodes.node_attributes, sets[j]) ;

			// save labels to assigned to node
			array_append (op->pending.nodes.node_labels, ctx->labelsId) ;
		}
	}
}

// prepare to create all edges for the current Record
static void _CreateEdges
(
	OpCreate *op,
	RecordBatch batch,
	GraphContext *gc
) {
	uint edges_to_create_count = array_len(op->pending.edges);
	if (edges_to_create_count == 0) {
		return ;
	}

	size_t batch_size = RecordBatch_Size (batch) ;
	EntityID ids[batch_size] ;

	for(uint i = 0 ; i < edges_to_create_count ; i++) {
		// get specified edge to create
		PendingEdgeCreations *pending_edge = op->pending.edges + i ;
		EdgeCreateCtx *ctx = &pending_edge->edges_to_create ;

		// for each record in batch
		for (uint j = 0 ; j < batch_size ; j++) {
			Record r = batch[j] ;

			// retrieve source and dest nodes
			GraphEntity *src_node =
				(GraphEntity*)Record_GetNode (r, ctx->src_idx) ;

			GraphEntity *dest_node =
				(GraphEntity*)Record_GetNode (r, ctx->dest_idx) ;

			// verify edge endpoints resolved properly, fail otherwise
			if(unlikely(!src_node                        ||
						!dest_node                       ||
						GraphEntity_IsDeleted (src_node) ||
						GraphEntity_IsDeleted (dest_node))) {
				ErrorCtx_RaiseRuntimeException (
						"Failed to create relationship; endpoint was not found.") ;
			}

			Edge edge = {0} ;
			edge.relationship = ctx->relation ;
			Edge_SetSrcNodeID  (&edge, ENTITY_GET_ID (src_node)) ;
			Edge_SetDestNodeID (&edge, ENTITY_GET_ID (dest_node)) ;
			Record_AddEdge (r, ctx->edge_idx, edge) ;
		}

		// convert query-level properties
		PropertyMap *map = ctx->properties ;
		AttributeSet sets[batch_size] ;
		memset (sets, 0, sizeof (AttributeSet) * batch_size) ;

		if (map != NULL) {
			ConvertPropertyMaps (sets, batch, map) ;
		}

		for (uint j = 0 ; j < batch_size ; j++) {
			// create the actual edge
			Record r = batch[j] ;
			Edge *edge = Record_GetEdge (r, ctx->edge_idx) ;

			// save edge for later insertion
			array_append (pending_edge->created_edges, edge) ;

			// save attributes to insert with node
			array_append (pending_edge->edge_attributes, sets[j]) ;
		}
	}
}

// return mode, emit a populated batches
static RecordBatch _handoff
(
	OpCreate *op
) {
	if (op->batch_idx < array_len (op->batches)) {
		return op->batches[op->batch_idx++] ;
	} else {
		return NULL ;
	}
}

static RecordBatch CreateConsume
(
	OpBase *opBase
) {
	OpCreate *op = (OpCreate *)opBase ;
	RecordBatch batch ;

	//--------------------------------------------------------------------------
	// return mode
	//--------------------------------------------------------------------------

	if (op->batches != NULL) {
		return _handoff (op) ;
	}

	//--------------------------------------------------------------------------
	// consume mode
	//--------------------------------------------------------------------------

	op->batches = array_new (RecordBatch, 32) ;

	OpBase       *child = NULL ;
	GraphContext *gc    = QueryCtx_GetGraphCtx() ;

	if (op->op.childCount == 0) {
		// no child operation to call
		batch = OpBase_CreateRecordBatch (opBase, 1) ;

		// create entities
		_CreateNodes (op, batch, gc) ;
		_CreateEdges (op, batch, gc) ;

		// save batch for later use
		array_append (op->batches, batch) ;
	} else {
		// pull data until child is depleted
		child = op->op.children[0];

		while ((batch = OpBase_Consume(child))) {
			// create entities
			_CreateNodes (op, batch, gc) ;
			_CreateEdges (op, batch, gc) ;

			// save batch for later use
			array_append (op->batches, batch) ;
		}
	}

	// done reading, we're not going to call consume any longer
	// there might be operations e.g. index scan that need to free
	// index R/W lock, as such reset all execution plan operation up the chain
	if (child != NULL) {
		OpBase_PropagateReset (child) ;
	}

	// create entities
	CommitNewEntities (&op->pending) ;

	// no one consumes our output, return NULL
	if (opBase->parent == NULL) {
		return NULL ;
	}

	// return record
	return _handoff (op) ;
}

static OpResult CreateReset
(
	OpBase *opBase
) {
	OpCreate *op = (OpCreate*) opBase ;

	FreeInternals (op) ;
	op->batch_idx = 0 ;

	// reset PendingCreations
	PendingCreations_Reset (&op->pending) ;

	return OP_OK ;
}

static OpBase *CreateClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT (plan != NULL) ;
	ASSERT (OpBase_Type (opBase) == OPType_CREATE) ;

	OpCreate *op = (OpCreate *)opBase ;

	NodeCreateCtx *nodes ;
	array_clone_with_cb (nodes, op->pending.nodes.nodes_to_create,
			NodeCreateCtx_Clone) ;

	EdgeCreateCtx *edges =
		array_new (EdgeCreateCtx, array_len (op->pending.edges)) ;

	for (uint i = 0 ; i < array_len (op->pending.edges) ; i++) {
		EdgeCreateCtx ctx =
			EdgeCreateCtx_Clone (op->pending.edges[i].edges_to_create) ;
		array_append (edges, ctx) ;
	}

	return NewCreateOp (plan, nodes, edges) ;
}

static void FreeInternals
(
	OpCreate *op
) {
	if (op->batches) {
		uint batch_count = array_len (op->batches) ;
		// batches[0..batch_idx-1] had already been emitted, skip them
		for (uint i = op->batch_idx ; i < batch_count ; i++) {
			RecordBatch_Free (op->batches + i) ;
		}

		array_free (op->batches) ;
		op->batches = NULL ;
	}
}

static void CreateFree
(
	OpBase *ctx
) {
	OpCreate *op = (OpCreate *)ctx ;

	FreeInternals (op) ;

	PendingCreationsFree (&op->pending) ;
}

