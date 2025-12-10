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
static Record CreateConsume (OpBase *opBase) ;
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
	Record r,
	GraphContext *gc
) {
	uint nodes_to_create_count = array_len(op->pending.nodes.nodes_to_create);
	for (uint i = 0; i < nodes_to_create_count; i++) {
		// get specified node to create
		NodeCreateCtx *n = op->pending.nodes.nodes_to_create + i;

		// create a new node
		Node newNode = Graph_ReserveNode(gc->g);

		// add new node to Record and save a reference to it
		Node *node_ref = Record_AddNode(r, n->node_idx, newNode);

		// convert query-level properties
		AttributeSet converted_attr = NULL;
		PropertyMap *map = n->properties;
		if(map != NULL) {
			ConvertPropertyMap(gc, &converted_attr, r, map, false);
		}

		// save node for later insertion
		array_append(op->pending.nodes.created_nodes, node_ref);

		// save attributes to insert with node
		array_append(op->pending.nodes.node_attributes, converted_attr);

		// save labels to assigned to node
		array_append(op->pending.nodes.node_labels, n->labelsId);
	}
}

// prepare to create all edges for the current Record
static void _CreateEdges
(
	OpCreate *op,
	Record r,
	GraphContext *gc
) {
	uint edges_to_create_count = array_len(op->pending.edges);
	for(uint i = 0; i < edges_to_create_count; i++) {
		PendingEdgeCreations *pending_edge = op->pending.edges + i;
		// get specified edge to create
		EdgeCreateCtx *e = &pending_edge->edges_to_create;

		// retrieve source and dest nodes
		GraphEntity *src_node  = (GraphEntity*)Record_GetNode(r, e->src_idx);
		GraphEntity *dest_node = (GraphEntity*)Record_GetNode(r, e->dest_idx);

		// verify edge endpoints resolved properly, fail otherwise
		if(unlikely(!src_node                       ||
					!dest_node                      ||
					GraphEntity_IsDeleted(src_node) ||
					GraphEntity_IsDeleted(dest_node))) {
			ErrorCtx_RaiseRuntimeException(
					"Failed to create relationship; endpoint was not found.");
		}

		// create the actual edge
		Edge newEdge = {0};
		newEdge.relationship = e->relation;
		Edge_SetSrcNodeID(&newEdge, ENTITY_GET_ID(src_node));
		Edge_SetDestNodeID(&newEdge, ENTITY_GET_ID(dest_node));
		Edge *edge_ref = Record_AddEdge(r, e->edge_idx, newEdge);

		// convert query-level properties
		PropertyMap *map = e->properties;
		AttributeSet converted_attr = NULL;
		if(map != NULL) {
			ConvertPropertyMap(gc, &converted_attr, r, map, false);
		}

		// save edge for later insertion
		array_append(pending_edge->created_edges, edge_ref);

		// save attributes to insert with node
		array_append(pending_edge->edge_attributes, converted_attr);
	}
}

// Return mode, emit a populated Record.
static Record _handoff
(
	OpCreate *op
) {
	if(op->rec_idx < array_len(op->records)) {
		return op->records[op->rec_idx++];
	} else {
		return NULL;
	}
}

static Record CreateConsume
(
	OpBase *opBase
) {
	OpCreate *op = (OpCreate *)opBase;
	Record r;

	// return mode, all data was consumed
	if(op->records) return _handoff(op);

	// consume mode
	op->records = array_new(Record, 32);

	OpBase       *child = NULL;
	GraphContext *gc    = QueryCtx_GetGraphCtx();

	if(op->op.childCount == 0) {
		// no child operation to call
		r = OpBase_CreateRecord(opBase);
		// create entities
		_CreateNodes(op, r, gc);
		_CreateEdges(op, r, gc);

		// save record for later use
		array_append(op->records, r);
	} else {
		// pull data until child is depleted
		child = op->op.children[0];
		while((r = OpBase_Consume(child))) {
			// create entities
			_CreateNodes(op, r, gc);
			_CreateEdges(op, r, gc);

			// save record for later use
			array_append(op->records, r);
		}
	}

	// done reading, we're not going to call consume any longer
	// there might be operations e.g. index scan that need to free
	// index R/W lock, as such reset all execution plan operation up the chain
	if(child) {
		OpBase_PropagateReset(child);
	}

	// create entities
	CommitNewEntities(&op->pending);

	// no one consumes our output, return NULL
	if (opBase->parent == NULL) {
		return NULL ;
	}

	// return record
	return _handoff(op);
}

static OpResult CreateReset
(
	OpBase *opBase
) {
	OpCreate *op = (OpCreate*) opBase ;

	FreeInternals (op) ;
	op->rec_idx = 0 ;

	// reset PendingCreations
	PendingCreations_Reset (&op->pending) ;

	return OP_OK ;
}

static OpBase *CreateClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_CREATE);

	OpCreate *op = (OpCreate *)opBase;
	NodeCreateCtx *nodes;
	EdgeCreateCtx *edges = array_new(EdgeCreateCtx, array_len(op->pending.edges));
	array_clone_with_cb(nodes, op->pending.nodes.nodes_to_create, NodeCreateCtx_Clone);

	for(uint i = 0; i < array_len(op->pending.edges); i++) {
		EdgeCreateCtx ctx = EdgeCreateCtx_Clone(op->pending.edges[i].edges_to_create);
		array_append(edges, ctx);
	}

	return NewCreateOp(plan, nodes, edges);
}

static void FreeInternals
(
	OpCreate *op
) {
	if (op->records) {
		uint rec_count = array_len (op->records) ;
		// records[0..rec_idx-1] had already been emitted, skip them
		for (uint i = op->rec_idx; i < rec_count; i++) {
			OpBase_DeleteRecord (op->records+i) ;
		}

		array_free (op->records) ;
		op->records = NULL ;
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

