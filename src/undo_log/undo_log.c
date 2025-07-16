/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "undo_log.h"
#include "query_ctx.h"
#include "../execution_plan/ops/shared/update_functions.h"
#include "../execution_plan/ops/shared/create_functions.h"
#include "../graph/entities/attribute_set.h"

// initial number of entries in undo-log
#define UNDOLOG_INIT_SIZE 32

#define UNDOLOG_GET_ITEM(log, i) DataBlock_GetItem(log, i)
#define UNDOLOG_ADD_OP(log, op) \
	*(UndoOp*)DataBlock_AllocateItem((log), NULL) = op;

static void _index_node
(
	QueryCtx *ctx,
	Node *n
) {
	uint label_count;
	NODE_GET_LABELS(ctx->gc->g, n, label_count);
	for(uint j = 0; j < label_count; j++) {
		Schema *s = GraphContext_GetSchemaByID(ctx->gc, labels[j], SCHEMA_NODE);
		ASSERT(s);

		if(Schema_HasIndices(s)) Schema_AddNodeToIndices(s, n);
	}
}

static void _index_node_with_labels
(
	QueryCtx *ctx,
	Node *n,
	int *labels,
	uint label_count
) {
	for(uint i = 0; i < label_count; i++) {
		Schema *s = GraphContext_GetSchemaByID(ctx->gc, labels[i], SCHEMA_NODE);
		ASSERT(s != NULL);

		if(Schema_HasIndices(s)) {
			Schema_AddNodeToIndices(s, n);
		}
	}
}

static void _index_edge
(
	QueryCtx *ctx,
	Edge *e
) {
	Schema *s = GraphContext_GetSchemaByID(ctx->gc, Edge_GetRelationID(e), SCHEMA_EDGE);
	ASSERT(s);

	if(Schema_HasIndices(s)) Schema_AddEdgeToIndices(s, e);
}

static void _index_delete_node_with_labels
(
	QueryCtx *ctx,
	Node *n,
	int *labels,
	uint label_count
) {
	for(uint i = 0; i < label_count; i++) {
		Schema *s = GraphContext_GetSchemaByID(ctx->gc, labels[i], SCHEMA_NODE);
		ASSERT(s != NULL);

		// update any indices this entity is represented in
		Schema_RemoveNodeFromIndices(s, n);
	}
}

static void _index_delete_node
(
	QueryCtx *ctx,
	Node *n
) {
	uint label_count;
	NODE_GET_LABELS(ctx->gc->g, n, label_count);
	for(uint j = 0; j < label_count; j++) {
		Schema *s = GraphContext_GetSchemaByID(ctx->gc, labels[j], SCHEMA_NODE);
		ASSERT(s);

		// update any indices this entity is represented in
		Schema_RemoveNodeFromIndices(s, n);
	}
}

static void _index_delete_edge
(
	QueryCtx *ctx,
	Edge *e
) {
	Schema *s = GraphContext_GetSchemaByID(ctx->gc, Edge_GetRelationID(e), SCHEMA_EDGE);
	ASSERT(s);

	// update any indices this entity is represented in
	Schema_RemoveEdgeFromIndices(s, e);
}

static void _UndoLog_Restore_Entity_Property
(
	GraphEntity *ge,
	Attribute_ID attr_id,
	SIValue value
) {
	// try to get current attribute value
	SIValue *old_value = GraphEntity_GetProperty(ge, attr_id);

	if(old_value == ATTRIBUTE_NOTFOUND) {
		// adding a new attribute; do nothing if its value is NULL
		if(SI_TYPE(value) != T_NULL) {
			AttributeSet_AddNoClone(ge->attributes, &attr_id, &value, 1, false);
		}
	} else {
		// update attribute
		AttributeSet_UpdateNoClone(ge->attributes, attr_id, value);
	}
}

// rollback the updates taken place by current query
static void _UndoLog_Rollback_Update_Entity
(
	QueryCtx *ctx,
	int seq_start,
	int seq_end
) {
	for(int i = seq_start; i > seq_end; --i) {
		UndoOp *op = UNDOLOG_GET_ITEM(ctx->undo_log, i);
		UndoUpdateOp *update_op = &op->update_op;

		// update indices
		if(update_op->entity_type == GETYPE_NODE) {
			// free current entity attribute-set
			AttributeSet_Free(update_op->n.attributes);
			// restore entity original attribute-set
			*update_op->n.attributes = update_op->set;
			_index_node(ctx, &update_op->n);
		} else {
			AttributeSet_Free(update_op->e.attributes);
			*update_op->e.attributes = update_op->set;
			_index_edge(ctx, &update_op->e);
		}
	}
}

static void _UndoLog_Rollback_Set_Labels
(
	QueryCtx *ctx,
	int seq_start,
	int seq_end
) {
	for(int i = seq_start; i > seq_end; --i) {
		Graph        *g                = QueryCtx_GetGraph();
		UndoOp       *op               = UNDOLOG_GET_ITEM(ctx->undo_log, i);
		UndoLabelsOp *update_labels_op = &(op->labels_op);
		uint         labels_count      = update_labels_op->labels_count;

		Graph_RemoveNodeLabels(g, update_labels_op->node.id,
				update_labels_op->label_ids, labels_count);

		_index_delete_node_with_labels(ctx, &(update_labels_op->node),
				update_labels_op->label_ids, labels_count);

		array_free(update_labels_op->label_ids);
	}
}

static void _UndoLog_Rollback_Remove_Labels
(
	QueryCtx *ctx,
	int seq_start,
	int seq_end
) {
	for(int i = seq_start; i > seq_end; --i) {
		Graph        *g                = QueryCtx_GetGraph();
		UndoOp       *op               = UNDOLOG_GET_ITEM(ctx->undo_log, i);
		UndoLabelsOp *update_labels_op = &(op->labels_op);
		uint         labels_count      = update_labels_op->labels_count;

		Graph_LabelNode(g, update_labels_op->node.id,
				update_labels_op->label_ids, labels_count);

		_index_node_with_labels(ctx, &(update_labels_op->node),
				update_labels_op->label_ids, labels_count);

		array_free(update_labels_op->label_ids);
	}
}

// undo node creation
static void _UndoLog_Rollback_Create_Node
(
	QueryCtx *ctx,
	int seq_start,
	int seq_end
) {
	ASSERT(seq_start > seq_end);

	uint node_count = seq_start - seq_end;

	Node *nodes = rm_malloc(sizeof(Node) * node_count);

	for(int i = seq_start; i > seq_end; --i) {
		UndoOp *op = UNDOLOG_GET_ITEM(ctx->undo_log, i);
		Node *n = &op->create_op.n;
		nodes[seq_start - i] = *n;
		_index_delete_node(ctx, n);
	}

	Graph_DeleteNodes(ctx->gc->g, nodes, node_count);
	rm_free(nodes);
}

// undo edge creation
static void _UndoLog_Rollback_Create_Edge
(
	QueryCtx *ctx,
	int seq_start,
	int seq_end
) {
	ASSERT(seq_start > seq_end);

	uint edge_count = seq_start - seq_end;

	Edge *edges = rm_malloc(sizeof(Edge) * edge_count);

	for(int i = seq_start; i > seq_end; --i) {
		UndoOp *op = UNDOLOG_GET_ITEM(ctx->undo_log, i);
		Edge *e = &op->create_op.e;
		edges[seq_start - i] = *e;
		_index_delete_edge(ctx, e);
	}

	Graph_DeleteEdges(ctx->gc->g, edges, edge_count);
	rm_free(edges);
}

// undo node deletion
static void _UndoLog_Rollback_Delete_Node
(
	QueryCtx *ctx,
	int seq_start,
	int seq_end
) {
	for(int i = seq_start; i > seq_end; --i) {
		Node n = GE_NEW_NODE();
		UndoOp *op = UNDOLOG_GET_ITEM(ctx->undo_log, i);
		UndoDeleteNodeOp *delete_op = &(op->delete_node_op);

		Graph_CreateNode(ctx->gc->g, &n, delete_op->labels,
				delete_op->label_count);
		*n.attributes = delete_op->set;

		// re-introduce node to indices
		_index_node(ctx, &n);

		// cleanup after undo rollback, as the op D'tor is not called
		rm_free(delete_op->labels);
	}
}

// undo edge deletion
static void _UndoLog_Rollback_Delete_Edge
(
	QueryCtx *ctx,
	int seq_start,
	int seq_end
) {
	for(int i = seq_start; i > seq_end; --i) {
		Edge e;
		UndoOp *op = UNDOLOG_GET_ITEM(ctx->undo_log, i);
		UndoDeleteEdgeOp delete_op = op->delete_edge_op;

		Graph_CreateEdge(ctx->gc->g, delete_op.src_id, delete_op.dest_id,
				delete_op.relationID, &e);
		*e.attributes = delete_op.set;

		_index_edge(ctx, &e);
	}
}

// undo schema addition
static void _UndoLog_Rollback_Add_Schema
(
	QueryCtx *ctx,
	int seq_start,
	int seq_end
) {
	for(int i = seq_start; i > seq_end; --i) {
		Edge e;
		UndoOp *op = UNDOLOG_GET_ITEM(ctx->undo_log, i);
		UndoAddSchemaOp schema_op = op->schema_op;
		int schema_id = schema_op.schema_id;
		int schema_count = GraphContext_SchemaCount(ctx->gc, schema_op.t);
		ASSERT(schema_id == schema_count - 1);
		GraphContext_RemoveSchema(ctx->gc, schema_id, schema_op.t);
		if(schema_op.t == SCHEMA_NODE) {
			Graph_RemoveLabel(ctx->gc->g, schema_id);
		} else {
			Graph_RemoveRelation(ctx->gc->g, schema_id);
		}
	}
}

// undo attribute addition
static void _UndoLog_Rollback_Add_Attribute
(
	QueryCtx *ctx,
	int seq_start,
	int seq_end
) {
	for(int i = seq_start; i > seq_end; --i) {
		UndoOp *op = UNDOLOG_GET_ITEM(ctx->undo_log, i);
		UndoAddAttributeOp attribute_op = op->attribute_op;
		int attribute_id = attribute_op.attribute_id;
		GraphContext_RemoveAttribute(ctx->gc, attribute_id);
	}
}

UndoLog UndoLog_New(void) {
	return (UndoLog)DataBlock_New(
		UNDOLOG_INIT_SIZE,
		UNDOLOG_INIT_SIZE,
		sizeof(UndoOp),
		NULL);
}

// returns number of entries in log
uint UndoLog_Length
(
	const UndoLog log  // log to query
) {
	ASSERT(log != NULL);
	return DataBlock_ItemCount(log);
}

//------------------------------------------------------------------------------
// Undo add changes
//------------------------------------------------------------------------------

// undo node creation
void UndoLog_CreateNode
(
	UndoLog log,           // undo log
	Node *node             // node created
) {
	ASSERT(log != NULL);

	UndoOp op;

	op.type        = UNDO_CREATE_NODE;
	op.create_op.n = *node;

	UNDOLOG_ADD_OP(log, op);
}

// undo edge creation
void UndoLog_CreateEdge
(
	UndoLog log,           // undo log
	Edge *edge             // edge created
) {
	ASSERT(log != NULL);

	UndoOp op;

	op.type        = UNDO_CREATE_EDGE;
	op.create_op.e = *edge;

	UNDOLOG_ADD_OP(log, op);
}

// undo node deletion
void UndoLog_DeleteNode
(
	UndoLog log,       // undo log
	Node *node         // node deleted
) {
	ASSERT(log != NULL);
	ASSERT(node != NULL);

	UndoOp op;

	op.type              = UNDO_DELETE_NODE;
	op.delete_node_op.id = node->id;

	// take ownership over node's attribute-set
	op.delete_node_op.set = *node->attributes;
	
	// mark node's attribute-set as read-only
	*node->attributes =
		(AttributeSet)ATTRIBUTE_SET_MARK_READONLY(*node->attributes);

	Graph *g = QueryCtx_GetGraph();
	NODE_GET_LABELS(g, node, op.delete_node_op.label_count);

	// save node's labels
	op.delete_node_op.labels =
		rm_malloc(sizeof(LabelID) * op.delete_node_op.label_count);

	memcpy(op.delete_node_op.labels, labels,
			sizeof(LabelID) * op.delete_node_op.label_count);

	UNDOLOG_ADD_OP(log, op);
}

// undo edge deletion
void UndoLog_DeleteEdge
(
	UndoLog log,   // undo log
	Edge *edge     // edge deleted
) {
	ASSERT(log != NULL);
	ASSERT(edge != NULL);

	UndoOp op;

	op.type                      = UNDO_DELETE_EDGE;
	op.delete_edge_op.id         = edge->id;
	op.delete_edge_op.src_id     = Edge_GetSrcNodeID(edge);
	op.delete_edge_op.dest_id    = Edge_GetDestNodeID(edge);
	op.delete_edge_op.relationID = Edge_GetRelationID(edge);

	// take ownership over edge's attribute-set
	op.delete_edge_op.set = *edge->attributes;

	// mark edge's attribute-set as read-only
	*edge->attributes =
		(AttributeSet)ATTRIBUTE_SET_MARK_READONLY(*edge->attributes);

	UNDOLOG_ADD_OP(log, op);
}

// undo entity update
void UndoLog_UpdateEntity
(
	UndoLog log,                 // undo log
	GraphEntity *ge,             // updated entity
	AttributeSet set,            // old attribute set
	GraphEntityType entity_type  // entity type
) {
	ASSERT(log != NULL);
	ASSERT(ge != NULL);

	UndoOp op;

	op.type                  = UNDO_UPDATE;
	op.update_op.set         = set;
	op.update_op.entity_type = entity_type;

	if(entity_type == GETYPE_NODE) {
		op.update_op.n = *(Node *)ge;
	} else {
		op.update_op.e = *(Edge *)ge;
	}

	UNDOLOG_ADD_OP(log, op);
}

// undo node add label
void UndoLog_AddLabels
(
	UndoLog log,                 // undo log
	Node *node,                  // updated node
	LabelID *label_ids,          // added labels
	size_t labels_count          // number of removed labels
) {
	ASSERT(node != NULL);
	ASSERT(label_ids != NULL);

	UndoOp op;

	op.type = UNDO_SET_LABELS;
	op.labels_op.node = *node;
	op.labels_op.label_ids = array_new(LabelID, labels_count);
	memcpy(op.labels_op.label_ids, label_ids, sizeof(LabelID)*labels_count);
	op.labels_op.labels_count = labels_count;
	UNDOLOG_ADD_OP(log, op);
}

// undo node remove label
void UndoLog_RemoveLabels
(
	UndoLog log,                 // undo log
	Node *node,                  // updated node
	LabelID *label_ids,          // removed labels
	size_t labels_count          // number of removed labels
) {
	ASSERT(node != NULL);
	ASSERT(label_ids != NULL);

	UndoOp op;

	op.type = UNDO_REMOVE_LABELS;
	op.labels_op.node = *node;
	op.labels_op.label_ids = array_new(LabelID, labels_count);
	memcpy(op.labels_op.label_ids, label_ids, sizeof(LabelID)*labels_count);
	op.labels_op.labels_count = labels_count;
	UNDOLOG_ADD_OP(log, op);
}

// undo schema addition
void UndoLog_AddSchema
(
	UndoLog log,    // undo log
	int schema_id,  // id of the schema
	SchemaType t    // type of the schema
) {
	ASSERT(log != NULL);
	UndoOp op;

	op.type = UNDO_ADD_SCHEMA;
	op.schema_op.schema_id = schema_id;
	op.schema_op.t = t;
	UNDOLOG_ADD_OP(log, op);
}

void UndoLog_AddAttribute
(
	UndoLog log,              // undo log
	Attribute_ID attribute_id // id of the attribute
) {
	ASSERT(log != NULL);
	UndoOp op;

	op.type = UNDO_ADD_ATTRIBUTE;
	op.attribute_op.attribute_id = attribute_id;
	UNDOLOG_ADD_OP(log, op);
}

//------------------------------------------------------------------------------
// rollback
//------------------------------------------------------------------------------

void UndoLog_Rollback
(
	UndoLog *log
) {
	ASSERT(log != NULL);

	UndoLog _log = *log;
	if(_log == NULL) return;

	QueryCtx *ctx  = QueryCtx_GetQueryCtx();
	uint64_t count = DataBlock_ItemCount(_log);

	// apply undo operations in reverse order for rollback correctness
	// find sequences of the same operation and rollback them as a bulk
	int seq_end = count - 1;
	while (seq_end >= 0) {
		UndoOp *op = UNDOLOG_GET_ITEM(_log, seq_end);
		UndoOpType cur_type = op->type;
		int seq_start = seq_end;
		seq_end--;
		while(seq_end >= 0) {
			op = UNDOLOG_GET_ITEM(_log, seq_end);
			if(op->type != cur_type) break;
			seq_end--;
		}

		switch(cur_type) {
			case UNDO_UPDATE:
				_UndoLog_Rollback_Update_Entity(ctx, seq_start, seq_end);
				break;
			case UNDO_CREATE_NODE:
				_UndoLog_Rollback_Create_Node(ctx, seq_start, seq_end);
				break;
			case UNDO_CREATE_EDGE:
				_UndoLog_Rollback_Create_Edge(ctx, seq_start, seq_end);
				break;
			case UNDO_DELETE_NODE:
				_UndoLog_Rollback_Delete_Node(ctx, seq_start, seq_end);
				break;
			case UNDO_DELETE_EDGE:
				_UndoLog_Rollback_Delete_Edge(ctx, seq_start, seq_end);
				break;
			case UNDO_SET_LABELS:
				_UndoLog_Rollback_Set_Labels(ctx, seq_start, seq_end);
				break;
			case UNDO_REMOVE_LABELS:
				_UndoLog_Rollback_Remove_Labels(ctx, seq_start, seq_end);
				break;
			case UNDO_ADD_SCHEMA:
				_UndoLog_Rollback_Add_Schema(ctx, seq_start, seq_end);
				break;
			case UNDO_ADD_ATTRIBUTE:
				_UndoLog_Rollback_Add_Attribute(ctx, seq_start, seq_end);
				break;
			default:
				ASSERT(false);
		}
 	}

	DataBlock_Free(_log);
	*log = NULL;
}

static void UndoLog_FreeOp
(
	UndoOp *op
) {
	ASSERT(op != NULL);

	switch(op->type) {
		case UNDO_UPDATE:
			AttributeSet_Free(&op->update_op.set);
			break;
		case UNDO_CREATE_NODE:
			break;
		case UNDO_CREATE_EDGE:
			break;
		case UNDO_DELETE_NODE:
			rm_free(op->delete_node_op.labels);
			AttributeSet_Free(&op->delete_node_op.set);
			break;
		case UNDO_DELETE_EDGE:
			AttributeSet_Free(&op->delete_edge_op.set);
			break;
		case UNDO_SET_LABELS:
		case UNDO_REMOVE_LABELS:
			array_free(op->labels_op.label_ids);
			break;
		case UNDO_ADD_SCHEMA:
		case UNDO_ADD_ATTRIBUTE:
			break;
		default:
			ASSERT(false);
	}
}

void UndoLog_Free
(
	UndoLog *log
) {
	ASSERT(log != NULL);

	UndoLog _log = *log;
	if(_log == NULL) return;

	DataBlockIterator *iter = DataBlock_Scan(_log);
	UndoOp *op;
	while((op = DataBlockIterator_Next(iter, NULL))) {
		UndoLog_FreeOp(op);
	}
	DataBlockIterator_Free(iter);
	DataBlock_Free(_log);
	*log = NULL;
}

