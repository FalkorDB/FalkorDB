/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "graph_hub.h"
#include "../query_ctx.h"
#include "../undo_log/undo_log.h"

// delete all references to a node from any relevant index
static void _DeleteNodeFromIndices
(
	GraphContext *gc,
	Node *n
) {
	ASSERT(n  != NULL);
	ASSERT(gc != NULL);

	Schema   *s      = NULL;
	Graph    *g      = gc->g;
	EntityID node_id = ENTITY_GET_ID(n);

	// retrieve node labels
	uint label_count;
	NODE_GET_LABELS(g, n, label_count);

	for(uint i = 0; i < label_count; i++) {
		int label_id = labels[i];
		s = GraphContext_GetSchemaByID(gc, label_id, SCHEMA_NODE);
		ASSERT(s != NULL);

		// update any indices this entity is represented in
		Schema_RemoveNodeFromIndices(s, n);
	}
}

static void _DeleteEdgeFromIndices
(
	GraphContext *gc,
	Edge *e
) {
	Schema  *s  =  NULL;
	Graph   *g  =  gc->g;

	int relation_id = Edge_GetRelationID(e);

	s = GraphContext_GetSchemaByID(gc, relation_id, SCHEMA_EDGE);

	// update any indices this entity is represented in
	Schema_RemoveEdgeFromIndices(s, e);
}

// add node to any relevant index
static void _AddNodeToIndices
(
	GraphContext *gc,
	Node *n
) {
	ASSERT(n  != NULL);
	ASSERT(gc != NULL);

	Schema    *s       =  NULL;
	Graph     *g       =  gc->g;
	EntityID  node_id  =  ENTITY_GET_ID(n);

	// retrieve node labels
	uint label_count;
	NODE_GET_LABELS(g, n, label_count);

	for(uint i = 0; i < label_count; i++) {
		int label_id = labels[i];
		s = GraphContext_GetSchemaByID(gc, label_id, SCHEMA_NODE);
		ASSERT(s != NULL);
		Schema_AddNodeToIndices(s, n);
	}
}

// add edge to any relevant index
static void _AddEdgeToIndices(GraphContext *gc, Edge *e) {
	Schema  *s  =  NULL;
	Graph   *g  =  gc->g;

	int relation_id = Edge_GetRelationID(e);

	s = GraphContext_GetSchemaByID(gc, relation_id, SCHEMA_EDGE);
	ASSERT(s != NULL);

	Schema_AddEdgeToIndices(s, e);
}

void CreateNode
(
	GraphContext *gc,
	Node *n,
	LabelID *labels,
	uint label_count,
	AttributeSet set,
	bool log
) {
	ASSERT(n  != NULL);
	ASSERT(gc != NULL);

	Graph_CreateNode(gc->g, n, labels, label_count);
	*n->attributes = set;

	// add node labels
	for(uint i = 0; i < label_count; i++) {
		Schema *s = GraphContext_GetSchemaByID(gc, labels[i], SCHEMA_NODE);
		ASSERT(s);
		Schema_AddNodeToIndices(s, n);
	}

	// add node creation operation to undo log
	if(log == true) {
		UndoLog undo_log = QueryCtx_GetUndoLog();
		UndoLog_CreateNode(undo_log, n);
		EffectsBuffer *eb = QueryCtx_GetEffectsBuffer();
		EffectsBuffer_AddCreateNodeEffect(eb, n, labels, label_count);
	}
}

void CreateEdge
(
	GraphContext *gc,
	Edge *e,
	NodeID src,
	NodeID dst,
	RelationID r,
	AttributeSet set,
	bool log
) {
	ASSERT(e  != NULL);
	ASSERT(gc != NULL);

	Graph_CreateEdge(gc->g, src, dst, r, e);
	*e->attributes = set;

	Schema *s = GraphContext_GetSchemaByID(gc, r, SCHEMA_EDGE);
	// all schemas have been created in the edge blueprint loop or earlier
	ASSERT(s != NULL);
	Schema_AddEdgeToIndices(s, e);

	// add edge creation operation to undo log
	if(log == true) {
		UndoLog undo_log = QueryCtx_GetUndoLog();
		UndoLog_CreateEdge(undo_log, e);
		EffectsBuffer *eb = QueryCtx_GetEffectsBuffer();
		EffectsBuffer_AddCreateEdgeEffect(eb, e);
	}
}

// delete a node
// remove the node from the relevant indexes
// add node deletion operation to undo-log
// return 1 on success, 0 otherwise
void DeleteNodes
(
	GraphContext *gc,
	Node *nodes,
	uint n,
	bool log
) {
	ASSERT(gc != NULL);
	ASSERT(nodes != NULL);

	bool has_indices = GraphContext_HasIndices(gc);

	UndoLog undo_log  = (log) ? QueryCtx_GetUndoLog() : NULL;
	EffectsBuffer *eb = (log) ? QueryCtx_GetEffectsBuffer() : NULL;
	for(uint i = 0; i < n; i++) {
		Node *n = nodes + i;

		if(log) {
			// add node deletion operation to undo log
			UndoLog_DeleteNode(undo_log, n);
			EffectsBuffer_AddDeleteNodeEffect(eb, n);
		}

		if(has_indices) {
			_DeleteNodeFromIndices(gc, n);
		}
	}

	Graph_DeleteNodes(gc->g, nodes, n);
}

void DeleteEdges
(
	GraphContext *gc,
	Edge *edges,
	uint64_t n,
	bool log
) {
	ASSERT(gc != NULL);
	ASSERT(n > 0);
	ASSERT(edges != NULL);

	// add edge deletion operation to undo log
	bool has_indecise = GraphContext_HasIndices(gc);

	UndoLog undo_log  = (log == true) ? QueryCtx_GetUndoLog() : NULL;
	EffectsBuffer *eb = (log == true) ? QueryCtx_GetEffectsBuffer() : NULL;

	if(has_indecise == true || log == true) {
		for (uint i = 0; i < n; i++) {
			if(log == true) {
				UndoLog_DeleteEdge(undo_log, edges + i);
				EffectsBuffer_AddDeleteEdgeEffect(eb, edges + i);
			}

			if(has_indecise == true) {
				_DeleteEdgeFromIndices(gc, edges + i);
			}
		}
	}

	Graph_DeleteEdges(gc->g, edges, n);
}

// updates a graph entity attribute set. Returns as out params the number
// of properties set and removed.
void UpdateEntityProperties
(
	GraphContext *gc,             // graph context
	GraphEntity *ge,              // updated entity
	const AttributeSet set,       // new attributes
	GraphEntityType entity_type,  // entity type
	bool log                      // log update in undo-log
) {
	ASSERT(gc != NULL);
	ASSERT(ge != NULL);

	AttributeSet old_set = GraphEntity_GetAttributes(ge);

	if(log == true) {
		UndoLog log = QueryCtx_GetUndoLog();
		UndoLog_UpdateEntity(log, ge, old_set, entity_type);
	}

	*ge->attributes = set;

	if(entity_type == GETYPE_NODE) {
		_AddNodeToIndices(gc, (Node *)ge);
	} else {
		_AddEdgeToIndices(gc, (Edge *)ge);
	}
}

void UpdateNodeProperty
(
	GraphContext *gc,             // graph context
	NodeID id,                    // node ID
	Attribute_ID attr_id,         // attribute ID
	SIValue v                     // new attribute value
) {
	Node n;
	int res = Graph_GetNode(gc->g, id, &n);

	// make sure entity was found
	UNUSED(res);
	ASSERT(res == true);

	if(attr_id == ATTRIBUTE_ID_ALL) {
		AttributeSet_Free(n.attributes);
	} else if(GraphEntity_GetProperty((GraphEntity *)&n, attr_id) == ATTRIBUTE_NOTFOUND) {
		AttributeSet_AddNoClone(n.attributes, &attr_id, &v, 1, true);
	} else {
		AttributeSet_UpdateNoClone(n.attributes, attr_id, v);
	}

	// retrieve node labels
	uint label_count;
	NODE_GET_LABELS(gc->g, &n, label_count);

	Schema *s;
	for(uint i = 0; i < label_count; i++) {
		int label_id = labels[i];
		s = GraphContext_GetSchemaByID(gc, label_id, SCHEMA_NODE);
		ASSERT(s != NULL);
		Schema_AddNodeToIndices(s, &n);
	}
}

void UpdateEdgeProperty
(
	GraphContext *gc,             // graph context
	EdgeID id,                    // edge ID
	RelationID r_id,              // relation ID
	NodeID src_id,                // source node ID
	NodeID dest_id,               // destination node ID
	Attribute_ID attr_id,         // attribute ID
	SIValue v                     // new attribute value
) {
	Edge e; // edge to delete

	// get src node, dest node and edge from the graph
	int res;
	UNUSED(res);

	res = Graph_GetEdge(gc->g, id, &e);
	ASSERT(res != 0);

	// set edge relation, src and destination node
	Edge_SetSrcNodeID(&e, src_id);
	Edge_SetDestNodeID(&e, dest_id);
	Edge_SetRelationID(&e, r_id);

	if(attr_id == ATTRIBUTE_ID_ALL) {
		AttributeSet_Free(e.attributes);
	} else if(GraphEntity_GetProperty((GraphEntity *)&e, attr_id) == ATTRIBUTE_NOTFOUND) {
		AttributeSet_AddNoClone(e.attributes, &attr_id, &v, 1, true);
	} else {
		AttributeSet_UpdateNoClone(e.attributes, attr_id, v);
	}

	Schema *schema = GraphContext_GetSchemaByID(gc, r_id, SCHEMA_EDGE);
	ASSERT(schema != NULL);
	Schema_AddEdgeToIndices(schema, &e);
}

void UpdateNodeLabels
(
	GraphContext *gc,            // graph context to update the entity
	Node *node,                  // the node to be updated
	const char **add_labels,     // labels to add to the node
	const char **remove_labels,  // labels to add to the node
	uint n_add_labels,           // number of labels to add
	uint n_remove_labels,        // number of labels to remove
	bool log                     // log this operation in undo-log
) {
	ASSERT(gc   != NULL);
	ASSERT(node != NULL);

	// quick return if there are no labels
	if(add_labels == NULL && remove_labels == NULL) {
		return;
	}

	// if add_labels is specified its count must be > 0
	ASSERT((add_labels != NULL && n_add_labels > 0) ||
		   (add_labels == NULL && n_add_labels == 0));

	// if remove_labels is specified its count must be > 0
	ASSERT((remove_labels != NULL && n_remove_labels > 0) ||
		   (remove_labels == NULL && n_remove_labels == 0));

	EffectsBuffer *eb = NULL; 
	UndoLog undo_log  = NULL;

	if(log == true) {
		eb = QueryCtx_GetEffectsBuffer();
		undo_log = QueryCtx_GetUndoLog();
	}

	if(add_labels != NULL) {
		int add_labels_ids[n_add_labels];
		uint add_labels_index = 0;

		for (uint i = 0; i < n_add_labels; i++) {
			const char *label = add_labels[i];
			// get or create label matrix
			const Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
			bool schema_created = false;
			if(s == NULL) {
				s = AddSchema(gc, label, SCHEMA_NODE, log);
				schema_created = true;
			}

			int  schema_id = Schema_GetID(s);
			bool node_labeled = Graph_IsNodeLabeled(gc->g, ENTITY_GET_ID(node),
					schema_id);

			if(!node_labeled) {
				// sync matrix
				// make sure label matrix is of the right dimensions
				if(schema_created) {
					RG_Matrix m = Graph_GetLabelMatrix(gc->g, schema_id);
				}
				// append label id
				add_labels_ids[add_labels_index++] = schema_id;
				// add to index
				Schema_AddNodeToIndices(s, node);
			}
		}

		if(add_labels_index > 0) {
			// update node's labels
			Graph_LabelNode(gc->g, ENTITY_GET_ID(node), add_labels_ids,
					add_labels_index);
			if(log == true) {
				UndoLog_AddLabels(undo_log, node, add_labels_ids,
						add_labels_index);
				EffectsBuffer_AddLabelsEffect(eb, node, add_labels_ids,
						add_labels_index);
			}
		}
	}

	if(remove_labels != NULL) {
		int remove_labels_ids[n_remove_labels];
		uint remove_labels_index = 0;

		for (uint i = 0; i < n_remove_labels; i++) {
			const char *label = remove_labels[i];

			// label removal
			// get or create label matrix
			const Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
			if(s == NULL) {
				// skip removal of none existing label
				continue;
			}

			if(!Graph_IsNodeLabeled(gc->g, ENTITY_GET_ID(node), Schema_GetID(s))) {
				// skip removal of none existing label
				continue;
			}

			// append label id
			remove_labels_ids[remove_labels_index++] = Schema_GetID(s);
			// remove node from index
			Schema_RemoveNodeFromIndices(s, node);
		}

		if(remove_labels_index > 0) {
			// update node's labels
			Graph_RemoveNodeLabels(gc->g, ENTITY_GET_ID(node),
					remove_labels_ids, remove_labels_index);
			if(log == true) {
				UndoLog_RemoveLabels(undo_log, node, remove_labels_ids,
						remove_labels_index);
				EffectsBuffer_AddRemoveLabelsEffect(eb, node, remove_labels_ids,
						remove_labels_index);
			}
		}
	}
}

Schema *AddSchema
(
	GraphContext *gc,   // graph context to add the schema
	const char *label,  // schema label
	SchemaType t,       // schema type (node/edge)
	bool log            // should operation be logged in the undo-log
) {
	ASSERT(gc != NULL);
	ASSERT(label != NULL);
	Schema *s = GraphContext_AddSchema(gc, label, t);

	if(log == true) {
		UndoLog undo_log = QueryCtx_GetUndoLog();
		UndoLog_AddSchema(undo_log, s->id, s->type);
		EffectsBuffer *eb = QueryCtx_GetEffectsBuffer();
		EffectsBuffer_AddNewSchemaEffect(eb, Schema_GetName(s), s->type);
	}

	return s;
}

Attribute_ID FindOrAddAttribute
(
	GraphContext *gc,       // graph context to add the attribute
	const char *attribute,  // attribute name
	bool log                // should operation be logged in the undo-log
) {
	ASSERT(gc != NULL);
	ASSERT(attribute != NULL);

	bool created;
	Attribute_ID attr_id = GraphContext_FindOrAddAttribute(gc, attribute,
			&created);

	// in case there was an append, the latest id should be tracked
	if(created == true && log == true) {
		UndoLog undo_log = QueryCtx_GetUndoLog();
		UndoLog_AddAttribute(undo_log, attr_id);
		EffectsBuffer *eb = QueryCtx_GetEffectsBuffer();
		EffectsBuffer_AddNewAttributeEffect(eb, attribute);
	}

	return attr_id;
}

