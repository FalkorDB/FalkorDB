/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "graph_hub.h"
#include "../query_ctx.h"

// create a node
// set the node labels and attributes
// add the node to the relevant indexes
// add node creation operation to undo-log
void GraphHub_CreateNode
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
		Schema_AddNodeToIndex(s, n);
	}

	// add node creation operation to undo log
	if(log == true) {
		UndoLog undo_log = QueryCtx_GetUndoLog();
		UndoLog_CreateNode(undo_log, n);

		EffectsBuffer *eb = QueryCtx_GetEffectsBuffer();
		EffectsBuffer_AddCreateNodeEffect(eb, n, labels, label_count);
	}
}

// batch create nodes
// all nodes share the same set of labels
// set the nodes labels and attributes
// add the nodes to the relevant indexes
// add nodes creation operation to undo-log
void GraphHub_CreateNodes
(
	GraphContext *gc,    // graph context
	Node **nodes,        // nodes to create
	AttributeSet *sets,  // nodes attributes
	uint node_count,     // number of nodes
	LabelID *labels,     // nodes labels
	uint label_count,    // number of labels
	bool log             // true if operation needs to be logged
) {
	ASSERT (gc    != NULL) ;
	ASSERT (nodes != NULL) ;
	ASSERT (node_count > 0) ;
	ASSERT (label_count == 0 || labels != NULL) ;

	// introduce nodes to graph
	Graph_CreateNodes (gc->g, nodes, sets, node_count, labels, label_count) ;

	//--------------------------------------------------------------------------
	// collect schemas with indices
	//--------------------------------------------------------------------------

	int s_idx = 0 ;
	Schema *schemas[label_count] ;

	for(uint i = 0; i < label_count; i++) {
		Schema *s = GraphContext_GetSchemaByID (gc, labels[i], SCHEMA_NODE) ;
		ASSERT (s != NULL) ;

		if (Schema_HasIndices (s)) {
			schemas[s_idx++] = s ;
		}
	}
	bool index = s_idx > 0 ;

	// add nodes creation operation to undo log
	if (log || index) {
		UndoLog undo_log  = NULL ;
		EffectsBuffer *eb = NULL ;

		if (log) {
			eb = QueryCtx_GetEffectsBuffer () ;
			undo_log = QueryCtx_GetUndoLog () ;
		}

		for (uint i = 0; i < node_count; i++) {
			Node *n = nodes[i] ;

			if (log == true) {
				UndoLog_CreateNode (undo_log, n) ;
				EffectsBuffer_AddCreateNodeEffect (eb, n, labels, label_count) ;
			}

			if (index) {
				for (uint j = 0; j < s_idx; j++) {
					Schema_AddNodeToIndex (schemas[j], n) ;
				}
			}
		}
	}
}

void GraphHub_CreateEdge
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
	Schema_AddEdgeToIndex(s, e);

	// add edge creation operation to undo log
	if(log == true) {
		UndoLog undo_log = QueryCtx_GetUndoLog();
		UndoLog_CreateEdge(undo_log, e);

		EffectsBuffer *eb = QueryCtx_GetEffectsBuffer();
		EffectsBuffer_AddCreateEdgeEffect(eb, e);
	}
}

void GraphHub_CreateEdges
(
	GraphContext *gc,
	Edge **edges,
	RelationID r,
	AttributeSet *sets,
	bool log
) {
	ASSERT (gc    != NULL) ;
	ASSERT (edges != NULL) ;

	Graph_CreateEdges (gc->g, r, edges, sets) ;

	Schema *s = GraphContext_GetSchemaByID (gc, r, SCHEMA_EDGE) ;
	ASSERT (s != NULL) ;
	bool has_indices = Schema_HasIndices (s) ;

	if (has_indices || log) {
		uint count = array_len (edges) ;
		UndoLog undo_log = NULL ;
		EffectsBuffer *eb = NULL ;

		if (log) {
			eb = QueryCtx_GetEffectsBuffer () ;
			undo_log = QueryCtx_GetUndoLog () ;
		}

		for (uint i = 0; i < count; i++) {
			Edge *e = edges[i] ;
			ASSERT (e->relationID == r) ;
			ASSERT (e->attributes != NULL) ;

			if (has_indices) {
				Schema_AddEdgeToIndex (s, e) ;
			}

			// add edge creation operation to undo log
			if (log) {
				UndoLog_CreateEdge (undo_log, e) ;
				EffectsBuffer_AddCreateEdgeEffect (eb, e) ;
			}
		}
	}
}

// delete a node
// remove the node from the relevant indexes
// add node deletion operation to undo-log
// return 1 on success, 0 otherwise
void GraphHub_DeleteNodes
(
	GraphContext *gc,
	Node *nodes,
	uint n,
	bool log
) {
	ASSERT (gc    != NULL) ;
	ASSERT (nodes != NULL) ;

	bool has_indices = GraphContext_HasIndices (gc) ;

	Graph *g          = NULL ;
	UndoLog undo_log  = NULL ;
	EffectsBuffer *eb = NULL ;

	if (log) {
		g = QueryCtx_GetGraph () ;
		eb = QueryCtx_GetEffectsBuffer () ;
		undo_log = QueryCtx_GetUndoLog () ;
	}

	if (log || has_indices) {
		for (uint i = 0; i < n; i++) {
			Node *node = nodes + i ;

			if (log) {
				// add node deletion operation to undo log
				size_t label_count ;
				NODE_GET_LABELS (g, node, label_count) ;
				UndoLog_DeleteNode (undo_log, node, labels, label_count) ;
				EffectsBuffer_AddDeleteNodeEffect (eb, node) ;
			}

			if (has_indices) {
				GraphContext_DeleteNodeFromIndices (gc, node, NULL, 0) ;
			}
		}
	}

	Graph_DeleteNodes (gc->g, nodes, n) ;
}

void GraphHub_DeleteEdges
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
	bool has_indices = GraphContext_HasIndices(gc);

	UndoLog undo_log  = NULL ;
	EffectsBuffer *eb = NULL ;

	if (log) {
		eb = QueryCtx_GetEffectsBuffer() ;
		undo_log = QueryCtx_GetUndoLog() ;
	}

	if (log == true || has_indices == true) {
		for (uint i = 0; i < n; i++) {
			Edge *e = edges + i ;
			if (log == true) {
				UndoLog_DeleteEdge (undo_log, e) ;
				EffectsBuffer_AddDeleteEdgeEffect (eb, e) ;
			}

			if (has_indices == true) {
				GraphContext_DeleteEdgeFromIndices (gc, e) ;
			}
		}
	}

	Graph_DeleteEdges(gc->g, edges, n);
}

// updates a graph entity attribute set
void GraphHub_UpdateEntityProperties
(
	GraphContext *gc,             // graph context
	GraphEntity *ge,              // updated entity
	const AttributeSet set,       // new attributes
	GraphEntityType entity_type,  // entity type
	bool log                      // log update in undo-log
) {
	ASSERT (gc != NULL) ;
	ASSERT (ge != NULL) ;

	AttributeSet old_set = GraphEntity_GetAttributes (ge) ;

	if (log == true) {
		UndoLog log = QueryCtx_GetUndoLog () ;
		if (entity_type == GETYPE_NODE) {
			UndoLog_UpdateNode (log, (Node *)ge, old_set) ;
		} else {
			UndoLog_UpdateEdge (log, (Edge *)ge, old_set) ;
		}
	}

	*ge->attributes = set ;

	if (entity_type == GETYPE_NODE) {
		GraphContext_AddNodeToIndices (gc, (Node *)ge) ;
	} else {
		GraphContext_AddEdgeToIndices (gc, (Edge *)ge) ;
	}
}

void GraphHub_UpdateNodeProperty
(
	GraphContext *gc,     // graph context
	NodeID id,            // node ID
	AttributeID attr_id,  // attribute ID
	SIValue v             // new attribute value
) {
	ASSERT(gc      != NULL);
	ASSERT(id      != INVALID_ENTITY_ID);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	Node n;  // node to update
	int res = Graph_GetNode(gc->g, id, &n);
	ASSERT(res == true);  // make sure entity was found

	if(attr_id == ATTRIBUTE_ID_ALL) {
		AttributeSet_Free(n.attributes);
	} else {
		AttributeSet_Update (NULL, n.attributes, &attr_id, &v, 1, false) ;
	}

	// retrieve node labels
	uint label_count;
	NODE_GET_LABELS(gc->g, &n, label_count);

	Schema *s;
	for(uint i = 0; i < label_count; i++) {
		int label_id = labels[i];
		s = GraphContext_GetSchemaByID(gc, label_id, SCHEMA_NODE);
		ASSERT(s != NULL);

		if(attr_id == ATTRIBUTE_ID_ALL) {
			// remove node from all indices
			Schema_RemoveNodeFromIndex(s, &n);
		} else {
			// index node if updated attribute is indexed
			Index idx = Schema_GetIndex(s, &attr_id, 1, INDEX_FLD_ANY, true);
			if(idx) Schema_AddNodeToIndex(s, &n);
		}
	}
}

void GraphHub_UpdateEdgeProperty
(
	GraphContext *gc,     // graph context
	EdgeID id,            // edge ID
	RelationID r_id,      // relation ID
	NodeID src_id,        // source node ID
	NodeID dest_id,       // destination node ID
	AttributeID attr_id,  // attribute ID
	SIValue v             // new attribute value
) {
	ASSERT(gc      != NULL);
	ASSERT(id      != INVALID_ENTITY_ID);
	ASSERT(r_id    != GRAPH_NO_RELATION);
	ASSERT(src_id  != INVALID_ENTITY_ID);
	ASSERT(dest_id != INVALID_ENTITY_ID);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	Edge e; // edge to update

	// get src node, dest node and edge from the graph
	int res = Graph_GetEdge(gc->g, id, &e);
	ASSERT(res != 0);

	// set edge relation, src and destination node
	Edge_SetRelationID(&e, r_id);
	Edge_SetSrcNodeID(&e,  src_id);
	Edge_SetDestNodeID(&e, dest_id);

	// get edge schema
	Schema *s = GraphContext_GetSchemaByID(gc, r_id, SCHEMA_EDGE);
	ASSERT(s != NULL);

	// clear all attributes
	if(attr_id == ATTRIBUTE_ID_ALL) {
		AttributeSet_Free(e.attributes);

		// remove edge from index
		Schema_RemoveEdgeFromIndex(s, &e);
		return;
	}

	GraphEntity *ge = (GraphEntity *)&e;

	AttributeSetChangeType change ;
	AttributeSet_Update (&change, e.attributes, &attr_id, &v, 1, false) ;
	bool update_idx = (change != CT_NONE) ;

	// update index if
	// 1. attribute was set/updated
	// 2. attribute is indexed
	if(update_idx == true) {
		// see if attribute is indexed
		Index idx = Schema_GetIndex(s, &attr_id, 1, INDEX_FLD_ANY, true);
		if(idx) Schema_AddEdgeToIndex(s, &e);
	}
}

void GraphHub_UpdateNodeLabels
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
				s = GraphHub_AddSchema(gc, label, SCHEMA_NODE, log);
				schema_created = true;
			}

			int  schema_id = Schema_GetID(s);
			bool node_labeled = Graph_IsNodeLabeled(gc->g, ENTITY_GET_ID(node),
					schema_id);

			if(!node_labeled) {
				// sync matrix
				// make sure label matrix is of the right dimensions
				if(schema_created) {
					Delta_Matrix m = Graph_GetLabelMatrix(gc->g, schema_id);
				}
				// append label id
				add_labels_ids[add_labels_index++] = schema_id;
				// add to index
				Schema_AddNodeToIndex(s, node);
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
			Schema_RemoveNodeFromIndex(s, node);
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

Schema *GraphHub_AddSchema
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
		// we've decided against the removal of a schema incase a query fails
		// this is due to the way execution plans are constructed
		// a plan is built without acquiering R/W lock, as such
		// it is possible for a thread to rely on a schme which will soon be
		// removed leading to all sort of issues
		// i've decided to leave the following lines commented for the time being
		//
		//UndoLog undo_log = QueryCtx_GetUndoLog();
		//UndoLog_AddSchema(undo_log, s->id, s->type);
		EffectsBuffer *eb = QueryCtx_GetEffectsBuffer();
		EffectsBuffer_AddNewSchemaEffect(eb, Schema_GetName(s), s->type);
	}

	return s;
}

AttributeID GraphHub_FindOrAddAttribute
(
	GraphContext *gc,       // graph context to add the attribute
	const char *attribute,  // attribute name
	bool log                // should operation be logged in the undo-log
) {
	ASSERT(gc != NULL);
	ASSERT(attribute != NULL);

	bool created;
	AttributeID attr_id = GraphContext_FindOrAddAttribute(gc, attribute,
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

// create index
Index GraphHub_AddIndex
(
	const char *label,   // label/relationship type
	const char *attr,    // attribute to index
	GraphEntityType et,  // entity type (node/edge)
	IndexFieldType t,    // type of index (range/fulltext/vector)
	SIValue options,     // index options
	bool log
) {
	ASSERT(label != NULL);
	ASSERT(attr != NULL);
	ASSERT(et != GETYPE_UNKNOWN);
	ASSERT(t == INDEX_FLD_FULLTEXT ||
		   t == INDEX_FLD_RANGE    ||
		   t == INDEX_FLD_VECTOR);

	GraphContext *gc = QueryCtx_GetGraphCtx();

	//--------------------------------------------------------------------------
	// make sure schema exists
	//--------------------------------------------------------------------------

	SchemaType st = (et == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE;
	Schema *s = GraphContext_GetSchema(gc, label, st);

	// schema missing, creating an index will create the schema
	if(s == NULL) {
		s = GraphHub_AddSchema(gc, label, st, log);
	}
	ASSERT(s != NULL);

	//--------------------------------------------------------------------------
	// make sure attribute exists
	//--------------------------------------------------------------------------

	// creating an index will create the attribute
	AttributeID attr_id = GraphHub_FindOrAddAttribute(gc, attr, log);

	//--------------------------------------------------------------------------
	// create index field
	//--------------------------------------------------------------------------

	Index idx = NULL;
	if(t == INDEX_FLD_RANGE) {
		idx = Index_RangeCreate(label, et, attr, attr_id);
	} else if(t == INDEX_FLD_FULLTEXT) {
		idx = Index_FulltextCreate(label, et, attr, attr_id, options);
	} else if(t == INDEX_FLD_VECTOR) {
		idx = Index_VectorCreate(label, et, attr, attr_id, options);
	} else {
		assert(false && "unknown index type");
	}

	//--------------------------------------------------------------------------
	// add create index operation to undo log
	//--------------------------------------------------------------------------

	if(idx != NULL && log == true) {
		UndoLog log = QueryCtx_GetUndoLog();

		// extract label and field from index
		IndexField *fld = Index_GetField(NULL, idx, attr_id);
		const char *field_name = IndexField_GetName(fld);
		const char *lbl = Index_GetLabel(idx);

		// add index create undo operation
		UndoLog_CreateIndex(log, st, lbl, field_name, t);
	}

	// index operation is not replicated via effects
	// remove all index creation side effects:
	// 1. schema creation
	// 2. attribute creation
	// from effects buffer, forcing query replication of the index
	EffectsBuffer_Reset(QueryCtx_GetEffectsBuffer());

	return idx;
}

