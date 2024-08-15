/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "create_functions.h"

#include "util.h"

#include "RG.h"
#include "../../../util/dict.h"
#include "../../../query_ctx.h"
#include "../../../errors/errors.h"
#include "../../../ast/ast_shared.h"
#include "../../../datatypes/array.h"
#include "../../../graph/graph_hub.h"


static uint64_t _stringHashFunc(const void *str) {
	const char* cast = str;

	unsigned long hash = 5381;
	int c;

	while ((c = *cast++)) {
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	}

	return hash;
}

static int _stringCompare(dict*, const void *key1, const void *key2) {
	return !memcmp(key1, key2, minu(sizeof(key1), sizeof(key2)));
}

dictType str_key_dt = {
	.hashFunction = _stringHashFunc,
	NULL,
	NULL,
	.keyCompare = _stringCompare,
	NULL,
	NULL
};

// commit node blueprints
static void _CommitNodesBlueprint
(
	PendingCreations *pending
) {
	GraphContext *gc = QueryCtx_GetGraphCtx();
	Graph *g = gc->g;

	// sync policy should be set to resize to capacity, no need to sync
	ASSERT(Graph_GetMatrixPolicy(g) == SYNC_POLICY_RESIZE);

	// create missing schemas
	// this loop iterates over the CREATE pattern, e.g.
	// CREATE (p:Person)
	// as such we're not expecting a large number of iterations
	uint blueprint_node_count = array_len(pending->nodes_to_create);
	for(uint i = 0; i < blueprint_node_count; i++) {
		NodeCreateCtx *node_ctx = pending->nodes_to_create + i;
		uint label_count = array_len(node_ctx->labels);

		for(uint j = 0; j < label_count; j++) {
			const char *label = node_ctx->labels[j];
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);

			if(s == NULL) {
				s = AddSchema(gc, label, SCHEMA_NODE, true);
				QueryCtx_GetResultSetStatistics()->labels_added++;
			}

			node_ctx->labelsId[j] = s->id;
			pending->node_labels[i][j] = s->id;

			// sync matrix, make sure label matrix is of the right dimensions
			Graph_GetLabelMatrix(g, Schema_GetID(s));
		}
		// sync matrix, make sure mapping matrix is of the right dimensions
		if(label_count > 0) Graph_GetNodeLabelMatrix(g);
	}
}

// commit nodes
static void _CommitNodes
(
	PendingCreations *pending
) {
	Node         *n                   = NULL;
	GraphContext *gc                  = QueryCtx_GetGraphCtx();
	Graph        *g                   = gc->g;
	uint         node_count           = array_len(pending->created_nodes);
	bool         constraint_violation = false;

	// sync policy should be set to NOP, no need to sync/resize
	ASSERT(Graph_GetMatrixPolicy(g) == SYNC_POLICY_NOP);

	for(int i = 0; i < node_count; i++) {
		n = pending->created_nodes[i];

		AttributeSet attr        = pending->node_attributes[i];
		int*         labels      = pending->node_labels[i];
		uint         label_count = array_len(labels);

		// introduce node into graph
		CreateNode(gc, n, labels, label_count, attr, true);

		//----------------------------------------------------------------------
		// enforce constraints
		//----------------------------------------------------------------------

		if(constraint_violation == false) {
			for(uint j = 0; j < label_count; j++) {
				Schema *s = GraphContext_GetSchemaByID(gc, labels[j], SCHEMA_NODE);
				char *err_msg = NULL;
				if(!Schema_EnforceConstraints(s, (GraphEntity*)n, &err_msg)) {
					// constraint violation
					ASSERT(err_msg != NULL);
					constraint_violation = true;
					ErrorCtx_SetError("%s", err_msg);
					free(err_msg);
					break;
				}
			}
		}
	}
}

// commit edge blueprints
static void _CommitEdgesBlueprint
(
	EdgeCreateCtx *blueprints
) {
	GraphContext *gc = QueryCtx_GetGraphCtx();
	Graph *g = gc->g;

	// sync policy should be set to resize to capacity, no need to sync
	ASSERT(Graph_GetMatrixPolicy(g) == SYNC_POLICY_RESIZE);

	// create missing schemas
	// this loop iterates over the CREATE pattern, e.g.
	// CREATE (p:Person)-[e:VISITED]->(q)
	// As such we're not expecting a large number of iterations
	uint blueprint_edge_count = array_len(blueprints);
	for(uint i = 0; i < blueprint_edge_count; i++) {
		EdgeCreateCtx *edge_ctx = blueprints + i;

		const char *relation = edge_ctx->relation;
		Schema *s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
		if(s == NULL) s = AddSchema(gc, relation, SCHEMA_EDGE, true);

		// calling Graph_GetRelationMatrix will make sure relationship matrix
		// is of the right dimensions
		Graph_GetRelationMatrix(g, Schema_GetID(s), false);
		Graph_GetMultiEdgeRelationMatrix(g, Schema_GetID(s));
	}

	// call Graph_GetAdjacencyMatrix will make sure the adjacency matrix
	// is of the right dimensions
	Graph_GetAdjacencyMatrix(g, false);
}

static dict* _CreateMultiEdgeDict(GraphContext* gc, const PendingCreations* pending, const size_t edge_count)
{
	Graph* g = gc->g;
	ASSERT(g != NULL);

	dict* multiEdgeCreationCtx = HashTableCreate(&str_key_dt);
	for(size_t i = 0; i < edge_count; i++) {
		Edge* edge = pending->created_edges[i];

		const Schema *schema = GraphContext_GetSchema(gc, edge->relationship, SCHEMA_EDGE);
		// all schemas have been created in the edge blueprint loop or earlier
		ASSERT(schema != NULL);

		edge->relationID = Schema_GetID(schema);
		const NodeID src_id  = Edge_GetSrcNodeID(edge);
		const NodeID dest_id = Edge_GetDestNodeID(edge);

		char* key;
		asprintf(&key, "%i_%lu_%lu", edge->relationID, src_id, dest_id);
		struct MultiEdgeCreationCtx* entry = HashTableFetchValue(multiEdgeCreationCtx, key);
		if (entry)
		{
			array_append(entry->edges_to_add, edge);
		} else
		{
			entry = rm_malloc(sizeof(struct MultiEdgeCreationCtx));
			entry->M = g->relations + edge->relationID;
			entry->relation_id = edge->relationID;
			entry->current_value = -1;
			entry->is_me = false;
			entry->src = src_id;
			entry->dest = dest_id;
			entry->creation_idx = i;
			entry->edges_to_add = array_new(Edge*, 1);
			array_append(entry->edges_to_add, edge);

			const int ret = HashTableAdd(multiEdgeCreationCtx, key, entry);
			ASSERT(ret == DICT_OK);
		}

		edge->attributes = Graph_AllocateAttribute(g, src_id, dest_id, edge->relationID, &edge->id);
		*edge->attributes = pending->edge_attributes[i];

		Schema_AddEdgeToIndex(schema, edge);
	}

	return multiEdgeCreationCtx;
}

// commit edges
static void _CommitEdges
(
	const PendingCreations *pending
) {
	GraphContext *gc                  = QueryCtx_GetGraphCtx();
	Graph *g                    = gc->g;

	// sync policy should be set to NOP, no need to sync/resize
	ASSERT(Graph_GetMatrixPolicy(g) == SYNC_POLICY_NOP);

	const uint edge_count             = array_len(pending->created_edges);
	dict* multiEdgeCreationCtx = _CreateMultiEdgeDict(gc, pending, edge_count);

	Graph_FormConnections(g, multiEdgeCreationCtx);

	// Individual members have to be release prior to releasing the hash table
	HashTableRelease(multiEdgeCreationCtx);

	//----------------------------------------------------------------------
	// enforce constraints
	//----------------------------------------------------------------------
	bool constraint_violation = false;
	for (size_t i = 0; i < edge_count; i++) {
		Edge* edge = pending->created_edges[i];

		UndoLog_CreateEdge(QueryCtx_GetUndoLog(), edge);
		EffectsBuffer_AddCreateEdgeEffect(QueryCtx_GetEffectsBuffer(), edge);

		if (!constraint_violation)
		{
			const Schema* schema = GraphContext_GetSchema(gc, edge->relationship, SCHEMA_EDGE);

			ASSERT(schema != NULL);
			char *err_msg = NULL;
			if (!Schema_EnforceConstraints(schema, (GraphEntity*) edge, &err_msg))
			{
				// constraint violated!
				ASSERT(err_msg != NULL);
				constraint_violation = true;
				ErrorCtx_SetError("%s", err_msg);
				free(err_msg);
			}
		}

	}
}

// Initialize all variables for storing pending creations.
void NewPendingCreationsContainer
(
	PendingCreations *pending,
	NodeCreateCtx *nodes,
	EdgeCreateCtx *edges
) {
	ASSERT(pending != NULL);

	pending->nodes_to_create = nodes;
	pending->edges_to_create = edges;
	pending->node_labels     = array_new(int *, 0);
	pending->created_nodes   = array_new(Node *, 0);
	pending->created_edges   = array_new(Edge *, 0);
	pending->node_attributes = array_new(AttributeSet, 0);
	pending->edge_attributes = array_new(AttributeSet, 0);
}

// Lock the graph and commit all changes introduced by the operation.
void CommitNewEntities
(
	OpBase *op,
	PendingCreations *pending
) {
	Graph *g = QueryCtx_GetGraph();
	uint node_count = array_len(pending->created_nodes);
	uint edge_count = array_len(pending->created_edges);

	// lock everything
	QueryCtx_LockForCommit();

	//--------------------------------------------------------------------------
	// commit nodes
	//--------------------------------------------------------------------------

	if(node_count > 0) {
		Graph_AllocateNodes(g, node_count);

		// set graph matrix sync policy to resize
		// no need to perform sync
		Graph_SetMatrixPolicy(g, SYNC_POLICY_RESIZE);
		_CommitNodesBlueprint(pending);

		// set graph matrix sync policy to NOP
		// no need to perform sync/resize
		Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);
		_CommitNodes(pending);

		// clear pending attributes array
		array_clear(pending->node_attributes);

		if(unlikely(ErrorCtx_EncounteredError())) {
			goto cleanup;
		}
	}

	//--------------------------------------------------------------------------
	// commit edges
	//--------------------------------------------------------------------------

	if(edge_count > 0) {
		Graph_AllocateEdges(g, edge_count);

		// set graph matrix sync policy to resize
		// no need to perform sync
		Graph_SetMatrixPolicy(g, SYNC_POLICY_RESIZE);
		_CommitEdgesBlueprint(pending->edges_to_create);

		// set graph matrix sync policy to NOP
		// no need to perform sync/resize
		Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);
		_CommitEdges(pending);

		// clear pending attributes array
		array_clear(pending->edge_attributes);

		if(unlikely(ErrorCtx_EncounteredError())) {
			goto cleanup;
		}
	}

cleanup:

	// restore matrix sync policy to default
	Graph_SetMatrixPolicy(g, SYNC_POLICY_FLUSH_RESIZE);
}

// resolve the properties specified in the query into constant values
void ConvertPropertyMap
(
	GraphContext* gc,
	AttributeSet *attributes,
	Record r,
	PropertyMap *map,
	bool fail_on_null
) {
	uint property_count = array_len(map->keys);
	SIValue vals[property_count];
	AttributeID ids[property_count];
	uint attrs_count = 0;
	for(int i = 0; i < property_count; i++) {
		// note that AR_EXP_Evaluate may raise a run-time exception
		// in which case the allocations in this function will leak
		// for example, this occurs in the query:
		// CREATE (a {val: 2}), (b {val: a.val})
		SIValue val = AR_EXP_Evaluate(map->values[i], r);
		if(!(SI_TYPE(val) & SI_VALID_PROPERTY_VALUE)) {
			// this value is of an invalid type
			if(!SIValue_IsNull(val)) {
				// if the value was a complex type, emit an exception
				SIValue_Free(val);
				for(int j = 0; j < i; j++) {
					SIValue_Free(vals[j]);
				}
				Error_InvalidPropertyValue();
				ErrorCtx_RaiseRuntimeException(NULL);
			}
			// the value was NULL
			// if this was prohibited in this context, raise an exception,
			// otherwise skip this value
			if(fail_on_null) {
				// emit an error and exit
				for(int j = 0; j < i; j++) {
					SIValue_Free(vals[j]);
				}
				ErrorCtx_RaiseRuntimeException("Cannot merge node using null property value");
			}

			// don't add null to attrribute set
			continue;
		}

		// emit an error and exit if we're trying to add
		// an array containing an invalid type
		if(SI_TYPE(val) == T_ARRAY) {
			SIType invalid_properties = ~SI_VALID_PROPERTY_VALUE;
			bool res = SIArray_ContainsType(val, invalid_properties);
			if(res) {
				// validation failed
				SIValue_Free(val);
				for(int j = 0; j < i; j++) {
					SIValue_Free(vals[j]);
				}
				Error_InvalidPropertyValue();
				ErrorCtx_RaiseRuntimeException(NULL);
			}
		}

		// set the converted attribute
		ids[attrs_count] = FindOrAddAttribute(gc, map->keys[i], true);
		vals[attrs_count++] = SI_CloneValue(val);
		SIValue_Free(val);
	}
	AttributeSet_AddNoClone(attributes, ids, vals, attrs_count, false);
}

// free all data associated with a completed create operation
void PendingCreationsFree
(
	PendingCreations *pending
) {
	if(pending->nodes_to_create) {
		uint nodes_to_create_count = array_len(pending->nodes_to_create);
		for(uint i = 0; i < nodes_to_create_count; i ++) {
			NodeCreateCtx_Free(pending->nodes_to_create[i]);
		}
		array_free(pending->nodes_to_create);
		pending->nodes_to_create = NULL;
	}

	if(pending->edges_to_create) {
		uint edges_to_create_count = array_len(pending->edges_to_create);
		for(uint i = 0; i < edges_to_create_count; i ++) {
			PropertyMap_Free(pending->edges_to_create[i].properties);
		}
		array_free(pending->edges_to_create);
		pending->edges_to_create = NULL;
	}

	if(pending->node_labels) {
		array_free(pending->node_labels);
		pending->node_labels = NULL;
	}

	if(pending->created_nodes) {
		array_free(pending->created_nodes);
		pending->created_nodes = NULL;
	}

	if(pending->created_edges) {
		array_free(pending->created_edges);
		pending->created_edges = NULL;
	}

	if(pending->node_attributes) {
		uint prop_count = array_len(pending->node_attributes);
		for(uint i = 0; i < prop_count; i ++) {
			AttributeSet_Free(pending->node_attributes + i);
		}
		array_free(pending->node_attributes);
		pending->node_attributes = NULL;
	}

	if(pending->edge_attributes) {
		uint prop_count = array_len(pending->edge_attributes);
 		for(uint i = 0; i < prop_count; i ++) {
 			AttributeSet_Free(pending->edge_attributes + i);
 		}
		array_free(pending->edge_attributes);
		pending->edge_attributes = NULL;
	}
}

