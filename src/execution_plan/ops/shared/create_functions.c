/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "create_functions.h"
#include "RG.h"
#include "../../../query_ctx.h"
#include "../../../errors/errors.h"
#include "../../../ast/ast_shared.h"
#include "../../../datatypes/array.h"
#include "../../../graph/graph_hub.h"

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
	uint blueprint_node_count = array_len(pending->nodes.nodes_to_create);
	for(uint i = 0; i < blueprint_node_count; i++) {
		NodeCreateCtx *node_ctx = pending->nodes.nodes_to_create + i;
		uint label_count = array_len(node_ctx->labels);

		for(uint j = 0; j < label_count; j++) {
			const char *label = node_ctx->labels[j];
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);

			if(s == NULL) {
				s = GraphHub_AddSchema(gc, label, SCHEMA_NODE, true);
				QueryCtx_GetResultSetStatistics()->labels_added++;
			}

			node_ctx->labelsId[j] = s->id;
			pending->nodes.node_labels[i][j] = s->id;

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
	Node         *n                   = NULL ;
	GraphContext *gc                  = QueryCtx_GetGraphCtx () ;
	Graph        *g                   = gc->g ;
	uint         node_count           = array_len (pending->nodes.created_nodes) ;
	bool         constraint_violation = false;

	// sync policy should be set to NOP, no need to sync/resize
	ASSERT (Graph_GetMatrixPolicy (g) == SYNC_POLICY_NOP) ;

	for (uint i = 0; i < node_count; i++) {
		n = pending->nodes.created_nodes[i] ;

		AttributeSet attr        = pending->nodes.node_attributes[i] ;
		LabelID     *labels      = pending->nodes.node_labels[i] ;
		uint         label_count = array_len (labels) ;

		// introduce node into graph
		GraphHub_CreateNode (gc, n, labels, label_count, attr, true) ;

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
	PendingCreations *pending
) {
	GraphContext *gc = QueryCtx_GetGraphCtx();
	Graph *g = gc->g;

	// sync policy should be set to resize to capacity, no need to sync
	ASSERT(Graph_GetMatrixPolicy(g) == SYNC_POLICY_RESIZE);

	// create missing schemas
	// this loop iterates over the CREATE pattern, e.g.
	// CREATE (p:Person)-[e:VISITED]->(q)
	// as such we're not expecting a large number of iterations
	uint blueprint_edge_count = array_len(pending->edges);
	for(uint i = 0; i < blueprint_edge_count; i++) {
		EdgeCreateCtx *edge_ctx = &pending->edges[i].edges_to_create;

		const char *relation = edge_ctx->relation;
		Schema *s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
		if(s == NULL) {
			s = GraphHub_AddSchema(gc, relation, SCHEMA_EDGE, true);
		}

		// calling Graph_GetRelationMatrix will make sure relationship matrix
		// is of the right dimensions
		Graph_GetRelationMatrix(g, Schema_GetID(s), false);
	}

	// call Graph_GetAdjacencyMatrix will make sure the adjacency matrix
	// is of the right dimensions
	Graph_GetAdjacencyMatrix(g, false);
}

// commit edges
static void _CommitEdges
(
	PendingCreations *pending
) {
	Edge         *e                   = NULL;
	GraphContext *gc                  = QueryCtx_GetGraphCtx();
	Graph        *g                   = gc->g;
	bool         constraint_violation = false;

	// sync policy should be set to NOP, no need to sync/resize
	ASSERT (Graph_GetMatrixPolicy (g) == SYNC_POLICY_NOP) ;

	// process batches
	uint count = array_len (pending->edges) ;
	for (uint i = 0; i < count; i++) {
		// processing current batch
		// get batched edges along with their corresponding attribute sets
		PendingEdgeCreations *pending_edge = pending->edges + i ;

		// all edges in a batch share the same relationship-type
		Schema *s = GraphContext_GetSchema (gc,
				pending_edge->edges_to_create.relation, SCHEMA_EDGE) ;

		// all schemas have been created in the edge blueprint loop or earlier
		ASSERT (s != NULL) ;

		int relation_id = Schema_GetID (s) ;
		Edge        **edges     = pending_edge->created_edges ;
		AttributeSet *attrs     = pending_edge->edge_attributes ;
		uint         edge_count = array_len (edges) ;

		// introduce all edges to the graph at once
		GraphHub_CreateEdges (gc, edges, relation_id, attrs, true) ;

		// no constraints, quick return
		if (!Schema_HasConstraints (s)) {
			continue ;
		}

		for (uint j = 0; j < edge_count; j++) {
			e = edges[j] ;

			//------------------------------------------------------------------
			// enforce constraints
			//------------------------------------------------------------------

			if (constraint_violation == false) {
				char *err_msg = NULL ;
				if (!Schema_EnforceConstraints (s, (GraphEntity*)e, &err_msg)) {
					// constraint violated!
					ASSERT (err_msg != NULL) ;
					constraint_violation = true ;
					ErrorCtx_SetError ("%s", err_msg) ;
					free(err_msg) ;
				}
			}
		}
	}
}

// initialize all variables for storing pending creations
void NewPendingCreationsContainer
(
	PendingCreations *pending,
	NodeCreateCtx *nodes,
	EdgeCreateCtx *edges
) {
	ASSERT(pending != NULL);

	pending->nodes.nodes_to_create = nodes;
	pending->nodes.node_labels     = array_new(LabelID *, 0);
	pending->nodes.created_nodes   = array_new(Node *, 0);
	pending->nodes.node_attributes = array_new(AttributeSet, 0);

	pending->edges = array_new(PendingEdgeCreations, array_len(edges));

	// create containers for each edge in the CREATE pattern
	// for example CREATE ()-[a:A]->()-[b:B]->()
	// we'll create two containers, one for all 'a' edges and their attributes and another one for the 'b' edges
	for(uint i = 0; i < array_len(edges); i++) {
		array_append(pending->edges, ((PendingEdgeCreations) {
			.edges_to_create = edges[i],
			.edge_attributes = array_new(AttributeSet, 0),
			.created_edges = array_new(Edge *, 0)
		}));
	}

	array_free(edges);
}

void PendingCreations_Reset
(
	PendingCreations *ctx
) {
	ASSERT (ctx != NULL) ;

	// clear nodes pending changes
	if (ctx->nodes.node_labels != NULL) {
		array_clear (ctx->nodes.node_labels) ;
	}

	if (ctx->nodes.created_nodes != NULL) {
		array_clear (ctx->nodes.created_nodes) ;
	}

	if (ctx->nodes.node_attributes != NULL) {
		uint n = array_len (ctx->nodes.node_attributes) ;
		for (uint i = 0; i < n; i++) {
			AttributeSet_Free (ctx->nodes.node_attributes + i) ;
		}
		array_clear (ctx->nodes.node_attributes) ;
	}

	for(uint i = 0; i < array_len(ctx->edges); i++) {
		array_clear (ctx->edges[i].created_edges) ;
		array_clear (ctx->edges[i].edge_attributes) ;
	}
}

// lock the graph and commit all pending changes
void CommitNewEntities
(
	PendingCreations *pending
) {
	Graph *g = QueryCtx_GetGraph();
	uint node_count = array_len(pending->nodes.created_nodes);
	// count number of edges created
	// edges are distributed among multiple containers, one for each edge in the CREATE/MERGE pattern
	uint edge_count = 0;
	for(uint i = 0; i < array_len(pending->edges); i++) {
		edge_count += array_len(pending->edges[i].created_edges);
	}

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
		array_clear(pending->nodes.node_attributes);

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
		_CommitEdgesBlueprint(pending);

		// set graph matrix sync policy to NOP
		// no need to perform sync/resize
		Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);
		_CommitEdges(pending);

		// clear pending attributes array
		for(uint i = 0; i < array_len(pending->edges); i++) {
			array_clear (pending->edges[i].edge_attributes) ;
		}

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
	uint property_count = array_len (map->keys) ;
	uint attrs_count    = 0;

	SIValue     vals[property_count] ;
	AttributeID ids [property_count] ;

	for (int i = 0; i < property_count; i++) {
		// note that AR_EXP_Evaluate may raise a run-time exception
		// in which case the allocations in this function will leak
		// for example, this occurs in the query:
		// CREATE (a {val: 2}), (b {val: a.val})
		SIValue val = AR_EXP_Evaluate (map->values[i], r) ;

		if (!(SI_TYPE(val) & SI_VALID_PROPERTY_VALUE)) {
			// this value is of an invalid type
			if (!SIValue_IsNull(val)) {
				// if the value was a complex type, emit an exception
				SIValue_Free (val) ;
				for (int j = 0; j < i; j++) {
					SIValue_Free (vals[j]) ;
				}

				Error_InvalidPropertyValue () ;
				ErrorCtx_RaiseRuntimeException (NULL) ;
			}

			// the value was NULL
			// if this was prohibited in this context, raise an exception,
			// otherwise skip this value
			if (fail_on_null) {
				// emit an error and exit
				for (int j = 0; j < i; j++) {
					SIValue_Free (vals[j]) ;
				}

				ErrorCtx_RaiseRuntimeException ("Cannot merge node using null property value") ;
			}

			// don't add null to attrribute set
			continue ;
		}

		// emit an error and exit if we're trying to add
		// an array containing an invalid type
		if (unlikely (SI_TYPE(val) == T_ARRAY)) {
			SIType invalid_properties = ~SI_VALID_PROPERTY_VALUE ;
			bool res = SIArray_ContainsType (val, invalid_properties) ;
			if (res) {
				// validation failed
				SIValue_Free (val) ;
				for (int j = 0; j < i; j++) {
					SIValue_Free (vals[j]) ;
				}

				Error_InvalidPropertyValue () ;
				ErrorCtx_RaiseRuntimeException (NULL) ;
			}
		}

		// set the converted attribute
		ids [attrs_count] = map->attr_ids[i] ;
		vals[attrs_count] = SI_CloneValue (val) ;
		attrs_count++ ;
		SIValue_Free (val) ;
	}

	// batch add
	AttributeSet_Add (attributes, ids, vals, attrs_count, false) ;
}

// free all data associated with a completed create operation
void PendingCreationsFree
(
	PendingCreations *pending
) {
	if(pending->nodes.nodes_to_create) {
		uint nodes_to_create_count = array_len(pending->nodes.nodes_to_create);
		for(uint i = 0; i < nodes_to_create_count; i++) {
			NodeCreateCtx_Free(pending->nodes.nodes_to_create[i]);
		}
		array_free(pending->nodes.nodes_to_create);
		pending->nodes.nodes_to_create = NULL;
	}

	if(pending->nodes.node_attributes) {
		uint prop_count = array_len(pending->nodes.node_attributes);
		for(uint i = 0; i < prop_count; i++) {
			AttributeSet_Free(pending->nodes.node_attributes + i);
		}
		array_free(pending->nodes.node_attributes);
		pending->nodes.node_attributes = NULL;
	}

	if(pending->nodes.node_labels) {
		array_free(pending->nodes.node_labels);
		pending->nodes.node_labels = NULL;
	}

	if(pending->nodes.created_nodes) {
		array_free(pending->nodes.created_nodes);
		pending->nodes.created_nodes = NULL;
	}

	if(pending->edges) {
		uint edges_count = array_len(pending->edges);
		for(uint i = 0; i < edges_count; i++) {
			PendingEdgeCreations *pending_edge = pending->edges + i;
			PropertyMap_Free(pending_edge->edges_to_create.properties);
			if(pending_edge->edge_attributes) {
				uint prop_count = array_len(pending_edge->edge_attributes);
				for(uint j = 0; j < prop_count; j ++) {
					AttributeSet_Free(pending_edge->edge_attributes + j);
				}
				array_free(pending_edge->edge_attributes);
				pending_edge->edge_attributes = NULL;
			}

			if(pending_edge->created_edges) {
				array_free(pending_edge->created_edges);
				pending_edge->created_edges = NULL;
			}
		}
		array_free(pending->edges);
		pending->edges = NULL;
	}
}

