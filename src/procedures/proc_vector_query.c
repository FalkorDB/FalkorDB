/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../graph/graph.h"
#include "../index/index.h"
#include "proc_vector_query.h"
#include "../datatypes/map.h"
#include "../datatypes/vector.h"
#include "../graph/graphcontext.h"
#include <string.h>

// KNN context
typedef struct {
	Node n;                   // retrieved node
	Edge e;                   // retrieved edge
	GraphEntityType t;        // entity type
	Graph *g;                 // graph
	RSIndex *idx;             // vector index
	RSResultsIterator *iter;  // iterator over query results
	SIValue output[2];        // yield array
	SIValue *yield_entity;    // yield node
	SIValue *yield_score;     // yield score
} VectorKNNCtx;

// create procedure private data
static VectorKNNCtx *_create_private_data
(
	GraphContext *gc,  // graph context
	RSIndex *idx,      // index
	RSQNode *root,     // RediSearch query
	GraphEntityType t  // entity type
) {
	VectorKNNCtx *ctx = (VectorKNNCtx*)rm_calloc(1, sizeof(VectorKNNCtx));

	ctx->t    = t;
	ctx->g    = gc->g;
	ctx->idx  = idx;
	ctx->iter = RediSearch_GetResultsIterator(root, idx);

	return ctx;
}

// process user's yield arguments
static void _process_yield
(
	VectorKNNCtx *ctx,
	const char **yield
) {
	ctx->yield_score  = NULL;
	ctx->yield_entity = NULL;

	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("entity", yield[i]) == 0) {
			ctx->yield_entity = ctx->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("score", yield[i]) == 0) {
			ctx->yield_score = ctx->output + idx;
			idx++;
			continue;
		}
	}
}

// extract KNN arguments from map
static bool _extractArgs
(
	const SIValue map,      // map holding KNN arguments
	int *k,                 // number of results to return
	GraphEntityType *type,  // type of entity to query
	char **label,           // entity label
	char **attribute,       // attribute to query
	SIValue *query_vector   // query vector
) {
	// expecting a map with the following structure:
	//
	// {
	//     type: 'NODE'/'RELATIONSHIP'
	//     label: 'Person'
	//     attribute: 'name'
	//     query_vector: vector32f([1,2])
	//     k:3
	// }

	if(Map_KeyCount(map) != 5) {
		return false;
	}

	SIValue v;  // current map argument

	// extract "type"
	if(!MAP_GET(map, "type", v) && SI_TYPE(v) != T_STRING) {
		return false;
	}
	if(strcasecmp(v.stringval, "node") == 0) {
		*type = GETYPE_NODE;
	} else if(strcasecmp(v.stringval, "relationship") == 0) {
		*type = GETYPE_EDGE;
	} else {
		return false;
	}

	// extract "label"
	if(!MAP_GET(map, "label", v) && SI_TYPE(v) != T_STRING) {
		return false;
	}
	*label = v.stringval;

	// extract "attribute"
	if(!MAP_GET(map, "attribute", v) && SI_TYPE(v) != T_STRING) {
		return false;
	}
	*attribute = v.stringval;

	// extract "query_vector"
	if(!MAP_GET(map, "query_vector", v) && SI_TYPE(v) != T_VECTOR32F) {
		return false;
	}
	*query_vector = v;

	// extract "k"
	if(!MAP_GET(map, "k", v) && SI_TYPE(v) != T_INT64) {
		return false;
	}
	*k = v.longval;

	return true;
}

// node iterator step function
static SIValue *Proc_NodeStep
(
	ProcedureCtx *ctx
) {
	VectorKNNCtx *pdata = (VectorKNNCtx *)ctx->privateData;

	ASSERT(pdata       != NULL);
	ASSERT(pdata->iter != NULL);

	// try to get a result out of the iterator
	// NULL is returned if iterator id depleted
	size_t len = 0;
	const NodeID *id = (NodeID*)RediSearch_ResultsIteratorNext(pdata->iter,
			pdata->idx, &len);

	// depleted
	if(!id) {
		return NULL;
	}

	// yield graph entity
	if(pdata->yield_entity) {
		bool res;
		// yield node
		// get graph entity
		Node *n = &pdata->n;
		res = Graph_GetNode(pdata->g, *(NodeID*)id, n);
		ASSERT(res == true);
		*pdata->yield_entity = SI_Node(n);
	}

	if(pdata->yield_score) {
		// get result score
		double score = RediSearch_ResultsIteratorGetScore(pdata->iter);
		*pdata->yield_score = SI_DoubleVal(score);
	}

	return pdata->output;
}

// edge iterator step function
static SIValue *Proc_EdgeStep
(
	ProcedureCtx *ctx
) {
	VectorKNNCtx *pdata = (VectorKNNCtx *)ctx->privateData;

	ASSERT(pdata       != NULL);
	ASSERT(pdata->iter != NULL);

	// try to get a result out of the iterator
	// NULL is returned if iterator id depleted
	size_t len = 0;
	const EdgeIndexKey *edge_key = RediSearch_ResultsIteratorNext(pdata->iter,
			pdata->idx, &len);

	// depleted
	if(!edge_key) {
		return NULL;
	}

	// yield graph entity
	if(pdata->yield_entity) {
		bool res;
		Edge *e = &pdata->e;
		EntityID edge_id = edge_key->edge_id;
		pdata->e.src_id  = edge_key->src_id;
		pdata->e.dest_id = edge_key->dest_id;
		res = Graph_GetEdge(pdata->g, edge_id, e);
		ASSERT(res == true);
		*pdata->yield_entity = SI_Edge(e);
	}

	if(pdata->yield_score) {
		// get result score
		double score = RediSearch_ResultsIteratorGetScore(pdata->iter);
		*pdata->yield_score = SI_DoubleVal(score);
	}

	return pdata->output;
}


// procedure invocation
// validate arguments and sets up internal context
ProcedureResult Proc_VectorKNNInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	const char **yield    // procedure output
) {
	// expecting a single map argument
	if(array_len((SIValue *)args) != 1 || SI_TYPE(args[0]) != T_MAP) {
		return PROCEDURE_ERR;
	}
	SIValue map = args[0];

	//--------------------------------------------------------------------------
	// extract procedure arguments from map
	//--------------------------------------------------------------------------

	int k;                 // number of results to return
	char *label;           // entity label
	char *attribute;       // attribute to query
	GraphEntityType et;    // entity type
	SIValue query_vector;  // query vector

	// extract arguments from map
	if(!_extractArgs(map,
					 &k,
					 &et,
					 &label,
					 &attribute,
					 &query_vector)) {
		//ErrorCtx_SetError("", err);
		return PROCEDURE_ERR;
	}

	GraphContext *gc = QueryCtx_GetGraphCtx();

	// depending on the entity type
	// set procedure step function and schema type
	SchemaType st;
	if(et == GETYPE_NODE) {
		st = SCHEMA_NODE;
		ctx->Step = Proc_NodeStep;
	} else  {
		st = SCHEMA_EDGE;
		ctx->Step = Proc_EdgeStep;
	}

	//--------------------------------------------------------------------------
	// make sure there's a vector index exists
	//--------------------------------------------------------------------------

	// get attribute ID
	Attribute_ID attr_id = GraphContext_GetAttributeID(gc, attribute);
	if(attr_id == ATTRIBUTE_ID_NONE) {
		//ErrorCtx_SetError("", err);
		return PROCEDURE_ERR;
	}

	Index idx = GraphContext_GetIndex(gc, label, &attr_id, 1, IDX_EXACT_MATCH,
			st);

	if(idx == NULL) {
		return PROCEDURE_ERR;
	}

	// make sure we're dealing with a vector index
	if(!(Index_GetFieldType(idx, attr_id) & INDEX_FLD_VECTOR)) {
		return PROCEDURE_ERR;
	}

	RSIndex *rsIdx = Index_RSIndex(idx);

	//--------------------------------------------------------------------------
	// construct a vector query
	//--------------------------------------------------------------------------

	// create a query vector
	float *vec = SIVector_Elements(query_vector);

	// create a redisearch query node
	RSQNode *root = RediSearch_CreateVecSimNode(rsIdx, attribute, (char*)vec,
			sizeof(float) * 2, k);

	// create procedure private data
	ctx->privateData = _create_private_data(gc, rsIdx, root, et);

	// process yield
	_process_yield((VectorKNNCtx*)ctx->privateData, yield);

	return PROCEDURE_OK;
}

// free procedure private data
ProcedureResult Proc_VectorKNNFree
(
	ProcedureCtx *ctx  // procedure context
) {
	// no private data, nothing to do
	if(!ctx->privateData) return PROCEDURE_OK;

	// free private data
	VectorKNNCtx *pdata = (VectorKNNCtx *)ctx->privateData;

	// free index iterator
	if(pdata->iter) {
		RediSearch_ResultsIteratorFree(pdata->iter);
	}

	rm_free(pdata);

	return PROCEDURE_OK;
}

// HNSW KNN procedure
//
// usage:
//
// CALL db.idx.vector.knn( {
// type: 'NODE'/'RELATIONSHIP',
// label: 'Person',
// attribute: 'name',
// query_vector: vector32f([1,2]),
// k:3 } ) YIELD entity

ProcedureCtx *Proc_VectorKNNGen() {
	ProcedureOutput *output    = array_new(ProcedureOutput, 1);
	ProcedureOutput out_entity = {.name = "entity", .type = SI_GRAPHENTITY};
	ProcedureOutput out_score  = {.name = "score", .type = T_DOUBLE};
	array_append(output, out_entity);
	array_append(output, out_score);

	ProcedureCtx *ctx = ProcCtxNew("db.idx.vector.knn",
								   1,
								   output,
								   NULL, // step func is determined by invoke
								   Proc_VectorKNNInvoke,
								   Proc_VectorKNNFree,
								   NULL,
								   true);
	return ctx;
}

