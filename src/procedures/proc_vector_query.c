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

typedef float (*distance_fp_t)
(
	SIValue a,  // first vector
	SIValue b   // second vector
);

// KNN context
typedef struct {
	Node n;                     // retrieved node
	Edge e;                     // retrieved edge
	GraphEntityType t;          // entity type
	Graph *g;                   // graph
	RSIndex *idx;               // vector index
	RSResultsIterator *iter;    // iterator over query results
	SIValue q;                  // query vector
	AttributeID attr_id;        // vector attribute ID
	distance_fp_t distance_fp;  // similarity function
	SIValue output[2];          // yield array
	SIValue *yield_entity;      // yield node
	SIValue *yield_score;       // yield score
} VectorKNNCtx;

// create procedure private data
static VectorKNNCtx *_create_private_data
(
	GraphContext *gc,      // graph context
	SIValue q,             // query vector
	AttributeID attr_id,   // vector attribute ID
	RSIndex *idx,          // index
	RSQNode *root,         // RediSearch query
	GraphEntityType t,     // entity type
	VecSimMetric sim_func  // similarity function
) {
	VectorKNNCtx *ctx = (VectorKNNCtx*)rm_calloc(1, sizeof(VectorKNNCtx));

	ctx->t           = t;
	ctx->q           = q;
	ctx->g           = gc->g;
	ctx->idx         = idx;
	ctx->iter        = RediSearch_GetResultsIterator(root, idx);
	ctx->attr_id     = attr_id;
	ctx->distance_fp = sim_func == VecSimMetric_L2
			? SIVector_EuclideanDistance
			: SIVector_CosineDistance;

	ASSERT(ctx->iter != NULL);

	return ctx;
}

// extract KNN arguments from args
static bool _extractArgs
(
	const SIValue *args,    // procedure arguments
	int *k,                 // number of results to return
	char **label,           // entity label
	char **attribute,       // attribute to query
	SIValue *query_vector   // query vector
) {
	// expecting four arguments
	//     arg[0] - label: 'Person'
	//     arg[1] - attribute: 'name'
	//     arg[2] - k:3
	//     arg[3] - query: vector32f([1,2])

	uint n = array_len((SIValue*)args);
	ASSERT(n == 4);

	SIValue v;  // current argument

	// extract "label"
	v = args[0];
	if(!(SI_TYPE(v) & T_STRING)) {
		return false;
	}
	*label = v.stringval;

	// extract "attribute"
	v = args[1];
	if(!(SI_TYPE(v) & T_STRING)) {
		return false;
	}
	*attribute = v.stringval;

	// extract "k"
	v = args[2];
	if(SI_TYPE(v) != T_INT64 || v.longval <= 0) {
		return false;
	}
	*k = v.longval;

	// extract "query"
	v = args[3];
	if(SI_TYPE(v) != T_VECTOR_F32) {
		return false;
	}
	*query_vector = v;

	return true;
}

// node iterator step function
static SIValue *Proc_NodeStep
(
	ProcedureCtx *ctx
) {
	VectorKNNCtx *pdata = (VectorKNNCtx *)ctx->privateData ;

	ASSERT (pdata != NULL) ;

	// try to get a result out of the iterator
	// NULL is returned if iterator id depleted
	size_t len = 0 ;
	const NodeID *id = (NodeID*)RediSearch_ResultsIteratorNext (pdata->iter,
			pdata->idx, &len) ;

	// depleted
	if (!id) {
		return NULL ;
	}

	Node *n  = &pdata->n ;
	bool res = Graph_GetNode (pdata->g, *(NodeID*)id, n) ;
	ASSERT (res == true) ;

	// yield graph entity
	if (pdata->yield_entity) {
		*pdata->yield_entity = SI_Node (n) ;
	}

	if (pdata->yield_score) {
		SIValue v ;
		bool found =
			GraphEntity_GetProperty((GraphEntity*)n, pdata->attr_id, &v);
		ASSERT (found) ;

		SIValue distance = SI_DoubleVal(pdata->distance_fp(pdata->q, v));
		*pdata->yield_score = distance;
	}

	return pdata->output ;
}

// edge iterator step function
static SIValue *Proc_EdgeStep
(
	ProcedureCtx *ctx
) {
	VectorKNNCtx *pdata = (VectorKNNCtx *)ctx->privateData;

	ASSERT(pdata != NULL);

	// try to get a result out of the iterator
	// NULL is returned if iterator id depleted
	size_t len = 0;
	const EdgeIndexKey *edge_key = RediSearch_ResultsIteratorNext(pdata->iter,
			pdata->idx, &len);

	// depleted
	if(!edge_key) {
		return NULL;
	}

	Edge *e = &pdata->e;
	pdata->e.src_id  = edge_key->src_id;
	pdata->e.dest_id = edge_key->dest_id;
	EntityID edge_id = edge_key->edge_id;

	bool res = Graph_GetEdge(pdata->g, edge_id, e);
	ASSERT(res == true);

	// yield graph entity
	if(pdata->yield_entity) {
		*pdata->yield_entity = SI_Edge(e);
	}

	if(pdata->yield_score) {
		SIValue v ;
		bool found =
			GraphEntity_GetProperty ((GraphEntity*)e, pdata->attr_id, &v) ;
		ASSERT (found) ;

		SIValue distance = SI_DoubleVal(pdata->distance_fp(pdata->q, v));
		*pdata->yield_score = distance;
	}

	return pdata->output ;
}

// procedure invocation
// validate arguments and sets up internal context
static ProcedureResult Proc_VectorQueryInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	GraphEntityType et
) {
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// validate args
	uint argc = array_len((SIValue*)args);
	if(argc != 4) return PROCEDURE_ERR;

	//--------------------------------------------------------------------------
	// extract procedure arguments
	//--------------------------------------------------------------------------

	int k;                 // number of results to return
	char *label;           // entity label
	char *attribute;       // attribute to query
	SIValue query_vector;  // query vector

	// extract arguments from map
	if(!_extractArgs(args, &k, &label, &attribute, &query_vector)) {
		return PROCEDURE_ERR;
	}

	// depending on the entity type
	// set procedure step function and schema type
	SchemaType st;
	if(et == GETYPE_NODE) {
		st = SCHEMA_NODE;
		ctx->Step = Proc_NodeStep;
	} else {
		st = SCHEMA_EDGE;
		ctx->Step = Proc_EdgeStep;
	}

	//--------------------------------------------------------------------------
	// make sure there's a vector index exists
	//--------------------------------------------------------------------------

	// get attribute ID
	AttributeID attr_id = GraphContext_GetAttributeID(gc, attribute);
	if(attr_id == ATTRIBUTE_ID_NONE) {
		ErrorCtx_SetError(EMSG_ACCESS_UNDEFINED_ATTRIBUTE);
		return PROCEDURE_ERR;
	}

	Index idx = GraphContext_GetIndex(gc, label, &attr_id, 1, INDEX_FLD_VECTOR,
			st);

	if(idx == NULL) {
		return PROCEDURE_ERR;
	}

	// make sure query vector dimension matches index dimension
	IndexField *f = Index_GetField(NULL, idx, attr_id);
	uint32_t idx_dim = IndexField_OptionsGetDimension(f);
	if(idx_dim != SIVector_Dim(query_vector)) {
		ErrorCtx_SetError(EMSG_VECTOR_DIMENSION_MISMATCH, idx_dim,
				SIVector_Dim(query_vector));
		return PROCEDURE_ERR;
	}

	VecSimMetric sim_func = IndexField_OptionsGetSimFunc(f);

	//--------------------------------------------------------------------------
	// construct a vector query
	//--------------------------------------------------------------------------

	// create a query vector
	float  *vec   = SIVector_Elements(query_vector);
	size_t nbytes = SIVector_ElementsByteSize(query_vector);

	// create a redisearch query node
	RSQNode *root = Index_BuildVectorQueryTree(idx, attribute, vec, nbytes, k);

	// create procedure private data
	RSIndex *rsIdx = Index_RSIndex(idx);
	ctx->privateData = _create_private_data(gc, query_vector, attr_id, rsIdx,
			root, et, sim_func);

	return PROCEDURE_OK;
}

// procedure invocation
// validate arguments and sets up internal context
ProcedureResult Proc_VectorQueryNodeInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	const char **yield    // procedure output
) {
	ProcedureResult res = Proc_VectorQueryInvoke(ctx, args, GETYPE_NODE);
	if(res != PROCEDURE_OK) {
		if(!ErrorCtx_EncounteredError()) {
			ErrorCtx_SetError(EMSG_PROC_INVALID_ARGUMENTS,
					"db.idx.vector.queryNodes");
		}
		return res;
	}

	//--------------------------------------------------------------------------
	// process yield
	//--------------------------------------------------------------------------

	VectorKNNCtx *pdata = ctx->privateData;
	pdata->yield_score  = NULL;
	pdata->yield_entity = NULL;

	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("node", yield[i]) == 0) {
			pdata->yield_entity = pdata->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("score", yield[i]) == 0) {
			pdata->yield_score = pdata->output + idx;
			idx++;
			continue;
		}
	}

	return PROCEDURE_OK;
}

// procedure invocation
// validate arguments and sets up internal context
ProcedureResult Proc_VectorQueryRelInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	const char **yield    // procedure output
) {
	ProcedureResult res = Proc_VectorQueryInvoke(ctx, args, GETYPE_EDGE);
	if(res != PROCEDURE_OK) {
		if(!ErrorCtx_EncounteredError()) {
			ErrorCtx_SetError(EMSG_PROC_INVALID_ARGUMENTS,
					"db.idx.vector.queryRelationships");
		}
		return res;
	}

	//--------------------------------------------------------------------------
	// process yield
	//--------------------------------------------------------------------------

	VectorKNNCtx *pdata = ctx->privateData;
	pdata->yield_score  = NULL;
	pdata->yield_entity = NULL;

	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("relationship", yield[i]) == 0) {
			pdata->yield_entity = pdata->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("score", yield[i]) == 0) {
			pdata->yield_score = pdata->output + idx;
			idx++;
			continue;
		}
	}

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
// CALL db.idx.vector.queryNodes( {
// label    : STRING,
// attribute: STRING,
// k        : INTEGER
// query    : Vectorf32,
// options  : map } ) YIELD node, score

ProcedureCtx *Proc_VectorQueryNodeCtx() {
	ProcedureOutput *output   = array_new(ProcedureOutput, 2);
	ProcedureOutput out_node  = {.name = "node", .type = T_NODE};
	ProcedureOutput out_score = {.name = "score", .type = T_DOUBLE};
	array_append(output, out_node);
	array_append(output, out_score);

	ProcedureCtx *ctx = ProcCtxNew("db.idx.vector.queryNodes",
								   4,
								   output,
								   NULL, // step func is determined by invoke
								   Proc_VectorQueryNodeInvoke,
								   Proc_VectorKNNFree,
								   NULL,
								   true);
	return ctx;
}

ProcedureCtx *Proc_VectorQueryRelCtx() {
	ProcedureOutput *output    = array_new(ProcedureOutput, 2);
	ProcedureOutput out_rel = {.name = "relationship", .type = T_EDGE};
	ProcedureOutput out_score  = {.name = "score", .type = T_DOUBLE};
	array_append(output, out_rel);
	array_append(output, out_score);

	ProcedureCtx *ctx = ProcCtxNew("db.idx.vector.queryRelationships",
								   4,
								   output,
								   NULL, // step func is determined by invoke
								   Proc_VectorQueryRelInvoke,
								   Proc_VectorKNNFree,
								   NULL,
								   true);
	return ctx;
}

