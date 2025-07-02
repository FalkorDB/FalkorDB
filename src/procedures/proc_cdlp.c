/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "LAGraphX.h"
#include "GraphBLAS.h"

#include "proc_cdlp.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "./utility/internal.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"
#include "../graph/graphcontext.h"

#define CDLP_MAX_ITER_DEFAULT 10

// CALL algo.labelPropagation({}) YIELD node, communityId
// CALL algo.labelPropagation(NULL) YIELD node, communityId
// CALL algo.labelPropagation({nodeLabels: ['L', 'P']}) YIELD node, communityId
// CALL algo.labelPropagation({relationshipTypes: ['R', 'E']}) YIELD node, communityId
// CALL algo.labelPropagation({nodeLabels: ['L'], relationshipTypes: ['E']}) YIELD node, communityId
// CALL algo.labelPropagation({nodeLabels: ['L'], maxIterations: 10})

typedef struct {
	Graph *g;                // graph
	GrB_Vector communities;  // communities[i]: community of node i
	GrB_Vector nodes;        // nodes participating in computation
	GrB_Info info;           // iterator state
	GxB_Iterator it;         // communities iterator
	GxB_Iterator cm_it;      // communities iterator
	Node node;               // node
	SIValue output[2];       // array with up to 2 entries [node, community id]
	SIValue *yield_node;     // yield node
	SIValue *yield_cid;      // yield community id
} CDLP_Context;

// process procedure yield
static void _process_yield
(
	CDLP_Context *ctx,
	const char **yield
) {
	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("node", yield[i]) == 0) {
			ctx->yield_node = ctx->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("communityId", yield[i]) == 0) {
			ctx->yield_cid = ctx->output + idx;
			idx++;
			continue;
		}
	}
}

// process procedure configuration argument
static bool _read_config
(
	SIValue config,         // procedure configuration
	LabelID **lbls,         // [output] labels
	RelationID **rels,      // [output] relationships
	int32_t *maxIterations  // [output] max number of iterations
) {
	// expecting configuration to be a map
	ASSERT(lbls            != NULL);
	ASSERT(rels            != NULL);
	ASSERT(maxIterations   != NULL);
	ASSERT(SI_TYPE(config) == T_MAP);

	// set outputs to NULL
	*lbls          = NULL;
	*rels          = NULL;
	*maxIterations = CDLP_MAX_ITER_DEFAULT;

	uint match_fields = 0;
	uint n = Map_KeyCount(config);
	if(n > 3) {
		// error config contains unknown key
		ErrorCtx_SetError("invalid labelPropagation configuration");
		return false;
	}

	SIValue v;
	LabelID *_lbls    = NULL;
	GraphContext *gc  = QueryCtx_GetGraphCtx();
	RelationID *_rels = NULL;

	if(MAP_GETCASEINSENSITIVE(config, "maxIterations", v)) {
		if(SI_TYPE(v) != T_INT64 || v.longval <= 0) {
			ErrorCtx_SetError("labelPropagation configuration, 'maxIterations' should be a positive integer");
			return false;
		}
		
		*maxIterations = v.longval;
		match_fields++;
	}

	if(MAP_GETCASEINSENSITIVE(config, "nodeLabels", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("labelPropagation configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			// error
			ErrorCtx_SetError("labelPropagation configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		_lbls = array_new(LabelID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue lbl = SIArray_Get(v, i);
			const char *label = lbl.stringval;
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
			if(s == NULL) {
				ErrorCtx_SetError("labelPropagation configuration, unknown label %s", label);
				goto error;
			}

			LabelID lbl_id = Schema_GetID(s);
			array_append(_lbls, lbl_id);
		}
		*lbls = _lbls;

		match_fields++;
	}

	if(MAP_GETCASEINSENSITIVE(config, "relationshipTypes", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("labelPropagation configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			ErrorCtx_SetError("labelPropagation configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		_rels = array_new(RelationID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue rel = SIArray_Get(v, i);
			const char *relation = rel.stringval;
			Schema *s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
			if(s == NULL) {
				ErrorCtx_SetError("labelPropagation configuration, unknown relationship-type %s", relation);
				goto error;
			}

			RelationID rel_id = Schema_GetID(s);
			array_append(_rels, rel_id);
		}
		*rels = _rels;

		match_fields++;
	}

	if(n != match_fields) {
		ErrorCtx_SetError("labelPropagation configuration contains unknown key");
		goto error;
	}

	return true;

error:
	if(_lbls != NULL) {
		array_free(_lbls);
		*lbls = NULL;
	}

	if(_rels != NULL) {
		array_free(_rels);
		*rels = NULL;
	}

	return false;
}

// invoke the procedure
ProcedureResult Proc_CDLPInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	const char **yield    // procedure outputs
) {
	// expecting 0 or 1 argument

	size_t l = array_len((SIValue *)args);

	if(l > 1) return PROCEDURE_ERR;

	SIValue config;

	if(l == 0 || SIValue_IsNull(args[0])) {
		config = SI_Map(0);
	} else {
		config = SI_CloneValue(args[0]);
	}

	// arg0 can be either a map or NULL
	SIType t = SI_TYPE(config);
	if(!(t & T_MAP)) {
		SIValue_Free(config);

		ErrorCtx_SetError("invalid argument to algo.labelPropagation");
		return PROCEDURE_ERR;
	}

	// read CDLP invoke configuration
	// {
	//	nodeLabels: ['A', 'B'],
	//	relationshipTypes: ['R'],
	//	maxIterations: 12
	// }

	LabelID    *lbls      = NULL;
	RelationID *rels      = NULL;
	int32_t maxIterations = CDLP_MAX_ITER_DEFAULT;

	//--------------------------------------------------------------------------
	// load configuration map
	//--------------------------------------------------------------------------

	bool config_ok = _read_config(config, &lbls, &rels, &maxIterations);

	SIValue_Free(config);

	if(!config_ok) {
		return PROCEDURE_ERR;
	}

	//--------------------------------------------------------------------------
	// setup procedure context
	//--------------------------------------------------------------------------

	Graph *g = QueryCtx_GetGraph();
	CDLP_Context *pdata = rm_calloc(1, sizeof(CDLP_Context));

	pdata->g = g;

	_process_yield(pdata, yield);

	// save private data
	ctx->privateData = pdata;

	//--------------------------------------------------------------------------
	// build adjacency matrix on which we'll run CDLP
	//--------------------------------------------------------------------------
	double tic[2];
	simple_tic(tic);
	GrB_Matrix A = NULL;
	get_sub_adjecency_matrix(&A, &pdata->nodes, g, lbls, array_len(lbls), rels,
			array_len(rels), true);
	double bm_time = simple_toc(tic);
	RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_WARNING, 
			"Build time: %f", bm_time);
	// free build matrix inputs
	if(lbls != NULL) array_free(lbls);
	if(rels != NULL) array_free(rels);

	//--------------------------------------------------------------------------
	// run CDLP
	//--------------------------------------------------------------------------

	// execute CLDP
	LAGraph_Graph G;
	char msg[LAGRAPH_MSG_LEN];

	GrB_Info info = LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg);
	ASSERT(info == GrB_SUCCESS);


	simple_tic(tic);
	GrB_Info cdlp_res = LAGraph_cdlp(&pdata->communities, G, maxIterations, msg);
	double cdlp_time = simple_toc(tic);
	RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_WARNING, 
			"CDLP time: %f", cdlp_time);
	info = LAGraph_Delete(&G, msg);
	ASSERT(info == GrB_SUCCESS);

	if(cdlp_res != GrB_SUCCESS) {
		return PROCEDURE_ERR;
	}

	//--------------------------------------------------------------------------
	// initialize iterator
	//--------------------------------------------------------------------------
	info = GxB_Iterator_new(&pdata->cm_it);
	ASSERT(info == GrB_SUCCESS);

	// iterate over participating nodes
	info = GxB_Vector_Iterator_attach(pdata->cm_it, pdata->communities, NULL);
	ASSERT(info == GrB_SUCCESS);

    pdata->info = GxB_Vector_Iterator_seek(pdata->cm_it, 0);

	info = GxB_Iterator_new(&pdata->it);
	ASSERT(info == GrB_SUCCESS);

	// iterate over participating nodes
	info = GxB_Vector_Iterator_attach(pdata->it, pdata->nodes, NULL);
	ASSERT(info == GrB_SUCCESS);

    pdata->info = GxB_Vector_Iterator_seek(pdata->it, 0);

	return PROCEDURE_OK;
}

// yield node and its score
// yields NULL if there are no additional nodes to return
SIValue *Proc_CDLPStep
(
	ProcedureCtx *ctx  // procedure context
) {
	ASSERT(ctx->privateData != NULL);

	CDLP_Context *pdata = (CDLP_Context *)ctx->privateData;

	// retrieve node from graph
	GrB_Index node_id;
	uint64_t community_id;
	while(pdata->info != GxB_EXHAUSTED) {
		// get current node id and its associated score
		node_id = GxB_Vector_Iterator_getIndex(pdata->it);

		if(Graph_GetNode(pdata->g, node_id, &pdata->node)) {
			break;
		}

		// move to the next entry in the components vector
		pdata->info = GxB_Vector_Iterator_next(pdata->it);
		pdata->info = GxB_Vector_Iterator_next(pdata->cm_it);
	}

	// depleted
	if(pdata->info == GxB_EXHAUSTED) {
		return NULL;
	}

	community_id = GxB_Iterator_get_UINT64(pdata->cm_it);

	// prep for next call to Proc_CDLPStep
	pdata->info = GxB_Vector_Iterator_next(pdata->it);
	pdata->info = GxB_Vector_Iterator_next(pdata->cm_it);
	
	//--------------------------------------------------------------------------
	// set outputs
	//--------------------------------------------------------------------------

	if(pdata->yield_node) {
		*pdata->yield_node = SI_Node(&pdata->node);
	}

	if(pdata->yield_cid) {
		*pdata->yield_cid = SI_LongVal(community_id);
	}

	return pdata->output;
}

ProcedureResult Proc_CDLPFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData != NULL) {
		CDLP_Context *pdata = ctx->privateData;

		if(pdata->it          != NULL) GrB_free(&pdata->it);
		if(pdata->cm_it       != NULL) GrB_free(&pdata->cm_it);
		if(pdata->nodes       != NULL) GrB_free(&pdata->nodes);
		if(pdata->communities != NULL) GrB_free(&pdata->communities);

		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

// CALL algo.labelPropagation({nodeLabels: ['Person'], relationshipTypes: ['KNOWS'],
// maxIterations:10}) YIELD node, communityId
// run Community detection label propagation 
ProcedureCtx *Proc_CDLPCtx(void) {
	void *privateData = NULL;

	ProcedureOutput *outputs         = array_new(ProcedureOutput, 2);
	ProcedureOutput output_node      = {.name = "node", .type = T_NODE};
	ProcedureOutput output_community = {.name = "communityId", .type = T_INT64};

	array_append(outputs, output_node);
	array_append(outputs, output_community);

	ProcedureCtx *ctx = ProcCtxNew("algo.labelPropagation",
								   PROCEDURE_VARIABLE_ARG_COUNT,
								   outputs,
								   Proc_CDLPStep,
								   Proc_CDLPInvoke,
								   Proc_CDLPFree,
								   privateData,
								   true);
	return ctx;
}
