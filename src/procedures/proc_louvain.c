/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "LAGraphX.h"
#include "GraphBLAS.h"
#include <string.h>

#include "proc_louvain.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"
#include "./utility/internal.h"
#include "../graph/graphcontext.h"

// LAGraph_louvain's own tests (LAGraph/experimental/test/test_louvain.c) use
// these values; louvain doesn't currently expose these as tunables here.
#define LOUVAIN_ITERMAX  6      // max modularity-improvement sweeps per level
#define LOUVAIN_LEVELMAX 2      // max improve-and-condense levels
#define LOUVAIN_EPSILON  1e-5f  // min modularity change considered an improvement

// CALL algo.louvain() YIELD node, communityId
// CALL algo.louvain(NULL) YIELD node, communityId
// CALL algo.louvain({nodeLabels: ['L', 'P']}) YIELD node, communityId
// CALL algo.louvain({relationshipTypes: ['R', 'E']}) YIELD node, communityId
// CALL algo.louvain({nodeLabels: ['L'], relationshipTypes: ['E']}) YIELD node, communityId

typedef struct {
	Graph *g;                // graph
	GrB_Vector communities;  // communities[i]: community label of compact row i
	GrB_Vector rows;         // participating node ids
	NodeID *node_ids;        // compact-row-index -> node id mapping
	uint64_t idx;            // current compact-row index
	Node node;               // current node
	SIValue output[2];       // array with up to 2 entries [node, community id]
	SIValue *yield_node;     // yield node
	SIValue *yield_cid;      // yield community id
} Louvain_Context;

static void _process_yield
(
	Louvain_Context *ctx,
	const char **yield
) {
	int idx = 0;
	for(uint i = 0; i < arr_len(yield); i++) {
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

static bool _read_config
(
	SIValue config,     // procedure configuration
	LabelID **lbls,     // [output] labels
	RelationID **rels   // [output] relationships
) {
	ASSERT(lbls            != NULL);
	ASSERT(rels            != NULL);
	ASSERT(SI_TYPE(config) == T_MAP);

	*lbls = NULL;
	*rels = NULL;

	uint match_fields = 0;
	uint n = Map_KeyCount(config);
	if(n > 2) {
		ErrorCtx_SetError("invalid louvain configuration");
		return false;
	}

	SIValue v;
	LabelID *_lbls    = NULL;
	GraphContext *gc  = QueryCtx_GetGraphCtx();
	RelationID *_rels = NULL;

	if(MAP_GETCASEINSENSITIVE(config, "nodeLabels", v)) {
		if(SI_TYPE(v) != T_ARRAY || !SIArray_AllOfType(v, T_STRING)) {
			ErrorCtx_SetError("louvain configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		_lbls = arr_new(LabelID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue lbl = SIArray_Get(v, i);
			Schema *s = GraphContext_GetSchema(gc, lbl.stringval, SCHEMA_NODE);
			if(s == NULL) {
				ErrorCtx_SetError(
					"louvain configuration contains non-existent label:%s",
					lbl.stringval);
				goto error;
			}

			arr_append(_lbls, Schema_GetID(s));
		}
		*lbls = _lbls;
		match_fields++;
	}

	if(MAP_GETCASEINSENSITIVE(config, "relationshipTypes", v)) {
		if(SI_TYPE(v) != T_ARRAY || !SIArray_AllOfType(v, T_STRING)) {
			ErrorCtx_SetError("louvain configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		_rels = arr_new(RelationID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue rel = SIArray_Get(v, i);
			Schema *s = GraphContext_GetSchema(gc, rel.stringval, SCHEMA_EDGE);
			if(s == NULL) {
				ErrorCtx_SetError(
					"louvain configuration contains non-existent type:%s",
					rel.stringval);
				goto error;
			}

			arr_append(_rels, Schema_GetID(s));
		}
		*rels = _rels;
		match_fields++;
	}

	if(n != match_fields) {
		ErrorCtx_SetError("louvain configuration contains unknown key");
		goto error;
	}

	return true;

error:
	if(_lbls != NULL) {
		arr_free(_lbls);
		*lbls = NULL;
	}

	if(_rels != NULL) {
		arr_free(_rels);
		*rels = NULL;
	}

	return false;
}

static void _build_node_map
(
	Louvain_Context *ctx
) {
	ASSERT(ctx != NULL);
	ASSERT(ctx->rows != NULL);

	GrB_Index nvals;
	GrB_Info info = GrB_Vector_nvals(&nvals, ctx->rows);
	ASSERT(info == GrB_SUCCESS);

	// NodeID is a GrB_Index, so the extracted indices can be written
	// directly into the node_ids array
	ctx->node_ids = arr_newlen(NodeID, nvals);

	info = GrB_Vector_extractTuples_BOOL(ctx->node_ids, NULL, &nvals,
		ctx->rows);
	ASSERT(info == GrB_SUCCESS);
}

static int64_t _community_id_at
(
	const Louvain_Context *ctx,
	GrB_Index idx
) {
	ASSERT(ctx != NULL);
	ASSERT(ctx->communities != NULL);

	uint64_t cid;
	GrB_Info info = GrB_Vector_extractElement_UINT64(&cid, ctx->communities, idx);
	ASSERT(info == GrB_SUCCESS);
	return (int64_t)cid;
}

ProcedureResult Proc_LouvainInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	size_t argc = arr_len((SIValue *)args);
	if(argc > 1) {
		ErrorCtx_SetError("algo.louvain expects a single argument");
		return PROCEDURE_ERR;
	}

	SIValue config;
	if(argc == 0 || SIValue_IsNull(args[0])) {
		config = SI_Map(0);
	} else {
		config = SI_CloneValue(args[0]);
	}

	if(SI_TYPE(config) != T_MAP) {
		SIValue_Free(config);
		ErrorCtx_SetError("invalid argument to algo.louvain");
		return PROCEDURE_ERR;
	}

	LabelID *lbls = NULL;
	RelationID *rels = NULL;

	bool config_ok = _read_config(config, &lbls, &rels);
	SIValue_Free(config);
	if(!config_ok) {
		return PROCEDURE_ERR;
	}

	Louvain_Context *pdata = rm_calloc(1, sizeof(Louvain_Context));
	pdata->g = QueryCtx_GetGraph();
	_process_yield(pdata, yield);
	ctx->privateData = pdata;

	GrB_Matrix A = NULL;
	// LAGraph_louvain requires a boolean, symmetric adjacency matrix.
	GrB_OK(Build_Matrix(&A, &pdata->rows, pdata->g,
		lbls, arr_len(lbls), rels, arr_len(rels), true, true));

	if(lbls != NULL) arr_free(lbls);
	if(rels != NULL) arr_free(rels);

	GrB_Index n = 0;
	GrB_OK(GrB_Matrix_nrows(&n, A));

	if(n > 0) {
		LAGraph_Graph G = NULL;
		char msg[LAGRAPH_MSG_LEN];
		GrB_Info info;
		msg[0] = '\0';

		info = LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg);
		if(info != GrB_SUCCESS) {
			GrB_OK(GrB_Matrix_free(&A));
			ErrorCtx_SetError("algo.louvain failed creating graph (status %d): %s",
				info, msg);
			return PROCEDURE_ERR;
		}

		msg[0] = '\0';
		info = LAGraph_louvain(&pdata->communities, G, LOUVAIN_ITERMAX,
			LOUVAIN_LEVELMAX, LOUVAIN_EPSILON, msg);
		char louvain_msg[LAGRAPH_MSG_LEN];
		louvain_msg[0] = '\0';
		if(msg[0] != '\0') {
			strncpy(louvain_msg, msg, LAGRAPH_MSG_LEN - 1);
			louvain_msg[LAGRAPH_MSG_LEN - 1] = '\0';
		}

		GrB_Info delete_info = LAGraph_Delete(&G, msg);
		if(delete_info != GrB_SUCCESS) {
			ErrorCtx_SetError("algo.louvain failed deleting graph (status %d): %s",
				delete_info, msg);
			return PROCEDURE_ERR;
		}

		if(info != GrB_SUCCESS) {
			ErrorCtx_SetError("algo.louvain failed running algorithm (status %d): %s",
				info, louvain_msg);
			return PROCEDURE_ERR;
		}
	} else {
		GrB_OK(GrB_Matrix_free(&A));
		GrB_OK(GrB_Vector_new(&pdata->communities, GrB_UINT64, 0));
	}

	_build_node_map(pdata);
	ASSERT(arr_len(pdata->node_ids) == (uint64_t)n);

	return PROCEDURE_OK;
}

SIValue *Proc_LouvainStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);
	Louvain_Context *pdata = ctx->privateData;

	uint64_t n = arr_len(pdata->node_ids);
	while(pdata->idx < n) {
		GrB_Index community_idx = pdata->idx;
		NodeID node_id = pdata->node_ids[pdata->idx++];

		if(!Graph_GetNode(pdata->g, node_id, &pdata->node)) {
			continue;
		}

		if(pdata->yield_node) {
			*pdata->yield_node = SI_Node(&pdata->node);
		}

		if(pdata->yield_cid) {
			*pdata->yield_cid = SI_LongVal(_community_id_at(pdata, community_idx));
		}

		return pdata->output;
	}

	return NULL;
}

ProcedureResult Proc_LouvainFree
(
	ProcedureCtx *ctx
) {
	if(ctx->privateData != NULL) {
		Louvain_Context *pdata = ctx->privateData;

		if(pdata->communities != NULL) GrB_free(&pdata->communities);
		if(pdata->rows != NULL) GrB_free(&pdata->rows);
		if(pdata->node_ids != NULL) arr_free(pdata->node_ids);

		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_LouvainCtx(void) {
	ProcedureOutput *outputs         = arr_new(ProcedureOutput, 2);
	ProcedureOutput output_node      = {.name = "node", .type = T_NODE};
	ProcedureOutput output_community = {.name = "communityId", .type = T_INT64};

	arr_append(outputs, output_node);
	arr_append(outputs, output_community);

	ProcedureCtx *ctx = ProcCtxNew("algo.louvain",
		PROCEDURE_VARIABLE_ARG_COUNT,
		outputs,
		Proc_LouvainStep,
		Proc_LouvainInvoke,
		Proc_LouvainFree,
		NULL,
		true);

	return ctx;
}
