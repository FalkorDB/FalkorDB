/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "LAGraphX.h"
#include "GraphBLAS.h"
#include <string.h>

#include "proc_leiden.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"
#include "./utility/internal.h"
#include "../graph/graphcontext.h"

// CALL algo.leiden() YIELD node, communityId
// CALL algo.leiden(NULL) YIELD node, communityId
// CALL algo.leiden({nodeLabels: ['L', 'P']}) YIELD node, communityId
// CALL algo.leiden({relationshipTypes: ['R', 'E']}) YIELD node, communityId
// CALL algo.leiden({nodeLabels: ['L'], relationshipTypes: ['E'], weightAttribute: 'cost'}) YIELD node, communityId

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
} Leiden_Context;

static void _process_yield
(
	Leiden_Context *ctx,
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
	SIValue config,         // procedure configuration
	LabelID **lbls,         // [output] labels
	RelationID **rels,      // [output] relationships
	AttributeID *weightAtt  // [output] relationship attribute used as weight
) {
	ASSERT(lbls            != NULL);
	ASSERT(rels            != NULL);
	ASSERT(weightAtt       != NULL);
	ASSERT(SI_TYPE(config) == T_MAP);

	*lbls      = NULL;
	*rels      = NULL;
	*weightAtt = ATTRIBUTE_ID_NONE;

	uint match_fields = 0;
	uint n = Map_KeyCount(config);
	if(n > 3) {
		ErrorCtx_SetError("invalid leiden configuration");
		return false;
	}

	SIValue v;
	LabelID *_lbls    = NULL;
	GraphContext *gc  = QueryCtx_GetGraphCtx();
	RelationID *_rels = NULL;

	if(MAP_GETCASEINSENSITIVE(config, "nodeLabels", v)) {
		if(SI_TYPE(v) != T_ARRAY || !SIArray_AllOfType(v, T_STRING)) {
			ErrorCtx_SetError("leiden configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		_lbls = arr_new(LabelID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue lbl = SIArray_Get(v, i);
			Schema *s = GraphContext_GetSchema(gc, lbl.stringval, SCHEMA_NODE);
			if(s == NULL) {
				ErrorCtx_SetError(
					"leiden configuration contains non-existent label:%s",
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
			ErrorCtx_SetError("leiden configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		_rels = arr_new(RelationID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue rel = SIArray_Get(v, i);
			Schema *s = GraphContext_GetSchema(gc, rel.stringval, SCHEMA_EDGE);
			if(s == NULL) {
				ErrorCtx_SetError(
					"leiden configuration contains non-existent type:%s",
					rel.stringval);
				goto error;
			}

			arr_append(_rels, Schema_GetID(s));
		}
		*rels = _rels;
		match_fields++;
	}

	bool has_weight_attribute =
		MAP_GETCASEINSENSITIVE(config, "weightAttribute", v);
	SIValue weight_property;
	bool has_weight_property =
		MAP_GETCASEINSENSITIVE(config, "weightProperty", weight_property);

	if(has_weight_attribute && has_weight_property) {
		ErrorCtx_SetError("leiden configuration can include either 'weightAttribute' or 'weightProperty', but not both");
		goto error;
	}

	if(has_weight_attribute || has_weight_property) {
		SIValue weight_cfg = has_weight_attribute ? v : weight_property;

		if(SI_TYPE(weight_cfg) != T_STRING) {
			ErrorCtx_SetError("leiden configuration, weight property should be a string");
			goto error;
		}

		*weightAtt = GraphContext_GetAttributeID(gc, weight_cfg.stringval);
		if(*weightAtt == ATTRIBUTE_ID_NONE) {
			ErrorCtx_SetError("leiden configuration, unknown attribute: %s", weight_cfg.stringval);
			goto error;
		}

		match_fields += has_weight_attribute ? 1 : 0;
		match_fields += has_weight_property ? 1 : 0;
	}

	if(n != match_fields) {
		ErrorCtx_SetError("leiden configuration contains unknown key");
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
	Leiden_Context *ctx
) {
	ASSERT(ctx != NULL);
	ASSERT(ctx->rows != NULL);

	ctx->node_ids = arr_new(NodeID, 0);

	struct GB_Iterator_opaque _it;
	GxB_Iterator it = &_it;

	GrB_Info info = GxB_Vector_Iterator_attach(it, ctx->rows, NULL);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_Vector_Iterator_seek(it, 0);
	while(info == GrB_SUCCESS) {
		arr_append(ctx->node_ids, (NodeID)GxB_Vector_Iterator_getIndex(it));
		info = GxB_Vector_Iterator_next(it);
	}

	ASSERT(info == GxB_EXHAUSTED);
}

static int64_t _community_id_at
(
	const Leiden_Context *ctx,
	GrB_Index idx
) {
	ASSERT(ctx != NULL);
	ASSERT(ctx->communities != NULL);

	int64_t cid;
	GrB_Info info = GrB_Vector_extractElement_INT64(&cid, ctx->communities, idx);
	ASSERT(info == GrB_SUCCESS);
	return cid;
}

ProcedureResult Proc_LeidenInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	size_t argc = arr_len((SIValue *)args);
	if(argc > 1) {
		ErrorCtx_SetError("algo.leiden expects a single argument");
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
		ErrorCtx_SetError("invalid argument to algo.leiden");
		return PROCEDURE_ERR;
	}

	LabelID *lbls = NULL;
	RelationID *rels = NULL;
	AttributeID weightAtt = ATTRIBUTE_ID_NONE;

	bool config_ok = _read_config(config, &lbls, &rels, &weightAtt);
	SIValue_Free(config);
	if(!config_ok) {
		return PROCEDURE_ERR;
	}

	Leiden_Context *pdata = rm_calloc(1, sizeof(Leiden_Context));
	pdata->g = QueryCtx_GetGraph();
	_process_yield(pdata, yield);
	ctx->privateData = pdata;

	GrB_Matrix A = NULL;
	GrB_Matrix A_w = NULL;
	// TODO: this should use an addition strategy
	GrB_OK(get_sub_weight_matrix(&A, &A_w, &pdata->rows, pdata->g,
		lbls, arr_len(lbls), rels, arr_len(rels), weightAtt, BWM_MAX, true));

	if(lbls != NULL) arr_free(lbls);
	if(rels != NULL) arr_free(rels);

	if(weightAtt == ATTRIBUTE_ID_NONE) {
		// Leiden requires positive weights; use a uniform weight of 1.0.
		GrB_OK(GrB_Matrix_assign_FP64(
			A_w, A, NULL, 1.0, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));
	}

	GrB_OK(GrB_free(&A));

	GrB_Index n = 0;
	GrB_OK(GrB_Matrix_nrows(&n, A_w));

	if(n > 0) {
		LAGraph_Graph G = NULL;
		char msg[LAGRAPH_MSG_LEN];
		GrB_Info info;
		msg[0] = '\0';

		info = LAGraph_New(&G, &A_w, LAGraph_ADJACENCY_UNDIRECTED, msg);
		if(info != GrB_SUCCESS) {
			ErrorCtx_SetError("algo.leiden failed creating graph (status %d): %s",
				info, msg);
			return PROCEDURE_ERR;
		}

		// LAGraph_Leiden requires G->emin to be cached.
		msg[0] = '\0';
		info = LAGraph_Cached_EMin(G, msg);
		if(info != GrB_SUCCESS) {
			LAGraph_Delete(&G, msg);
			ErrorCtx_SetError("algo.leiden failed caching minimum edge weight (status %d): %s",
				info, msg);
			return PROCEDURE_ERR;
		}

		msg[0] = '\0';
		int leiden_res = LAGraph_Leiden(&pdata->communities, G, 0, msg);
		char leiden_msg[LAGRAPH_MSG_LEN];
		leiden_msg[0] = '\0';
		if(msg[0] != '\0') {
			strncpy(leiden_msg, msg, LAGRAPH_MSG_LEN - 1);
			leiden_msg[LAGRAPH_MSG_LEN - 1] = '\0';
		}

		info = LAGraph_Delete(&G, msg);
		if(info != GrB_SUCCESS) {
			ErrorCtx_SetError("algo.leiden failed deleting graph (status %d): %s",
				info, msg);
			return PROCEDURE_ERR;
		}

		if(leiden_res != GrB_SUCCESS) {
			ErrorCtx_SetError("algo.leiden failed running algorithm (status %d): %s",
				leiden_res, leiden_msg);
			return PROCEDURE_ERR;
		}
	} else {
		GrB_OK(GrB_Matrix_free(&A_w));
		GrB_OK(GrB_Vector_new(&pdata->communities, GrB_UINT64, 0));
	}

	_build_node_map(pdata);
	ASSERT(arr_len(pdata->node_ids) == (uint64_t)n);

	return PROCEDURE_OK;
}

SIValue *Proc_LeidenStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);
	Leiden_Context *pdata = ctx->privateData;

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

ProcedureResult Proc_LeidenFree
(
	ProcedureCtx *ctx
) {
	if(ctx->privateData != NULL) {
		Leiden_Context *pdata = ctx->privateData;

		if(pdata->communities != NULL) GrB_free(&pdata->communities);
		if(pdata->rows != NULL) GrB_free(&pdata->rows);
		if(pdata->node_ids != NULL) arr_free(pdata->node_ids);

		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_LeidenCtx(void) {
	ProcedureOutput *outputs         = arr_new(ProcedureOutput, 2);
	ProcedureOutput output_node      = {.name = "node", .type = T_NODE};
	ProcedureOutput output_community = {.name = "communityId", .type = T_INT64};

	arr_append(outputs, output_node);
	arr_append(outputs, output_community);

	ProcedureCtx *ctx = ProcCtxNew("algo.leiden",
		PROCEDURE_VARIABLE_ARG_COUNT,
		outputs,
		Proc_LeidenStep,
		Proc_LeidenInvoke,
		Proc_LeidenFree,
		NULL,
		true);

	return ctx;
}
