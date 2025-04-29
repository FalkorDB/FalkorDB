/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"

#include "proc_wcc.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../datatypes/map.h"
#include "../algorithms/wcc.h"
#include "../datatypes/array.h"
#include "../graph/graphcontext.h"

// CALL algo.wcc({}) YIELD node, componentId
// CALL algo.wcc(NULL) YIELD node, componentId
// CALL algo.wcc({nodeLabels: ['L', 'P']}) YIELD node, componentId
// CALL algo.wcc({relationshipTypes: ['R', 'E']}) YIELD node, componentId
// CALL algo.wcc({nodeLabels: ['L'], relationshipTypes: ['E']}) YIELD node, componentId

typedef struct {
	LabelID *lbls;             // considered labels
	RelationID *rels;          // considered relationship-types
	GrB_Vector components;     // computed components
	GrB_Vector N;              // nodes participating in WCC
	GrB_Info info;             // iterator state
	GxB_Iterator it;           // components iterator
	Node node;                 // current node
	Graph *g;                  // graph
	SIValue output[2];         // array with up to 2 entries [node, component]
	SIValue *yield_node;       // yield node
	SIValue *yield_component;  // yield component
} WCC_Context;

static void _process_yield
(
	WCC_Context *ctx,
	const char **yield
) {
	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("node", yield[i]) == 0) {
			ctx->yield_node = ctx->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("componentId", yield[i]) == 0) {
			ctx->yield_component = ctx->output + idx;
			idx++;
			continue;
		}
	}
}

// process procedure configuration argument
static bool _read_config
(
	SIValue config,    // procedure configuration
	LabelID **lbls,    // [output] labels
	RelationID **rels  // [output] relationships
) {
	// expecting configuration to be a map
	ASSERT(lbls            != NULL);
	ASSERT(rels            != NULL);
	ASSERT(SI_TYPE(config) == T_MAP);

	// set outputs to NULL
	*lbls = NULL;
	*rels = NULL;

	uint match_fields = 0;
	uint n = Map_KeyCount(config);
	if(n > 2) {
		// error config contains unknown key
		ErrorCtx_SetError("invalid wcc configuration");
		return false;
	}

	SIValue v;
	GraphContext *gc  = QueryCtx_GetGraphCtx();
	LabelID *_lbls    = NULL;
	RelationID *_rels = NULL;

	if(MAP_GETCASEINSENSITIVE(config, "nodeLabels", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("wcc configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			// error
			ErrorCtx_SetError("wcc configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		_lbls = array_new(LabelID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue lbl = SIArray_Get(v, i);
			const char *label = lbl.stringval;
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
			if(s == NULL) {
				ErrorCtx_SetError("wcc configuration, unknown label %s", label);
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
			ErrorCtx_SetError("wcc configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			ErrorCtx_SetError("wcc configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		_rels = array_new(RelationID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue rel = SIArray_Get(v, i);
			const char *relation = rel.stringval;
			Schema *s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
			if(s == NULL) {
				ErrorCtx_SetError("wcc configuration, unknown relationship-type %s", relation);
				goto error;
			}

			RelationID rel_id = Schema_GetID(s);
			array_append(_rels, rel_id);
		}
		*rels = _rels;

		match_fields++;
	}

	if(n - match_fields > 0) {
		ErrorCtx_SetError("wcc configuration contains unknown key");
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
ProcedureResult Proc_WCCInvoke
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
		ErrorCtx_SetError("invalid argument to algo.wcc");
		return PROCEDURE_ERR;
	}

	// read WCC invoke configuration
	// {
	//	nodeLabels: ['A', 'B'],
	//	relationshipTypes: ['R']
	// }

	LabelID    *lbls = NULL;
	RelationID *rels = NULL;

	bool config_ok = _read_config(config, &lbls, &rels);
	SIValue_Free(config);

	if(!config_ok) {
		return PROCEDURE_ERR;
	}

	// setup context
	Graph *g = QueryCtx_GetGraph();
	WCC_Context *pdata = rm_calloc(1, sizeof(WCC_Context));

	pdata->g               = g;
	pdata->N               = NULL;
	pdata->it              = NULL;
	pdata->lbls            = lbls;
	pdata->rels            = rels;
	pdata->components      = NULL;
	pdata->yield_node      = NULL;
	pdata->yield_component = NULL;

	_process_yield(pdata, yield);

	// save private data
	ctx->privateData = pdata;

	//--------------------------------------------------------------------------
	// run the WCC algorithm
	//--------------------------------------------------------------------------

	GrB_Info info = WCC(&pdata->components, &pdata->N, g, lbls, array_len(lbls),
			rels, array_len(rels));

	if(info != GrB_SUCCESS) {
		return PROCEDURE_ERR;
	}

	//--------------------------------------------------------------------------
	// initialize iterator
	//--------------------------------------------------------------------------

	info = GxB_Iterator_new(&pdata->it);
	ASSERT(info == GrB_SUCCESS);

	// iterate over participating nodes
	info = GxB_Vector_Iterator_attach(pdata->it, pdata->N, NULL);
	ASSERT(info == GrB_SUCCESS);

    pdata->info = GxB_Vector_Iterator_seek(pdata->it, 0);

	return PROCEDURE_OK;
}

// yield node and its component
// yields NULL if there are no additional nodes to return
SIValue *Proc_WCCStep
(
	ProcedureCtx *ctx  // procedure context
) {
	ASSERT(ctx->privateData != NULL);

	WCC_Context *pdata = (WCC_Context *)ctx->privateData;

	// retrieve node from graph
	GrB_Index node_id;
	while(pdata->info != GxB_EXHAUSTED) {
		// get current node id and its associated component id
		node_id = GxB_Vector_Iterator_getIndex(pdata->it);

		if(Graph_GetNode(pdata->g, node_id, &pdata->node)) {
			break;
		}

		// move to the next entry in the components vector
		pdata->info = GxB_Vector_Iterator_next(pdata->it);
	}

	// depleted
	if(pdata->info == GxB_EXHAUSTED) {
		return NULL;
	}

	// prep for next call to Proc_WCCStep
	pdata->info = GxB_Vector_Iterator_next(pdata->it);

	uint64_t component_id;
	GrB_Info info = GrB_Vector_extractElement_UINT64(&component_id,
			pdata->components, node_id);

	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// set output(s)
	//--------------------------------------------------------------------------

	if(pdata->yield_node) {
		*pdata->yield_node = SI_Node(&pdata->node);
	}

	if(pdata->yield_component) {
		*pdata->yield_component = SI_LongVal(component_id);
	}

	return pdata->output;
}

ProcedureResult Proc_WCCFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData != NULL) {
		WCC_Context *pdata = ctx->privateData;

		if(pdata->N          != NULL) GrB_free(&pdata->N);
		if(pdata->it         != NULL) GrB_free(&pdata->it);
		if(pdata->lbls       != NULL) array_free(pdata->lbls);
		if(pdata->rels       != NULL) array_free(pdata->rels);
		if(pdata->components != NULL) GrB_free(&pdata->components);

		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_WCCCtx(void) {
	void *privateData = NULL;

	ProcedureOutput *outputs         = array_new(ProcedureOutput, 2);
	ProcedureOutput output_node      = {.name = "node", .type = T_NODE};
	ProcedureOutput output_component = {.name = "componentId", .type = T_INT64};

	array_append(outputs, output_node);
	array_append(outputs, output_component);

	ProcedureCtx *ctx = ProcCtxNew("algo.WCC",
								   1,
								   outputs,
								   Proc_WCCStep,
								   Proc_WCCInvoke,
								   Proc_WCCFree,
								   privateData,
								   true);
	return ctx;
}

