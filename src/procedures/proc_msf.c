/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "LAGraphX.h"
#include "GraphBLAS.h"

#include "proc_msf.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"
#include "./utility/internal.h"
#include "../graph/graphcontext.h"

// CALL algo.MSF({}) YIELD edge, weight
// CALL algo.MSF(NULL) YIELD edge, weight
// CALL algo.MSF({nodeLabels: ['L', 'P']}) YIELD edge, weight
// CALL algo.MSF({relationshipTypes: ['R', 'E']}) YIELD edge, weight
// CALL algo.MSF({nodeLabels: ['L'], relationshipTypes: ['E'], weightAttribute: 'cost'}) YIELD edge, weight
// CALL algo.MSF({nodeLabels: ['L'], objective: minimum})

typedef struct {
	Graph *g;              	// graph
	GrB_Matrix tree;		// The MSF
	GrB_Matrix w_tree;		// The weighted MSF
	GrB_Vector nodes; 	   	// nodes participating in computation
	int *relationIDs;       // edge type(s) to traverse.
	int relationCount;      // length of relationIDs.
	GrB_Info info;         	// iterator state
	GxB_Iterator it;       	// iterator
	Node node;             	// node
	Edge edge;             	// edge
	AttributeID weight_prop;// weight attribute id
	SIValue output[2];     	// array with up to 2 entries [edge, weight]
	// SIValue *yield_node;   	// nodes 
	SIValue *yield_edge;   	// edges
	SIValue *yield_weight; 	// edge weights
} MSF_Context;

// process procedure yield
static void _process_yield
(
	MSF_Context *ctx,
	const char **yield
) {
	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		// if(strcasecmp("node", yield[i]) == 0) {
		// 	ctx->yield_node = ctx->output + idx;
		// 	idx++;
		// 	continue;
		// }

		if(strcasecmp("edge", yield[i]) == 0) {
			ctx->yield_edge = ctx->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("weight", yield[i]) == 0) {
			ctx->yield_weight = ctx->output + idx;
			idx++;
			continue;
		}
	}
}

// TODO should return enum rather than bool?
// process procedure configuration argument
static bool _read_config
(
	SIValue config,         // procedure configuration
	LabelID **lbls,         // [output] labels
	RelationID **rels,      // [output] relationships
	AttributeID *weightAtt, // [output] relationship used as weight
	bool *maxSF             // [output] true if maximum spanning forest
) {
	// expecting configuration to be a map
	ASSERT(lbls            != NULL);
	ASSERT(rels            != NULL);
	ASSERT(weightAtt      != NULL);
	ASSERT(maxSF   	       != NULL);
	ASSERT(SI_TYPE(config) == T_MAP);

	// set outputs to NULL
	*lbls = NULL;
	*rels = NULL;
	*weightAtt = ATTRIBUTE_ID_NONE;
	*maxSF = false;

	uint match_fields = 0;
	uint n = Map_KeyCount(config);
	if(n > 4) {
		// error config contains unknown key
		ErrorCtx_SetError("invalid msf configuration");
		return false;
	}

	SIValue v;
	LabelID *_lbls    = NULL;
	GraphContext *gc  = QueryCtx_GetGraphCtx();
	RelationID *_rels = NULL;

	if(MAP_GETCASEINSENSITIVE(config, "nodeLabels", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("msf configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			// error
			ErrorCtx_SetError("msf configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		_lbls = array_new(LabelID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue lbl = SIArray_Get(v, i);
			const char *label = lbl.stringval;
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
			if(s == NULL) {
				// log non-existant label
                RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_WARNING, 
                    "Skipping non-existent label: '%s'.", label);
				continue;
			}

			LabelID lbl_id = Schema_GetID(s);
			array_append(_lbls, lbl_id);
		}
		*lbls = _lbls;

		match_fields++;
	}

	if(MAP_GETCASEINSENSITIVE(config, "relationshipTypes", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("msf configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			ErrorCtx_SetError("msf configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		_rels = array_new(RelationID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue rel = SIArray_Get(v, i);
			const char *relation = rel.stringval;
			Schema *s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
			if(s == NULL) {
				// log non-existant relation
                RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_WARNING, 
                    "Skipping non-existent relation: '%s'.", relation);
				continue;
			}

			RelationID rel_id = Schema_GetID(s);
			array_append(_rels, rel_id);
		}
		*rels = _rels;

		match_fields++;
	}
	if(MAP_GETCASEINSENSITIVE(config, "weightAttribute", v)) {
		if(SI_TYPE(v) != T_STRING) {
			ErrorCtx_SetError("msf configuration, 'weightAttribute' should be a string");
			goto error;
		}

		const char *relation = v.stringval;
		*weightAtt = GraphContext_GetAttributeID(gc, relation);
		if(*weightAtt == ATTRIBUTE_ID_NONE) {
			ErrorCtx_SetError("msf configuration, unknown attribute-type %s", relation);
			goto error;
		}
		match_fields++;
	}
	if(MAP_GETCASEINSENSITIVE(config, "objective", v)) {
		if(SI_TYPE(v) != T_STRING) {
			ErrorCtx_SetError("msf configuration, 'objective' should be a string");
			goto error;
		}
		const char *objective = v.stringval;
		if(strncasecmp(objective, "min", 3) == 0)
			*maxSF = false;
		else if(strncasecmp(objective, "max", 3) == 0)
			*maxSF = true;
		else{
			ErrorCtx_SetError("msf configuration, unknown objective %s", objective);
			goto error;
		}
		match_fields++;
	}

	if(n != match_fields) {
		ErrorCtx_SetError("msf configuration contains unknown key");
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
ProcedureResult Proc_MSFInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	const char **yield    // procedure outputs
) {
	// expecting a single argument
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
	if(t != T_MAP) {
		SIValue_Free(config);

		ErrorCtx_SetError("invalid argument to algo.MSF");
		return PROCEDURE_ERR;
	}

	// read MSF invoke configuration
	// {
	//	nodeLabels: ['A', 'B'],
	//	relationshipTypes: ['R'],
	//	weightRelationship: 'R',
	//	objective: 'minimize'
	// }

	LabelID    *lbls      = NULL;
	RelationID *rels      = NULL;
	AttributeID weightAtt = ATTRIBUTE_ID_NONE;
	bool        maxSF     = false;

	//--------------------------------------------------------------------------
	// load configuration map
	//--------------------------------------------------------------------------

	bool config_ok = _read_config(config, &lbls, &rels, &weightAtt, &maxSF);
	
	SIValue_Free(config);

	if(!config_ok) {
		//error set by _read_config
		return PROCEDURE_ERR;
	}


	//--------------------------------------------------------------------------
	// setup procedure context
	//--------------------------------------------------------------------------

	Graph *g = QueryCtx_GetGraph();
	MSF_Context *pdata = rm_calloc(1, sizeof(MSF_Context));

	pdata->g              = g;
	pdata->weight_prop    = weightAtt;
	pdata->relationIDs    = rels;
	pdata->relationCount  = array_len(rels);
	_process_yield(pdata, yield);

	// save private data
	ctx->privateData = pdata;

	GrB_Matrix A = NULL, A_w = NULL;
	GrB_Info info;

	//makes into a symetric matrix
	info =  Build_Weighted_Matrix (
			&A, &A_w, &pdata->nodes, g, lbls, array_len(lbls), rels,
			array_len(rels), weightAtt, maxSF? BWM_MAX: BWM_MIN, true, true);
	ASSERT(info == GrB_SUCCESS);
	
	// free build matrix inputs
	if(lbls != NULL) array_free(lbls);
	// if(rels != NULL) array_free(rels);

	//--------------------------------------------------------------------------
	// run MSF centrality
	//--------------------------------------------------------------------------

	// Make weights negative if looking for Maximum Spanning Tree
	if(maxSF)
	{
		info = GrB_Matrix_apply(A_w, NULL, NULL, GrB_AINV_FP64, A_w, NULL);
		ASSERT(info == GrB_SUCCESS);
	}
	char msg[LAGRAPH_MSG_LEN];
	// execute Minimum Spanning Forest
	GrB_Info msf_res =
		LAGraph_msf(&pdata->w_tree, A_w, false, msg);
	if(msf_res != GrB_SUCCESS) {
		GrB_free(&A);
		GrB_free(&A_w);
		return PROCEDURE_ERR;
	}
	if(maxSF)
	{
		info = GrB_Matrix_apply(
			pdata->w_tree, NULL, NULL, GrB_AINV_FP64, pdata->w_tree, NULL) ;
		ASSERT(info == GrB_SUCCESS);
	}
	GrB_Index n;
	info = GrB_Matrix_nrows(&n, pdata->w_tree);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&pdata->tree, GrB_UINT64, n, n);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_assign(
		pdata->tree, pdata->w_tree, NULL, A, GrB_ALL, n, GrB_ALL, n, GrB_DESC_S
	);
	ASSERT(info == GrB_SUCCESS);
	
	// clean up algorithm inputs
	info = GrB_free(&A_w);
	info |= GrB_free(&A);
	ASSERT(info == GrB_SUCCESS);
	
	//--------------------------------------------------------------------------
	// initialize iterator
	//--------------------------------------------------------------------------

	info = GxB_Iterator_new(&pdata->it);
	ASSERT(info == GrB_SUCCESS);

	// iterate over spanning tree
	info = GxB_Matrix_Iterator_attach(pdata->it, pdata->tree, NULL);
	ASSERT(info == GrB_SUCCESS);
    pdata->info = GxB_Matrix_Iterator_seek(pdata->it, 0);
	return PROCEDURE_OK;
}

// yield edge and its weight.
// yields NULL if there are no additional edges to return
SIValue *Proc_MSFStep
(
	ProcedureCtx *ctx  // procedure context
) {
	ASSERT(ctx->privateData != NULL);

	MSF_Context *pdata = (MSF_Context *)ctx->privateData;

	// depleted
	if(pdata->info == GxB_EXHAUSTED) {
		return NULL;
	}

	// retrieve node from graph
	GrB_Index node_i, node_j;
	GxB_Matrix_Iterator_getIndex(pdata->it, &node_i, &node_j);
	Edge edge;
	EdgeID edgeID = (EdgeID) GxB_Iterator_get_UINT64(pdata->it);
	ASSERT(SCALAR_ENTRY(edgeID)) ;
	
	// prep for next call to Proc_BetweennessStep
	pdata->info = GxB_Matrix_Iterator_next(pdata->it);

	//--------------------------------------------------------------------------
	// set outputs
	//--------------------------------------------------------------------------
	if(pdata->yield_edge || pdata->yield_weight)
	{
		bool edge_flag = Graph_GetEdge(pdata->g, edgeID, &pdata->edge);
		ASSERT(edge_flag) ;
	}
	if(pdata->yield_edge) 
	{
		pdata->edge.src_id = (NodeID) node_i;
		pdata->edge.dest_id = (NodeID) node_j;
		pdata->edge.relationID = GRAPH_UNKNOWN_RELATION;
		if(pdata->relationCount > 0) {
			for(RelationID relID = 0; relID < pdata->relationCount; relID++) {
				if(Graph_CheckAndSetEdgeRelationID(
					pdata->g, &pdata->edge, pdata->relationIDs[relID])) {
					break;
				}
			}
			if(pdata->edge.relationID == GRAPH_UNKNOWN_RELATION){
				pdata->edge.src_id = (NodeID) node_j;
				pdata->edge.dest_id = (NodeID) node_i;
				for(RelationID relID = 0; relID < pdata->relationCount; relID++) {
					if(Graph_CheckAndSetEdgeRelationID(
						pdata->g, &pdata->edge, pdata->relationIDs[relID])) {
						break;
					}
				}
			}
		} else {
			Graph_FindAndSetEdgeRelationID(pdata->g, &pdata->edge);
			if(pdata->edge.relationID == GRAPH_UNKNOWN_RELATION){
				pdata->edge.src_id = (NodeID) node_j;
				pdata->edge.dest_id = (NodeID) node_i;
				Graph_FindAndSetEdgeRelationID(pdata->g, &pdata->edge);
			}
		}
		ASSERT(pdata->edge.relationID != GRAPH_UNKNOWN_RELATION);
		*pdata->yield_edge = SI_Edge(&pdata->edge);
	}
	if(pdata->yield_weight) {
		double weight_val = 0; 
		GrB_Info info = GrB_Matrix_extractElement_FP64(
			&weight_val, pdata->w_tree, node_i, node_j);
		// must be found since w_tree and tree have the same structure. 
		ASSERT(info == GrB_SUCCESS);
		*pdata->yield_weight = SI_DoubleVal(weight_val);
	}
	return pdata->output;
}

ProcedureResult Proc_MSFFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData != NULL) {
		MSF_Context *pdata = ctx->privateData;

		if(pdata->tree		 != NULL) GrB_free(&pdata->tree);
		if(pdata->w_tree	 != NULL) GrB_free(&pdata->w_tree);
		if(pdata->it         != NULL) GrB_free(&pdata->it);
		if(pdata->nodes      != NULL) GrB_free(&pdata->nodes);
		array_free(pdata->relationIDs);
		rm_free(ctx->privateData);
	}
	return PROCEDURE_OK;
}

// CALL algo.MSF({nodeLabels: ['Person'], relationshipTypes: ['KNOWS'],
// attribute: 'Years', objective: 'Minimum'}) YIELD node, score
ProcedureCtx *Proc_MSFCtx(void) {
	void *privateData = NULL;

	ProcedureOutput *outputs         = array_new(ProcedureOutput, 2);
	ProcedureOutput output_edge      = {.name = "edge", .type = T_EDGE};
	ProcedureOutput output_weight = {.name = "weight", .type = T_DOUBLE};
	// ProcedureOutput output_weight = {.name = "weight", .type = SI_NUMERIC};

	array_append(outputs, output_edge);
	array_append(outputs, output_weight);

	ProcedureCtx *ctx = ProcCtxNew("algo.MSF",
								   PROCEDURE_VARIABLE_ARG_COUNT,
								   outputs,
								   Proc_MSFStep,
								   Proc_MSFInvoke,
								   Proc_MSFFree,
								   privateData,
								   true);
	return ctx;
}

