/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"

#include "LAGraphX.h"
#include "GraphBLAS.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../datatypes/map.h"
#include "../util/mt19937-64.h"
#include "../datatypes/array.h"
#include "./utility/internal.h"
#include "../graph/graphcontext.h"
#include "proc_harmonic_centrality.h"

#include <string.h>
#include <stdio.h>
#include "procedures/proc_ctx.h"

// closeness invoke examples:
//
// CALL algo.closeness() YIELD node, score
// CALL algo.closeness(NULL) YIELD node, score
// CALL algo.closeness({nodeLabels: ['L', 'P']}) YIELD node, score
// CALL algo.closeness({relationshipTypes: ['R', 'E']}) YIELD node, score
// CALL algo.closeness({nodeLabels: ['L'], relationshipTypes: ['E'],
//   weightAttribute: 'value', default:0}) YIELD node, score

// Harmonic Centrality procedure context
typedef struct {
	const Graph *g;                // graph
	AttributeID  weight_prop;      // weight attribute id
	GrB_Vector   scores;           // harmonic centrality scores (FP64)
	GrB_Vector   reachable_nodes;  // reachable node estimates (INT64)
	GrB_Info     info;             // iterator state
	GxB_Iterator it;               // iterator over scores
	GxB_Iterator it_reach;         // iterator over reachable_nodes (same sparsity as scores)
	Node         node;             // current node
	SIValue     *yield_node;       // nodes
	SIValue     *yield_score;      // score
	SIValue     *yield_reachable;  // reachable node estimate
	SIValue      output[3];        // array with up to 3 entries [node, score, reachable]
} HarmonicCentrality_Context ;

// process procedure yield
static void _process_yield
(
	HarmonicCentrality_Context *ctx,
	const char **yield
) {
	int idx = 0 ;

	for (uint i = 0 ; i < arr_len (yield) ; i++) {
		if (strcasecmp ("node", yield [i]) == 0) {
			ctx->yield_node = ctx->output + idx ;
			idx++ ;
		}

		else if (strcasecmp ("score", yield [i]) == 0) {
			ctx->yield_score = ctx->output + idx ;
			idx++ ;
		}

		else if (strcasecmp ("reachable", yield[i]) == 0) {
			ctx->yield_reachable = ctx->output + idx ;
			idx++ ;
		}

		else {
			// unknown yield fields are silently skipped;
			// the query parser validates yields before reaching here
		}
	}
}

// process procedure configuration argument
static bool _read_config
(
	SIValue config,          // procedure configuration
	LabelID **lbls,          // [output] labels
	RelationID **rels,       // [output] relationships
	AttributeID *weightAtt,  // [output] relationship used as weight
	SIValue *defaultW        // [output] the default value of the weightAtt
) {
	ASSERT (lbls             != NULL) ;
	ASSERT (rels             != NULL) ;
	ASSERT (defaultW         != NULL) ;
	ASSERT (weightAtt        != NULL) ;
	ASSERT (SI_TYPE (config) == T_MAP) ;  // expecting configuration to be a map

	// initialize outputs
	*lbls      = NULL ;
	*rels      = NULL ;
	*defaultW  = SI_NullVal();
	*weightAtt = ATTRIBUTE_ID_NONE ;

	uint match_fields = 0 ;
	uint n = Map_KeyCount (config) ;
	if (n > 4) {
		// error config contains unknown key
		ErrorCtx_SetError ("invalid HarmonicCentrality configuration") ;
		return false ;
	}

	SIValue v ;
	LabelID *_lbls    = NULL ;
	GraphContext *gc  = QueryCtx_GetGraphCtx () ;
	RelationID *_rels = NULL ;

	//--------------------------------------------------------------------------
	// read labels
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE (config, "nodeLabels", v)) {
		if (SI_TYPE (v) != T_ARRAY) {
			ErrorCtx_SetError ("harmonic centrality configuration, "
					"'nodeLabels' should be an array of strings") ;
			goto error ;
		}

		if (!SIArray_AllOfType (v, T_STRING)) {
			// error
			ErrorCtx_SetError ("harmonic centrality configuration, "
					"'nodeLabels' should be an array of strings") ;
			goto error ;
		}

		_lbls = arr_new (LabelID, 0) ;
		u_int32_t l = SIArray_Length (v) ;
		for (u_int32_t i = 0; i < l; i++) {
			SIValue lbl = SIArray_Get (v, i) ;
			const char *label = lbl.stringval ;
			Schema *s = GraphContext_GetSchema (gc, label, SCHEMA_NODE) ;
			if (s == NULL) {
				// error
				ErrorCtx_SetError ("harmonic centrality configuration contains "
						"non-existent label:%s", label) ;
				goto error ;
			}

			LabelID lbl_id = Schema_GetID (s) ;
			arr_append (_lbls, lbl_id) ;
		}
		*lbls = _lbls ;

		match_fields++ ;
	}

	//--------------------------------------------------------------------------
	// read relationship-types
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE(config, "relationshipTypes", v)) {
		if (SI_TYPE (v) != T_ARRAY) {
			ErrorCtx_SetError ("harmonic centrality configuration, "
					"'relationshipTypes' should be an array of strings") ;
			goto error ;
		}

		if (!SIArray_AllOfType (v, T_STRING)) {
			ErrorCtx_SetError ("harmonic centrality configuration, "
					"'relationshipTypes' should be an array of strings") ;
			goto error ;
		}

		_rels = arr_new (RelationID, 0) ;
		u_int32_t l = SIArray_Length (v) ;
		for (u_int32_t i = 0; i < l; i++) {
			SIValue rel = SIArray_Get (v, i) ;
			const char *relation = rel.stringval ;
			Schema *s = GraphContext_GetSchema (gc, relation, SCHEMA_EDGE) ;
			if (s == NULL) {
				// error
				ErrorCtx_SetError ("harmonic centrality configuration contains "
						"non-existent type:%s", relation) ;
				goto error ;
			}

			RelationID rel_id = Schema_GetID (s) ;
			arr_append (_rels, rel_id) ;
		}
		*rels = _rels ;

		match_fields++ ;
	}

	//--------------------------------------------------------------------------
	// read weight attribute
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE (config, "weightAttribute", v)) {
		if (SI_TYPE (v) != T_STRING) {
			ErrorCtx_SetError ("harmonic centrality configuration, "
					"'weightAttribute' should be a string") ;
			goto error ;
		}

		const char *attr = v.stringval ;
		*weightAtt = GraphContext_GetAttributeID (gc, attr) ;
		if (*weightAtt == ATTRIBUTE_ID_NONE) {
			ErrorCtx_SetError ("harmonic centrality configuration, "
					"unknown attribute: %s", attr) ;
			goto error ;
		}
		match_fields++ ;
	}

	//--------------------------------------------------------------------------
	// read default weight value
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE (config, "defaultWeight", v)) {
		if (SI_TYPE (v) != T_INT64 || v.longval < 0) {
			ErrorCtx_SetError ("harmonic centrality configuration, "
					"'defaultWeight' should be non negative integer") ;
			goto error ;
		}
		*defaultW = v ;

		match_fields++ ;
	}

	// validate no unknown configuration fields
	if (n != match_fields) {
		ErrorCtx_SetError ("harmonic centrality configuration contains unknown key") ;
		goto error ;
	}

	// defaultWeight requires weightAttribute
	if (!SIValue_IsNull (*defaultW) && *weightAtt == ATTRIBUTE_ID_NONE) {
		ErrorCtx_SetError ("harmonic centrality configuration, "
				"'defaultWeight' requires 'weightAttribute'") ;
		goto error ;
	}

	return true ;

error:
	// clean up

	if (_lbls != NULL) {
		arr_free (_lbls) ;
		*lbls = NULL;
	}

	if (_rels != NULL) {
		arr_free (_rels) ;
		*rels = NULL ;
	}

	return false ;
}


// invoke the procedure
ProcedureResult Proc_CentralityInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	const char **yield    // procedure outputs
) {
	// expecting a single argument
	size_t l = arr_len ((SIValue *) args) ;

	if (l > 1) {
		ErrorCtx_SetError ("algo.HarmonicCentrality expects a single argument");
		return PROCEDURE_ERR ;
	}

	SIValue config ;

	// empty config map incase one wasn't provided
	if (l == 0 || SIValue_IsNull (args[0])) {
		config = SI_Map (0) ;
	} else {
		config = SI_CloneValue (args[0]) ;
	}

	// arg0 can be either a map or NULL
	SIType t = SI_TYPE (config) ;
	if (t != T_MAP) {
		SIValue_Free (config) ;

		ErrorCtx_SetError ("invalid argument to algo.HarmonicCentrality") ;
		return PROCEDURE_ERR ;
	}

	// read HarmonicCentrality invoke configuration
	// {
	//	nodeLabels: ['A', 'B'],
	//	relationshipTypes: ['R'],
	//	weightAttribute: 'score',
	//	defaultWeight: 0
	// }

	LabelID    *lbls = NULL ;  // filtered labels
	RelationID *rels = NULL ;  // filtered relationships
	SIValue     defaultW = SI_NullVal () ;  // default weight
	AttributeID weightAtt = ATTRIBUTE_ID_NONE ;

	//--------------------------------------------------------------------------
	// load configuration map
	//--------------------------------------------------------------------------

	bool config_ok = _read_config (config, &lbls, &rels, &weightAtt, &defaultW) ;

	// free input config
	SIValue_Free (config) ;

	if (!config_ok) {
		// error set by _read_config
		return PROCEDURE_ERR ;
	}

	//--------------------------------------------------------------------------
	// setup procedure context
	//--------------------------------------------------------------------------

	Graph *g = QueryCtx_GetGraph () ;
	HarmonicCentrality_Context *pdata =
		rm_calloc (1, sizeof (HarmonicCentrality_Context)) ;

	pdata->g           = g ;
	pdata->weight_prop = weightAtt ;

	_process_yield (pdata, yield) ;

	// save private data
	ctx->privateData = pdata ;

	//--------------------------------------------------------------------------
	// run centrality
	//--------------------------------------------------------------------------

	GrB_Matrix A               = NULL;   // matrix with the edges of specified types
	GrB_Vector nodes           = NULL;   // vector of participating nodes
	GrB_Vector scores          = NULL;   // score[i] is the centrality of node i
	GrB_Vector reachable_nodes = NULL;   // reachable[i] is the estimated reachable count

	LAGraph_Graph G = NULL ;
	bool sym        = false ;  // matrix is directed
	bool compact    = true ;

	GrB_OK (Build_Matrix (&A, &nodes, g, lbls, arr_len (lbls), rels,
			arr_len(rels), sym, compact)) ;

	ASSERT (A     != NULL) ;
	ASSERT (nodes != NULL) ;

	if (lbls != NULL) {
		arr_free (lbls) ;
	}

	if (rels != NULL) {
		arr_free (rels) ;
	}

	if (weightAtt == ATTRIBUTE_ID_NONE) {
		GrB_OK (GrB_Vector_assign_BOOL (
			nodes, nodes, NULL, true, GrB_ALL, 0, GrB_DESC_S));
	} else {
		if (!get_node_attribute (nodes, g, weightAtt, defaultW,
			T_BOOL | T_INT64)) {
			ErrorCtx_SetError ("harmonic centrality weight attribute specified"
				" with no default value and non-existent or non-integer"
				" attribute was found");
			GrB_free (&A) ;
			GrB_free (&nodes) ;
			return PROCEDURE_ERR;
		}
	}

	char msg [LAGRAPH_MSG_LEN] ;
	GrB_Info info = LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg) ;
	ASSERT (info == GrB_SUCCESS) ;

	GrB_Vector *rnodes = pdata->yield_reachable ? &reachable_nodes : NULL;
	info = LAGr_HarmonicCentrality (&scores, rnodes, G, nodes, msg) ;

	LAGraph_Delete (&G, msg) ;
	GrB_free (&nodes) ;

	if (info != GrB_SUCCESS) {
		GrB_free (&scores) ;
		GrB_free (&reachable_nodes) ;
		return PROCEDURE_ERR ;
	}

	pdata->scores          = scores ;
	pdata->reachable_nodes = reachable_nodes ;

	//--------------------------------------------------------------------------
	// initialize iterator directly over scores (index = nodeID, value = score)
	//--------------------------------------------------------------------------

	// score must be returned by LAGraph, so to simplify the code, we always
	// add an iterator to the scores vector
	GrB_OK (GxB_Iterator_new (&pdata->it)) ;
	GrB_OK (GxB_Vector_Iterator_attach (pdata->it, pdata->scores, NULL)) ;
	pdata->info = GxB_Vector_Iterator_seek (pdata->it, 0) ;

	//--------------------------------------------------------------------------
	// initialize iterator over reachable_nodes (same sparsity pattern as scores)
	//--------------------------------------------------------------------------

	if (pdata->yield_reachable != NULL) {
		GrB_OK (GxB_Iterator_new (&pdata->it_reach)) ;
		GrB_OK (GxB_Vector_Iterator_attach (pdata->it_reach,
			pdata->reachable_nodes, NULL)) ;
		GrB_OK (GxB_Vector_Iterator_seek (pdata->it_reach, 0)) ;
	}

	return PROCEDURE_OK ;
}

// yields node and its score
// yields NULL if there are no additional nodes to return
SIValue *Proc_CentralityStep
(
	ProcedureCtx *ctx  // procedure context
) {
	ASSERT (ctx->privateData != NULL) ;

	HarmonicCentrality_Context *pdata =
		(HarmonicCentrality_Context *) ctx->privateData ;

	// skip any node IDs that no longer exist in the graph
	GrB_Index node_id ;
	while (pdata->info != GxB_EXHAUSTED) {
		node_id = GxB_Vector_Iterator_getIndex (pdata->it) ;

		if (Graph_GetNode (pdata->g, node_id, &pdata->node)) {
			break ;
		}

		pdata->info = GxB_Vector_Iterator_next (pdata->it) ;
		if (pdata->it_reach != NULL) {
			GxB_Vector_Iterator_next (pdata->it_reach) ;
		}
	}

	if (pdata->info == GxB_EXHAUSTED) {
		return NULL ;
	}

	// reachable_nodes has the same sparsity pattern as scores
	ASSERT (pdata->it_reach == NULL ||
		GxB_Vector_Iterator_getIndex (pdata->it_reach) == node_id) ;

	//--------------------------------------------------------------------------
	// set outputs
	//--------------------------------------------------------------------------

	if (pdata->yield_node != NULL) {
		*pdata->yield_node = SI_Node (&pdata->node) ;
	}

	if (pdata->yield_score != NULL) {
		*pdata->yield_score = SI_DoubleVal (GxB_Iterator_get_FP64 (pdata->it)) ;
	}

	if (pdata->yield_reachable != NULL && pdata->it_reach != NULL) {
		*pdata->yield_reachable =
			SI_LongVal (GxB_Iterator_get_INT64 (pdata->it_reach)) ;
	}

	// advance both iterators in lockstep for next call
	pdata->info = GxB_Vector_Iterator_next (pdata->it) ;
	if (pdata->it_reach != NULL) {
		GxB_Vector_Iterator_next (pdata->it_reach) ;
	}

	return pdata->output ;
}

ProcedureResult Proc_CentralityFree
(
	ProcedureCtx *ctx
) {
	if (ctx->privateData != NULL) {
		HarmonicCentrality_Context *pdata = ctx->privateData ;

		if (pdata->it     != NULL) {
			GrB_free (&pdata->it) ;
		}
		if (pdata->it_reach != NULL) {
			GrB_free (&pdata->it_reach) ;
		}
		if (pdata->scores != NULL) {
			GrB_free (&pdata->scores) ;
		}
		if (pdata->reachable_nodes != NULL) {
			GrB_free (&pdata->reachable_nodes) ;
		}

		rm_free (ctx->privateData) ;
	}

	return PROCEDURE_OK ;
}

// CALL algo.HarmonicCentrality({
//     nodeLabels:         ['Person'],
//     relationshipTypes:  ['KNOWS'],
//     weightAttribute:    'Power',
//     defaultWeight:      0
// })
// YIELD node, score, reachable
//
// nodeLabels: optional array of strings. Error on non-existent lable name.
// relationshipTypes: optional array of strings. Error on non-existent
//                    relationship name.
// weightAttribute:   optional string. Error on non-existent name.
// defaultWeight:     optional default weight, non-negative integer.
//                    weightAttribute must have been specified already.
//                    If not given, will error on non-integer or non-existent
//                    weight attribute values. If given, these values will be
//                    treated as the default.

ProcedureCtx *Proc_HarmonicCentralityCtx(void) {
	ProcedureOutput *outputs         = arr_new (ProcedureOutput, 3) ;
	ProcedureOutput output_node      = {.name = "node",      .type = T_NODE}   ;
	ProcedureOutput output_score     = {.name = "score",     .type = T_DOUBLE} ;
	ProcedureOutput output_reachable = {.name = "reachable", .type = T_INT64}  ;

	arr_append (outputs, output_node) ;
	arr_append (outputs, output_score) ;
	arr_append (outputs, output_reachable) ;

	ProcedureCtx *ctx = ProcCtxNew ("algo.HarmonicCentrality",
								   PROCEDURE_VARIABLE_ARG_COUNT,
								   outputs,
								   Proc_CentralityStep,
								   Proc_CentralityInvoke,
								   Proc_CentralityFree,
								   NULL,
								   true) ;
	return ctx ;
}

