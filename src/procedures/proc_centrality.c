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

#include <string.h>
#include "hll/hll.h"

#define HLL_BITS 10              // 1024 registers, ~3.25% estimation error
#define CENTRALITY_MAX_ITER 100  // maximum BFS levels to propagate

// closeness invoke examples:
//
// CALL algo.closeness() YIELD node, score
// CALL algo.closeness(NULL) YIELD node, score
// CALL algo.closeness({nodeLabels: ['L', 'P']}) YIELD node, score
// CALL algo.closeness({relationshipTypes: ['R', 'E']}) YIELD node, score
// CALL algo.closeness({nodeLabels: ['L'], relationshipTypes: ['E'],
//   weightAttribute: 'value', default:0}) YIELD node, score

// Centrality procedure context
typedef struct {
	const Graph *g;           // graph
	AttributeID weight_prop;  // weight attribute id
	GrB_Vector  scores;       // harmonic centrality scores (FP64, index = nodeID)
	GrB_Info    info;         // iterator state
	GxB_Iterator it;          // iterator over scores
	Node        node;         // current node
	SIValue output[2];        // array with up to 2 entries [node, score]
	SIValue *yield_node;      // nodes
	SIValue *yield_score;     // score
} Centrality_Context;

// process procedure yield
static void _process_yield
(
	Centrality_Context *ctx,
	const char **yield
) {
	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("node", yield[i]) == 0) {
			ctx->yield_node = ctx->output + idx;
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

// process procedure configuration argument
static bool _read_config
(
	SIValue config,         // procedure configuration
	LabelID **lbls,         // [output] labels
	RelationID **rels,      // [output] relationships
	AttributeID *weightAtt, // [output] relationship used as weight
	SIValue *defaultW       // [output] the default value of the weightAtt
) {
	ASSERT(lbls            != NULL);
	ASSERT(rels            != NULL);
	ASSERT(defaultW        != NULL);
	ASSERT(weightAtt       != NULL);
	ASSERT(SI_TYPE(config) == T_MAP);  // expecting configuration to be a map

	// initialize outputs
	*lbls      = NULL;
	*rels      = NULL;
	*defaultW  = SI_NullVal();
	*weightAtt = ATTRIBUTE_ID_NONE;

	uint match_fields = 0;
	uint n = Map_KeyCount(config);
	if(n > 4) {
		// error config contains unknown key
		ErrorCtx_SetError("invalid centrality configuration");
		return false;
	}

	SIValue v;
	LabelID *_lbls    = NULL;
	GraphContext *gc  = QueryCtx_GetGraphCtx();
	RelationID *_rels = NULL;

	//--------------------------------------------------------------------------
	// read labels
	//--------------------------------------------------------------------------

	if(MAP_GETCASEINSENSITIVE(config, "nodeLabels", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("harmonic centrality configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			// error
			ErrorCtx_SetError("harmonic centrality configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		_lbls = array_new(LabelID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue lbl = SIArray_Get(v, i);
			const char *label = lbl.stringval;
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
			if(s == NULL) {
				// error
				ErrorCtx_SetError(
					"harmonic centrality configuration contains non-existent label:%s", label);
				goto error;
			}

			LabelID lbl_id = Schema_GetID(s);
			array_append(_lbls, lbl_id);
		}
		*lbls = _lbls;

		match_fields++;
	}

	//--------------------------------------------------------------------------
	// read relationship-types
	//--------------------------------------------------------------------------

	if(MAP_GETCASEINSENSITIVE(config, "relationshipTypes", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("harmonic centrality configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			ErrorCtx_SetError("harmonic centrality configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		_rels = array_new(RelationID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue rel = SIArray_Get(v, i);
			const char *relation = rel.stringval;
			Schema *s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
			if(s == NULL) {
				// error
				ErrorCtx_SetError(
					"harmonic centrality configuration contains non-existent type:%s", relation);
				goto error;
			}

			RelationID rel_id = Schema_GetID(s);
			array_append(_rels, rel_id);
		}
		*rels = _rels;

		match_fields++;
	}

	//--------------------------------------------------------------------------
	// read weight attribute
	//--------------------------------------------------------------------------

	if(MAP_GETCASEINSENSITIVE(config, "weightAttribute", v)) {
		if(SI_TYPE(v) != T_STRING) {
			ErrorCtx_SetError("harmonic centrality configuration, 'weightAttribute' should be a string");
			goto error;
		}

		const char *attr = v.stringval;
		*weightAtt = GraphContext_GetAttributeID(gc, attr);
		if(*weightAtt == ATTRIBUTE_ID_NONE) {
			ErrorCtx_SetError("harmonic centrality configuration, unknown attribute: %s", attr);
			goto error;
		}
		match_fields++;
	}

	//--------------------------------------------------------------------------
	// read objective (min/max)
	//--------------------------------------------------------------------------

	if(MAP_GETCASEINSENSITIVE(config, "defaultWeight", v)) {
		if(SI_TYPE(v) != T_INT64) {
			ErrorCtx_SetError("harmonic centrality configuration, 'defaultWeight' should be a string");
			goto error;
		}
		memcpy(defaultW, &v, sizeof(SIValue));
		match_fields++;
	}

	// validate no unknown configuration fields
	if(n != match_fields) {
		ErrorCtx_SetError("harmonic centrality configuration contains unknown key");
		goto error;
	}

	return true;

error:
	// clean up

	if (_lbls != NULL) {
		array_free(_lbls);
		*lbls = NULL;
	}

	if (_rels != NULL) {
		array_free(_rels);
		*rels = NULL;
	}

	return false;
}

// compute harmonic closeness centrality estimates using HLL BFS propagation
//
// each node maintains an HLL sketch of "reachable nodes seen so far".
// at BFS level d, sketches are propagated along edges.
// the harmonic contribution delta/d is accumulated into each node's score,
// where delta is the number of new nodes discovered at that level.
static ProcedureResult _calculate_centrality
(
	GrB_Vector *scores,            // output: FP64 scores by original node ID
	const GrB_Matrix A,            // adjacency matrix (original indices)
	const GrB_Vector node_weights  // participating nodes (original indices)
) {
	ASSERT(scores       != NULL);
	ASSERT(A            != NULL);
	ASSERT(node_weights != NULL);

	GrB_Matrix    _A         = NULL;
	GxB_Container score_cont = NULL;
	GxB_Container A_cont     = NULL;
	GrB_Index     nrows      = 0;
	GrB_Index     nvals      = 0;

	GrB_OK(GrB_Vector_size(&nrows, node_weights));
	GrB_OK(GrB_Vector_nvals(&nvals, node_weights));

	//--------------------------------------------------------------------------
	// build compact submatrix _A
	//--------------------------------------------------------------------------

	GrB_Descriptor desc = NULL;
	GrB_OK(GrB_Descriptor_new(&desc));
	GrB_OK(GrB_set(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST));
	GrB_OK(GrB_set(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST));
	GrB_OK(GrB_Matrix_new(&_A, GrB_BOOL, nvals, nvals));
	GrB_set(_A, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT);
	GrB_set(_A, GxB_SPARSE, GxB_SPARSITY_CONTROL);
	GrB_OK(GxB_Matrix_extract_Vector(
		_A, NULL, NULL, A, node_weights, node_weights, desc));
	GrB_free(&desc);

	//--------------------------------------------------------------------------
	// create scores vector (0.0 at each participating node)
	//--------------------------------------------------------------------------

	GrB_OK(GrB_Vector_new(scores, GrB_FP64, nrows));
	GrB_OK(GrB_Vector_assign_FP64(*scores, node_weights, NULL, 0.0,
			GrB_ALL, 0, GrB_DESC_S));
	GrB_OK (GrB_set (*scores, GxB_SPARSE | GxB_FULL, GxB_SPARSITY_CONTROL));
	GrB_OK(GxB_Container_new(&score_cont));
	GrB_OK(GxB_unload_Vector_into_Container(*scores, score_cont, NULL));


	//--------------------------------------------------------------------------
	// unload compact matrix into container for raw CSR access
	//--------------------------------------------------------------------------

	GrB_OK(GxB_Container_new(&A_cont));
	GrB_OK(GxB_unload_Matrix_into_Container(_A, A_cont, NULL));
	GrB_free(&_A);

	uint32_t *A_p = NULL;
	uint32_t *A_i = NULL;
	GrB_Type  p_type, i_type;
	uint64_t  p_n, i_n, p_size, i_size;
	int       p_handling, i_handling;

	GrB_OK(GxB_Vector_unload(A_cont->p, (void **) &A_p, &p_type,
			&p_n, &p_size, &p_handling, NULL));
	GrB_OK(GxB_Vector_unload(A_cont->i, (void **) &A_i, &i_type,
			&i_n, &i_size, &i_handling, NULL));
	GrB_OK (GxB_Container_free(&A_cont));

	if((p_type != GrB_INT32 && p_type != GrB_UINT32)
		|| (i_type != GrB_INT32 && i_type != GrB_UINT32)) {
		ErrorCtx_SetError(
			"algo.Centrality: unexpected index type (graph too large)");
		rm_free(A_p);
		rm_free(A_i);
		GrB_free(scores);
		GxB_Container_free(&score_cont);
		return PROCEDURE_ERR;
	}

	//--------------------------------------------------------------------------
	// HLL BFS propagation
	//--------------------------------------------------------------------------

	struct HLL *new_sets    = rm_calloc(nvals, sizeof(struct HLL));
	struct HLL *old_sets    = rm_calloc(nvals, sizeof(struct HLL));
	double     *flat_scores = rm_calloc(nvals, sizeof(double));
	size_t      reg_size    = (size_t) 1 << HLL_BITS;

	// initialize each node's HLL with node_weights[nodeID] distinct
	// hashes. For boolean weights this is 1 hash per node, but the structure
	// supports weighted nodes where more hashes reflect a larger contribution
	GrB_Type weight_t = NULL;
	GrB_OK (GxB_Vector_type(&weight_t, node_weights));
	ASSERT(weight_t == GrB_BOOL || weight_t == GrB_INT64);
	if (weight_t == GrB_INT64){
		int64_t max_w = 0, min_w = 0;
		GrB_OK (GrB_Vector_reduce_INT64(
			&max_w, NULL, GrB_MAX_MONOID_INT64, node_weights, NULL));
		GrB_OK (GrB_Vector_reduce_INT64(
			&min_w, NULL, GrB_MIN_MONOID_INT64, node_weights, NULL));
		if (min_w < 0 || max_w > 127){
			ErrorCtx_SetError(
				"algo.Centrality: node weights to large (>127)");
			rm_free(A_p);
			rm_free(A_i);
			GrB_free(scores);
			GxB_Container_free(&score_cont);
			return PROCEDURE_ERR;
		}
	}

	GxB_Iterator nw_it;
	GrB_OK(GxB_Iterator_new(&nw_it));
	GrB_OK(GxB_Vector_Iterator_attach(nw_it, node_weights, NULL));
	GrB_Info nw_info = GxB_Vector_Iterator_seek(nw_it, 0);

	for(GrB_Index k = 0; nw_info != GxB_EXHAUSTED; k++) {
		hll_init(&new_sets[k], HLL_BITS);
		hll_init(&old_sets[k], HLL_BITS);

		GrB_Index w = weight_t == GrB_INT64 ?
			GxB_Iterator_get_INT64(nw_it) : GxB_Iterator_get_BOOL(nw_it);
		for(GrB_Index h = 0; h < w; h++) {
			GrB_Index seed[2] = {k, h};
			hll_add(&new_sets[k], seed, sizeof(seed));
		}
		nw_info = GxB_Vector_Iterator_next(nw_it);
	}

	GrB_free(&nw_it);

	for(int d = 1; d <= CENTRALITY_MAX_ITER; d++) {
		// snapshot current sketches before propagating
		for(GrB_Index k = 0; k < nvals; k++) {
			memcpy(old_sets[k].registers, new_sets[k].registers, reg_size);
		}

		bool changed = false;

		// merge each neighbor's pre-round set into this node's set;
		// each node i only writes new_sets[i] and reads old_sets (snapshot),
		// so iterations are fully independent
		for(GrB_Index i = 0; i < nvals; i++) {
			double old_count = hll_count(&old_sets[i]);

			for(GrB_Index k = A_p[i]; k < A_p[i + 1]; k++) {
				hll_merge(&new_sets[i], &old_sets[A_i[k]]);
			}

			double delta = hll_count(&new_sets[i]) - old_count;
			if(delta > 0) {
				flat_scores[i] += delta / d;
				changed = true;
			}
		}

		// stop when no HLL cardinality changed this round
		if(!changed) break;
	}

	//--------------------------------------------------------------------------
	// write flat_scores into score_cont->x, clear iso, reload scores vector
	//--------------------------------------------------------------------------

	// discard the 1-element iso x, create a fresh vector, then load flat_scores
	GrB_OK(GrB_free(&score_cont->x));
	GrB_OK(GrB_Vector_new(&score_cont->x, GrB_FP64, nvals));
	GrB_OK(GxB_Vector_load(score_cont->x, (void **)&flat_scores, GrB_FP64,
			nvals, nvals * sizeof(double), GrB_DEFAULT, NULL));
	score_cont->iso = false;

	GrB_OK (GxB_load_Vector_from_Container(*scores, score_cont, NULL));
	GrB_OK (GxB_Container_free(&score_cont));

	GrB_Index check_nvals = 0;
	GrB_OK(GrB_Vector_nvals(&check_nvals, *scores));
	ASSERT(check_nvals == nvals);

	//--------------------------------------------------------------------------
	// cleanup
	//--------------------------------------------------------------------------

	for(GrB_Index k = 0; k < nvals; k++) {
		hll_destroy(&new_sets[k]);
		hll_destroy(&old_sets[k]);
	}

	rm_free(new_sets);
	rm_free(old_sets);
	rm_free(A_p);
	rm_free(A_i);
	GrB_free(&score_cont);
	GrB_free (&A_cont);
	GrB_free (&_A);

	return PROCEDURE_OK;
}

// invoke the procedure
ProcedureResult Proc_CentralityInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	const char **yield    // procedure outputs
) {
	// expecting a single argument
	size_t l = array_len((SIValue *) args);

	if(l > 1) {
		ErrorCtx_SetError("algo.centrality expects a single argument");
		return PROCEDURE_ERR;
	}

	SIValue config;

	// empty config map incase one wasn't provided
	if(l == 0 || SIValue_IsNull(args[0])) {
		config = SI_Map(0);
	} else {
		config = SI_CloneValue(args[0]);
	}

	// arg0 can be either a map or NULL
	SIType t = SI_TYPE(config);
	if(t != T_MAP) {
		SIValue_Free(config);

		ErrorCtx_SetError("invalid argument to algo.Centrality");
		return PROCEDURE_ERR;
	}

	// read Centrality invoke configuration
	// {
	//	nodeLabels: ['A', 'B'],
	//	relationshipTypes: ['R'],
	//	weightAttribute: 'score',
	//	defaultWeight: 0
	// }

	LabelID    *lbls      = NULL;   // filtered labels
	RelationID *rels      = NULL;   // filtered relationships
	SIValue     defaultW  = {0};    // default weight
	AttributeID weightAtt = ATTRIBUTE_ID_NONE;

	//--------------------------------------------------------------------------
	// load configuration map
	//--------------------------------------------------------------------------

	bool config_ok = _read_config(config, &lbls, &rels, &weightAtt, &defaultW);
	
	SIValue_Free(config);

	if(!config_ok) {
		//error set by _read_config
		return PROCEDURE_ERR;
	}

	//--------------------------------------------------------------------------
	// setup procedure context
	//--------------------------------------------------------------------------

	Graph *g = QueryCtx_GetGraph();
	Centrality_Context *pdata = rm_calloc(1, sizeof(Centrality_Context));

	pdata->g             = g;
	pdata->weight_prop   = weightAtt;

	_process_yield(pdata, yield);

	// save private data
	ctx->privateData = pdata;

	//--------------------------------------------------------------------------
	// run centrality
	//--------------------------------------------------------------------------

	GrB_Matrix A      = NULL;
	GrB_Vector nodes  = NULL;
	GrB_Vector scores = NULL;
	bool sym          = false;
	bool compact      = true;

	GrB_OK(Build_Matrix(&A, &nodes, g, lbls, array_len(lbls), rels,
			array_len(rels), sym, compact));

	array_free(lbls);
	array_free(rels);

	ASSERT(A     != NULL);
	ASSERT(nodes != NULL);

	if (weightAtt == ATTRIBUTE_ID_NONE) {
		GrB_OK (GrB_Vector_assign_BOOL(
			nodes, nodes, NULL, true, GrB_ALL, 0, GrB_DESC_S));
	} else {
		get_node_attribute (nodes, g, weightAtt, defaultW, T_BOOL | T_INT64) ;
	}


	ProcedureResult res = _calculate_centrality(&scores, A, nodes);

	GrB_free(&A);
	GrB_free(&nodes);

	if(res != PROCEDURE_OK) {
		GrB_free (&scores);
		return PROCEDURE_ERR;
	}

	pdata->scores = scores;

	//--------------------------------------------------------------------------
	// initialize iterator directly over scores (index = nodeID, value = score)
	//--------------------------------------------------------------------------

	GrB_OK(GxB_Iterator_new(&pdata->it));
	GrB_OK(GxB_Vector_Iterator_attach(pdata->it, pdata->scores, NULL));
	pdata->info = GxB_Vector_Iterator_seek(pdata->it, 0);

	return PROCEDURE_OK;
}



// yields node and its score
// yields NULL if there are no additional nodes to return
SIValue *Proc_CentralityStep
(
	ProcedureCtx *ctx  // procedure context
) {
	ASSERT(ctx->privateData != NULL);

	Centrality_Context *pdata = (Centrality_Context *) ctx->privateData;

	// skip any node IDs that no longer exist in the graph
	GrB_Index node_id;
	while(pdata->info != GxB_EXHAUSTED) {
		node_id = GxB_Vector_Iterator_getIndex(pdata->it);

		if(Graph_GetNode(pdata->g, node_id, &pdata->node)) {
			break;
		}

		pdata->info = GxB_Vector_Iterator_next(pdata->it);
	}

	if(pdata->info == GxB_EXHAUSTED) {
		return NULL;
	}

	//--------------------------------------------------------------------------
	// set outputs
	//--------------------------------------------------------------------------

	if(pdata->yield_node != NULL) {
		*pdata->yield_node = SI_Node(&pdata->node);
	}

	if(pdata->yield_score != NULL) {
		*pdata->yield_score = SI_DoubleVal(GxB_Iterator_get_FP64(pdata->it));
	}

	// advance iterator for next call
	pdata->info = GxB_Vector_Iterator_next(pdata->it);

	return pdata->output;
}

ProcedureResult Proc_CentralityFree
(
	ProcedureCtx *ctx
) {
	if(ctx->privateData != NULL) {
		Centrality_Context *pdata = ctx->privateData;

		if(pdata->it     != NULL) GrB_free(&pdata->it);
		if(pdata->scores != NULL) GrB_free(&pdata->scores);

		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

// CALL algo.centrality({
//     nodeLabels:         ['Person'],
//     relationshipTypes:  ['KNOWS'],
//     weightAttribute:    'Power',
//     defaultWeight:      '0'
// })
// YIELD node, score
ProcedureCtx *Proc_CentralityCtx(void) {
	ProcedureOutput *outputs      = array_new(ProcedureOutput, 2);
	ProcedureOutput output_node   = {.name = "node",  .type = T_NODE};
	ProcedureOutput output_score  = {.name = "score", .type = T_DOUBLE};

	array_append(outputs, output_node);
	array_append(outputs, output_score);

	ProcedureCtx *ctx = ProcCtxNew("algo.Centrality",
								   PROCEDURE_VARIABLE_ARG_COUNT,
								   outputs,
								   Proc_CentralityStep,
								   Proc_CentralityInvoke,
								   Proc_CentralityFree,
								   NULL,
								   true);
	return ctx;
}

