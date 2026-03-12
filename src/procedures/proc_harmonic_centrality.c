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
#include <stdio.h>
#include "hll/hll.h"
#include "../util/simple_timer.h"

#define HLL_BITS 10              // 1024 registers, ~3.25% estimation error
#define CENTRALITY_MAX_ITER 100  // maximum BFS levels to propagate

//------------------------------------------------------------------------------
// GraphBLAS Ops
//------------------------------------------------------------------------------
void fdb_hll_init(
	struct HLL *z,
	const uint64_t *x,
	GrB_Index i,
	GrB_Index j,
	bool theta
) {
	int64_t weight = *x;

	hll_init(z, HLL_BITS);

	for(int64_t h = 0; h < weight; h++) {
		GrB_Index seed[2] = {i, h};
		hll_add(z, seed, sizeof(seed));
	}
}

void fdb_hll_merge
(
	struct HLL *z,
	const struct HLL *x,
	const struct HLL *y
) {
	//FIXME: this requires so much memory but needed for pointer safety
	// also probably leaks memory
	void *reg = rm_calloc(z->size, 1);
	memcpy (reg, x->registers, z->size);
	z->registers = reg;
	hll_merge(z, y);
}

void fdb_hll_delta
(
	double *z,
	const struct HLL *x,
	const struct HLL *y
) {
	*z = 0;
	size_t s = (size_t)1 << HLL_BITS;
	bool diff = 0 != memcmp(x->registers, y->registers, s);
	if(diff) {
		*z = hll_count(x) - hll_count(y);
		memcpy(x->registers, y->registers, s);
	}
}

void fdb_hll_second
(
	struct HLL *z,
	const struct HLL *x, //unused
	const struct HLL *y
) {
	z->bits = y->bits;
	z->size = y->size;
	z->registers = y->registers;
}

int64_t print_hll
(
	char *string,        // value is printed to the string
	size_t string_size,  // size of the string array
	const void *value,   // HLL value to print
	int verbose          // if >0, print verbosely; else tersely
) {
	const struct HLL *hll = (const struct HLL *)value;

	if(verbose > 0) {
		return snprintf(string, string_size,
			"HLL{bits=%u, size=%zu, count=%.2f}",
			hll->bits, hll->size, hll_count(hll));
	}

	return snprintf(string, string_size, "HLL{count=%.2f}", hll_count(hll));
}

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
	const Graph *g;            // graph
	AttributeID  weight_prop;  // weight attribute id
	GrB_Vector   scores;       // harmonic centrality scores (FP64, index = nodeID)
	GrB_Info     info;         // iterator state
	GxB_Iterator it;           // iterator over scores
	Node         node;         // current node
	SIValue     *yield_node;   // nodes
	SIValue     *yield_score;  // score
	SIValue      output[2];    // array with up to 2 entries [node, score]
} HarmonicCentrality_Context ;

// process procedure yield
static void _process_yield
(
	HarmonicCentrality_Context *ctx,
	const char **yield
) {
	int idx = 0 ;
	for (uint i = 0; i < array_len(yield); i++) {
		if (strcasecmp ("node", yield [i]) == 0) {
			ctx->yield_node = ctx->output + idx ;
			idx++ ;
			continue ;
		}

		else if (strcasecmp ("score", yield[i]) == 0) {
			ctx->yield_score = ctx->output + idx ;
			idx++ ;
			continue ;
		}

		else {
			ASSERT (false && "unsupported yield") ;
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
	ASSERT (lbls            != NULL) ;
	ASSERT (rels            != NULL) ;
	ASSERT (defaultW        != NULL) ;
	ASSERT (weightAtt       != NULL) ;
	ASSERT (SI_TYPE (config) == T_MAP) ;  // expecting configuration to be a map

	// initialize outputs
	*lbls      = NULL ;
	*rels      = NULL ;
	*defaultW  = SI_DoubleVal (0) ;
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

		_lbls = array_new (LabelID, 0) ;
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
			array_append (_lbls, lbl_id) ;
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

		_rels = array_new (RelationID, 0) ;
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
			array_append (_rels, rel_id) ;
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
	// read objective (min/max)
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE (config, "defaultWeight", v)) {
		if (SI_TYPE (v) != T_INT64) {
			ErrorCtx_SetError ("harmonic centrality configuration, "
					"'defaultWeight' should be an integer") ;
			goto error;
		}
		*defaultW = v ;
		match_fields++ ;
	}

	// validate no unknown configuration fields
	if (n != match_fields) {
		ErrorCtx_SetError ("harmonic centrality configuration contains unknown key") ;
		goto error ;
	}

	return true ;

error:
	// clean up

	if (_lbls != NULL) {
		array_free (_lbls) ;
		*lbls = NULL;
	}

	if (_rels != NULL) {
		array_free (_rels) ;
		*rels = NULL ;
	}

	return false ;
}

// compute harmonic closeness centrality estimates using HLL BFS propagation
//
// each node maintains an HLL sketch of "reachable nodes seen so far"
// at BFS level d, sketches are propagated along edges
// the harmonic contribution delta/d is accumulated into each node's score,
// where delta is the number of new nodes discovered at that level
static ProcedureResult _calculate_centrality
(
	GrB_Vector *scores,            // output: FP64 scores by original node ID
	const GrB_Matrix A,            // adjacency matrix (original indices)
	const GrB_Vector node_weights  // participating nodes (original indices)
) {
	ASSERT (A            != NULL) ;
	ASSERT (scores       != NULL) ;
	ASSERT (node_weights != NULL) ;

	GrB_Matrix    _A         = NULL;
	GxB_Container score_cont = NULL;
	GrB_Index     nrows      = 0;
	GrB_Index     nvals      = 0;

	// simple_timer_t t_total;
	// simple_timer_t t_phase;
	// simple_tic(t_total);

	GrB_OK (GrB_Vector_size(&nrows, node_weights));
	GrB_OK (GrB_Vector_nvals(&nvals, node_weights));
	GrB_OK (GrB_Vector_new(scores, GrB_FP64, nrows));

	if (nvals == 0) {
		return PROCEDURE_OK;
	}

	// printf("[centrality] n_participating=%llu\n", (unsigned long long) nvals);
	
	// double check weight type and maximum weight requirements
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
			return PROCEDURE_ERR;
		}
	}

	//--------------------------------------------------------------------------
	// build compact submatrix _A
	//--------------------------------------------------------------------------

	// simple_tic(t_phase);

	GrB_Descriptor desc = NULL ;
	GrB_OK (GrB_Descriptor_new (&desc)) ;
	GrB_OK (GrB_set (desc, GxB_USE_INDICES, GxB_COLINDEX_LIST)) ;
	GrB_OK (GrB_set (desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST)) ;
	GrB_OK (GrB_Matrix_new (&_A, GrB_BOOL, nvals, nvals)) ;
	GrB_set (_A, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT) ;
	GrB_set (_A, GxB_SPARSE, GxB_SPARSITY_CONTROL) ;
	GrB_OK (GxB_Matrix_extract_Vector (
		_A, NULL, NULL, A, node_weights, node_weights, desc)) ;
	GrB_free (&desc) ;

	//--------------------------------------------------------------------------
	// create scores vector (0.0 at each participating node)
	//--------------------------------------------------------------------------

	GrB_OK (GrB_Vector_assign_FP64(*scores, node_weights, NULL, 0.0,
			GrB_ALL, 0, GrB_DESC_S));
	GrB_OK (GrB_set (*scores, GxB_SPARSE | GxB_FULL, GxB_SPARSITY_CONTROL));
	GrB_OK (GxB_Container_new(&score_cont));
	GrB_OK (GxB_unload_Vector_into_Container(*scores, score_cont, NULL));

	//--------------------------------------------------------------------------
	// Initialize HLL
	//--------------------------------------------------------------------------

	// simple_tic(t_phase);
	GrB_Type      hll_t       = NULL;
	GrB_Vector    new_sets    = NULL;
	GrB_Vector    old_sets    = NULL;
	GxB_Container old_cont    = NULL;
	GrB_Vector    flat_scores = NULL;
	GrB_Vector    flat_weight = NULL;
	GrB_Vector    delta_vec   = NULL;

	GrB_IndexUnaryOp init_hlls      = NULL;
	GrB_BinaryOp     shallow_second = NULL;
	GrB_BinaryOp     merge_hll_biop = NULL;
	GrB_Monoid       merge_hll      = NULL;
	GrB_Semiring     merge_second   = NULL;
	// WARNING: op has side effects
	GrB_BinaryOp     delta_hll      = NULL;


	GrB_OK (GrB_Type_new(&hll_t, sizeof(struct HLL)));
	GrB_OK (GrB_Vector_new (&new_sets, hll_t, nvals));
	GrB_OK (GrB_Vector_new (&old_sets, hll_t, nvals));
	GrB_OK (GxB_Container_new (&old_cont)) ;
	GrB_OK (GrB_Vector_new(&flat_scores, GrB_FP64, nvals));
	GrB_OK (GrB_Vector_new(&flat_weight, GrB_INT64, nvals));
	GrB_OK (GxB_Vector_extractTuples_Vector(
		NULL, flat_weight, node_weights, NULL));

	GrB_OK(GrB_free(&score_cont->x));
	score_cont->x = flat_scores;

	// init op: weight (INT64) at row index i → HLL seeded with 'weight' hashes
	GrB_OK(GrB_IndexUnaryOp_new(&init_hlls,
		(GxB_index_unary_function) fdb_hll_init, hll_t, GrB_INT64, GrB_BOOL));

	// merge binary op (HLL, HLL) -> HLL  in-place: z == x required
	GrB_OK(GrB_BinaryOp_new(&merge_hll_biop,
		(GxB_binary_function) fdb_hll_merge, hll_t, hll_t, hll_t));

	// second op
	GrB_OK(GrB_BinaryOp_new(&shallow_second,
		(GxB_binary_function) fdb_hll_second, hll_t, GrB_BOOL, hll_t));

	// merge monoid — identity is an empty (all-zero) HLL sketch
	struct HLL hll_zero;
	hll_init(&hll_zero, HLL_BITS);
	GrB_OK(GrB_Monoid_new_UDT(&merge_hll, merge_hll_biop, &hll_zero));

	// semiring: add = merge monoid, multiply = copy (pass-through second operand)
	GrB_OK(GrB_Semiring_new(&merge_second, merge_hll, shallow_second));

	// delta op: (HLL_old, HLL_new) → FP64 cardinality change
	// WARNING: side-effect: overwrites old registers with new on any difference
	GrB_OK(GrB_BinaryOp_new(&delta_hll,
		(GxB_binary_function) fdb_hll_delta, GrB_FP64, hll_t, hll_t));

	// delta output: one FP64 entry per participating node
	GrB_OK(GrB_Vector_new(&delta_vec, GrB_FP64, nvals));

	//--------------------------------------------------------------------------
	// Load vector values
	//--------------------------------------------------------------------------
	// initialize HLLs
	GrB_OK (GrB_Vector_apply_IndexOp_BOOL (
		new_sets, NULL, NULL, init_hlls, flat_weight, false, NULL));
	GrB_OK (GrB_Vector_apply_IndexOp_BOOL (
		old_sets, NULL, NULL, init_hlls, flat_weight, false, NULL));
	GrB_OK (GrB_set (old_sets, GxB_BITMAP, GxB_SPARSITY_CONTROL));
	GrB_OK (GrB_Type_set_VOID (hll_t, (void **) &print_hll, GxB_PRINT_FUNCTION,
		sizeof(GxB_print_function)));

	GrB_OK (GrB_free (&flat_weight));
	GrB_OK (GrB_Vector_assign_FP64 (
		flat_scores, NULL, NULL, 0.0, GrB_ALL, 0, NULL));

	//--------------------------------------------------------------------------
	// HLL BFS propagation
	//--------------------------------------------------------------------------

	int64_t changes = 0 ;
	// simple_tic(t_phase);
	for (int d = 1; d <= CENTRALITY_MAX_ITER; d++) {
		simple_timer_t t_loop ;

		changes = 0 ;
		// simple_tic(t_loop);

		// foward bfs
		// merge each neighbor's pre-round set into this node's set
		// target kernel for inplace adjustments
		// GrB_set(GrB_GLOBAL, true, GxB_BURBLE);
		GrB_OK (GrB_mxv(
			new_sets, NULL, merge_hll_biop, merge_second, _A, old_sets, NULL)) ;
		// GrB_set(GrB_GLOBAL, false, GxB_BURBLE);
		// printf("[centrality] merge time: %.2f ms\n",
		// 	TIMER_GET_ELAPSED_MILLISECONDS(t_loop));

		GrB_OK (GxB_unload_Vector_into_Container(old_sets, old_cont, NULL));
		// simple_tic(t_loop);
		// find the delta between last round and this one

		GrB_OK (GrB_eWiseMult(
			delta_vec, NULL, NULL, delta_hll, new_sets, old_cont->x, NULL));
		int32_t status = 0;
		GrB_OK (GrB_get(delta_vec, &status, GxB_SPARSITY_STATUS));
		ASSERT (status == GxB_FULL);

		// old_set bitmap is the set of nodes with non-zero deltas
		GrB_OK (GrB_apply(
			old_cont->b, NULL, NULL, GrB_IDENTITY_BOOL, delta_vec, NULL));
		GrB_OK (GrB_Vector_reduce_INT64(
			&changes, NULL, GrB_PLUS_MONOID_INT64, old_cont->b, NULL));
		old_cont->nvals = changes;
		GrB_OK (GxB_load_Vector_from_Container(old_sets, old_cont, NULL));

		// use the deltas to update the score
		GrB_OK (GrB_apply(flat_scores, NULL, GrB_PLUS_FP64, GrB_DIV_FP64,
			delta_vec, (double) d, NULL));

		// stop when no HLL cardinality changed this round
		if (changes == 0) {
			break ;
		}
	}

	// printf("[centrality] HLL BFS propagation: %.2f ms\n",
	// 	TIMER_GET_ELAPSED_MILLISECONDS(t_phase));

	//--------------------------------------------------------------------------
	// write flat_scores into score_cont->x, clear iso, reload scores vector
	//--------------------------------------------------------------------------

	// simple_tic(t_phase);
	score_cont->iso = false;
	GrB_OK (GxB_load_Vector_from_Container(*scores, score_cont, NULL));
	GrB_OK (GxB_Container_free(&score_cont));

	// printf("[centrality] score write-back: %.2f ms\n",
	// 	TIMER_GET_ELAPSED_MILLISECONDS(t_phase));

	//--------------------------------------------------------------------------
	// cleanup
	//--------------------------------------------------------------------------

	// free operators (semiring before monoid)
	GrB_free(&merge_second);
	GrB_free(&merge_hll);
	GrB_free(&shallow_second);
	GrB_free(&merge_hll_biop);
	GrB_free(&delta_hll);
	GrB_free(&init_hlls);

	// free the monoid identity's register allocation
	hll_destroy(&hll_zero);

	GrB_free(&delta_vec);
	GrB_free(&new_sets);
	GrB_free(&old_sets);
	GrB_free(&old_cont);
	GrB_free(&hll_t);
	GrB_free(&_A);

	// printf("[centrality] _calculate_centrality total: %.2f ms\n",
	// 	TIMER_GET_ELAPSED_MILLISECONDS(t_total));

	return PROCEDURE_OK ;
}

// invoke the procedure
ProcedureResult Proc_CentralityInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	const char **yield    // procedure outputs
) {
	// expecting a single argument
	size_t l = array_len ((SIValue *) args) ;

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

	LabelID    *lbls      ;  // filtered labels
	RelationID *rels      ;  // filtered relationships
	SIValue     defaultW  ;  // default weight
	AttributeID weightAtt ;

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

	GrB_Matrix A      = NULL;
	GrB_Vector nodes  = NULL;
	GrB_Vector scores = NULL;
	bool sym          = false;
	bool compact      = true;

	// simple_timer_t t_invoke;
	// simple_timer_t t_phase;
	// simple_tic(t_invoke);
	// simple_tic(t_phase);

	GrB_OK (Build_Matrix (&A, &nodes, g, lbls, array_len (lbls), rels,
			array_len(rels), sym, compact)) ;

	ASSERT (A     != NULL) ;
	ASSERT (nodes != NULL) ;

	// printf("[centrality] Build_Matrix: %.2f ms\n",
	// 	TIMER_GET_ELAPSED_MILLISECONDS(t_phase));

	array_free (lbls) ;
	array_free (rels) ;

	if (weightAtt == ATTRIBUTE_ID_NONE) {
		GrB_OK (GrB_Vector_assign_BOOL(
			nodes, nodes, NULL, true, GrB_ALL, 0, GrB_DESC_S));
	} else {
		// simple_tic(t_phase);
		get_node_attribute (nodes, g, weightAtt, defaultW, T_BOOL | T_INT64) ;
		// printf("[centrality] get_node_attribute: %.2f ms\n",
		// 	TIMER_GET_ELAPSED_MILLISECONDS(t_phase));
	}


	ProcedureResult res = _calculate_centrality (&scores, A, nodes) ;

	GrB_free (&A) ;
	GrB_free (&nodes) ;

	if (res != PROCEDURE_OK) {
		GrB_free (&scores) ;
		return PROCEDURE_ERR ;
	}

	// printf("[centrality] Proc_CentralityInvoke total: %.2f ms\n",
	// 	TIMER_GET_ELAPSED_MILLISECONDS(t_invoke));

	pdata->scores = scores ;

	//--------------------------------------------------------------------------
	// initialize iterator directly over scores (index = nodeID, value = score)
	//--------------------------------------------------------------------------

	GrB_OK (GxB_Iterator_new (&pdata->it)) ;
	GrB_OK (GxB_Vector_Iterator_attach (pdata->it, pdata->scores, NULL)) ;
	pdata->info = GxB_Vector_Iterator_seek (pdata->it, 0) ;

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
	}

	if (pdata->info == GxB_EXHAUSTED) {
		return NULL ;
	}

	//--------------------------------------------------------------------------
	// set outputs
	//--------------------------------------------------------------------------

	if (pdata->yield_node != NULL) {
		*pdata->yield_node = SI_Node (&pdata->node) ;
	}

	if (pdata->yield_score != NULL) {
		*pdata->yield_score = SI_DoubleVal (GxB_Iterator_get_FP64 (pdata->it)) ;
	}

	// advance iterator for next call
	pdata->info = GxB_Vector_Iterator_next (pdata->it) ;

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
		if (pdata->scores != NULL) {
			GrB_free (&pdata->scores) ;
		}

		rm_free (ctx->privateData) ;
	}

	return PROCEDURE_OK ;
}

// CALL algo.HarmonicCentrality({
//     nodeLabels:         ['Person'],
//     relationshipTypes:  ['KNOWS'],
//     weightAttribute:    'Power',
//     defaultWeight:      '0'
// })
// YIELD node, score
ProcedureCtx *Proc_HarmonicCentralityCtx(void) {
	ProcedureOutput *outputs      = array_new (ProcedureOutput, 2) ;
	ProcedureOutput output_node   = {.name = "node",  .type = T_NODE} ;
	ProcedureOutput output_score  = {.name = "score", .type = T_DOUBLE} ;

	array_append (outputs, output_node) ;
	array_append (outputs, output_score) ;

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

