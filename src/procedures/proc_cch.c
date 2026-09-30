/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "proc_cch.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../effects/effects.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"
#include "../graph/graph_hub.h"
#include "../graph/graphcontext.h"
#include "../graph/tensor/tensor.h"
#include "../algorithms/cch.h"

#include <math.h>

// CALL algo.CCH({relTypes: ['ROAD'],
//                weightProp: 'cost',
//                shortcutRelType: 'SHORTCUT',
//                rankProp: 'rank'}) YIELD shortcutsCreated
//
// Builds a Customizable Contraction Hierarchy over the sub-graph induced by
// 'relTypes', for the metric 'weightProp', and commits it to the graph:
//   - every improving chordal arc becomes a 'shortcutRelType' edge carrying
//     the customized weight under 'weightProp'
//   - every node receives 'rankProp' = its elimination rank
// A later rank-aware bidirectional Dijkstra over ROAD + SHORTCUT edges then
// answers point-to-point shortest paths quickly.

typedef struct {
	int64_t  shortcuts_created;         // total shortcut edges created
	bool     done;                      // true once Step emitted its one row
	SIValue  output[1];                 // result returned
	SIValue *yield_shortcuts_created;   // yield shortcutsCreated
} CCHProcCtx;

// resolve which output slots the caller asked to YIELD
static void _process_yield
(
	CCHProcCtx *ctx,      // [in/out] procedure private context to populate
	const char **yield    // caller-requested yield column names
) {
	ctx->yield_shortcuts_created = NULL ;

	for (uint i = 0; i < arr_len (yield); i++) {
		if (strcasecmp ("shortcutsCreated", yield [i]) == 0) {
			ctx->yield_shortcuts_created = ctx->output ;
			return ;
		}
	}
}

// validate config map and resolve every key to its concrete id. every key is
// required; an unresolvable relType / weightProp is a hard error -- this
// procedure has real side effects, so silently running over less data than the
// caller asked for is worse than failing loudly.
static bool _read_config
(
	SIValue config,             // procedure configuration
	RelationID **relTypeIDs,    // [output] relation types forming the graph
	AttributeID *weightAtt,     // [output] edge weight attribute
	RelationID *shortcutRelID,  // [output] relation type for shortcut edges
	AttributeID *rankAttrID,    // [output] node attribute for elimination rank
	AttributeID *middleAttrID   // [output] shortcut-edge attribute for middle id
) {
	ASSERT (weightAtt        != NULL)  ;
	ASSERT (relTypeIDs       != NULL)  ;
	ASSERT (rankAttrID       != NULL)  ;
	ASSERT (shortcutRelID    != NULL)  ;
	ASSERT (middleAttrID     != NULL)  ;
	ASSERT (SI_TYPE (config) == T_MAP) ;

	*weightAtt     = ATTRIBUTE_ID_NONE ;
	*relTypeIDs    = NULL ;
	*rankAttrID    = ATTRIBUTE_ID_NONE ;
	*shortcutRelID = GRAPH_NO_RELATION ;
	*middleAttrID  = ATTRIBUTE_ID_NONE ;

	uint n = Map_KeyCount (config) ;
	if (n != 5) {
		ErrorCtx_SetError ("algo.CCH configuration expects exactly 5 keys: "
				"relTypes, weightProp, shortcutRelType, rankProp, "
				"middleProp") ;
		return false ;
	}

	SIValue v ;
	GraphContext *gc = QueryCtx_GetGraphCtx () ;
	RelationID *_relTypeIDs = NULL ;

	//--------------------------------------------------------------------------
	// relTypes
	//--------------------------------------------------------------------------

	if (!MAP_GETCASEINSENSITIVE (config, "relTypes", v)) {
		ErrorCtx_SetError ("algo.CCH configuration missing required key: "
				"relTypes") ;
		goto error ;
	}

	if (SI_TYPE (v) != T_ARRAY || !SIArray_AllOfType (v, T_STRING)) {
		ErrorCtx_SetError ("algo.CCH configuration, 'relTypes' should be an "
				"array of strings") ;
		goto error ;
	}

	u_int32_t rel_count = SIArray_Length (v) ;
	if (rel_count == 0) {
		ErrorCtx_SetError ("algo.CCH configuration, 'relTypes' must not be "
				"empty") ;
		goto error ;
	}

	_relTypeIDs = arr_new (RelationID, rel_count) ;
	for (u_int32_t i = 0; i < rel_count; i++) {
		SIValue rel = SIArray_Get (v, i) ;
		const char *type = rel.stringval ;
		Schema *s = GraphContext_GetSchema (gc, type, SCHEMA_EDGE) ;
		if (s == NULL) {
			ErrorCtx_SetError ("algo.CCH configuration contains non-existent "
					"relationship type: %s", type) ;
			goto error ;
		}

		arr_append (_relTypeIDs, Schema_GetID (s)) ;
	}
	*relTypeIDs = _relTypeIDs ;

	//--------------------------------------------------------------------------
	// weightProp
	//--------------------------------------------------------------------------

	if (!MAP_GETCASEINSENSITIVE (config, "weightProp", v)) {
		ErrorCtx_SetError ("algo.CCH configuration missing required key: "
				"weightProp") ;
		goto error ;
	}

	if (!(SI_TYPE (v) & T_STRING)) {
		ErrorCtx_SetError ("algo.CCH configuration, 'weightProp' should be a "
				"string") ;
		goto error ;
	}

	*weightAtt = GraphContext_GetAttributeID (gc, v.stringval) ;
	if (*weightAtt == ATTRIBUTE_ID_NONE) {
		ErrorCtx_SetError ("algo.CCH configuration, unknown attribute: %s",
				v.stringval) ;
		goto error ;
	}

	//--------------------------------------------------------------------------
	// shortcutRelType
	//--------------------------------------------------------------------------

	if (!MAP_GETCASEINSENSITIVE (config, "shortcutRelType", v)) {
		ErrorCtx_SetError ("algo.CCH configuration missing required key: "
				"shortcutRelType") ;
		goto error ;
	}

	if (!(SI_TYPE (v) & T_STRING)) {
		ErrorCtx_SetError ("algo.CCH configuration, 'shortcutRelType' should "
				"be a string") ;
		goto error ;
	}

	Schema *s = GraphHub_AddSchema (gc, v.stringval, SCHEMA_EDGE, true) ;
	*shortcutRelID = Schema_GetID (s) ;

	//--------------------------------------------------------------------------
	// rankProp
	//--------------------------------------------------------------------------

	if (!MAP_GETCASEINSENSITIVE (config, "rankProp", v)) {
		ErrorCtx_SetError ("algo.CCH configuration missing required key: "
				"rankProp") ;
		goto error ;
	}

	if (!(SI_TYPE (v) & T_STRING)) {
		ErrorCtx_SetError ("algo.CCH configuration, 'rankProp' should be a "
				"string") ;
		goto error ;
	}

	*rankAttrID = GraphHub_FindOrAddAttribute (gc, v.stringval, true) ;

	//--------------------------------------------------------------------------
	// middleProp -- shortcut-edge attribute holding the middle node's id, used
	// to unpack a shortcut back into the two (road or shortcut) edges it spans
	//--------------------------------------------------------------------------

	if (!MAP_GETCASEINSENSITIVE (config, "middleProp", v)) {
		ErrorCtx_SetError ("algo.CCH configuration missing required key: "
				"middleProp") ;
		goto error ;
	}

	if (!(SI_TYPE (v) & T_STRING)) {
		ErrorCtx_SetError ("algo.CCH configuration, 'middleProp' should be a "
				"string") ;
		goto error ;
	}

	*middleAttrID = GraphHub_FindOrAddAttribute (gc, v.stringval, true) ;

	return true ;

error:
	if (_relTypeIDs != NULL) {
		arr_free (_relTypeIDs) ;
		*relTypeIDs = NULL ;
	}
	return false ;
}

// resolves a single edge's weight, defaulting to 1 if 'attr_id' is
// missing/non-numeric on this particular edge -- matches Dijkstra/AStar's
// per-edge fallback convention
static double _edge_weight
(
	const Graph *g,       // graph owning the edge
	AttributeID attr_id,  // weight attribute to read
	EdgeID id             // edge whose weight is resolved
) {
	Edge e ;
	bool found = Graph_GetEdge (g, id, &e) ;
	ASSERT (found == true) ;

	SIValue w = GraphEntity_GetNumericPropertyOrDefault ((GraphEntity *)&e, attr_id,
			SI_LongVal (1)) ;
	return SI_GET_NUMERIC (w) ;
}

// context for the GraphBLAS IndexUnaryOp that resolves each matrix entry to a
// weight
typedef struct {
	const Graph *g;       // graph being queried
	AttributeID attr_id;  // attribute id that holds the weight
} EdgeWeightContext;

// GraphBLAS IndexUnaryOp callback: reads the weight attribute off the edge(s)
// at (i,j) and writes it to *z. a tensor cell is either a scalar EdgeID
// (SCALAR_ENTRY) or, for parallel edges, a GrB_Vector of EdgeIDs (AS_VECTOR) --
// in the latter case the cheapest parallel edge wins.
static void _get_edge_weight
(
	double *z,                    // [output] weight value
	const void *x,                // entry value (EdgeID, scalar or tagged vector)
	GrB_Index i,                  // row index -- unused
	GrB_Index j,                  // col index -- unused
	const EdgeWeightContext *ctx  // user-supplied context (theta)
) {
	uint64_t entry = *(const uint64_t *)x ;

	if (SCALAR_ENTRY (entry)) {
		*z = _edge_weight (ctx->g, ctx->attr_id, (EdgeID)entry) ;
		return ;
	}

	// multi-edge cell: the vector's stored indices are the parallel edges'
	// EdgeIDs -- take the cheapest
	GrB_Vector ids = AS_VECTOR (entry) ;

	struct GB_Iterator_opaque _it ;
	GxB_Iterator it = &_it ;
	GrB_OK (GxB_Vector_Iterator_attach (it, ids, NULL)) ;

	double min_w = INFINITY ;
	GrB_Info info = GxB_Vector_Iterator_seek (it, 0) ;
	while (info != GxB_EXHAUSTED) {
		EdgeID id = (EdgeID) GxB_Vector_Iterator_getIndex (it) ;
		double w = _edge_weight (ctx->g, ctx->attr_id, id) ;
		if (w < min_w) {
			min_w = w ;
		}
		info = GxB_Vector_Iterator_next (it) ;
	}

	*z = min_w ;
}

static GrB_Type         ctx_type    = NULL              ;
static GrB_IndexUnaryOp get_weight  = NULL              ;
static pthread_once_t index_op_once = PTHREAD_ONCE_INIT ;

static void _init_tensor_ops
(
	void
) {
	GrB_OK (GrB_Type_new (&ctx_type, sizeof (EdgeWeightContext))) ;

	GrB_OK (GrB_IndexUnaryOp_new (&get_weight,
			(GxB_index_unary_function)_get_edge_weight, GrB_FP64, GrB_UINT64,
			ctx_type)) ;
}

// TODO: switch to get_sub_weight_matrix
// builds a plain GrB_FP64 weight matrix over 'g's full NodeID space (row/col k
// IS NodeID k, always). each relation type's matrix is exported and its EdgeID
// entries resolved to weights in bulk via the IndexUnaryOp above; entries from
// different relation types landing on the same (src,dst) pair collapse to the
// cheapest one via GrB_MIN_FP64.
static GrB_Matrix _build_weight_matrix
(
	Graph *g,                     // graph providing the relation matrices
	const RelationID *relTypeIDs, // relation types forming the sub-graph
	uint relCount,                // number of relation types
	AttributeID weightAtt         // edge attribute holding the weight
) {
	GrB_Index dim = Graph_RequiredMatrixDim (g) ;

	GrB_Matrix A_w = NULL ;
	GrB_OK (GrB_Matrix_new (&A_w, GrB_FP64, dim, dim)) ;

	EdgeWeightContext w_ctx = { .g = g, .attr_id = weightAtt } ;

	GrB_Scalar ctx_scalar = NULL ;

	pthread_once (&index_op_once, _init_tensor_ops) ;

	GrB_OK (GrB_Scalar_new (&ctx_scalar, ctx_type)) ;
	GrB_OK (GrB_Scalar_setElement_UDT (ctx_scalar, (void *)&w_ctx)) ;

	for (uint r = 0; r < relCount; r++) {
		Delta_Matrix R = Graph_GetRelationMatrix (g, relTypeIDs [r], false) ;

		GrB_Matrix U = NULL ;
		GrB_OK (Delta_Matrix_export (&U, R, GrB_UINT64, NULL)) ;

		GrB_Matrix Wr = NULL ;
		GrB_OK (GrB_Matrix_new (&Wr, GrB_FP64, dim, dim)) ;
		GrB_OK (GrB_Matrix_apply_IndexOp_Scalar (Wr, NULL, NULL, get_weight, U,
					ctx_scalar, NULL)) ;
		GrB_OK (GrB_free (&U)) ;

		// combine relation types by taking the cheapest parallel edge
		GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp (A_w, NULL, NULL, GrB_MIN_FP64,
					A_w, Wr, NULL)) ;
		GrB_OK (GrB_free (&Wr)) ;
	}

	GrB_OK (GrB_free (&ctx_scalar)) ;

	return A_w ;
}

// materialize every entry of 'S' (the improving shortcuts computed by
// CCH_ExtractShortcuts) as a real shortcut-typed edge, carrying the shortcut's
// weight under 'weightAtt'. returns the number of edges created.
static int64_t _materialize_shortcuts
(
	GraphContext *gc,          // graph context receiving the edges
	GrB_Matrix S,              // improving shortcuts: (r,c) -> customized weight
	GrB_Matrix M,              // middle nodes: (r,c) -> node id splitting the arc
	RelationID shortcutRelID,  // relation type assigned to the created edges
	AttributeID weightAtt,     // attribute the weight is written under
	AttributeID middleAtt      // attribute the middle-node id is written under
) {
	GrB_Index nvals ;
	GrB_OK (GrB_Matrix_nvals (&nvals, S)) ;

	if (nvals == 0) {
		return 0 ;
	}

	Graph *g = GraphContext_GetGraph (gc) ;

	// pre-size the edge DataBlock for the whole batch. no matrix-policy change
	// is needed: this is a single GraphHub_CreateEdges call, so there is no
	// inter-batch matrix flush to defer (unlike the looped bulk-insert path)
	Graph_AllocateEdges (g, nvals) ;

	// collect every shortcut up front so they can all be introduced to the
	// graph in a single GraphHub_CreateEdges batch
	Edge         **edges = arr_new (Edge *,       nvals) ;
	AttributeSet  *sets  = arr_new (AttributeSet, nvals) ;

	GxB_Iterator it ;
	GxB_Iterator_new (&it) ;
	GrB_OK (GxB_Matrix_Iterator_attach (it, S, NULL)) ;
	GrB_Info info = GxB_Matrix_Iterator_seek (it, 0) ;

	while (info == GrB_SUCCESS) {
		GrB_Index r, c ;
		GxB_Matrix_Iterator_getIndex (it, &r, &c) ;
		double w = GxB_Iterator_get_FP64 (it) ;

		// middle node id for this shortcut -- M has an entry for exactly the
		// same (r,c) pairs S does (see CCH_ExtractShortcuts)
		int64_t mid ;
		GrB_OK (GrB_Matrix_extractElement_INT64 (&mid, M, r, c)) ;

		// src_id/dest_id are the only fields the caller must set --
		// GraphHub_CreateEdges fills in id/relationID/attributes itself
		Edge *e = rm_calloc (1, sizeof (Edge)) ;
		e->src_id  = (NodeID) r ;
		e->dest_id = (NodeID) c ;
		arr_append (edges, e) ;

		AttributeSet set = NULL ;
		SIValue wv = SI_DoubleVal (w) ;
		SIValue mv = SI_LongVal (mid) ;
		AttributeSet_Add (&set, &weightAtt, &wv, 1, true) ;
		AttributeSet_Add (&set, &middleAtt, &mv, 1, true) ;
		arr_append (sets, set) ;

		info = GxB_Matrix_Iterator_next (it) ;
	}

	GrB_OK (GrB_free (&it)) ;

	int64_t created = arr_len (edges) ;

	GraphHub_CreateEdges (gc, edges, shortcutRelID, sets, true) ;

	for (int64_t i = 0; i < created; i++) {
		rm_free (edges[i]) ;
	}
	arr_free (edges) ;
	arr_free (sets) ;

	return created ;
}

// build a GrB_INT64 vector holding every real node's elimination rank
// (rank[nodeID] = cch->iperm[nodeID]). only real (non-tombstoned) nodes are
// included so _set_ranks never dereferences a missing node.
static GrB_Vector _build_rank_vector
(
	Graph     *g,     // graph whose real nodes are ranked
	const CCH *cch    // completed hierarchy providing iperm (node -> rank)
) {
	GrB_Vector rank = NULL ;
	GrB_OK (GrB_Vector_new (&rank, GrB_INT64, cch->n)) ;

	GrB_OK (GxB_Vector_load (rank, (void **)(&cch->iperm), GrB_INT64, cch->n,
				sizeof (int64_t) * cch->n, GrB_DEFAULT, NULL)) ;

	GrB_OK (GrB_Vector_wait (rank, GrB_MATERIALIZE)) ;
	return rank ;
}

// sets 'rankAttrID' on every node carrying an entry in 'rank'. mirrors the real
// SET-clause write path (AttributeSet_Update + GraphHub_UpdateEntityProperties
// for the undo-log, EffectsBuffer for replication), not
// GraphEntity_AddProperty, which has neither.
static void _set_ranks
(
	GraphContext *gc,        // graph context (undo-log + effects routing)
	GrB_Vector rank,         // nodeID -> elimination rank
	AttributeID rankAttrID   // node attribute the rank is written under
) {
	Graph *g = GraphContext_GetGraph (gc) ;
	EffectsBuffer *eb = QueryCtx_GetEffectsBuffer () ;

	GxB_Iterator it ;
	GxB_Iterator_new (&it) ;
	GrB_OK (GxB_Vector_Iterator_attach (it, rank, NULL)) ;
	GrB_Info info = GxB_Vector_Iterator_seek (it, 0) ;

	while (info != GxB_EXHAUSTED) {
		GrB_Index idx = GxB_Vector_Iterator_getIndex (it) ;
		int64_t   r   = GxB_Iterator_get_INT64 (it) ;

		Node n = GE_NEW_NODE () ;
		if (!Graph_GetNode (g, (NodeID)idx, &n)) {
			info = GxB_Vector_Iterator_next (it) ;
			continue ;
		}

		AttributeSet cur     = GraphEntity_GetAttributes ((GraphEntity *)&n) ;
		AttributeSet new_set = AttributeSet_Clone (cur) ;

		AttributeSetChangeType change ;
		SIValue rv = SI_LongVal (r) ;
		AttributeSet_Update (&change, &new_set, &rankAttrID, &rv, 1, true) ;

		GraphHub_UpdateEntityProperties (gc, (GraphEntity *)&n, new_set,
				GETYPE_NODE, true) ;

		if (change == CT_ADD) {
			EffectsBuffer_AddEntityAddAttributeEffect (eb, (GraphEntity *)&n,
					rankAttrID, rv, GETYPE_NODE) ;
		} else if (change == CT_UPDATE) {
			EffectsBuffer_AddEntityUpdateAttributeEffect (eb, (GraphEntity *)&n,
					rankAttrID, rv, GETYPE_NODE) ;
		}

		info = GxB_Vector_Iterator_next (it) ;
	}

	GrB_OK (GrB_free (&it)) ;
}

// procedure entry point: validate the config, build the CCH for the requested
// metric, and commit the shortcut edges + node ranks to the graph. all the heavy
// work happens here; Step merely emits the single summary row afterwards.
static ProcedureResult Proc_CCHInvoke
(
	ProcedureCtx *ctx,     // procedure context (receives privateData)
	const SIValue *args,   // invocation args: expects a single config map
	const char **yield     // caller-requested yield column names
) {
	if (arr_len ((SIValue *)args) != 1 || SI_TYPE (args[0]) != T_MAP) {
		ErrorCtx_SetError ("algo.CCH expects a single map argument") ;
		return PROCEDURE_ERR ;
	}

	RelationID *relTypeIDs    = NULL ;
	AttributeID weightAtt     = ATTRIBUTE_ID_NONE ;
	RelationID  shortcutRelID = GRAPH_NO_RELATION ;
	AttributeID rankAttrID    = ATTRIBUTE_ID_NONE ;
	AttributeID middleAttrID  = ATTRIBUTE_ID_NONE ;

	if (!_read_config (args [0], &relTypeIDs, &weightAtt, &shortcutRelID,
				&rankAttrID, &middleAttrID)) {
		return PROCEDURE_ERR ;
	}

	CCHProcCtx *pdata = rm_calloc (1, sizeof (CCHProcCtx)) ;
	ctx->privateData = pdata ;
	_process_yield (pdata, yield) ;

	GraphContext   *gc     = QueryCtx_GetGraphCtx () ;
	Graph          *g      = QueryCtx_GetGraph () ;
	RedisModuleCtx *rm_ctx = QueryCtx_GetRedisModuleCtx () ;

	//--------------------------------------------------------------------------
	// build the road weight matrix (also serves as the CCH topology)
	//--------------------------------------------------------------------------

	RedisModule_Log (rm_ctx, "notice",
			"algo.CCH: building weight matrix over %u relationship type(s)",
			arr_len (relTypeIDs)) ;

	GrB_Matrix W = _build_weight_matrix (g, relTypeIDs, arr_len (relTypeIDs),
			weightAtt) ;
	arr_free (relTypeIDs) ;

	int64_t dim = (int64_t) Graph_RequiredMatrixDim (g) ;

	//--------------------------------------------------------------------------
	// CCH: preprocessing (metric-independent) + customization (metric)
	//--------------------------------------------------------------------------

	RedisModule_Log (rm_ctx, "notice",
			"algo.CCH: computing elimination order + chordal triangulation") ;

	CCH *cch = CCH_New (dim) ;
	CCH_EliminationOrder      (cch, W) ;   // Phase 1a
	CCH_ChordalTriangulation  (cch) ;      // Phase 1b
	CCH_Customize             (cch, W) ;   // Phase 2

	//--------------------------------------------------------------------------
	// commit everything to the graph, retain nothing
	//--------------------------------------------------------------------------

	GrB_Matrix S = NULL ;
	GrB_Matrix M = NULL ;
	CCH_ExtractShortcuts (cch, W, &S, &M) ;
	GrB_OK (GrB_free (&W)) ;

	// TODO: we don't need the rank vector, simply use cch->iperm to set node
	// ranks
	GrB_Vector rank = _build_rank_vector (g, cch) ;
	CCH_Free (cch) ;   // no CCH structure lingers in RAM

	pdata->shortcuts_created =
		_materialize_shortcuts (gc, S, M, shortcutRelID, weightAtt,
				middleAttrID) ;
	_set_ranks (gc, rank, rankAttrID) ;

	GrB_OK (GrB_free (&S)) ;
	GrB_OK (GrB_free (&M)) ;
	GrB_OK (GrB_free (&rank)) ;

	RedisModule_Log (rm_ctx, "notice",
			"algo.CCH: done (%lld shortcut edges created)",
			(long long) pdata->shortcuts_created) ;

	return PROCEDURE_OK ;
}

// emit the single result row (shortcutsCreated) on the first call, then NULL on
// every subsequent call to signal end-of-stream
static SIValue *Proc_CCHStep
(
	ProcedureCtx *ctx    // procedure context (holds privateData)
) {
	ASSERT (ctx->privateData != NULL) ;

	CCHProcCtx *pdata = ctx->privateData ;

	if (pdata->done) {
		return NULL ;
	}

	pdata->done = true ;

	if (pdata->yield_shortcuts_created != NULL) {
		*pdata->yield_shortcuts_created = SI_LongVal (pdata->shortcuts_created) ;
	}

	return pdata->output ;
}

// release the procedure's private context
static ProcedureResult Proc_CCHFree
(
	ProcedureCtx *ctx    // procedure context to tear down
) {
	if (ctx->privateData != NULL) {
		rm_free (ctx->privateData) ;
	}

	return PROCEDURE_OK ;
}

// construct the algo.CCH procedure descriptor: one INT64 output
// (shortcutsCreated); write-capable, so readOnly = false
ProcedureCtx *Proc_CCHCtx (void) {
	ProcedureOutput *outputs = arr_newlen (ProcedureOutput, 1) ;
	outputs[0] = (ProcedureOutput){.name = "shortcutsCreated", .type = T_INT64} ;

	return ProcCtxNew ("algo.CCH", 1, outputs, Proc_CCHStep, Proc_CCHInvoke,
			Proc_CCHFree, NULL, false) ;
}
