/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "proc_contraction_hierarchies.h"
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
#include "../algorithms/utils/node_map.h"
#include "../algorithms/contraction_hierarchies.h"

// CALL algo.contractionHierarchies({relTypes: ['ROAD'],
//                                    weightProp: 'cost',
//                                    shortcutRelType: 'SHORTCUT',
//                                    rankProperty: 'rank'}) YIELD shortcutsCreated

typedef struct {
	int64_t shortcuts_created;        // total shortcut edges created
	bool    done;                     // true once Step has emitted the single result
	SIValue output[1];                // result returned
	SIValue *yield_shortcuts_created;  // yield shortcutsCreated
} CHCtx;

// process procedure yield
static void _process_yield
(
	CHCtx *ctx,
	const char **yield
) {
	ctx->yield_shortcuts_created = NULL ;

	int idx = 0 ;
	for (uint i = 0; i < arr_len (yield); i++) {
		if (strcasecmp ("shortcutsCreated", yield [i]) == 0) {
			ctx->yield_shortcuts_created = ctx->output + idx ;
			return ;
		}
	}
}

// validate config map and resolve every key to its concrete id
// every key here is required and an unresolvable relType / weightProp is a
// hard error this procedure has real side effects, so silently running over
// less data than the caller asked for is worse than failing loudly
static bool _read_config
(
	SIValue config,             // procedure configuration
	RelationID **relTypeIDs,    // [output] relation types to traverse/compact
	AttributeID *weightAtt,     // [output] edge weight attribute
	RelationID *shortcutRelID,  // [output] relation type for shortcut edges
	AttributeID *rankAttrID     // [output] node attribute for contraction rank
) {
	ASSERT (weightAtt        != NULL)  ;
	ASSERT (relTypeIDs       != NULL)  ;
	ASSERT (rankAttrID       != NULL)  ;
	ASSERT (shortcutRelID    != NULL)  ;
	ASSERT (SI_TYPE (config) == T_MAP) ;

	*weightAtt     = ATTRIBUTE_ID_NONE ;
	*relTypeIDs    = NULL ;
	*rankAttrID    = ATTRIBUTE_ID_NONE ;
	*shortcutRelID = GRAPH_NO_RELATION ;

	uint n = Map_KeyCount (config) ;
	if (n != 4) {
		ErrorCtx_SetError ("algo.contractionHierarchies configuration expects "
				"exactly 4 keys: relTypes, weightProp, shortcutRelType, "
				"rankProperty") ;
		return false ;
	}

	SIValue v ;
	GraphContext *gc = QueryCtx_GetGraphCtx () ;
	RelationID *_relTypeIDs = NULL ;

	//--------------------------------------------------------------------------
	// relTypes
	//--------------------------------------------------------------------------

	if (!MAP_GETCASEINSENSITIVE (config, "relTypes", v)) {
		ErrorCtx_SetError ("algo.contractionHierarchies configuration missing "
				"required key: relTypes") ;
		goto error ;
	}

	if (SI_TYPE (v) != T_ARRAY || !SIArray_AllOfType (v, T_STRING)) {
		ErrorCtx_SetError ("algo.contractionHierarchies configuration, "
				"'relTypes' should be an array of strings") ;
		goto error ;
	}

	u_int32_t rel_count = SIArray_Length (v) ;
	if (rel_count == 0) {
		ErrorCtx_SetError ("algo.contractionHierarchies configuration, "
				"'relTypes' must not be empty") ;
		goto error ;
	}

	_relTypeIDs = arr_new (RelationID, rel_count) ;
	for (u_int32_t i = 0; i < rel_count; i++) {
		SIValue rel = SIArray_Get (v, i) ;
		const char *type = rel.stringval ;
		Schema *s = GraphContext_GetSchema (gc, type, SCHEMA_EDGE) ;
		if (s == NULL) {
			ErrorCtx_SetError ("algo.contractionHierarchies configuration "
					"contains non-existent relationship type: %s", type) ;
			goto error ;
		}

		arr_append (_relTypeIDs, Schema_GetID (s)) ;
	}
	*relTypeIDs = _relTypeIDs ;

	//--------------------------------------------------------------------------
	// weightProp
	//--------------------------------------------------------------------------

	if (!MAP_GETCASEINSENSITIVE (config, "weightProp", v)) {
		ErrorCtx_SetError ("algo.contractionHierarchies configuration missing "
				"required key: weightProp") ;
		goto error ;
	}

	if (!(SI_TYPE (v) & T_STRING)) {
		ErrorCtx_SetError ("algo.contractionHierarchies configuration, "
				"'weightProp' should be a string") ;
		goto error ;
	}

	*weightAtt = GraphContext_GetAttributeID (gc, v.stringval) ;
	if (*weightAtt == ATTRIBUTE_ID_NONE) {
		ErrorCtx_SetError ("algo.contractionHierarchies configuration, "
				"unknown attribute: %s", v.stringval) ;
		goto error ;
	}

	//--------------------------------------------------------------------------
	// shortcutRelType
	//--------------------------------------------------------------------------

	if (!MAP_GETCASEINSENSITIVE (config, "shortcutRelType", v)) {
		ErrorCtx_SetError ("algo.contractionHierarchies configuration missing "
				"required key: shortcutRelType") ;
		goto error ;
	}

	if (!(SI_TYPE (v) & T_STRING)) {
		ErrorCtx_SetError ("algo.contractionHierarchies configuration, "
				"'shortcutRelType' should be a string") ;
		goto error ;
	}

	Schema *s = GraphHub_AddSchema (gc, v.stringval, SCHEMA_EDGE, true) ;
	*shortcutRelID = Schema_GetID (s) ;

	//--------------------------------------------------------------------------
	// rankProperty
	//--------------------------------------------------------------------------

	if (!MAP_GETCASEINSENSITIVE (config, "rankProperty", v)) {
		ErrorCtx_SetError ("algo.contractionHierarchies configuration missing "
				"required key: rankProperty") ;
		goto error ;
	}

	if (!(SI_TYPE (v) & T_STRING)) {
		ErrorCtx_SetError ("algo.contractionHierarchies configuration, "
				"'rankProperty' should be a string") ;
		goto error ;
	}

	*rankAttrID = GraphHub_FindOrAddAttribute (gc, v.stringval, true) ;

	return true ;

error:
	if (_relTypeIDs != NULL) {
		arr_free (_relTypeIDs) ;
		*relTypeIDs = NULL ;
	}
	return false ;
}

// context for the GraphBLAS IndexUnaryOp that resolves each matrix entry
// to a weight -- mirrors proc_maxflow.c's EdgeCapacityContext /
// _get_edge_capacity
typedef struct {
	const Graph *g;       // graph being queried
	AttributeID attr_id;  // attribute id that holds the weight
} EdgeWeightContext;

// resolves a single edge's weight, defaulting to 1 if 'attr_id' is
// missing/non-numeric on this particular edge -- matches
// Dijkstra_ShortestPath/AStar_ShortestPath's existing per-edge fallback
// convention
static double _edge_weight
(
	const Graph *g,
	AttributeID attr_id,
	EdgeID id
) {
	Edge e;
	bool found = Graph_GetEdge(g, id, &e);
	ASSERT(found == true);

	SIValue w = _get_value_or_default((GraphEntity *)&e, attr_id,
			SI_LongVal(1));
	return SI_GET_NUMERIC(w);
}

// GraphBLAS IndexUnaryOp callback: reads the weight attribute off the
// edge(s) at (i,j) and writes it to *z. a tensor cell is either a scalar
// EdgeID (SCALAR_ENTRY) or, when the relation type has multiple parallel
// edges between the same node pair, a GrB_Vector of EdgeIDs (AS_VECTOR) --
// in the latter case the cheapest parallel edge wins, the same "cheapest
// parallel edge" convention ContractNode itself relies on when merging
// shortcuts into A/AT (contraction_hierarchies.c)
static void _get_edge_weight
(
	double *z,                    // [output] weight value
	const void *x,                // entry value (EdgeID, scalar or tagged vector)
	GrB_Index i,                  // row index -- unused
	GrB_Index j,                  // col index -- unused
	const EdgeWeightContext *ctx  // user-supplied context (theta)
) {
	uint64_t entry = *(const uint64_t *)x;

	if(SCALAR_ENTRY(entry)) {
		*z = _edge_weight(ctx->g, ctx->attr_id, (EdgeID)entry);
		return;
	}

	// multi-edge cell: the vector's stored indices are the parallel
	// edges' EdgeIDs (see tensor.h) -- take the cheapest
	GrB_Vector ids = AS_VECTOR(entry);

	struct GB_Iterator_opaque _it;
	GxB_Iterator it = &_it;
	GrB_OK(GxB_Vector_Iterator_attach(it, ids, NULL));

	double min_w = INFINITY;
	GrB_Info info = GxB_Vector_Iterator_seek(it, 0);
	while(info != GxB_EXHAUSTED) {
		EdgeID id = (EdgeID)GxB_Vector_Iterator_getIndex(it);
		double w = _edge_weight(ctx->g, ctx->attr_id, id);
		if(w < min_w) {
			min_w = w;
		}
		info = GxB_Vector_Iterator_next(it);
	}

	*z = min_w;
}

// builds a plain GrB_FP64 weight matrix over 'g's full NodeID space (never
// compacted/renumbered -- row/col k IS NodeID k, always, even across
// tombstoned/never-allocated slots), following the same approach
// algo.maxFlow uses to build its capacity matrix: each relation type's
// matrix is exported directly (Delta_Matrix_export) and its EdgeID
// entries resolved to weights in bulk via a GraphBLAS IndexUnaryOp
// (_get_edge_weight above), rather than a manual per-node/per-edge scan.
// entries from different relation types landing on the same (src, dst)
// pair collapse to the cheapest one via GrB_MIN_FP64
static GrB_Matrix _build_weight_matrix
(
	Graph *g,
	const RelationID *relTypeIDs,
	uint relCount,
	AttributeID weightAtt
) {
	GrB_Index dim = Graph_RequiredMatrixDim (g) ;

	GrB_Matrix A_w = NULL ;
	GrB_OK (GrB_Matrix_new (&A_w, GrB_FP64, dim, dim)) ;

	EdgeWeightContext w_ctx = { .g = g, .attr_id = weightAtt };

	GrB_Type         ctx_type   = NULL ;
	GrB_Scalar       ctx_scalar = NULL ;
	GrB_IndexUnaryOp get_weight = NULL ;

	GrB_OK (GrB_Type_new (&ctx_type, sizeof (EdgeWeightContext))) ;
	GrB_OK (GrB_Scalar_new (&ctx_scalar, ctx_type)) ;
	GrB_OK (GrB_Scalar_setElement_UDT (ctx_scalar, (void *)&w_ctx)) ;
	GrB_OK (GrB_IndexUnaryOp_new (&get_weight,
			(GxB_index_unary_function)_get_edge_weight,
			GrB_FP64, GrB_UINT64, ctx_type)) ;

	for (uint r = 0; r < relCount; r++) {
		Delta_Matrix R = Graph_GetRelationMatrix (g, relTypeIDs [r], false) ;

		GrB_Matrix U = NULL ;
		GrB_OK(Delta_Matrix_export (&U, R, GrB_UINT64, NULL)) ;

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
	GrB_OK (GrB_free (&ctx_type)) ;
	GrB_OK (GrB_free (&get_weight)) ;

	return A_w ;
}

// materialize every entry of 'S' (the shortcut overlay computed by
// ContractionHierarchies_Contract) as a real SHORTCUT-typed edge in the
// graph, carrying the shortcut's weight under 'weightAtt'. returns the
// number of edges created.
static int64_t _materialize_shortcuts
(
	GraphContext *gc,
	Graph *g,
	GrB_Matrix S,
	RelationID shortcutRelID,
	AttributeID weightAtt
) {
	GrB_Index nvals ;
	GrB_OK (GrB_Matrix_nvals (&nvals, S)) ;

	if (nvals == 0) {
		return 0 ;
	}

	MATRIX_POLICY policy = Graph_GetMatrixPolicy (g) ;
	Graph_AllocateEdges (g, nvals) ;
	Graph_SetMatrixPolicy (g, SYNC_POLICY_NOP) ;

	// collect every shortcut up front so they can all be introduced to the
	// graph in a single GraphHub_CreateEdges batch
	Edge         **edges = arr_new (Edge *,       nvals) ;
	AttributeSet *sets   = arr_new (AttributeSet, nvals) ;

	GxB_Iterator it ;
	GxB_Iterator_new (&it) ;
	GrB_OK (GxB_Matrix_Iterator_attach (it, S, NULL)) ;
	GrB_Info info = GxB_Matrix_Iterator_seek (it, 0) ;

	while (info == GrB_SUCCESS) {
		GrB_Index r, c ;
		GxB_Matrix_Iterator_getIndex (it, &r, &c) ;
		double w = GxB_Iterator_get_FP64 (it) ;

		// src_id/dest_id are the only fields the caller must set --
		// Graph_CreateEdges fills in id/relationID/attributes itself
		Edge *e = rm_calloc (1, sizeof (Edge)) ;
		e->src_id  = (NodeID) r ;
		e->dest_id = (NodeID) c ;
		arr_append (edges, e) ;

		AttributeSet set = NULL ;
		SIValue wv = SI_DoubleVal (w) ;
		AttributeSet_Add (&set, &weightAtt, &wv, 1, true) ;
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

	Graph_SetMatrixPolicy (g, policy) ;

	return created ;
}

// sets 'rankAttrID' on every node with an entry in 'rank' (see
// ContractionHierarchies_Contract -- only actually-contracted nodes have
// one). mirrors the real SET-clause write path (AttributeSet_Update +
// GraphHub_UpdateEntityProperties for undo-log, EffectsBuffer for
// replication), not GraphEntity_AddProperty, which has neither.
static void _set_ranks
(
	GraphContext *gc,
	Graph *g,
	GrB_Vector rank,
	AttributeID rankAttrID
) {
	EffectsBuffer *eb = QueryCtx_GetEffectsBuffer();

	GxB_Iterator it;
	GxB_Iterator_new(&it);
	GrB_OK(GxB_Vector_Iterator_attach(it, rank, NULL));
	GrB_Info info = GxB_Vector_Iterator_seek(it, 0);

	while(info != GxB_EXHAUSTED) {
		GrB_Index idx = GxB_Vector_Iterator_getIndex(it);
		int64_t r = GxB_Iterator_get_INT64(it);

		Node n = GE_NEW_NODE();
		bool found = Graph_GetNode(g, (NodeID)idx, &n);
		ASSERT(found);

		AttributeSet cur = GraphEntity_GetAttributes((GraphEntity *)&n);
		AttributeSet new_set = AttributeSet_Clone(cur);

		AttributeSetChangeType change;
		SIValue rv = SI_LongVal(r);
		AttributeSet_Update(&change, &new_set, &rankAttrID, &rv, 1, true);

		GraphHub_UpdateEntityProperties(gc, (GraphEntity *)&n, new_set,
				GETYPE_NODE, true);

		if(change == CT_ADD) {
			EffectsBuffer_AddEntityAddAttributeEffect(eb, (GraphEntity *)&n,
					rankAttrID, rv, GETYPE_NODE);
		} else if(change == CT_UPDATE) {
			EffectsBuffer_AddEntityUpdateAttributeEffect(eb, (GraphEntity *)&n,
					rankAttrID, rv, GETYPE_NODE);
		}

		info = GxB_Vector_Iterator_next(it);
	}

	GrB_OK(GrB_free(&it));
}

static ProcedureResult Proc_ContractionHierarchiesInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	if (arr_len ((SIValue *)args) != 1 || SI_TYPE (args[0]) != T_MAP) {
		ErrorCtx_SetError ("algo.contractionHierarchies expects a single map "
				"argument") ;
		return PROCEDURE_ERR ;
	}

	RelationID *relTypeIDs    = NULL ;
	AttributeID weightAtt     = ATTRIBUTE_ID_NONE ;
	RelationID  shortcutRelID = GRAPH_NO_RELATION ;
	AttributeID rankAttrID    = ATTRIBUTE_ID_NONE ;

	if (!_read_config (args [0], &relTypeIDs, &weightAtt, &shortcutRelID,
				&rankAttrID)) {
		return PROCEDURE_ERR ;
	}

	CHCtx *pdata = rm_calloc (1, sizeof (CHCtx)) ;
	ctx->privateData = pdata ;
	_process_yield (pdata, yield) ;

	GraphContext    *gc     = QueryCtx_GetGraphCtx () ;
	Graph           *g      = QueryCtx_GetGraph () ;
	RedisModuleCtx  *rm_ctx = QueryCtx_GetRedisModuleCtx () ;

	RedisModule_Log (rm_ctx, "notice",
			"algo.contractionHierarchies: building weight matrix over %u "
			"relationship type(s)", arr_len (relTypeIDs)) ;

	GrB_Matrix A_w = _build_weight_matrix (g, relTypeIDs, arr_len(relTypeIDs),
			weightAtt) ;
	arr_free (relTypeIDs) ;

	GrB_Index _log_edge_count ;
	GrB_OK (GrB_Matrix_nvals (&_log_edge_count, A_w)) ;
	RedisModule_Log (rm_ctx, "notice",
			"algo.contractionHierarchies: weight matrix built (%llu edges), "
			"starting contraction",
			(unsigned long long) _log_edge_count) ;

	GrB_Matrix S    = NULL ;
	GrB_Vector rank = NULL ;
	ContractionHierarchies_Contract (A_w, &S, &rank) ;
	GrB_OK (GrB_free (&A_w)) ;

	GrB_Index _log_shortcut_count, _log_node_count ;
	GrB_OK (GrB_Matrix_nvals (&_log_shortcut_count, S)) ;
	GrB_OK (GrB_Vector_nvals (&_log_node_count, rank)) ;
	RedisModule_Log (rm_ctx, "notice",
			"algo.contractionHierarchies: contraction complete (%llu nodes "
			"contracted, %llu shortcuts identified), materializing results",
			(unsigned long long) _log_node_count,
			(unsigned long long) _log_shortcut_count) ;

	pdata->shortcuts_created =
		_materialize_shortcuts (gc, g, S, shortcutRelID, weightAtt) ;
	_set_ranks (gc, g, rank, rankAttrID) ;

	GrB_OK (GrB_free (&S)) ;
	GrB_OK (GrB_free (&rank)) ;

	RedisModule_Log (rm_ctx, "notice",
			"algo.contractionHierarchies: done (%lld shortcut edges created)",
			(long long) pdata->shortcuts_created) ;

	return PROCEDURE_OK ;
}

static SIValue *Proc_ContractionHierarchiesStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);

	CHCtx *pdata = ctx->privateData;

	if(pdata->done) {
		return NULL ;
	}

	pdata->done = true ;

	if (pdata->yield_shortcuts_created != NULL) {
		*pdata->yield_shortcuts_created = SI_LongVal (pdata->shortcuts_created) ;
	}

	return pdata->output ;
}

static ProcedureResult Proc_ContractionHierarchiesFree
(
	ProcedureCtx *ctx
) {
	if (ctx->privateData != NULL) {
		rm_free (ctx->privateData) ;
	}

	return PROCEDURE_OK ;
}

ProcedureCtx *Proc_ContractionHierarchiesCtx (void) {
	ProcedureOutput *outputs = arr_newlen (ProcedureOutput, 1) ;
	outputs[0] = (ProcedureOutput){.name = "shortcutsCreated", .type = T_INT64} ;

	return ProcCtxNew ("algo.contractionHierarchies", 1, outputs,
			Proc_ContractionHierarchiesStep, Proc_ContractionHierarchiesInvoke,
			Proc_ContractionHierarchiesFree, NULL, false) ;
}

