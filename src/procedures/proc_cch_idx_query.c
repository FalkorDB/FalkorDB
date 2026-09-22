/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "proc_cch_idx_query.h"
#include "../value.h"
#include "../util/arr.h"
#include "../util/dict.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"
#include "../datatypes/path/path.h"
#include "../datatypes/path/sipath.h"
#include "../index/cch_index.h"
#include "../graph/graphcontext.h"
#include "../algorithms/cch.h"
#include "../algorithms/utils/priority_heap.h"

#include <math.h>

// CALL db.idx.cch.query({sourceNode: s, targetNode: t,
//                        relTypes: ['ROAD'], weightProp: 'w'})
//      YIELD pathWeight, path
//
// The hierarchy is queried entirely in rank space: both the forward search
// (from the source) and the backward search (from the target) only ever climb
// rank via the upward graph 'up[]', differing solely in which per-arc weight
// they read -- up_w for the forward direction, dn_w for the backward one. They
// meet at the highest-rank common node; the resulting shortcut arcs are then
// unpacked, via each arc's stored middle rank, into the original road edges.

#define KEY(r) ((void *)(uintptr_t)(r))

typedef struct {
	Path    *path;              // reconstructed road path (NULL if unreachable)
	double   weight;            // total path weight
	bool     done;             // true once Step emitted its row
	SIValue  output[2];        // results returned
	SIValue *yield_weight;     // yield pathWeight
	SIValue *yield_path;       // yield path
} CCHIdxQueryCtx;

// per-rank search record
typedef struct {
	double  dist;       // best known distance from the search's start
	int64_t pred;       // predecessor rank on the best path
	bool    has_pred;   // false only for the start rank
	bool    finalized;  // popped with its optimal distance
} SRec;

// a directed shortcut-level arc, in rank space
typedef struct {
	int64_t f;  // from rank
	int64_t t;  // to rank
} RankArc;

//------------------------------------------------------------------------------
// helpers
//------------------------------------------------------------------------------

// read an edge's weight (defaults to 1 when missing), populating e->attributes
static double _edge_weight
(
	const Graph *g,
	AttributeID weightAtt,
	Edge *e
) {
	Graph_GetEdge (g, e->id, e) ;   // populate attributes
	SIValue w = GraphEntity_GetNumericPropertyOrDefault ((GraphEntity *)e,
			weightAtt, SI_LongVal (1)) ;
	return SI_GET_NUMERIC (w) ;
}

// rank-pruned Dijkstra from 'start' over the upward graph. forward=true relaxes
// each upper neighbor with its up_w (forward) weight; forward=false with its
// dn_w (backward) weight. both directions only ever climb rank. fills 'recs'
// (rank -> SRec*) and appends every discovered rank to 'visited'.
static void _search
(
	const CCH *cch,      // hierarchy being queried
	int64_t    start,    // start rank
	bool       forward,  // forward (up_w) vs backward (dn_w)
	dict      *recs,     // [out] rank -> SRec*
	int64_t  **visited   // [out] arr of discovered ranks
) {
	NodeWeightHeap heap ;
	NodeWeightHeap_init (&heap) ;

	SRec *s0 = rm_calloc (1, sizeof (SRec)) ;
	s0->dist = 0 ; s0->has_pred = false ;
	HashTableAdd (recs, KEY (start), s0) ;
	arr_append (*visited, start) ;
	NodeWeightHeap_offer (&heap, (NodeWeightItem){ .node = (NodeID)start,
			.weight = 0.0 }) ;

	NodeWeightItem it ;
	while (NodeWeightHeap_poll (&heap, &it)) {
		int64_t cur = (int64_t)it.node ;
		SRec *cr = HashTableFetchValue (recs, KEY (cur)) ;
		if (cr->finalized) continue ;   // stale duplicate
		cr->finalized = true ;

		double cur_w = cr->dist ;

		int64_t *nbrs = cch->up   [cur] ;              // upper neighbors (ranks)
		double  *ws   = forward ? cch->up_w [cur]
		                        : cch->dn_w [cur] ;    // parallel weights
		uint32_t m = arr_len (nbrs) ;
		for (uint32_t i = 0; i < m; i++) {
			double w = ws [i] ;
			if (!isfinite (w)) continue ;   // directed arc absent this way
			int64_t b  = nbrs [i] ;
			double  nd = cur_w + w ;

			SRec *nr = HashTableFetchValue (recs, KEY (b)) ;
			if (nr == NULL) {
				nr = rm_calloc (1, sizeof (SRec)) ;
				nr->dist = INFINITY ;
				HashTableAdd (recs, KEY (b), nr) ;
				arr_append (*visited, b) ;
			}

			if (!nr->finalized && nd < nr->dist) {
				nr->dist     = nd ;
				nr->pred     = cur ;
				nr->has_pred = true ;
				NodeWeightHeap_offer (&heap, (NodeWeightItem){ .node = (NodeID)b,
						.weight = nd }) ;
			}
		}
	}

	NodeWeightHeap_free (&heap) ;
}

// locate the chordal arc directed rf -> rt and return its customized weight and
// middle rank. the arc is stored at the lower-ranked endpoint; the direction
// picks up_w/up_mid (rf < rt, upward) or dn_w/dn_mid (rf > rt, downward).
// returns false if no such arc exists (hierarchy inconsistent with the request)
static bool _find_arc
(
	const CCH *cch,   // hierarchy
	int64_t    rf,    // from rank
	int64_t    rt,    // to rank
	double    *w,     // [out] customized weight of arc rf -> rt
	int64_t   *mid    // [out] middle rank, or -1 for an original road edge
) {
	int64_t lo = (rf < rt) ? rf : rt ;
	int64_t hi = (rf < rt) ? rt : rf ;

	int64_t *nbrs = cch->up [lo] ;
	uint32_t m = arr_len (nbrs) ;
	for (uint32_t i = 0; i < m; i++) {
		if (nbrs [i] == hi) {
			if (rf < rt) {
				*w   = cch->up_w   [lo][i] ;
				*mid = cch->up_mid [lo][i] ;
			} else {
				*w   = cch->dn_w   [lo][i] ;
				*mid = cch->dn_mid [lo][i] ;
			}
			return true ;
		}
	}

	return false ;
}

// cheapest real edge realizing the road hop perm[rf] -> perm[rt], across the
// index's relationship types. writes it to '*out' and returns true, or false
// when no such edge exists.
static bool _best_road_edge
(
	Graph            *g,
	NodeID            from,
	NodeID            to,
	const RelationID *rels,
	int               nrels,
	AttributeID       weightAtt,
	Edge             *out       // [output] realizing edge (set iff true)
) {
	Edge *tmp = arr_new (Edge, 4) ;

	for (int i = 0; i < nrels; i++) {
		Graph_GetEdgesConnectingNodes (g, from, to, rels [i], &tmp) ;
	}

	double bw    = INFINITY ;
	bool   found = false ;
	for (uint32_t i = 0; i < arr_len (tmp); i++) {
		Edge e = tmp [i] ;
		double w = _edge_weight (g, weightAtt, &e) ;
		if (w < bw) { bw = w ; *out = e ; found = true ; }
	}

	arr_free (tmp) ;
	return found ;
}

// expand the shortcut arc rf -> rt into road edges, appended in order to 'out'.
// an original arc (middle == -1) resolves to the cheapest real road edge; a
// shortcut recurses through its middle rank (rf -> mid, mid -> rt). returns
// false if the hierarchy is inconsistent (a missing arc or a leaf with no
// realizing road edge), so the caller can fail loudly instead of returning a
// garbage path.
static bool _unpack
(
	const CCH        *cch,
	Graph            *g,
	const RelationID *rels,
	int               nrels,
	AttributeID       weightAtt,
	int64_t           rf,
	int64_t           rt,
	Edge            **out
) {
	double  w ;
	int64_t mid ;
	if (!_find_arc (cch, rf, rt, &w, &mid)) {
		return false ;
	}

	if (mid == -1) {
		// original road edge perm[rf] -> perm[rt]
		NodeID from = (NodeID)cch->perm [rf] ;
		NodeID to   = (NodeID)cch->perm [rt] ;
		Edge   e ;
		if (!_best_road_edge (g, from, to, rels, nrels, weightAtt, &e)) {
			return false ;
		}
		arr_append (*out, e) ;
		return true ;
	}

	// shortcut: unpack the two halves rf -> mid and mid -> rt
	return _unpack (cch, g, rels, nrels, weightAtt, rf,  mid, out) &&
	       _unpack (cch, g, rels, nrels, weightAtt, mid, rt,  out) ;
}

// free a search's records dict + its SRec values
static void _free_recs
(
	dict    *recs,
	int64_t *visited
) {
	for (uint32_t i = 0; i < arr_len (visited); i++) {
		SRec *r = HashTableFetchValue (recs, KEY (visited [i])) ;
		if (r != NULL) rm_free (r) ;
	}
	HashTableRelease (recs) ;
	arr_free (visited) ;
}

//------------------------------------------------------------------------------
// config
//------------------------------------------------------------------------------

static void _process_yield
(
	CCHIdxQueryCtx *ctx,
	const char **yield
) {
	ctx->yield_weight = NULL ;
	ctx->yield_path   = NULL ;

	int idx = 0 ;
	for (uint i = 0; i < arr_len (yield); i++) {
		if (strcasecmp ("pathWeight", yield [i]) == 0) {
			ctx->yield_weight = ctx->output + idx ; idx++ ; continue ;
		}
		if (strcasecmp ("path", yield [i]) == 0) {
			ctx->yield_path = ctx->output + idx ; idx++ ; continue ;
		}
	}
}

static bool _read_config
(
	SIValue      config,
	Node        *src,          // [out]
	Node        *dst,          // [out]
	RelationID **rels,         // [out] relation types the index spans
	int         *relCount,     // [out]
	AttributeID *weightAtt     // [out]
) {
	if (SI_TYPE (config) != T_MAP) {
		ErrorCtx_SetError ("db.idx.cch.query expects a single map argument") ;
		return false ;
	}

	GraphContext *gc = QueryCtx_GetGraphCtx () ;
	SIValue v ;

	// sourceNode / targetNode
	SIValue s, t ;
	if (!MAP_GETCASEINSENSITIVE (config, "sourceNode", s) ||
	    !MAP_GETCASEINSENSITIVE (config, "targetNode", t)) {
		ErrorCtx_SetError ("db.idx.cch.query requires sourceNode and "
				"targetNode") ;
		return false ;
	}
	if (SI_TYPE (s) != T_NODE || SI_TYPE (t) != T_NODE) {
		ErrorCtx_SetError ("db.idx.cch.query, sourceNode/targetNode must be "
				"nodes") ;
		return false ;
	}
	*src = *(Node *)s.ptrval ;
	*dst = *(Node *)t.ptrval ;

	// relTypes
	if (!MAP_GETCASEINSENSITIVE (config, "relTypes", v) ||
	    SI_TYPE (v) != T_ARRAY || !SIArray_AllOfType (v, T_STRING)) {
		ErrorCtx_SetError ("db.idx.cch.query, relTypes must be an array of "
				"strings") ;
		return false ;
	}
	uint32_t rel_count = SIArray_Length (v) ;
	if (rel_count == 0) {
		ErrorCtx_SetError ("db.idx.cch.query, relTypes must not be empty") ;
		return false ;
	}
	RelationID *_rels = arr_new (RelationID, rel_count) ;
	for (uint32_t i = 0; i < rel_count; i++) {
		SIValue rt = SIArray_Get (v, i) ;
		Schema *rs = GraphContext_GetSchema (gc, rt.stringval, SCHEMA_EDGE) ;
		if (rs == NULL) {
			ErrorCtx_SetError ("db.idx.cch.query, unknown relationship type: %s",
					rt.stringval) ;
			arr_free (_rels) ;
			return false ;
		}
		arr_append (_rels, Schema_GetID (rs)) ;
	}
	*rels     = _rels ;
	*relCount = arr_len (_rels) ;

	// weightProp
	if (!MAP_GETCASEINSENSITIVE (config, "weightProp", v) ||
	    !(SI_TYPE (v) & T_STRING)) {
		ErrorCtx_SetError ("db.idx.cch.query requires string weightProp") ;
		arr_free (_rels) ;
		return false ;
	}
	*weightAtt = GraphContext_GetAttributeID (gc, v.stringval) ;
	if (*weightAtt == ATTRIBUTE_ID_NONE) {
		ErrorCtx_SetError ("db.idx.cch.query, unknown attribute: %s",
				v.stringval) ;
		arr_free (_rels) ;
		return false ;
	}

	return true ;
}

//------------------------------------------------------------------------------
// procedure entry points
//------------------------------------------------------------------------------

static ProcedureResult Proc_CCHIdxQueryInvoke
(
	ProcedureCtx  *ctx,
	const SIValue *args,
	const char   **yield
) {
	if (arr_len ((SIValue *)args) != 1) {
		ErrorCtx_SetError ("db.idx.cch.query expects a single map argument") ;
		return PROCEDURE_ERR ;
	}

	Node        src, dst ;
	RelationID *rels      = NULL ;
	int         relCount  = 0 ;
	AttributeID weightAtt = ATTRIBUTE_ID_NONE ;

	if (!_read_config (args [0], &src, &dst, &rels, &relCount, &weightAtt)) {
		return PROCEDURE_ERR ;
	}

	GraphContext *gc = QueryCtx_GetGraphCtx () ;
	Graph        *g  = QueryCtx_GetGraph () ;

	// locate the CCH index for this (relTypes, weightProp)
	CCHIndex *idx = GraphContext_GetCCHIndex (gc, rels, relCount, weightAtt) ;
	if (idx == NULL || idx->cch == NULL) {
		ErrorCtx_SetError ("db.idx.cch.query: no CCH index over these "
				"relationship types and weight attribute; create one with "
				"db.idx.cch.create") ;
		arr_free (rels) ;
		return PROCEDURE_ERR ;
	}
	const CCH *cch = idx->cch ;

	CCHIdxQueryCtx *pdata = rm_calloc (1, sizeof (CCHIdxQueryCtx)) ;
	pdata->weight = INFINITY ;
	ctx->privateData = pdata ;
	_process_yield (pdata, yield) ;

	NodeID s_id = ENTITY_GET_ID (&src) ;
	NodeID t_id = ENTITY_GET_ID (&dst) ;

	// degenerate: source == target -> single-node path, weight 0
	if (s_id == t_id) {
		pdata->weight = 0 ;
		pdata->path   = Path_New (1) ;
		Node n = GE_NEW_NODE () ;
		Graph_GetNode (g, s_id, &n) ;
		Path_AppendNode (pdata->path, n) ;
		arr_free (rels) ;
		return PROCEDURE_OK ;
	}

	// nodes outside the ranked id-space cannot be on any hierarchy path
	if (s_id >= (NodeID)cch->n || t_id >= (NodeID)cch->n) {
		arr_free (rels) ;
		return PROCEDURE_OK ;   // no path
	}

	int64_t rs = cch->iperm [s_id] ;
	int64_t rt = cch->iperm [t_id] ;

	// forward (from source) and backward (from target) rank-pruned searches
	dict    *fwd  = HashTableCreate (&def_dt) ;
	dict    *bwd  = HashTableCreate (&def_dt) ;
	int64_t *fvis = arr_new (int64_t, 16) ;
	int64_t *bvis = arr_new (int64_t, 16) ;

	_search (cch, rs, true,  fwd, &fvis) ;
	_search (cch, rt, false, bwd, &bvis) ;

	// meet at the shared node minimizing fwd.dist + bwd.dist
	double  best = INFINITY ;
	int64_t meet = -1 ;
	for (uint32_t i = 0; i < arr_len (fvis); i++) {
		int64_t r  = fvis [i] ;
		SRec   *fr = HashTableFetchValue (fwd, KEY (r)) ;
		SRec   *br = HashTableFetchValue (bwd, KEY (r)) ;
		if (fr == NULL || br == NULL) continue ;
		double tot = fr->dist + br->dist ;
		if (tot < best) { best = tot ; meet = r ; }
	}

	ProcedureResult res = PROCEDURE_OK ;

	if (meet != -1) {
		// shortcut-level arc sequence rs -> ... -> meet -> ... -> rt
		RankArc *arcs = arr_new (RankArc, 16) ;

		// forward half: walk preds meet -> rs, collect reversed
		RankArc *fa  = arr_new (RankArc, 8) ;
		int64_t  cur = meet ;
		while (true) {
			SRec *r = HashTableFetchValue (fwd, KEY (cur)) ;
			if (!r->has_pred) break ;
			arr_append (fa, ((RankArc){ .f = r->pred, .t = cur })) ;
			cur = r->pred ;
		}
		for (int64_t i = (int64_t)arr_len (fa) - 1; i >= 0; i--) {
			arr_append (arcs, fa [i]) ;
		}
		arr_free (fa) ;

		// backward half: walk preds meet -> rt, already in forward order
		cur = meet ;
		while (true) {
			SRec *r = HashTableFetchValue (bwd, KEY (cur)) ;
			if (!r->has_pred) break ;
			arr_append (arcs, ((RankArc){ .f = cur, .t = r->pred })) ;
			cur = r->pred ;
		}

		// unpack every shortcut arc into its underlying road edges
		Edge *road = arr_new (Edge, 32) ;
		bool  ok   = true ;
		for (uint32_t i = 0; i < arr_len (arcs); i++) {
			if (!_unpack (cch, g, rels, relCount, weightAtt, arcs [i].f,
						arcs [i].t, &road)) {
				ok = false ;
				break ;
			}
		}
		arr_free (arcs) ;

		if (ok) {
			// build the path: n0 -e0- n1 -e1- ... nK
			pdata->weight = best ;
			pdata->path   = Path_New (arr_len (road) + 1) ;

			Node n0 = GE_NEW_NODE () ;
			Graph_GetNode (g, s_id, &n0) ;
			Path_AppendNode (pdata->path, n0) ;

			for (uint32_t i = 0; i < arr_len (road); i++) {
				Edge e = road [i] ;
				Graph_GetEdge (g, e.id, &e) ;   // populate attributes for the path
				Path_AppendEdge (pdata->path, e) ;

				Node nn = GE_NEW_NODE () ;
				Graph_GetNode (g, e.dest_id, &nn) ;
				Path_AppendNode (pdata->path, nn) ;
			}
		} else {
			// a shortcut had no middle arc or a leaf had no realizing road
			// edge: the hierarchy is inconsistent with the graph. fail loudly
			// rather than return a garbage path
			ErrorCtx_SetError ("db.idx.cch.query: inconsistent hierarchy "
					"(rebuild it with db.idx.cch.create)") ;
			res = PROCEDURE_ERR ;
		}
		arr_free (road) ;
	}

	_free_recs (fwd, fvis) ;
	_free_recs (bwd, bvis) ;
	arr_free (rels) ;

	return res ;
}

static SIValue *Proc_CCHIdxQueryStep
(
	ProcedureCtx *ctx
) {
	ASSERT (ctx->privateData != NULL) ;
	CCHIdxQueryCtx *pdata = ctx->privateData ;

	if (pdata->done || pdata->path == NULL) return NULL ;
	pdata->done = true ;

	if (pdata->yield_weight) *pdata->yield_weight = SI_DoubleVal (pdata->weight) ;
	if (pdata->yield_path)   *pdata->yield_path   = SIPath_Wrap (&pdata->path) ;
	else                     Path_Free (pdata->path) ;
	pdata->path = NULL ;   // ownership transferred (or freed) above

	return pdata->output ;
}

static ProcedureResult Proc_CCHIdxQueryFree
(
	ProcedureCtx *ctx
) {
	if (ctx->privateData != NULL) {
		CCHIdxQueryCtx *pdata = ctx->privateData ;
		if (pdata->path != NULL) Path_Free (pdata->path) ;
		rm_free (pdata) ;
	}
	return PROCEDURE_OK ;
}

ProcedureCtx *Proc_CCHIdxQueryCtx (void) {
	ProcedureOutput *outputs = arr_newlen (ProcedureOutput, 2) ;
	outputs[0] = (ProcedureOutput){ .name = "pathWeight", .type = T_DOUBLE } ;
	outputs[1] = (ProcedureOutput){ .name = "path",       .type = T_PATH } ;

	return ProcCtxNew ("db.idx.cch.query", 1, outputs, Proc_CCHIdxQueryStep,
			Proc_CCHIdxQueryInvoke, Proc_CCHIdxQueryFree, NULL, true) ;
}
