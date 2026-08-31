/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "proc_cch_query.h"
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
#include "../graph/graphcontext.h"

#include <math.h>

// CALL algo.CCH.query({sourceNode: s, targetNode: t,
//                      relTypes: ['ROAD'], shortcutRelType: 'SHORTCUT',
//                      weightProp: 'w', rankProperty: 'rank',
//                      middleProp: 'mid'}) YIELD pathWeight, path

#define KEY(id) ((void *)(uintptr_t)(id))

typedef struct {
	Path    *path;              // reconstructed road path (NULL if unreachable)
	double   weight;            // total path weight
	bool     done;             // true once Step emitted its row
	SIValue  output[2];        // results returned
	SIValue *yield_weight;     // yield pathWeight
	SIValue *yield_path;       // yield path
} CCHQueryCtx;

//------------------------------------------------------------------------------
// tiny binary min-heap of (weight, node) keyed on weight
//------------------------------------------------------------------------------

typedef struct { double w; NodeID id; } HItem;
typedef struct { HItem *a; int n; int cap; } MinHeap;

static void _heap_init(MinHeap *h) {
	h->cap = 64; h->n = 0;
	h->a = rm_malloc(sizeof(HItem) * h->cap);
}

static void _heap_free(MinHeap *h) { rm_free(h->a); }

static void _heap_push(MinHeap *h, double w, NodeID id) {
	if(h->n == h->cap) {
		h->cap *= 2;
		h->a = rm_realloc(h->a, sizeof(HItem) * h->cap);
	}
	int i = h->n++;
	h->a[i] = (HItem){ .w = w, .id = id };
	while(i > 0) {
		int p = (i - 1) / 2;
		if(h->a[p].w <= h->a[i].w) break;
		HItem t = h->a[p]; h->a[p] = h->a[i]; h->a[i] = t;
		i = p;
	}
}

static bool _heap_pop(MinHeap *h, HItem *out) {
	if(h->n == 0) return false;
	*out = h->a[0];
	h->a[0] = h->a[--h->n];
	int i = 0;
	while(true) {
		int l = 2 * i + 1, r = 2 * i + 2, s = i;
		if(l < h->n && h->a[l].w < h->a[s].w) s = l;
		if(r < h->n && h->a[r].w < h->a[s].w) s = r;
		if(s == i) break;
		HItem t = h->a[s]; h->a[s] = h->a[i]; h->a[i] = t;
		i = s;
	}
	return true;
}

//------------------------------------------------------------------------------
// helpers
//------------------------------------------------------------------------------

static inline SIValue _get_value_or_default
(
	GraphEntity *ge,
	AttributeID id,
	SIValue default_value
) {
	SIValue v;
	if(!GraphEntity_GetProperty(ge, id, &v)) return default_value;
	if(SI_TYPE(v) & SI_NUMERIC) return v;
	return default_value;
}

// read an edge's weight (defaults to 1 when missing), populating e->attributes
static double _edge_weight(Graph *g, AttributeID weightAtt, Edge *e) {
	Graph_GetEdge(g, e->id, e);   // populate attributes
	SIValue w = _get_value_or_default((GraphEntity *)e, weightAtt, SI_LongVal(1));
	return SI_GET_NUMERIC(w);
}

// elimination rank of a node, cached across lookups (stored as rank+1 so a
// genuine 0 rank isn't confused with "absent")
static int64_t _node_rank
(
	Graph *g,
	AttributeID rankAtt,
	dict *cache,
	NodeID id
) {
	void *v = HashTableFetchValue(cache, KEY(id));
	if(v != NULL) return (int64_t)(intptr_t)v - 1;

	Node n = GE_NEW_NODE();
	Graph_GetNode(g, id, &n);

	SIValue rv;
	int64_t rank = -1;
	if(GraphEntity_GetProperty((GraphEntity *)&n, rankAtt, &rv) &&
			(SI_TYPE(rv) & SI_NUMERIC)) {
		rank = (int64_t)SI_GET_NUMERIC(rv);
	}

	HashTableAdd(cache, KEY(id), (void *)(intptr_t)(rank + 1));
	return rank;
}

// per-node search record
typedef struct {
	double  dist;       // best known distance from the search's start
	NodeID  pred_node;  // predecessor on the best path
	Edge    pred_edge;  // edge (pred_node,this) [fwd] / (this,pred_node) [bwd]
	bool    has_pred;   // false only for the start node
	bool    finalized;  // popped with its optimal distance
} SRec;

// rank-pruned Dijkstra from 'start'. forward=true explores upward via outgoing
// edges (relaxing neighbors of strictly higher rank); forward=false explores
// upward via incoming edges (backward search from the target). fills 'recs'
// (nodeid -> SRec*) and appends every discovered node id to 'visited'.
static void _search
(
	Graph            *g,
	NodeID            start,
	bool              forward,
	const RelationID *rels,      // road types followed by the shortcut type
	int               relCount,
	AttributeID       weightAtt,
	AttributeID       rankAtt,
	dict             *rankCache,
	dict             *recs,      // [out] nodeid -> SRec*
	NodeID          **visited    // [out] arr of discovered node ids
) {
	MinHeap heap;
	_heap_init(&heap);

	SRec *s0 = rm_calloc(1, sizeof(SRec));
	s0->dist = 0; s0->has_pred = false;
	HashTableAdd(recs, KEY(start), s0);
	arr_append(*visited, start);
	_heap_push(&heap, 0.0, start);

	GRAPH_EDGE_DIR dir =
		forward ? GRAPH_EDGE_DIR_OUTGOING : GRAPH_EDGE_DIR_INCOMING;

	Edge *edges = arr_new(Edge, 16);

	HItem it;
	while(_heap_pop(&heap, &it)) {
		SRec *cr = HashTableFetchValue(recs, KEY(it.id));
		if(cr->finalized) continue;        // stale duplicate
		cr->finalized = true;

		double  cur_w    = cr->dist;
		NodeID  cur      = it.id;
		int64_t cur_rank = _node_rank(g, rankAtt, rankCache, cur);

		Node cn = GE_NEW_NODE();
		Graph_GetNode(g, cur, &cn);

		for(int r = 0; r < relCount; r++) {
			arr_clear(edges);
			Graph_GetNodeEdges(g, &cn, dir, rels[r], &edges);

			uint32_t m = arr_len(edges);
			for(uint32_t i = 0; i < m; i++) {
				Edge e = edges[i];
				NodeID nb = forward ? e.dest_id : e.src_id;
				if(nb == cur) continue;    // self-loop

				// rank pruning: both searches only ever climb in rank
				int64_t nb_rank = _node_rank(g, rankAtt, rankCache, nb);
				if(nb_rank <= cur_rank) continue;

				double ew = _edge_weight(g, weightAtt, &e);
				e.relationID = rels[r];    // preserve rel across Graph_GetEdge
				double nd = cur_w + ew;

				SRec *nr = HashTableFetchValue(recs, KEY(nb));
				if(nr == NULL) {
					nr = rm_calloc(1, sizeof(SRec));
					nr->dist = INFINITY;
					HashTableAdd(recs, KEY(nb), nr);
					arr_append(*visited, nb);
				}

				if(!nr->finalized && nd < nr->dist) {
					nr->dist      = nd;
					nr->pred_node = cur;
					nr->pred_edge = e;
					nr->has_pred  = true;
					_heap_push(&heap, nd, nb);
				}
			}
		}
	}

	arr_free(edges);
	_heap_free(&heap);
}

// find the edge realizing arc x -> y: the shortcut if one exists (it always
// improves on any road edge), otherwise the cheapest road edge. exactly one
// applies, since a chordal arc is materialized as a shortcut precisely when it
// beats the road edge there.
static Edge _best_subedge
(
	Graph      *g,
	NodeID      x,
	NodeID      y,
	RelationID  shortcutRelID,
	AttributeID weightAtt
) {
	Edge *tmp = arr_new(Edge, 4);

	Graph_GetEdgesConnectingNodes(g, x, y, shortcutRelID, &tmp);
	if(arr_len(tmp) > 0) {
		Edge e = tmp[0];
		e.relationID = shortcutRelID;
		arr_free(tmp);
		return e;
	}

	// cheapest road (non-shortcut) edge
	arr_clear(tmp);
	Graph_GetEdgesConnectingNodes(g, x, y, GRAPH_NO_RELATION, &tmp);
	Edge   best;
	double bw    = INFINITY;
	bool   found = false;
	for(uint32_t i = 0; i < arr_len(tmp); i++) {
		Edge e = tmp[i];
		if(e.relationID == shortcutRelID) continue;
		double w = _edge_weight(g, weightAtt, &e);
		if(w < bw) { bw = w; best = e; found = true; }
	}
	arr_free(tmp);

	ASSERT(found && "sub-arc of a shortcut has no realizing edge");
	return best;
}

// expand 'e' into road edges, appended in order to 'out'. a shortcut recurses
// through its middle node (e -> src..mid, mid..dst); a road edge is emitted
// as-is.
static void _unpack
(
	Graph      *g,
	Edge        e,
	RelationID  shortcutRelID,
	AttributeID weightAtt,
	AttributeID middleAtt,
	Edge      **out
) {
	if(e.relationID != shortcutRelID) {
		arr_append(*out, e);           // a genuine road hop
		return;
	}

	// shortcut: read its middle node id, then unpack the two halves
	Graph_GetEdge(g, e.id, &e);        // ensure attributes
	SIValue mv;
	bool ok = GraphEntity_GetProperty((GraphEntity *)&e, middleAtt, &mv);
	ASSERT(ok && (SI_TYPE(mv) & SI_NUMERIC));
	NodeID mid = (NodeID)SI_GET_NUMERIC(mv);

	Edge a = _best_subedge(g, e.src_id,  mid,        shortcutRelID, weightAtt);
	Edge b = _best_subedge(g, mid,       e.dest_id,  shortcutRelID, weightAtt);
	_unpack(g, a, shortcutRelID, weightAtt, middleAtt, out);
	_unpack(g, b, shortcutRelID, weightAtt, middleAtt, out);
}

// free a search's records dict + its SRec values
static void _free_recs(dict *recs, NodeID *visited) {
	for(uint32_t i = 0; i < arr_len(visited); i++) {
		SRec *r = HashTableFetchValue(recs, KEY(visited[i]));
		if(r != NULL) rm_free(r);
	}
	HashTableRelease(recs);
	arr_free(visited);
}

//------------------------------------------------------------------------------
// config
//------------------------------------------------------------------------------

static void _process_yield(CCHQueryCtx *ctx, const char **yield) {
	ctx->yield_weight = NULL;
	ctx->yield_path   = NULL;

	int idx = 0;
	for(uint i = 0; i < arr_len(yield); i++) {
		if(strcasecmp("pathWeight", yield[i]) == 0) {
			ctx->yield_weight = ctx->output + idx; idx++; continue;
		}
		if(strcasecmp("path", yield[i]) == 0) {
			ctx->yield_path = ctx->output + idx; idx++; continue;
		}
	}
}

static bool _read_config
(
	SIValue      config,
	Node        *src,           // [out]
	Node        *dst,           // [out]
	RelationID **rels,          // [out] road types + shortcut type
	int         *relCount,      // [out]
	RelationID  *shortcutRelID, // [out]
	AttributeID *weightAtt,     // [out]
	AttributeID *rankAtt,       // [out]
	AttributeID *middleAtt      // [out]
) {
	if(SI_TYPE(config) != T_MAP) {
		ErrorCtx_SetError("algo.CCH.query expects a single map argument");
		return false;
	}

	GraphContext *gc = QueryCtx_GetGraphCtx();
	SIValue v;

	// sourceNode / targetNode
	SIValue s, t;
	if(!MAP_GETCASEINSENSITIVE(config, "sourceNode", s) ||
	   !MAP_GETCASEINSENSITIVE(config, "targetNode", t)) {
		ErrorCtx_SetError("algo.CCH.query requires sourceNode and targetNode");
		return false;
	}
	if(SI_TYPE(s) != T_NODE || SI_TYPE(t) != T_NODE) {
		ErrorCtx_SetError("algo.CCH.query, sourceNode/targetNode must be nodes");
		return false;
	}
	*src = *(Node *)s.ptrval;
	*dst = *(Node *)t.ptrval;

	// shortcutRelType (must already exist)
	if(!MAP_GETCASEINSENSITIVE(config, "shortcutRelType", v) ||
			!(SI_TYPE(v) & T_STRING)) {
		ErrorCtx_SetError("algo.CCH.query requires string shortcutRelType");
		return false;
	}
	Schema *ss = GraphContext_GetSchema(gc, v.stringval, SCHEMA_EDGE);
	if(ss == NULL) {
		ErrorCtx_SetError("algo.CCH.query, unknown shortcutRelType: %s",
				v.stringval);
		return false;
	}
	*shortcutRelID = Schema_GetID(ss);

	// relTypes (road types)
	if(!MAP_GETCASEINSENSITIVE(config, "relTypes", v) ||
			SI_TYPE(v) != T_ARRAY || !SIArray_AllOfType(v, T_STRING)) {
		ErrorCtx_SetError("algo.CCH.query, relTypes must be an array of strings");
		return false;
	}
	uint32_t road_count = SIArray_Length(v);
	RelationID *_rels = arr_new(RelationID, road_count + 1);
	for(uint32_t i = 0; i < road_count; i++) {
		SIValue rt = SIArray_Get(v, i);
		Schema *rs = GraphContext_GetSchema(gc, rt.stringval, SCHEMA_EDGE);
		if(rs == NULL) {
			ErrorCtx_SetError("algo.CCH.query, unknown relationship type: %s",
					rt.stringval);
			arr_free(_rels);
			return false;
		}
		arr_append(_rels, Schema_GetID(rs));
	}
	arr_append(_rels, *shortcutRelID);   // search traverses road + shortcut
	*rels     = _rels;
	*relCount = arr_len(_rels);

	// weightProp / rankProperty / middleProp
	const char *props[3] = { "weightProp", "rankProperty", "middleProp" };
	AttributeID *outs[3]  = { weightAtt, rankAtt, middleAtt };
	for(int i = 0; i < 3; i++) {
		if(!MAP_GETCASEINSENSITIVE(config, props[i], v) ||
				!(SI_TYPE(v) & T_STRING)) {
			ErrorCtx_SetError("algo.CCH.query requires string %s", props[i]);
			arr_free(_rels);
			return false;
		}
		*outs[i] = GraphContext_GetAttributeID(gc, v.stringval);
		if(*outs[i] == ATTRIBUTE_ID_NONE) {
			ErrorCtx_SetError("algo.CCH.query, unknown attribute: %s",
					v.stringval);
			arr_free(_rels);
			return false;
		}
	}

	return true;
}

//------------------------------------------------------------------------------
// procedure entry points
//------------------------------------------------------------------------------

static ProcedureResult Proc_CCHQueryInvoke
(
	ProcedureCtx  *ctx,
	const SIValue *args,
	const char   **yield
) {
	if(arr_len((SIValue *)args) != 1) {
		ErrorCtx_SetError("algo.CCH.query expects a single map argument");
		return PROCEDURE_ERR;
	}

	Node        src, dst;
	RelationID *rels          = NULL;
	int         relCount      = 0;
	RelationID  shortcutRelID = GRAPH_NO_RELATION;
	AttributeID weightAtt = ATTRIBUTE_ID_NONE, rankAtt = ATTRIBUTE_ID_NONE,
	            middleAtt = ATTRIBUTE_ID_NONE;

	if(!_read_config(args[0], &src, &dst, &rels, &relCount, &shortcutRelID,
				&weightAtt, &rankAtt, &middleAtt)) {
		return PROCEDURE_ERR;
	}

	CCHQueryCtx *pdata = rm_calloc(1, sizeof(CCHQueryCtx));
	pdata->weight = INFINITY;
	ctx->privateData = pdata;
	_process_yield(pdata, yield);

	Graph *g = QueryCtx_GetGraph();

	NodeID s_id = ENTITY_GET_ID(&src);
	NodeID t_id = ENTITY_GET_ID(&dst);

	// degenerate: source == target -> single-node path, weight 0
	if(s_id == t_id) {
		pdata->weight = 0;
		pdata->path   = Path_New(1);
		Node n = GE_NEW_NODE();
		Graph_GetNode(g, s_id, &n);
		Path_AppendNode(pdata->path, n);
		arr_free(rels);
		return PROCEDURE_OK;
	}

	dict *rankCache = HashTableCreate(&def_dt);

	// forward (from source) and backward (from target) rank-pruned searches
	dict   *fwd  = HashTableCreate(&def_dt);
	dict   *bwd  = HashTableCreate(&def_dt);
	NodeID *fvis = arr_new(NodeID, 16);
	NodeID *bvis = arr_new(NodeID, 16);

	_search(g, s_id, true,  rels, relCount, weightAtt, rankAtt, rankCache,
			fwd, &fvis);
	_search(g, t_id, false, rels, relCount, weightAtt, rankAtt, rankCache,
			bwd, &bvis);

	// meet at the shared ancestor minimizing fwd.dist + bwd.dist
	double best = INFINITY;
	NodeID meet = INVALID_ENTITY_ID;
	for(uint32_t i = 0; i < arr_len(fvis); i++) {
		NodeID nid = fvis[i];
		SRec *fr = HashTableFetchValue(fwd, KEY(nid));
		SRec *br = HashTableFetchValue(bwd, KEY(nid));
		if(fr == NULL || br == NULL) continue;
		double tot = fr->dist + br->dist;
		if(tot < best) { best = tot; meet = nid; }
	}

	if(meet != (NodeID)INVALID_ENTITY_ID) {
		// shortcut-level edge sequence src -> ... -> meet -> ... -> dst
		Edge *hops = arr_new(Edge, 16);

		// forward half: walk preds meet -> src, collect reversed
		Edge *fh = arr_new(Edge, 8);
		NodeID cur = meet;
		while(true) {
			SRec *r = HashTableFetchValue(fwd, KEY(cur));
			if(!r->has_pred) break;
			arr_append(fh, r->pred_edge);   // edge pred_node -> cur
			cur = r->pred_node;
		}
		for(int64_t i = (int64_t)arr_len(fh) - 1; i >= 0; i--) {
			arr_append(hops, fh[i]);
		}
		arr_free(fh);

		// backward half: walk preds meet -> dst, already in forward order
		cur = meet;
		while(true) {
			SRec *r = HashTableFetchValue(bwd, KEY(cur));
			if(!r->has_pred) break;
			arr_append(hops, r->pred_edge);  // edge cur -> pred_node
			cur = r->pred_node;
		}

		// unpack every hop into road edges
		Edge *road = arr_new(Edge, 32);
		for(uint32_t i = 0; i < arr_len(hops); i++) {
			_unpack(g, hops[i], shortcutRelID, weightAtt, middleAtt, &road);
		}
		arr_free(hops);

		// build the path: n0 -e0- n1 -e1- ... nK
		pdata->weight = best;
		pdata->path   = Path_New(arr_len(road) + 1);

		Node n0 = GE_NEW_NODE();
		Graph_GetNode(g, s_id, &n0);
		Path_AppendNode(pdata->path, n0);

		for(uint32_t i = 0; i < arr_len(road); i++) {
			Edge e = road[i];
			Graph_GetEdge(g, e.id, &e);      // populate attributes for the path
			Path_AppendEdge(pdata->path, e);

			Node nn = GE_NEW_NODE();
			Graph_GetNode(g, e.dest_id, &nn);
			Path_AppendNode(pdata->path, nn);
		}
		arr_free(road);
	}

	_free_recs(fwd, fvis);
	_free_recs(bwd, bvis);
	HashTableRelease(rankCache);
	arr_free(rels);

	return PROCEDURE_OK;
}

static SIValue *Proc_CCHQueryStep(ProcedureCtx *ctx) {
	ASSERT(ctx->privateData != NULL);
	CCHQueryCtx *pdata = ctx->privateData;

	if(pdata->done || pdata->path == NULL) return NULL;
	pdata->done = true;

	if(pdata->yield_weight) *pdata->yield_weight = SI_DoubleVal(pdata->weight);
	if(pdata->yield_path)   *pdata->yield_path   = SIPath_Wrap(&pdata->path);
	else                    Path_Free(pdata->path);
	pdata->path = NULL;   // ownership transferred (or freed) above

	return pdata->output;
}

static ProcedureResult Proc_CCHQueryFree(ProcedureCtx *ctx) {
	if(ctx->privateData != NULL) {
		CCHQueryCtx *pdata = ctx->privateData;
		if(pdata->path != NULL) Path_Free(pdata->path);
		rm_free(pdata);
	}
	return PROCEDURE_OK;
}

ProcedureCtx *Proc_CCHQueryCtx(void) {
	ProcedureOutput *outputs = arr_newlen(ProcedureOutput, 2);
	outputs[0] = (ProcedureOutput){ .name = "pathWeight", .type = T_DOUBLE };
	outputs[1] = (ProcedureOutput){ .name = "path",       .type = T_PATH };

	return ProcCtxNew("algo.CCH.query", 1, outputs, Proc_CCHQueryStep,
			Proc_CCHQueryInvoke, Proc_CCHQueryFree, NULL, true);
}
