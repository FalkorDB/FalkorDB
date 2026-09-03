/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "AStar.h"
#include "../value.h"
#include "../util/arr.h"
#include "../util/heap.h"
#include "../util/dict.h"
#include "../util/rmalloc.h"
#include "utils/node_map.h"
#include "utils/priority_heap.h"

#include <float.h>
#include <math.h>

//------------------------------------------------------------------------------
// heuristic: haversine great-circle distance to dst
//------------------------------------------------------------------------------

#define ASTAR_EARTH_RADIUS 6378140.0
#define ASTAR_DegreeToRadians(d) ((d) * M_PI / 180.0)

// haversine great-circle distance in meters between two (lat, lon) pairs
// given in degrees. duplicates the math in AR_DISTANCE (point_funcs.c) but
// operates on two raw double pairs rather than two SIValue.point values,
// since latitudeProperty/longitudeProperty are two independent scalar node
// properties here, not a single T_POINT property.
static inline double _haversine_meters
(
	double lat1, double lon1,
	double lat2, double lon2
) {
	double rlat1 = ASTAR_DegreeToRadians(lat1);
	double rlat2 = ASTAR_DegreeToRadians(lat2);
	double dlat  = rlat2 - rlat1;
	double dlon  = ASTAR_DegreeToRadians(lon2) - ASTAR_DegreeToRadians(lon1);

	double a = pow(sin(dlat / 2), 2) +
		cos(rlat1) * cos(rlat2) * pow(sin(dlon / 2), 2);
	double c = 2 * atan2(sqrt(a), sqrt(1 - a));

	return ASTAR_EARTH_RADIUS * c;
}

// read a node's (lat, lon) via lat_prop/lon_prop. returns false if either
// property is missing or non-numeric, in which case *lat/*lon are left
// untouched and the caller should treat h(node) as 0.
static inline bool _get_node_latlon
(
	Graph *g,
	NodeID id,
	AttributeID lat_prop,
	AttributeID lon_prop,
	double *lat,
	double *lon
) {
	Node n = GE_NEW_NODE();
	Graph_GetNode(g, id, &n);

	SIValue vlat;
	SIValue vlon;
	if(!GraphEntity_GetProperty((GraphEntity *) &n, lat_prop, &vlat) ||
	   !GraphEntity_GetProperty((GraphEntity *) &n, lon_prop, &vlon) ||
	   !(SI_TYPE(vlat) & SI_NUMERIC)                                 ||
	   !(SI_TYPE(vlon) & SI_NUMERIC)) {
		return false;
	}

	*lat = SI_GET_NUMERIC(vlat);
	*lon = SI_GET_NUMERIC(vlon);

	return true;
}

// admissible heuristic: haversine distance from 'id' to the fixed goal
// (dst_lat, dst_lon) scaled into the weight's units by 'heur_scale', or 0 if
// dst's coordinates were never resolved or 'id' itself lacks a numeric lat/lon
// -- either case degrades gracefully to Dijkstra-like behavior for the affected
// node rather than erroring mid-search.
//
// 'heur_scale' converts raw meters to the weightProp's units and MUST be a
// lower bound on the weight accrued per meter of straight-line progress for the
// heuristic to stay admissible (see AStar.h). e.g. weightProp == distance in
// meters -> heur_scale == 1; weightProp == travel time in hours -> heur_scale
// == 1 / (max_speed in meters-per-hour).
static inline double _heuristic
(
	Graph *g,
	NodeID id,
	AttributeID lat_prop,
	AttributeID lon_prop,
	double heur_scale,
	bool dst_has_coords,
	double dst_lat,
	double dst_lon
) {
	if(!dst_has_coords) {
		return 0;
	}

	double lat, lon;
	if(!_get_node_latlon(g, id, lat_prop, lon_prop, &lat, &lon)) {
		return 0;
	}

	return heur_scale * _haversine_meters(lat, lon, dst_lat, dst_lon);
}

// per-node search record used by the A* search below (best-known cost g, cached
// heuristic h, predecessor and connecting edge). "label" in the label-setting
// sense -- unrelated to graph node labels.
typedef struct {
	NodeID parent;    // predecessor in the shortest-path tree
	Edge   edge;      // edge connecting parent -> this node
	double g_score;   // current best known true cost to reach this node
	double h;         // cached heuristic, computed once on first discovery
	bool   finalized; // true once popped from the heap with its optimal g_score
} AStarLabel;

//------------------------------------------------------------------------------
// AStarCtx: reusable A* engine
//------------------------------------------------------------------------------

// A reusable A* search engine, mirroring the Dijkstra engine (DijkstraCtx) but
// self-contained in the A* module: it keeps its own lat/lon-driven heuristic
// rather than pushing geographic concerns into the generic shortest-path code.
//
// as with DijkstraCtx, the expensive per-search setup -- attaching one
// TensorIterator per (direction, relation) pair -- is done once at construction
// and amortized across many AStarCtx_Run calls, which re-seek (not re-attach)
// those iterators. this is what makes the k-shortest driver below affordable:
// it runs O(k * path-length) spur searches, each a fresh A* from a different
// spur node to the same dst, over a subgraph with some nodes/edges blocked.
typedef struct AStarCtx {
	// graph + search parameters (borrowed, owned by the caller)
	Graph          *g;                 // graph to traverse
	GRAPH_EDGE_DIR  dir;               // traverse direction
	RelationID     *relationIDs;       // edge type(s) to traverse
	Tensor         *relationMatrices;  // relation matrix per relationIDs entry
	int             relationCount;     // length of relationIDs
	AttributeID     weight_prop;       // weight attribute id
	AttributeID     lat_prop;          // latitude attribute id (heuristic)
	AttributeID     lon_prop;          // longitude attribute id (heuristic)
	double          heur_scale;        // meters -> weight-units heuristic scale

	// optional landmark potential: another AStarCtx run single-source (see
	// AStarCtx_Run with dst == INVALID_ENTITY_ID) on the *reverse* graph from
	// this search's destination, so AStarCtx_Distance(potential, n) is the exact
	// remaining cost h(n) = dist(n -> dst). when set it supersedes the haversine
	// heuristic (it's exact, hence a tighter admissible+consistent bound); NULL
	// (the default) keeps the haversine/heur_scale behavior. borrowed -- must
	// outlive this ctx and stay un-rerun while attached.
	const struct AStarCtx *potential;

	// expansion directions derived from 'dir', computed once at construction
	int             ndirs;
	GRAPH_EDGE_DIR  dirs[2];

	// one TensorIterator per (direction, relation) pair, attached once and
	// re-seeked on every node expansion, across every AStarCtx_Run call
	TensorIterator *iters;

	// per-run scratch, reset (not reallocated) between runs
	NodeMap         record_idx;  // node id -> 1-based slot in 'records'
	AStarLabel     *records;     // one search record per discovered node
	NodeWeightHeap  heap;        // priority queue keyed by f = g + h
} AStarCtx;

static AStarCtx *AStarCtx_New
(
	Graph *g,
	GRAPH_EDGE_DIR dir,
	RelationID *relationIDs,
	Tensor *relationMatrices,
	int relationCount,
	AttributeID weight_prop,
	AttributeID lat_prop,
	AttributeID lon_prop,
	double heur_scale
) {
	AStarCtx *ac = rm_calloc(1, sizeof(AStarCtx));

	ac->g                = g;
	ac->dir              = dir;
	ac->relationIDs      = relationIDs;
	ac->relationMatrices = relationMatrices;
	ac->relationCount    = relationCount;
	ac->weight_prop      = weight_prop;
	ac->lat_prop         = lat_prop;
	ac->lon_prop         = lon_prop;
	ac->heur_scale       = heur_scale;

	ac->ndirs = 0;
	if(dir == GRAPH_EDGE_DIR_OUTGOING || dir == GRAPH_EDGE_DIR_BOTH) {
		ac->dirs[ac->ndirs++] = GRAPH_EDGE_DIR_OUTGOING;
	}
	if(dir == GRAPH_EDGE_DIR_INCOMING || dir == GRAPH_EDGE_DIR_BOTH) {
		ac->dirs[ac->ndirs++] = GRAPH_EDGE_DIR_INCOMING;
	}

	ac->iters = rm_malloc(ac->ndirs * relationCount * sizeof(TensorIterator));
	for(int d = 0; d < ac->ndirs; d++) {
		bool transpose = (ac->dirs[d] == GRAPH_EDGE_DIR_INCOMING);
		for(int r = 0; r < relationCount; r++) {
			TensorIterator_Attach(&ac->iters[d * relationCount + r],
					relationMatrices[r], transpose);
		}
	}

	NodeMap_init(&ac->record_idx);
	ac->records = arr_new(AStarLabel, 64);
	NodeWeightHeap_init(&ac->heap);

	return ac;
}

// reset per-run scratch so the engine can run another search. iterators are
// intentionally left attached (re-seeked per node via IterateRow), which is
// the whole point of reusing the context.
static inline void _AStarCtx_Reset
(
	AStarCtx *ac
) {
	NodeMap_clear(&ac->record_idx);
	arr_clear(ac->records);
	NodeWeightHeap_clear(&ac->heap);
}

// forward declaration: the landmark heuristic reads distances out of a
// completed reverse search via this accessor (defined below).
static bool AStarCtx_Distance(const AStarCtx *ac, NodeID v, double *weight);

// per-node heuristic h(n) used to order the frontier by f = g + h. exactly one
// source is active per search:
//   - landmark potential attached: h(n) = exact remaining cost dist(n -> dst),
//     read from the reverse sweep; a node absent from the sweep can't reach dst,
//     so h = +inf (DBL_MAX) parks it behind dst and it's never expanded first --
//     correct, since it lies on no src->dst path.
//   - otherwise: the haversine estimate scaled by heur_scale (0 when
//     coordinates/scale are unavailable, i.e. plain Dijkstra).
static inline double _astar_node_h
(
	AStarCtx *ac,
	NodeID id,
	bool dst_has_coords,
	double dst_lat,
	double dst_lon
) {
	if(ac->potential != NULL) {
		double d;
		return AStarCtx_Distance(ac->potential, id, &d) ? d : DBL_MAX;
	}

	return _heuristic(ac->g, id, ac->lat_prop, ac->lon_prop,
			ac->heur_scale, dst_has_coords, dst_lat, dst_lon);
}

// run an A* search from src. internal scratch is reset at entry so the context
// is reusable. blocked_nodes / blocked_edges, when non-NULL, are membership sets
// (keyed by (uintptr_t)id) of nodes/edges to skip during relaxation -- as if
// removed from the graph for this run (Yen's spur searches).
//
// dst_id == INVALID_ENTITY_ID runs single-source: no goal, no early exit -- the
// search finalizes every reachable node and returns true, so its g_scores can be
// read back via AStarCtx_Distance (this is how a landmark potential is built on
// the reverse graph). otherwise the search stops when dst is finalized and
// returns whether it was reached.
static bool AStarCtx_Run
(
	AStarCtx *ac,
	NodeID src_id,
	NodeID dst_id,
	const dict *blocked_nodes,
	const dict *blocked_edges
) {
	_AStarCtx_Reset(ac);

	bool single_source = (dst_id == (NodeID)INVALID_ENTITY_ID);

	// resolve dst's coordinates once, up front: every heuristic evaluation
	// during this search targets this fixed goal. if dst has no numeric lat/lon,
	// heur_scale is non-positive, this is a single-source sweep, or a landmark
	// potential is driving h, the haversine term degrades to 0 -- the potential
	// branch in _astar_node_h ignores these anyway.
	double dst_lat = 0;
	double dst_lon = 0;
	bool dst_has_coords = !single_source                          &&
			(ac->weight_prop != ATTRIBUTE_ID_NONE)               &&
			(ac->heur_scale > 0)                                 &&
			_get_node_latlon(ac->g, dst_id, ac->lat_prop, ac->lon_prop, &dst_lat, &dst_lon);

	// seed the source: g_score 0, priority f = 0 + h(src).
	double src_h = _astar_node_h(ac, src_id, dst_has_coords, dst_lat, dst_lon);
	AStarLabel src_label =
		{ .parent = src_id, .g_score = 0, .h = src_h, .finalized = false };

	arr_append(ac->records, src_label);

	uint32_t *src_slot = NodeMap_findOrInsert(&ac->record_idx, src_id, NULL);

	// set the one-based position in records array
	*src_slot = arr_len(ac->records);

	NodeWeightItem seed = { .node = src_id, .weight = src_h };
	NodeWeightHeap_offer (&ac->heap, seed);

	bool found = false;

	// main A* loop: repeatedly extract the not-yet-finalized node with the
	// smallest priority (f = g + h) and finalize it -- optimal since the
	// haversine heuristic is admissible and consistent. stops when dst is
	// finalized (found) or the heap empties (dst unreachable).
	while(!found) {
		NodeWeightItem item;
		if (!NodeWeightHeap_poll (&ac->heap, &item)) {
			break;  // heap exhausted: dst is unreachable
		}

		NodeID cur = item.node;

		uint32_t cur_idx = NodeMap_find(&ac->record_idx, cur);

		ASSERT(cur_idx != 0);
		if(ac->records[cur_idx - 1].finalized) {
			continue;  // stale duplicate entry
		}

		ac->records[cur_idx - 1].finalized = true;

		if(cur == dst_id) {
			found = true;
			break;
		}

		double cur_g = ac->records[cur_idx - 1].g_score;

		// relaxation: stream edges straight from the tensor iterator (no scratch
		// array); fetch an edge's attribute set from the datablock only when a
		// weight is actually read or an improving edge is stored for the path.
		bool need_weight = (ac->weight_prop != ATTRIBUTE_ID_NONE);

        int num_iters = ac->ndirs * ac->relationCount;
		for(int r = 0; r < num_iters; r++) {
			TensorIterator *it = &ac->iters[r];
			TensorIterator_IterateRow (it, cur);

			// iters is laid out [dir][relation]; map the flat index back to its
			// relation so relationIDs[] (length relationCount) isn't read out of
			// bounds when ndirs == 2 (relDirection:'both').
			Edge e = { .relationID = ac->relationIDs[r % ac->relationCount] };
			while (TensorIterator_next (it, &e.src_id, &e.dest_id, &e.id, NULL))
			{
				e.attributes = NULL;
				ASSERT (e.src_id == cur || e.dest_id == cur) ;

				// nid is whichever node is NOT cur
				NodeID nid = (e.src_id == cur) ? e.dest_id : e.src_id;

				if (nid == cur) {
					continue;  // ignore self-loops
				}

				// blocked-set filters (Yen spur searches); skipped when NULL.
				if (blocked_edges != NULL &&
					HashTableFind((dict *)blocked_edges,
						(void *)(uintptr_t)e.id) != NULL) {
					continue;
				}
				if (blocked_nodes != NULL &&
					HashTableFind((dict *)blocked_nodes,
						(void *)(uintptr_t)nid) != NULL) {
					continue;
				}

				// candidate g_score to 'nid' through 'cur' (default weight 1
				// when no weight property). weightProp is assumed
				// non-negative (see AStar.h).
				double edge_w = 1;
				if(need_weight) {
					Graph_GetEdge(ac->g, e.id, &e);  // populate e.attributes
					SIValue w = GraphEntity_GetNumericPropertyOrDefault((GraphEntity *)&e,
							ac->weight_prop, SI_LongVal(1));
					edge_w = SI_GET_NUMERIC(w);
				}
				double new_g = cur_g + edge_w;

				bool is_new;
				uint32_t *nslot =
					NodeMap_findOrInsert(&ac->record_idx, nid, &is_new);

				double h;
				if(!is_new) {
					AStarLabel *nlabel = ac->records + (*nslot - 1);
					if(nlabel->finalized || new_g >= nlabel->g_score) {
						continue;
					}
					h = nlabel->h;  // heuristic is fixed per node
				} else {
					// first discovery: compute and cache h(nid).
					h = _astar_node_h(ac, nid, dst_has_coords, dst_lat, dst_lon);
					AStarLabel fresh = { .h = h, .finalized = false };
					arr_append(ac->records, fresh);
					*nslot = arr_len(ac->records);
				}

				// store the improving edge; ensure attributes are populated
				// (already done above when weighted) for the returned path.
				if(e.attributes == NULL) {
					Graph_GetEdge (ac->g, e.id, &e);
				}

				AStarLabel *nlabel = ac->records + (*nslot - 1);
				nlabel->edge       = e;
				nlabel->parent     = cur;
				nlabel->g_score    = new_g;

				// queue (or re-queue) 'nid' at priority f = g + h(nid). a landmark
				// h of +inf (DBL_MAX, unreachable-to-dst node) saturates the
				// priority, parking it behind dst so it is never expanded.
				NodeWeightItem qi = { .node = nid, .weight = new_g + h };
				NodeWeightHeap_offer(&ac->heap, qi);
			}
		}
	}

	// single-source sweep always "succeeds": every reachable node was finalized
	// and its distance is readable via AStarCtx_Distance (used to build a
	// landmark potential). single-pair reports whether dst was reached.
	return single_source ? true : found;
}

// after a run: report the finalized g_score (true cost) to 'v'. returns false
// if 'v' was never discovered by the last run.
static bool AStarCtx_Distance
(
	const AStarCtx *ac,
	NodeID v,
	double *weight
) {
	uint32_t idx = NodeMap_find(&ac->record_idx, v);
	if(idx == 0) {
		return false;
	}

	if(weight != NULL) {
		*weight = ac->records[idx - 1].g_score;
	}

	return true;
}

// after a run that reached dst: reconstruct the src -> dst path by walking
// parent pointers. caller owns the returned Path.
static Path *AStarCtx_Path
(
	const AStarCtx *ac,
	NodeID src_id,
	NodeID dst_id
) {
	NodeID cur = dst_id;
	Path *p = Path_New(8);

	while(cur != src_id) {
		uint32_t idx = NodeMap_find(&ac->record_idx, cur);
		ASSERT(idx != 0);
		AStarLabel *label = ac->records + (idx - 1);

		Node n = GE_NEW_NODE();
		Graph_GetNode(ac->g, cur, &n);
		Path_AppendNode(p, n);
		Path_AppendEdge(p, label->edge);

		cur = label->parent;
	}

	Node srcNode = GE_NEW_NODE();
	Graph_GetNode(ac->g, src_id, &srcNode);
	Path_AppendNode(p, srcNode);

	Path_Reverse(p);

	return p;
}

static void AStarCtx_Free
(
	AStarCtx *ac
) {
	if(ac == NULL) {
		return;
	}

	NodeWeightHeap_free(&ac->heap);
	NodeMap_free(&ac->record_idx);
	arr_free(ac->records);
	rm_free(ac->iters);
	rm_free(ac);
}

//------------------------------------------------------------------------------
// AStar_ShortestPath: single-pair wrapper over the engine
//------------------------------------------------------------------------------

bool AStar_ShortestPath
(
	Path **path,
	double *weight,
	Graph *g,
	NodeID src_id,
	NodeID dst_id,
	GRAPH_EDGE_DIR dir,
	RelationID *relationIDs,
	Tensor *relationMatrices,
	int relationCount,
	AttributeID weight_prop,
	AttributeID lat_prop,
	AttributeID lon_prop,
	double heur_scale
) {
	ASSERT(g      != NULL);
	ASSERT(path   != NULL);
	ASSERT(weight != NULL);

	AStarCtx *ac = AStarCtx_New(g, dir, relationIDs, relationMatrices,
			relationCount, weight_prop, lat_prop, lon_prop, heur_scale);

	bool found = AStarCtx_Run(ac, src_id, dst_id, NULL, NULL);

	if(found) {
		*path = AStarCtx_Path(ac, src_id, dst_id);
		AStarCtx_Distance(ac, dst_id, weight);
	}

	AStarCtx_Free(ac);

	return found;
}

//------------------------------------------------------------------------------
// AStar_KShortestPaths: Yen's algorithm driven by A* spur searches
//------------------------------------------------------------------------------
//
// this mirrors the Yen orchestration in yen.c; the two are kept deliberately
// separate (each engine self-contained) rather than sharing a generic driver,
// so the A* variant stays fully enclosed in this module.

// a Yen candidate path waiting in the candidate heap
typedef struct {
	Path   *path;    // the candidate path
	double  weight;  // its total weight
} AStarCandidate;

// candidate-heap comparator: the heap keeps the *greatest* element (per this
// cmp) on top, so invert the natural order -- smallest weight (then shortest
// path) becomes the top, i.e. the next path to accept.
static int _astar_cand_cmp
(
	const void *a,
	const void *b,
	void *udata
) {
	const AStarCandidate *ca = (const AStarCandidate *)a;
	const AStarCandidate *cb = (const AStarCandidate *)b;

	if(ca->weight != cb->weight) {
		return (ca->weight < cb->weight) ? 1 : -1;
	}

	size_t la = Path_Len(ca->path);
	size_t lb = Path_Len(cb->path);
	if(la != lb) {
		return (la < lb) ? 1 : -1;
	}

	return 0;
}

// 64-bit FNV-1a hash of a path's edge-id sequence -- its identity for dedup.
// (not SIPath_HashCode: that hashes node/edge SIValues and would need each
// candidate wrapped in an SIValue; the edge-id sequence is exactly the identity
// Yen needs -- it distinguishes parallel edges -- and costs no allocation.)
static uint64_t _path_key
(
	const Path *p
) {
	uint64_t h = 1469598103934665603ULL;  // FNV offset basis

	uint ec = Path_EdgeCount(p);
	for(uint i = 0; i < ec; i++) {
		uint64_t e = ENTITY_GET_ID(Path_GetEdge(p, i));
		h ^= e;
		h *= 1099511628211ULL;  // FNV prime
	}

	return h;
}

// record path 'p' as seen; returns true if new, false if already known.
static bool _mark_seen
(
	dict *seen,
	const Path *p
) {
	uint64_t h = _path_key(p);

	if(HashTableFind(seen, (void *)(uintptr_t)h) != NULL) {
		return false;
	}

	HashTableAdd(seen, (void *)(uintptr_t)h, (void *)(uintptr_t)1);
	return true;
}

// does 'p' share prev's root prefix (its first 'i' edges) and have an edge at
// index 'i' to block?
static bool _shares_root
(
	const Path *p,
	const Path *prev,
	uint i
) {
	if(Path_EdgeCount(p) <= i) {
		return false;
	}

	for(uint j = 0; j < i; j++) {
		if(ENTITY_GET_ID(Path_GetEdge(p, j)) !=
			ENTITY_GET_ID(Path_GetEdge(prev, j))) {
			return false;
		}
	}

	return true;
}

// total weight of prev's first 'i' edges (the root path up to the spur node)
static double _root_weight
(
	const Path *prev,
	uint i,
	AttributeID weight_prop
) {
	double w = 0;

	for(uint j = 0; j < i; j++) {
		SIValue v = GraphEntity_GetNumericPropertyOrDefault(
				(GraphEntity *)Path_GetEdge(prev, j),
				weight_prop, SI_LongVal(1));
		w += SI_GET_NUMERIC(v);
	}

	return w;
}

// build root(prev, i) ++ spur: prev's first i edges / i+1 nodes, then the spur
// path (whose first node is prev.nodes[i], already included, so it is skipped).
static Path *_concat
(
	const Path *prev,
	uint i,
	const Path *spur
) {
	uint spur_nodes = Path_NodeCount(spur);
	uint spur_edges = Path_EdgeCount(spur);

	Path *total = Path_New(i + 1 + spur_nodes);

	for(uint j = 0; j <= i; j++) {
		Path_AppendNode(total, *Path_GetNode(prev, j));
	}
	for(uint j = 0; j < i; j++) {
		Path_AppendEdge(total, *Path_GetEdge(prev, j));
	}

	for(uint j = 1; j < spur_nodes; j++) {
		Path_AppendNode(total, *Path_GetNode(spur, j));
	}
	for(uint j = 0; j < spur_edges; j++) {
		Path_AppendEdge(total, *Path_GetEdge(spur, j));
	}

	return total;
}

// A[0] hop count below which building a landmark reverse sweep (~one full
// Dijkstra) costs more than the spur searches it would accelerate. below it the
// driver runs its base-heuristic spurs (plain Dijkstra when h==0, haversine when
// coordinates are set); at/above it the sweep is built once and amortized across
// all spurs. tunable; validated on the NVDB road network.
#define ASTAR_LANDMARK_MIN_HOPS 200

// reverse of a traversal direction: the landmark potential searches from dst
// *against* the query direction, so its finalized g_scores are the exact
// remaining costs dist(n -> dst) on the forward graph.
static inline GRAPH_EDGE_DIR _reverse_dir
(
	GRAPH_EDGE_DIR dir
) {
	if(dir == GRAPH_EDGE_DIR_OUTGOING) return GRAPH_EDGE_DIR_INCOMING;
	if(dir == GRAPH_EDGE_DIR_INCOMING) return GRAPH_EDGE_DIR_OUTGOING;
	return GRAPH_EDGE_DIR_BOTH;  // undirected is its own reverse
}

// shared Yen k-shortest driver over a prepared engine 'ac' (its base heuristic
// -- haversine+scale, or h==0 for plain Dijkstra -- is whatever the caller
// configured). after A[0], when it pays off (k > 1 and a long A[0]) it builds a
// landmark potential: one reverse single-source sweep from dst whose exact
// dist(n -> dst) then steers every spur search (goal-directed Yen). results are
// identical to plain Yen; only the spur exploration shrinks. 'ac' is borrowed --
// the caller creates and frees it.
static uint _astar_kshortest
(
	AStarCtx    *ac,
	NodeID       src,
	NodeID       dst,
	uint64_t     k,
	AttributeID  weight_prop,
	Path      ***paths,
	double     **weights
) {
	// A: accepted paths (ascending weight); AW: their weights (parallel)
	Path   **A  = arr_new(Path *, 0);
	double  *AW = arr_new(double, 0);

	*paths   = A;
	*weights = AW;

	if(k == 0 || src == dst) {
		return 0;
	}

	// A[0]: the global shortest path (base heuristic; no landmark yet -- its
	// length is what the landmark gate keys off). if dst is unreachable, none.
	if(!AStarCtx_Run(ac, src, dst, NULL, NULL)) {
		return 0;
	}

	Path  *p0 = AStarCtx_Path(ac, src, dst);
	double w0;
	AStarCtx_Distance(ac, dst, &w0);
	arr_append(A, p0);
	arr_append(AW, w0);

	// landmark gate: only build the reverse sweep (~one full Dijkstra) when the
	// spur phase is big enough to amortize it -- k > 1 and a long A[0]. below the
	// threshold the base-heuristic spurs are cheaper than paying for the sweep.
	// once built, it steers every spur search via ac->potential.
	AStarCtx *rev = NULL;
	if(k > 1 && Path_EdgeCount(p0) >= ASTAR_LANDMARK_MIN_HOPS) {
		rev = AStarCtx_New(ac->g, _reverse_dir(ac->dir), ac->relationIDs,
				ac->relationMatrices, ac->relationCount, ac->weight_prop,
				ATTRIBUTE_ID_NONE, ATTRIBUTE_ID_NONE, 0);
		// single-source sweep from dst over the reverse graph (plain Dijkstra,
		// h==0): finalizes exact dist(n -> dst) for every reachable n.
		AStarCtx_Run(rev, dst, (NodeID)INVALID_ENTITY_ID, NULL, NULL);
		ac->potential = rev;
	}

	// B: candidate min-heap; seen: dedup set of path keys (covers A and B).
	heap_t *B    = Heap_new(_astar_cand_cmp, NULL);
	dict   *seen = HashTableCreate(&def_dt);
	_mark_seen(seen, p0);

	// blocked node/edge sets, reused (emptied) across spur searches.
	dict *blocked_nodes = HashTableCreate(&def_dt);
	dict *blocked_edges = HashTableCreate(&def_dt);

	while(arr_len(A) < k) {
		Path *prev       = arr_tail (A);
		uint  prev_nodes = Path_NodeCount(prev);

		// spur node ranges over prev's nodes [0 .. prev_nodes-2] (its last node
		// is dst, which can't spur).
		for(uint i = 0; i + 1 < prev_nodes; i++) {
			NodeID spur = ENTITY_GET_ID(Path_GetNode(prev, i));

			// block the edge leaving the spur node in every already-found path
			// that shares prev's root prefix.
			HashTableEmpty(blocked_edges, NULL);
			for(uint a = 0; a < arr_len(A); a++) {
				if(_shares_root(A[a], prev, i)) {
					EdgeID be = ENTITY_GET_ID(Path_GetEdge(A[a], i));
					HashTableAdd(blocked_edges, (void *)(uintptr_t)be,
							(void *)(uintptr_t)1);
				}
			}

			// block the root nodes strictly before the spur node (keeps the
			// total loopless).
			HashTableEmpty(blocked_nodes, NULL);
			for(uint j = 0; j < i; j++) {
				NodeID bn = ENTITY_GET_ID(Path_GetNode(prev, j));
				HashTableAdd(blocked_nodes, (void *)(uintptr_t)bn,
						(void *)(uintptr_t)1);
			}

			// A* search spurNode -> dst on the graph minus the blocked sets.
			if(!AStarCtx_Run(ac, spur, dst, blocked_nodes, blocked_edges)) {
				continue;  // no spur path from here
			}

			Path  *spur_path = AStarCtx_Path(ac, spur, dst);
			double spur_w;
			AStarCtx_Distance(ac, dst, &spur_w);

			// candidate = root(prev, i) ++ spur_path
			Path   *total   = _concat(prev, i, spur_path);
			double  total_w = _root_weight(prev, i, weight_prop) + spur_w;
			Path_Free(spur_path);

			if(_mark_seen(seen, total)) {
				AStarCandidate *c = rm_malloc(sizeof(AStarCandidate));
				c->path   = total;
				c->weight = total_w;
				Heap_offer(&B, c);
			} else {
				Path_Free(total);  // already generated before
			}
		}

		// accept the lightest candidate; if none remain, we're done.
		AStarCandidate *best = Heap_poll(B);
		if(best == NULL) {
			break;
		}

		arr_append(A, best->path);
		arr_append(AW, best->weight);
		rm_free(best);
	}

	// drain and free any candidates left unaccepted.
	AStarCandidate *c;
	while((c = Heap_poll(B)) != NULL) {
		Path_Free(c->path);
		rm_free(c);
	}

	Heap_free(B);
	HashTableRelease(seen);
	HashTableRelease(blocked_nodes);
	HashTableRelease(blocked_edges);

	// detach and free the landmark sweep if we built one; 'ac' is the caller's.
	if(rev != NULL) {
		ac->potential = NULL;
		AStarCtx_Free(rev);
	}

	*paths   = A;
	*weights = AW;
	return arr_len(A);
}

//------------------------------------------------------------------------------
// public k-shortest entry point
//------------------------------------------------------------------------------

// Yen's k-shortest driven by A* spur searches. the geographic (haversine)
// heuristic scaled by heur_scale accelerates A[0] and short-route spurs; for
// long routes the driver additionally builds an exact landmark potential (see
// _astar_kshortest) that supersedes it. algo.SPpaths calls this with lat/lon ==
// ATTRIBUTE_ID_NONE and heur_scale == 0 -- a plain-Dijkstra base plus the same
// landmark acceleration.
uint AStar_KShortestPaths
(
	Graph *g,
	NodeID src,
	NodeID dst,
	uint64_t k,
	GRAPH_EDGE_DIR dir,
	RelationID *relationIDs,
	Tensor *relationMatrices,
	int relationCount,
	AttributeID weight_prop,
	AttributeID lat_prop,
	AttributeID lon_prop,
	double heur_scale,
	Path ***paths,
	double **weights
) {
	ASSERT(g       != NULL);
	ASSERT(paths   != NULL);
	ASSERT(weights != NULL);

	AStarCtx *ac = AStarCtx_New(g, dir, relationIDs, relationMatrices,
			relationCount, weight_prop, lat_prop, lon_prop, heur_scale);

	uint n = _astar_kshortest(ac, src, dst, k, weight_prop, paths, weights);

	AStarCtx_Free(ac);
	return n;
}
