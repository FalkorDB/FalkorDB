/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "AStar.h"
#include "../value.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "utils/node_map.h"
#include "utils/priority_heap.h"

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
	if(!GraphEntity_GetProperty((GraphEntity *)&n, lat_prop, &vlat) ||
	   !GraphEntity_GetProperty((GraphEntity *)&n, lon_prop, &vlon) ||
	   !(SI_TYPE(vlat) & SI_NUMERIC)                                ||
	   !(SI_TYPE(vlon) & SI_NUMERIC)) {
		return false;
	}

	*lat = SI_GET_NUMERIC(vlat);
	*lon = SI_GET_NUMERIC(vlon);

	return true;
}

// admissible heuristic: haversine distance from 'id' to the fixed goal
// (dst_lat, dst_lon), or 0 if dst's coordinates were never resolved or
// 'id' itself lacks a numeric lat/lon -- either case degrades gracefully
// to Dijkstra-like behavior for the affected node rather than erroring
// mid-search.
static inline double _heuristic
(
	Graph *g,
	NodeID id,
	AttributeID lat_prop,
	AttributeID lon_prop,
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

	return _haversine_meters(lat, lon, dst_lat, dst_lon);
}

// per-node label used by the search below
typedef struct {
	NodeID parent;    // predecessor in the shortest-path tree
	Edge   edge;      // edge connecting parent -> this node
	double g_score;   // current best known true cost to reach this node
	double h;         // cached heuristic, computed once on first discovery
	bool   finalized; // true once popped from the heap with its optimal g_score
} AStarLabel;

//------------------------------------------------------------------------------
// A*
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
	AttributeID lon_prop
) {
	// 'labels' holds one AStarLabel per node ever discovered (tentative or
	// finalized best-known g_score, its parent and connecting edge).
	// 'label_idx' maps a node id to its 1-based slot in 'labels' (0 means
	// "not yet discovered").
	// 'heap' is the A* priority queue: pending (node, f = g_score + h)
	// candidates ordered so the next NodeWeightHeap_poll always returns the
	// smallest-priority candidate discovered so far.
	NodeMap label_idx;
	NodeMap_init(&label_idx);

	AStarLabel *labels = arr_new(AStarLabel, 64);

	NodeWeightHeap heap;
	NodeWeightHeap_init(&heap);

	// scratch buffer for edge expansion, reused (via arr_clear) across
	// every neighbor scan performed by this search
	Edge *neighbors = arr_new(Edge, 32);

	// build the list of edge directions to expand through when scanning a
	// node's neighbors: OUTGOING, INCOMING, or both, per the caller's
	// requested traversal direction.
	int ndirs = 0;
	GRAPH_EDGE_DIR dirs[2];
	if(dir == GRAPH_EDGE_DIR_OUTGOING || dir == GRAPH_EDGE_DIR_BOTH) {
		dirs[ndirs++] = GRAPH_EDGE_DIR_OUTGOING;
	}
	if(dir == GRAPH_EDGE_DIR_INCOMING || dir == GRAPH_EDGE_DIR_BOTH) {
		dirs[ndirs++] = GRAPH_EDGE_DIR_INCOMING;
	}

	// one TensorIterator per (direction, relation) pair, attached once up
	// front and reseeked (rather than re-attached) for every node popped
	// from the heap below -- attaching a TensorIterator is expensive (it
	// re-derives the underlying matrices' sparsity/format), while reseeking
	// it to a different row is cheap, and the relation matrices themselves
	// don't change over the course of a single search
	TensorIterator *iters = rm_malloc(ndirs * relationCount * sizeof(TensorIterator));
	for(int d = 0; d < ndirs; d++) {
		bool transpose = (dirs[d] == GRAPH_EDGE_DIR_INCOMING);
		for(int r = 0; r < relationCount; r++) {
			TensorIterator_Attach(&iters[d * relationCount + r],
					relationMatrices[r], transpose);
		}
	}

	// resolve dst's coordinates once, up front: every heuristic evaluation
	// during this search targets this fixed goal. if dst has no numeric
	// lat/lon, the heuristic degrades to 0 for the entire search (i.e.
	// plain Dijkstra) rather than erroring.
	double dst_lat = 0;
	double dst_lon = 0;
	bool dst_has_coords =
		_get_node_latlon(g, dst_id, lat_prop, lon_prop, &dst_lat, &dst_lon);

	// initialization: seed the source node with g_score 0 and no parent
	// (it parents itself, which also makes the path-reconstruction loop's
	// "cur != src_id" stop condition correct). every other node is
	// implicitly at g_score +inf until first discovered below.
	AStarLabel src_label =
		{ .parent = src_id, .g_score = 0, .finalized = false };

	arr_append(labels, src_label);

	uint32_t *src_slot = NodeMap_findOrInsert(&label_idx, src_id, NULL);
	*src_slot = arr_len(labels);

	// push the source onto the priority queue so the main loop below has
	// somewhere to start. its priority is g_score (0) + h(src); since it's
	// the only entry, it's popped first regardless of this value.
	double src_h = _heuristic(g, src_id, lat_prop, lon_prop,
			dst_has_coords, dst_lat, dst_lon);
	NodeWeightItem seed = { .node = src_id, .weight = src_h };

	NodeWeightHeap_offer(&heap, seed);

	bool found = false;

	// main A* loop: repeatedly extract the not-yet-finalized node with the
	// smallest priority (f = g_score + h) and finalize it -- that g_score
	// is now guaranteed optimal, since the heuristic is admissible (never
	// overestimates true remaining cost) and consistent (haversine
	// distance satisfies the triangle inequality), the same guarantee
	// Dijkstra gets from non-negative edge weights alone. stops either
	// when dst is finalized (found) or the heap empties (dst unreachable
	// from src).
	while(!found) {
		// extract the minimum-priority candidate. this may be a stale
		// duplicate left over from a relaxation performed after this
		// entry was queued (see the lazy-deletion note on NodeWeightItem);
		// staleness is detected below via the label's 'finalized' flag
		// rather than by removing superseded heap entries in place.
		NodeWeightItem item;
		if(!NodeWeightHeap_poll(&heap, &item)) {
			break;  // heap exhausted: dst is unreachable
		}

		NodeID cur = item.node;

		uint32_t cur_idx = NodeMap_find(&label_idx, cur);

		ASSERT(cur_idx != 0);
		if(labels[cur_idx - 1].finalized) {
			continue;  // stale duplicate entry
		}

		// finalize 'cur': its current label g_score is its true shortest
		// cost from src and will never be improved again (label setting --
		// each node is finalized exactly once).
		labels[cur_idx - 1].finalized = true;

		// dst just got finalized, its shortest path is settled: stop
		// early instead of exploring the rest of the reachable graph.
		if(cur == dst_id) {
			found = true;
			break;
		}

		double cur_g = labels[cur_idx - 1].g_score;

		Node curNode = GE_NEW_NODE();
		Graph_GetNode(g, cur, &curNode);

		// relaxation step: examine every edge leaving (or entering, per
		// 'dirs') 'cur', across every relationship type the caller
		// allows, and try to improve each neighbor's tentative g_score
		// through 'cur'.
		for(int d = 0; d < ndirs; d++) {
			for(int r = 0; r < relationCount; r++) {
				Graph_GetNodeEdgesFromIterator (g, &curNode, dirs[d],
						&iters[d * relationCount + r], relationIDs[r],
						&neighbors) ;
			}

			uint32_t n = arr_len (neighbors) ;
			for (uint32_t j = 0; j < n; j++) {
				Edge *e = neighbors + j;
				NodeID nid = (dirs[d] == GRAPH_EDGE_DIR_OUTGOING)
					? Edge_GetDestNodeID(e)
					: Edge_GetSrcNodeID(e);

				if(nid == cur) {
					continue;  // ignore self-loops
				}

				// candidate g_score to 'nid' going through 'cur' via this
				// edge: cur's finalized g_score plus the edge's weight.
				// NOTE: weightProp is assumed non-negative here (see the
				// function-level comment above); a negative value would
				// silently make this search's result incorrect.
				SIValue w = _get_value_or_default((GraphEntity *)e,
						weight_prop, SI_LongVal(1));
				double new_g = cur_g + SI_GET_NUMERIC(w);

				// look up (or reserve) 'nid's slot in 'labels'
				bool is_new;
				AStarLabel *nlabel = NULL ;
				uint32_t *nslot = NodeMap_findOrInsert(&label_idx, nid, &is_new);

				if(!is_new) {
					// 'nid' already labeled: this is the relaxation
					// comparison proper. skip if it's already finalized
					// (its g_score is final and can't improve) or if
					// going through 'cur' isn't strictly better than what
					// it already has.
					nlabel = labels + (*nslot - 1);
					if(nlabel->finalized || new_g >= nlabel->g_score) {
						continue;
					}

					// found a strictly shorter route to 'nid' through
					// 'cur': update its label in place with the new best
					// g_score, parent and connecting edge.
					nlabel->edge    = *e;
					nlabel->parent  = cur;
					nlabel->g_score = new_g;
				} else {
					// first time 'nid' is discovered: create its label
					// with 'cur' as parent and 'new_g' as its (so far
					// unbeaten) tentative g_score.

					double h = _heuristic (g, nid, lat_prop, lon_prop,
							dst_has_coords, dst_lat, dst_lon) ;

					AStarLabel lbl = {.parent = cur, .edge = *e,
						.g_score = new_g, .h = h, .finalized = false } ;

					arr_append (labels, lbl) ;

					*nslot = arr_len (labels) ;
					nlabel = labels + (*nslot - 1) ;
				}

				// queue (or re-queue) 'nid' at its updated priority
				// f = g_score + h(nid). any older, now-superseded heap
				// entry for 'nid' is left in place and simply skipped
				// later as a stale duplicate once popped.
				NodeWeightItem qi = { .node = nid, .weight = new_g + nlabel->h };
				NodeWeightHeap_offer(&heap, qi);
			}

			arr_clear(neighbors);
		}
	}

	// search is over (dst found or heap exhausted): entries are stored by
	// value, so there's nothing to drain, just free the heap itself.
	NodeWeightHeap_free(&heap);
	arr_free(neighbors);
	rm_free(iters);

	if(!found) {
		// dst is unreachable from src: nothing to report.
		arr_free(labels);
		NodeMap_free(&label_idx);
		return false;
	}

	// reconstruct the path by walking parent pointers from dst back to
	// src, one finalized label at a time.
	NodeID cur = dst_id;
	Path *p = Path_New(8);

	while(cur != src_id) {
		uint32_t idx = NodeMap_find(&label_idx, cur);
		ASSERT(idx != 0);
		AStarLabel *label = labels + (idx - 1);

		// append 'cur' and the edge that reached it from its parent; the
		// path is being built tail-first (dst towards src) and will be
		// reversed once the walk reaches src.
		Node n = GE_NEW_NODE();
		Graph_GetNode(g, cur, &n);
		Path_AppendNode(p, n);
		Path_AppendEdge(p, label->edge);

		cur = label->parent;
	}

	// walk terminated at src: append it (it has no incoming edge on this
	// path) and flip the path from dst->src order into src->dst order.
	Node srcNode = GE_NEW_NODE();
	Graph_GetNode(g, src_id, &srcNode);
	Path_AppendNode(p, srcNode);

	Path_Reverse(p);

	// dst's finalized label already holds the total shortest g_score from
	// src, accumulated incrementally throughout the relaxation loop.
	uint32_t dst_idx = NodeMap_find(&label_idx, dst_id);

	*path   = p;
	*weight = labels[dst_idx - 1].g_score;

	arr_free(labels);
	NodeMap_free(&label_idx);

	return true;
}

