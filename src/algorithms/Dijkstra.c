/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "Dijkstra.h"
#include "../value.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "utils/node_map.h"
#include "utils/priority_heap.h"

// per-node label used by the Dijkstra search below
typedef struct {
	NodeID parent;    // predecessor in the shortest-path tree
	Edge   edge;      // edge connecting parent -> this node
	double weight;    // current best known weight to reach this node
	bool   finalized; // true once popped from the heap with its optimal weight
} DijkstraLabel;

//------------------------------------------------------------------------------
// Dijkstra
//------------------------------------------------------------------------------

bool Dijkstra_ShortestPath
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
	AttributeID weight_prop
) {
	// 'labels' holds one DijkstraLabel per node ever discovered (tentative
	// or finalized best-known distance, its parent and connecting edge).
	// 'label_idx' maps a node id to its 1-based slot in 'labels' (0 means
	// "not yet discovered").
	// 'heap' is the Dijkstra priority queue: pending (node, weight)
	// candidates ordered so the next NodeWeightHeap_poll always returns the
	// smallest-weight candidate discovered so far.
	NodeMap label_idx;
	NodeMap_init(&label_idx);

	DijkstraLabel *labels = arr_new(DijkstraLabel, 64);

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

	// initialization: seed the source node with distance 0 and no parent
	// (it parents itself, which also makes the path-reconstruction loop's
	// "cur != src_id" stop condition correct). every other node is
	// implicitly at distance +inf until first discovered below.
	DijkstraLabel src_label =
		{ .parent = src_id, .weight = 0, .finalized = false };

	arr_append(labels, src_label);

	uint32_t *src_slot = NodeMap_findOrInsert(&label_idx, src_id, NULL);
	*src_slot = arr_len(labels);

	// push the source onto the priority queue so the main loop below has
	// somewhere to start.
	NodeWeightItem seed = { .node = src_id, .weight = 0 };

	NodeWeightHeap_offer(&heap, seed);

	bool found = false;

	// main Dijkstra loop: repeatedly extract the not-yet-finalized node
	// with the smallest tentative distance and finalize it -- that
	// distance is now guaranteed optimal, since all edge weights are
	// non-negative and every unexplored candidate is at least as large.
	// stops either when dst is finalized (found) or the heap empties
	// (dst unreachable from src).
	while(!found) {
		// extract the minimum-weight candidate. this may be a stale
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

		// finalize 'cur': its current label weight is its true shortest
		// distance from src and will never be improved again (label
		// setting -- each node is finalized exactly once).
		labels[cur_idx - 1].finalized = true;

		// dst just got finalized, its shortest path is settled: stop
		// early instead of exploring the rest of the reachable graph.
		if(cur == dst_id) {
			found = true;
			break;
		}

		double cur_weight = labels[cur_idx - 1].weight;

		Node curNode = GE_NEW_NODE();
		Graph_GetNode(g, cur, &curNode);

		// relaxation step: examine every edge leaving (or entering, per
		// 'dirs') 'cur', across every relationship type the caller
		// allows, and try to improve each neighbor's tentative distance
		// through 'cur'.
		for(int d = 0; d < ndirs; d++) {
			for(int r = 0; r < relationCount; r++) {
				Graph_GetNodeEdgesFromIterator(g, &curNode, dirs[d],
						&iters[d * relationCount + r], relationIDs[r], &neighbors);
			}

			uint32_t n = arr_len(neighbors);
			for(uint32_t j = 0; j < n; j++) {
				Edge *e = neighbors + j;
				NodeID nid = (dirs[d] == GRAPH_EDGE_DIR_OUTGOING)
					? Edge_GetDestNodeID(e)
					: Edge_GetSrcNodeID(e);

				if(nid == cur) {
					continue;  // ignore self-loops
				}

				// candidate distance to 'nid' going through 'cur' via
				// this edge: cur's finalized distance plus the edge's
				// weight.
				// NOTE: weightProp is assumed non-negative here (see the
				// function-level comment above); a negative value would
				// silently make this search's result incorrect.
				SIValue w = _get_value_or_default((GraphEntity *)e,
						weight_prop, SI_LongVal(1));
				double new_weight = cur_weight + SI_GET_NUMERIC(w);

				// look up (or reserve) 'nid's slot in 'labels'
				bool is_new;
				uint32_t *nslot = NodeMap_findOrInsert(&label_idx, nid, &is_new);

				if(!is_new) {
					// 'nid' already labeled: this is the relaxation
					// comparison proper. skip if it's already finalized
					// (its distance is final and can't improve) or if
					// going through 'cur' isn't strictly better than what
					// it already has.
					DijkstraLabel *nlabel = labels + (*nslot - 1);
					if(nlabel->finalized || new_weight >= nlabel->weight) {
						continue;
					}

					// found a strictly shorter route to 'nid' through
					// 'cur': update its label in place with the new best
					// distance, parent and connecting edge.
					nlabel->edge   = *e;
					nlabel->parent = cur;
					nlabel->weight = new_weight;
				} else {
					// first time 'nid' is discovered: create its label
					// with 'cur' as parent and 'new_weight' as its (so
					// far unbeaten) tentative distance.
					DijkstraLabel nlabel = { .parent = cur, .edge = *e,
						.weight = new_weight, .finalized = false };

					arr_append(labels, nlabel);
					*nslot = arr_len(labels);
				}

				// queue (or re-queue) 'nid' at its updated tentative
				// weight. any older, now-superseded heap entry for 'nid'
				// is left in place and simply skipped later as a stale
				// duplicate once popped.
				NodeWeightItem qi = { .node = nid, .weight = new_weight };
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
		DijkstraLabel *label = labels + (idx - 1);

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

	// dst's finalized label already holds the total shortest weight from
	// src, accumulated incrementally throughout the relaxation loop.
	uint32_t dst_idx = NodeMap_find(&label_idx, dst_id);

	*path   = p;
	*weight = labels[dst_idx - 1].weight;

	arr_free(labels);
	NodeMap_free(&label_idx);

	return true;
}

