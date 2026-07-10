/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "Dijkstra.h"
#include "../value.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"

// get numeric attribute value of an entity otherwise return default value
static inline SIValue _get_value_or_default
(
	GraphEntity *ge,
	AttributeID id,
	SIValue default_value
) {
	SIValue v;

	if(!GraphEntity_GetProperty(ge, id, &v)) {
		return default_value;
	}

	if(SI_TYPE(v) & SI_NUMERIC) {
		return v;
	}

	return default_value;
}

//------------------------------------------------------------------------------
// DijkstraHeap: min-heap of (node, weight) candidates
//------------------------------------------------------------------------------

// per-node label used by the Dijkstra search below
typedef struct {
	NodeID parent;    // predecessor in the shortest-path tree
	Edge   edge;      // edge connecting parent -> this node
	double weight;    // current best known weight to reach this node
	bool   finalized; // true once popped from the heap with its optimal weight
} DijkstraLabel;

// heap entry: a candidate (node, weight) pair waiting to be finalized.
// duplicate/stale entries for the same node are allowed (lazy deletion);
// they're skipped at pop time via DijkstraLabel.finalized.
typedef struct {
	NodeID node;
	double weight;  // weight at the time this entry was queued (heap key)
} DijkstraItem;

// min-heap of DijkstraItem, ordered by ascending weight.
//
// unlike a generic heap (items stored via a runtime-sized memcpy/void* cmp
// callback), this is fixed to DijkstraItem: element moves are plain struct
// assignments the compiler can inline as two register moves, and the
// weight comparison is inlined rather than called through a function
// pointer. entries are stored by value (no per-push allocation) since
// DijkstraItem is a small POD.
typedef struct {
	DijkstraItem *items;  // contiguous buffer of 'cap' slots
	uint32_t count;        // number of items currently held
	uint32_t cap;           // number of slots currently allocated
} DijkstraHeap;

#define DIJKSTRA_HEAP_DEFAULT_CAP 64

static void DijkstraHeap_init
(
	DijkstraHeap *hp
) {
	hp->cap   = DIJKSTRA_HEAP_DEFAULT_CAP ;
	hp->count = 0 ;
	hp->items = rm_calloc (hp->cap, sizeof (DijkstraItem)) ;
}

// move 'item' up from 'idx', treating 'idx' as an empty hole until the
// final resting place is found, then write 'item' there once (one struct
// assignment per level, instead of a swap's three)
static void _dijkstra_heap_sift_up
(
	DijkstraHeap *hp,
	uint32_t idx,
	DijkstraItem item
) {
	while(idx > 0) {
		uint32_t parent = (idx - 1) / 2;

		if(item.weight >= hp->items[parent].weight) {
			break;
		}

		hp->items[idx] = hp->items[parent];
		idx = parent;
	}

	hp->items[idx] = item;
}

// move 'item' down from 'idx' using the same hole technique as
// _dijkstra_heap_sift_up
static void _dijkstra_heap_sift_down
(
	DijkstraHeap *hp,
	uint32_t idx,
	DijkstraItem item
) {
	while(true) {
		uint32_t l = idx * 2 + 1;
		uint32_t r = idx * 2 + 2;
		uint32_t smallest   = idx;
		double   smallest_w = item.weight;

		if(l < hp->count && hp->items[l].weight < smallest_w) {
			smallest   = l;
			smallest_w = hp->items[l].weight;
		}
		if(r < hp->count && hp->items[r].weight < smallest_w) {
			smallest = r;
		}

		if(smallest == idx) {
			break;
		}

		hp->items[idx] = hp->items[smallest];
		idx = smallest;
	}

	hp->items[idx] = item;
}

static void DijkstraHeap_offer
(
	DijkstraHeap *hp,
	DijkstraItem item
) {
	if(hp->count == hp->cap) {
		hp->cap *= 2;
		hp->items = rm_realloc(hp->items, (size_t)hp->cap * sizeof(DijkstraItem));
	}

	_dijkstra_heap_sift_up(hp, hp->count, item);
	hp->count++;
}

static bool DijkstraHeap_poll
(
	DijkstraHeap *hp,
	DijkstraItem *out
) {
	if(hp->count == 0) {
		return false;
	}

	*out = hp->items[0];

	hp->count--;
	if(hp->count > 0) {
		_dijkstra_heap_sift_down(hp, 0, hp->items[hp->count]);
	}

	return true;
}

static void DijkstraHeap_free
(
	DijkstraHeap *hp
) {
	rm_free(hp->items);
}

//------------------------------------------------------------------------------
// NodeMap: NodeID -> label index
//------------------------------------------------------------------------------

// maps a discovered NodeID to its 1-based slot in 'labels' (0 means "not
// present"). specialized open-addressing hash map (linear probing,
// power-of-two capacity, no tombstones) rather than a generic chained
// dict: keys are only ever inserted or looked up during a single search
// and the whole map is torn down in one shot at the end, so there's no
// need for per-entry allocation, deletion support, or incremental
// rehashing -- all of which dominate a generic dict's cost here (a
// malloc/free pair per discovered node, plus chain-walking and
// incremental-rehash bookkeeping on every lookup)
typedef struct {
	NodeID   key;
	uint32_t val;  // 1-based index into 'labels'; 0 means the slot is empty
} NodeMapEntry;

typedef struct {
	NodeMapEntry *slots;
	uint32_t count;  // occupied slots
	uint32_t cap;    // number of slots, always a power of two
} NodeMap;

#define NODE_MAP_DEFAULT_CAP 64

static void NodeMap_init
(
	NodeMap *m
) {
	m->cap   = NODE_MAP_DEFAULT_CAP;
	m->count = 0;
	m->slots = rm_calloc(m->cap, sizeof(NodeMapEntry));
}

// fibonacci hashing: spreads a NodeID (often sequential/dense) across the
// table with a single multiply before masking down to 'cap'
static inline uint32_t _node_map_hash
(
	NodeID key,
	uint32_t cap
) {
	return (uint32_t)((key * 0x9E3779B97F4A7C15ULL) >> 32) & (cap - 1);
}

static void _node_map_grow
(
	NodeMap *m
) {
	uint32_t old_cap        = m->cap;
	NodeMapEntry *old_slots = m->slots;

	m->cap  *= 2;
	m->slots = rm_calloc(m->cap, sizeof(NodeMapEntry));

	for(uint32_t i = 0; i < old_cap; i++) {
		if(old_slots[i].val == 0) {
			continue;  // empty slot
		}

		uint32_t idx = _node_map_hash(old_slots[i].key, m->cap);
		while(m->slots[idx].val != 0) {
			idx = (idx + 1) & (m->cap - 1);
		}
		m->slots[idx] = old_slots[i];
	}

	rm_free(old_slots);
}

// find 'key's slot, inserting a fresh (empty, val == 0) one if absent. the
// returned pointer is only valid until the next call that may grow the
// table. 'is_new', if not NULL, reports which case occurred
static uint32_t *NodeMap_findOrInsert
(
	NodeMap *m,
	NodeID key,
	bool *is_new
) {
	if((m->count + 1) * 2 >= m->cap) {  // load factor >= 0.5
		_node_map_grow(m);
	}

	uint32_t idx = _node_map_hash(key, m->cap);
	while(m->slots[idx].val != 0) {
		if(m->slots[idx].key == key) {
			if(is_new) *is_new = false;
			return &m->slots[idx].val;
		}
		idx = (idx + 1) & (m->cap - 1);
	}

	m->slots[idx].key = key;
	m->count++;
	if(is_new) *is_new = true;

	return &m->slots[idx].val;
}

// find 'key's value, 0 if not present
static uint32_t NodeMap_find
(
	const NodeMap *m,
	NodeID key
) {
	uint32_t idx = _node_map_hash(key, m->cap);
	while(m->slots[idx].val != 0) {
		if(m->slots[idx].key == key) {
			return m->slots[idx].val;
		}
		idx = (idx + 1) & (m->cap - 1);
	}

	return 0;
}

static void NodeMap_free
(
	NodeMap *m
) {
	rm_free(m->slots);
}

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
	// candidates ordered so the next DijkstraHeap_poll always returns the
	// smallest-weight candidate discovered so far.
	NodeMap label_idx;
	NodeMap_init(&label_idx);

	DijkstraLabel *labels = arr_new(DijkstraLabel, 64);

	DijkstraHeap heap;
	DijkstraHeap_init(&heap);

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
	DijkstraItem seed = { .node = src_id, .weight = 0 };

	DijkstraHeap_offer(&heap, seed);

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
		// entry was queued (see the lazy-deletion note on DijkstraItem);
		// staleness is detected below via the label's 'finalized' flag
		// rather than by removing superseded heap entries in place.
		DijkstraItem item;
		if(!DijkstraHeap_poll(&heap, &item)) {
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
				DijkstraItem qi = { .node = nid, .weight = new_weight };
				DijkstraHeap_offer(&heap, qi);
			}

			arr_clear(neighbors);
		}
	}

	// search is over (dst found or heap exhausted): entries are stored by
	// value, so there's nothing to drain, just free the heap itself.
	DijkstraHeap_free(&heap);
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

