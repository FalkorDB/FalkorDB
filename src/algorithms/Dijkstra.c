/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "Dijkstra.h"
#include "../value.h"
#include "../util/arr.h"
#include "../util/dict.h"
#include "../util/rmalloc.h"

#include <string.h>

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

// reset the heap to empty without releasing its buffer, so it can be reused
// by a subsequent search (see DijkstraCtx).
static inline void DijkstraHeap_clear
(
	DijkstraHeap *hp
) {
	hp->count = 0;
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

// reset the map to empty without releasing its buffer, so it can be reused
// by a subsequent search (see DijkstraCtx). capacity is retained; only the
// occupied slots are zeroed back out.
static inline void NodeMap_clear
(
	NodeMap *m
) {
	memset(m->slots, 0, (size_t)m->cap * sizeof(NodeMapEntry));
	m->count = 0;
}

static void NodeMap_free
(
	NodeMap *m
) {
	rm_free(m->slots);
}

//------------------------------------------------------------------------------
// DijkstraCtx: reusable Dijkstra engine
//------------------------------------------------------------------------------

struct DijkstraCtx {
	// graph + search parameters (borrowed, owned by the caller)
	Graph          *g;                 // graph to traverse
	GRAPH_EDGE_DIR  dir;               // traverse direction
	RelationID     *relationIDs;       // edge type(s) to traverse
	Tensor         *relationMatrices;  // relation matrix per relationIDs entry
	int             relationCount;     // length of relationIDs
	AttributeID     weight_prop;       // weight attribute id

	// expansion directions derived from 'dir', computed once at construction
	int             ndirs;             // number of active directions (1 or 2)
	GRAPH_EDGE_DIR  dirs[2];           // the active directions themselves

	// one TensorIterator per (direction, relation) pair, attached once at
	// construction and re-seeked (not re-attached) on every node expansion,
	// across every DijkstraCtx_Run call -- attaching is expensive (re-derives
	// matrix sparsity/format), re-seeking is cheap, and the matrices don't
	// change over the engine's lifetime (held under the graph read lock)
	TensorIterator *iters;

	// per-run scratch, reset (not reallocated) between runs
	NodeMap         label_idx;   // node id -> 1-based slot in 'labels'
	DijkstraLabel  *labels;      // one DijkstraLabel per discovered node
	DijkstraHeap    heap;        // priority queue of pending candidates
	Edge           *neighbors;   // reused buffer for each neighbor scan
};

DijkstraCtx *DijkstraCtx_New
(
	Graph *g,
	GRAPH_EDGE_DIR dir,
	RelationID *relationIDs,
	Tensor *relationMatrices,
	int relationCount,
	AttributeID weight_prop
) {
	DijkstraCtx *dc = rm_calloc(1, sizeof(DijkstraCtx));

	dc->g                = g;
	dc->dir              = dir;
	dc->relationIDs      = relationIDs;
	dc->relationMatrices = relationMatrices;
	dc->relationCount    = relationCount;
	dc->weight_prop      = weight_prop;

	// build the list of edge directions to expand through when scanning a
	// node's neighbors: OUTGOING, INCOMING, or both, per the caller's
	// requested traversal direction.
	dc->ndirs = 0;
	if(dir == GRAPH_EDGE_DIR_OUTGOING || dir == GRAPH_EDGE_DIR_BOTH) {
		dc->dirs[dc->ndirs++] = GRAPH_EDGE_DIR_OUTGOING;
	}
	if(dir == GRAPH_EDGE_DIR_INCOMING || dir == GRAPH_EDGE_DIR_BOTH) {
		dc->dirs[dc->ndirs++] = GRAPH_EDGE_DIR_INCOMING;
	}

	// attach one TensorIterator per (direction, relation) pair, once.
	dc->iters =
		rm_malloc(dc->ndirs * relationCount * sizeof(TensorIterator));
	for(int d = 0; d < dc->ndirs; d++) {
		bool transpose = (dc->dirs[d] == GRAPH_EDGE_DIR_INCOMING);
		for(int r = 0; r < relationCount; r++) {
			TensorIterator_Attach(&dc->iters[d * relationCount + r],
					relationMatrices[r], transpose);
		}
	}

	NodeMap_init(&dc->label_idx);
	dc->labels    = arr_new(DijkstraLabel, 64);
	DijkstraHeap_init(&dc->heap);
	dc->neighbors = arr_new(Edge, 32);

	return dc;
}

// reset per-run scratch (map/labels/heap/neighbors) so the engine can run
// another search. iterators are intentionally left attached -- that's the
// whole point of reusing the context.
static inline void _DijkstraCtx_Reset
(
	DijkstraCtx *dc
) {
	NodeMap_clear(&dc->label_idx);
	arr_clear(dc->labels);
	DijkstraHeap_clear(&dc->heap);
	arr_clear(dc->neighbors);
}

bool DijkstraCtx_Run
(
	DijkstraCtx *dc,
	NodeID src_id,
	NodeID dst_id,
	const dict *blocked_nodes,
	const dict *blocked_edges
) {
	_DijkstraCtx_Reset(dc);

	// single-pair mode stops as soon as 'dst' is finalized; single-source
	// mode (dst == INVALID_ENTITY_ID) runs to completion over the whole
	// reachable component.
	bool early = (dst_id != (NodeID)INVALID_ENTITY_ID);

	// initialization: seed the source node with distance 0 and no parent
	// (it parents itself, which also makes DijkstraCtx_Path's "cur != src_id"
	// stop condition correct). every other node is implicitly at distance
	// +inf until first discovered below.
	DijkstraLabel src_label =
		{ .parent = src_id, .weight = 0, .finalized = false };

	arr_append(dc->labels, src_label);

	uint32_t *src_slot = NodeMap_findOrInsert(&dc->label_idx, src_id, NULL);
	*src_slot = arr_len(dc->labels);

	DijkstraItem seed = { .node = src_id, .weight = 0 };
	DijkstraHeap_offer(&dc->heap, seed);

	bool found = false;

	// main Dijkstra loop: repeatedly extract the not-yet-finalized node with
	// the smallest tentative distance and finalize it -- that distance is now
	// guaranteed optimal, since all edge weights are non-negative and every
	// unexplored candidate is at least as large. stops either when dst is
	// finalized (single-pair) or the heap empties (single-source, or dst
	// unreachable).
	while(true) {
		// extract the minimum-weight candidate. this may be a stale duplicate
		// left over from a relaxation performed after this entry was queued
		// (lazy deletion); staleness is detected below via the label's
		// 'finalized' flag rather than by removing superseded heap entries.
		DijkstraItem item;
		if(!DijkstraHeap_poll(&dc->heap, &item)) {
			break;  // heap exhausted
		}

		NodeID cur = item.node;

		uint32_t cur_idx = NodeMap_find(&dc->label_idx, cur);

		ASSERT(cur_idx != 0);
		if(dc->labels[cur_idx - 1].finalized) {
			continue;  // stale duplicate entry
		}

		// finalize 'cur': its current label weight is its true shortest
		// distance from src and will never be improved again (label setting --
		// each node is finalized exactly once).
		dc->labels[cur_idx - 1].finalized = true;

		// in single-pair mode, dst just got finalized: its shortest path is
		// settled, stop early instead of exploring the rest of the graph.
		if(early && cur == dst_id) {
			found = true;
			break;
		}

		double cur_weight = dc->labels[cur_idx - 1].weight;

		Node curNode = GE_NEW_NODE();
		Graph_GetNode(dc->g, cur, &curNode);

		// relaxation step: examine every edge leaving (or entering, per
		// 'dirs') 'cur', across every relationship type the caller allows,
		// and try to improve each neighbor's tentative distance through 'cur'.
		for(int d = 0; d < dc->ndirs; d++) {
			for(int r = 0; r < dc->relationCount; r++) {
				Graph_GetNodeEdgesFromIterator(dc->g, &curNode, dc->dirs[d],
						&dc->iters[d * dc->relationCount + r],
						dc->relationIDs[r], &dc->neighbors);
			}

			uint32_t n = arr_len(dc->neighbors);
			for(uint32_t j = 0; j < n; j++) {
				Edge *e = dc->neighbors + j;
				NodeID nid = (dc->dirs[d] == GRAPH_EDGE_DIR_OUTGOING)
					? Edge_GetDestNodeID(e)
					: Edge_GetSrcNodeID(e);

				if(nid == cur) {
					continue;  // ignore self-loops
				}

				// blocked-set filters: skip edges/nodes the caller has
				// virtually removed from the graph for this run (Yen's spur
				// searches). both checks are skipped entirely when the
				// corresponding set is NULL, so the common case pays nothing.
				if(blocked_edges != NULL &&
					HashTableFind((dict *)blocked_edges,
						(void *)(uintptr_t)ENTITY_GET_ID(e)) != NULL) {
					continue;
				}
				if(blocked_nodes != NULL &&
					HashTableFind((dict *)blocked_nodes,
						(void *)(uintptr_t)nid) != NULL) {
					continue;
				}

				// candidate distance to 'nid' going through 'cur' via this
				// edge: cur's finalized distance plus the edge's weight.
				// NOTE: weightProp is assumed non-negative here (see the
				// header comment); a negative value would silently make this
				// search's result incorrect.
				SIValue w = _get_value_or_default((GraphEntity *)e,
						dc->weight_prop, SI_LongVal(1));
				double new_weight = cur_weight + SI_GET_NUMERIC(w);

				// look up (or reserve) 'nid's slot in 'labels'
				bool is_new;
				uint32_t *nslot =
					NodeMap_findOrInsert(&dc->label_idx, nid, &is_new);

				if(!is_new) {
					// 'nid' already labeled: this is the relaxation comparison
					// proper. skip if it's already finalized (its distance is
					// final and can't improve) or if going through 'cur' isn't
					// strictly better than what it already has.
					DijkstraLabel *nlabel = dc->labels + (*nslot - 1);
					if(nlabel->finalized || new_weight >= nlabel->weight) {
						continue;
					}

					// found a strictly shorter route to 'nid' through 'cur':
					// update its label in place with the new best distance,
					// parent and connecting edge.
					nlabel->edge   = *e;
					nlabel->parent = cur;
					nlabel->weight = new_weight;
				} else {
					// first time 'nid' is discovered: create its label with
					// 'cur' as parent and 'new_weight' as its (so far
					// unbeaten) tentative distance.
					DijkstraLabel nlabel = { .parent = cur, .edge = *e,
						.weight = new_weight, .finalized = false };

					arr_append(dc->labels, nlabel);
					*nslot = arr_len(dc->labels);
				}

				// queue (or re-queue) 'nid' at its updated tentative weight.
				// any older, now-superseded heap entry for 'nid' is left in
				// place and simply skipped later as a stale duplicate.
				DijkstraItem qi = { .node = nid, .weight = new_weight };
				DijkstraHeap_offer(&dc->heap, qi);
			}

			arr_clear(dc->neighbors);
		}
	}

	// single-pair: report whether dst was reached. single-source: the run
	// always completes; the caller reads results via DijkstraCtx_Distance.
	return early ? found : true;
}

bool DijkstraCtx_Distance
(
	const DijkstraCtx *dc,
	NodeID v,
	double *weight
) {
	uint32_t idx = NodeMap_find(&dc->label_idx, v);
	if(idx == 0) {
		return false;  // 'v' was never discovered by the last run
	}

	if(weight != NULL) {
		*weight = dc->labels[idx - 1].weight;
	}

	return true;
}

Path *DijkstraCtx_Path
(
	const DijkstraCtx *dc,
	NodeID src_id,
	NodeID dst_id
) {
	// reconstruct the path by walking parent pointers from dst back to src,
	// one finalized label at a time.
	NodeID cur = dst_id;
	Path *p = Path_New(8);

	while(cur != src_id) {
		uint32_t idx = NodeMap_find(&dc->label_idx, cur);
		ASSERT(idx != 0);
		DijkstraLabel *label = dc->labels + (idx - 1);

		// append 'cur' and the edge that reached it from its parent; the path
		// is built tail-first (dst towards src) and reversed once we hit src.
		Node n = GE_NEW_NODE();
		Graph_GetNode(dc->g, cur, &n);
		Path_AppendNode(p, n);
		Path_AppendEdge(p, label->edge);

		cur = label->parent;
	}

	// walk terminated at src: append it (it has no incoming edge on this path)
	// and flip the path from dst->src order into src->dst order.
	Node srcNode = GE_NEW_NODE();
	Graph_GetNode(dc->g, src_id, &srcNode);
	Path_AppendNode(p, srcNode);

	Path_Reverse(p);

	return p;
}

void DijkstraCtx_Free
(
	DijkstraCtx *dc
) {
	if(dc == NULL) {
		return;
	}

	DijkstraHeap_free(&dc->heap);
	NodeMap_free(&dc->label_idx);
	arr_free(dc->labels);
	arr_free(dc->neighbors);
	rm_free(dc->iters);
	rm_free(dc);
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
	// thin wrapper over the reusable engine: build a one-shot context, run a
	// single-pair search, reconstruct the path if dst was reached.
	DijkstraCtx *dc = DijkstraCtx_New(g, dir, relationIDs, relationMatrices,
			relationCount, weight_prop);

	bool found = DijkstraCtx_Run(dc, src_id, dst_id, NULL, NULL);

	if(found) {
		*path = DijkstraCtx_Path(dc, src_id, dst_id);
		DijkstraCtx_Distance(dc, dst_id, weight);
	}

	DijkstraCtx_Free(dc);

	return found;
}
