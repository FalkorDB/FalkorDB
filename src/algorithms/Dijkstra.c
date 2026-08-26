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
#include "utils/entity_value.h"

#include <float.h>
#include <string.h>

//------------------------------------------------------------------------------
// DijkstraHeap: min-heap of (node, weight) candidates
//------------------------------------------------------------------------------

// per-node search record used by the Dijkstra search below (best-known weight,
// its predecessor and the connecting edge). "label" in the label-setting sense
// of Dijkstra's algorithm -- unrelated to graph node labels.
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
// NodeMap: NodeID -> record index
//------------------------------------------------------------------------------

// maps a discovered NodeID to its 1-based slot in 'records' (0 means "not
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
	uint32_t val;  // 1-based index into 'records'; 0 means the slot is empty
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
// by a subsequent search (see DijkstraCtx). capacity is retained; the whole
// slot buffer is cleared (O(cap)) -- for a context reused across many searches
// this trades a per-run clear for never reallocating.
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
	const Graph          *g;                 // graph to traverse
	GRAPH_EDGE_DIR  dir;                     // traverse direction
	const RelationID     *relationIDs;       // edge type(s) to traverse
	int             relationCount;           // length of relationIDs
	AttributeID     weight_prop;             // weight attribute id

	// expansion directions derived from 'dir', computed once at construction
	int             ndirs;    // number of active directions (1 or 2)
	GRAPH_EDGE_DIR  dirs[2];  // the active directions themselves

	// one TensorIterator per (direction, relation) pair, attached once at
	// construction and re-seeked (not re-attached) on every node expansion,
	// across every DijkstraCtx_Run call -- attaching is expensive (re-derives
	// matrix sparsity/format), re-seeking is cheap, and the matrices don't
	// change over the engine's lifetime (held under the graph read lock)
	TensorIterator *iters;

	// per-run scratch, reset (not reallocated) between runs
	NodeMap         record_idx;  // node id -> 1-based slot in 'records'
	DijkstraLabel  *records;     // one search record per discovered node
	DijkstraHeap    heap;        // priority queue of pending candidates
};

// create a reusable Dijkstra engine over the given graph/direction/relations,
// attaching the tensor iterators once (see DijkstraCtx / Dijkstra.h). the
// relation arrays are borrowed and must outlive the returned context.
DijkstraCtx *DijkstraCtx_New
(
	const Graph *g,                  // graph to traverse
	GRAPH_EDGE_DIR dir,              // traverse direction
	const RelationID *relationIDs,   // edge type(s) to traverse
	const Tensor *relationMatrices,  // relation matrix per relationIDs entry
	int relationCount,               // length of relationIDs
	AttributeID weight_prop          // weight attribute id
) {
	DijkstraCtx *dc = rm_calloc (1, sizeof(DijkstraCtx));

	dc->g                = g;
	dc->dir              = dir;
	dc->relationIDs      = relationIDs;
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

	NodeMap_init(&dc->record_idx);
	dc->records = arr_new(DijkstraLabel, 64);
	DijkstraHeap_init(&dc->heap);

	return dc;
}

// reset per-run scratch (record map, records, heap) so the engine can run
// another search. iterators are intentionally left attached -- that's the
// whole point of reusing the context.
static inline void _DijkstraCtx_Reset
(
	DijkstraCtx *dc
) {
	NodeMap_clear(&dc->record_idx);
	arr_clear(dc->records);
	DijkstraHeap_clear(&dc->heap);
}

// core search shared by the public entry points below. 'early_exit' stops the
// moment dst is finalized (single-pair). otherwise the search is single-source:
// it finalizes every node whose distance is <= 'dist_bound' and stops when the
// next-closest node exceeds it (DBL_MAX = unbounded, i.e. the whole reachable
// component); if dst is also given, reaching it tightens 'dist_bound' to dst's
// distance, so the search finalizes exactly the ball of nodes within the
// src->dst distance -- what the all-shortest DAG needs from each sweep.
static bool _dijkstra_run
(
	DijkstraCtx *dc,
	NodeID src_id,
	NodeID dst_id,
	bool early_exit,
	double dist_bound,
	const dict *blocked_nodes,
	const dict *blocked_edges
) {
	_DijkstraCtx_Reset(dc);

	// initialization: seed the source node with distance 0 and no parent
	// (it parents itself, which also makes DijkstraCtx_Path's "cur != src_id"
	// stop condition correct). every other node is implicitly at distance
	// +inf until first discovered below.
	DijkstraLabel src_label =
		{ .parent = src_id, .weight = 0, .finalized = false };

	arr_append(dc->records, src_label);

	uint32_t *src_slot = NodeMap_findOrInsert(&dc->record_idx, src_id, NULL);
	*src_slot = arr_len(dc->records);

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

		uint32_t cur_idx = NodeMap_find(&dc->record_idx, cur);

		ASSERT(cur_idx != 0);
		if(dc->records[cur_idx - 1].finalized) {
			continue;  // stale duplicate entry
		}

		// bounded single-source: nodes pop in nondecreasing distance, so once
		// one exceeds the bound every remaining one does too -- stop. inert for
		// the unbounded / single-pair paths, where dist_bound is DBL_MAX.
		if(item.weight > dist_bound) {
			break;
		}

		// finalize 'cur': its current label weight is its true shortest
		// distance from src and will never be improved again (label setting --
		// each node is finalized exactly once).
		dc->records[cur_idx - 1].finalized = true;
		double cur_weight = dc->records[cur_idx - 1].weight;

		// dst finalized: single-pair stops immediately; the bounded single-
		// source path instead tightens the bound to dst's distance, so it goes
		// on to finalize the rest of the src->dst ball and then stops.
		if(dst_id != (NodeID)INVALID_ENTITY_ID && cur == dst_id) {
			found = true;
			if(early_exit) {
				break;
			}
			dist_bound = cur_weight;
		}

		// relaxation step: examine every edge leaving (or entering, per
		// 'dirs') 'cur', across every relationship type the caller allows,
		// and try to improve each neighbor's tentative distance through 'cur'.
		//
		// edges are streamed straight from the tensor iterator rather than
		// collected into a scratch array; the edge's attribute set is fetched
		// (from the edges datablock) only when it's actually needed -- to read
		// a weight property, or to store an improving edge for the eventual
		// returned path -- so the many non-improving edges cost nothing beyond
		// the id triplet the iterator already produces.
		bool need_weight = (dc->weight_prop != ATTRIBUTE_ID_NONE);

        int num_iters = dc->ndirs * dc->relationCount;
		for(int r = 0; r < num_iters; r++) {
			TensorIterator *it = &dc->iters[r];
			TensorIterator_IterateRow (it, cur);

			// iters is laid out [dir][relation]; map the flat index back to its
			// relation so relationIDs[] (length relationCount) isn't read out of
			// bounds when ndirs == 2 (relDirection:'both').
			Edge e = { .relationID = dc->relationIDs[r % dc->relationCount] };
			while (TensorIterator_next (it, &e.src_id, &e.dest_id, &e.id, NULL))
			{
				e.attributes = NULL;
				ASSERT (e.src_id == cur || e.dest_id == cur) ;

				// nid is whichever node is NOT cur
				NodeID nid = (e.src_id == cur) ? e.dest_id : e.src_id;
				if(nid == cur) {
					continue;  // ignore self-loops
				}

				// blocked-set filters: skip edges/nodes the caller has
				// virtually removed from the graph for this run (Yen's spur
				// searches). both checks are skipped entirely when the
				// corresponding set is NULL, so the common case pays nothing.
				if(blocked_edges != NULL &&
					HashTableFind((dict *)blocked_edges,
						(void *)(uintptr_t)e.id) != NULL) {
					continue;
				}
				if(blocked_nodes != NULL &&
					HashTableFind((dict *)blocked_nodes,
						(void *)(uintptr_t)nid) != NULL) {
					continue;
				}

				// candidate distance to 'nid' through 'cur': cur's
				// finalized distance plus this edge's weight (default 1 when
				// no weight property is set). reading the weight also
				// populates e.attributes, reused below when storing.
				// NOTE: weightProp is assumed non-negative (see the header);
				// a negative value would silently make the result incorrect.
				double edge_w = 1;
				if(need_weight) {
					Graph_GetEdge(dc->g, e.id, &e);  // populate e.attributes
					SIValue w = _get_value_or_default((GraphEntity *)&e,
							dc->weight_prop, SI_LongVal(1));
					edge_w = SI_GET_NUMERIC(w);
				}
				double new_weight = cur_weight + edge_w;

				// look up (or reserve) 'nid's slot in 'records'
				bool is_new;
				uint32_t *nslot =
					NodeMap_findOrInsert(&dc->record_idx, nid, &is_new);

				if(!is_new) {
					// already discovered: skip if finalized (distance is
					// final) or if going through 'cur' isn't strictly better.
					DijkstraLabel *nlabel = dc->records + (*nslot - 1);
					if(nlabel->finalized || new_weight >= nlabel->weight) {
						continue;
					}
				} else {
					// first time 'nid' is seen: append a placeholder record
					// (filled in just below) and point its slot at it.
					DijkstraLabel fresh = { .finalized = false };
					arr_append(dc->records, fresh);
					*nslot = arr_len(dc->records);
				}

				// store the improving edge; ensure its attribute set is
				// populated (already done above when weighted) so the
				// reconstructed path carries usable edges.
				if(e.attributes == NULL) {
					Graph_GetEdge(dc->g, e.id, &e);
				}
				DijkstraLabel *nlabel = dc->records + (*nslot - 1);
				nlabel->edge   = e;
				nlabel->parent = cur;
				nlabel->weight = new_weight;

				// queue (or re-queue) 'nid' at its updated tentative weight;
				// any superseded heap entry is skipped later as a stale dup.
				DijkstraItem qi = { .node = nid, .weight = new_weight };
				DijkstraHeap_offer(&dc->heap, qi);
			}
		}
	}
	// when a dst was given, report whether it was reached; a pure single-source
	// run (dst == INVALID) always "succeeds" -- results read via _Distance.
	return (dst_id != (NodeID)INVALID_ENTITY_ID) ? found : true;
}

// run a search from 'src_id', resetting per-run scratch first so the context is
// reusable. dst_id == INVALID_ENTITY_ID runs to completion (single-source);
// otherwise the search stops when dst is finalized (single-pair). see
// Dijkstra.h for the full contract.
bool DijkstraCtx_Run
(
	DijkstraCtx *dc,            // engine
	NodeID src_id,              // source node
	NodeID dst_id,              // destination, or INVALID_ENTITY_ID for all
	const dict *blocked_nodes,  // nodes to skip, or NULL
	const dict *blocked_edges   // edges to skip, or NULL
) {
	bool early = (dst_id != (NodeID)INVALID_ENTITY_ID);
	return _dijkstra_run(dc, src_id, dst_id, early, DBL_MAX,
			blocked_nodes, blocked_edges);
}

// bounded single-source run: finalizes every node within 'dist_bound' of src
// and stops. if dst_id is given, reaching it tightens the bound to dst's
// distance -- so the search finalizes exactly the ball of nodes no farther than
// src->dst, and returns whether dst was reached. see Dijkstra.h.
bool DijkstraCtx_RunBounded
(
	DijkstraCtx *dc,            // engine
	NodeID src_id,             // source node
	NodeID dst_id,             // target whose distance bounds the ball, or
	                           //   INVALID_ENTITY_ID to use dist_bound as-is
	double dist_bound,         // finalize nodes within this distance (DBL_MAX = none)
	const dict *blocked_nodes,  // nodes to skip, or NULL
	const dict *blocked_edges   // edges to skip, or NULL
) {
	return _dijkstra_run(dc, src_id, dst_id, false, dist_bound,
			blocked_nodes, blocked_edges);
}

// report the finalized shortest-path weight to 'v' from the last run, or
// false if 'v' was never discovered (see Dijkstra.h).
bool DijkstraCtx_Distance
(
	const DijkstraCtx *dc,  // engine
	NodeID v,               // node to query
	double *weight          // [output] shortest-path weight to 'v'
) {
	uint32_t idx = NodeMap_find(&dc->record_idx, v);
	if(idx == 0) {
		return false;  // 'v' was never discovered by the last run
	}

	if(weight != NULL) {
		*weight = dc->records[idx - 1].weight;
	}

	return true;
}

// reconstruct the src -> dst path from the last (single-pair) run that reached
// dst; caller owns the returned Path (see Dijkstra.h).
Path *DijkstraCtx_Path
(
	const DijkstraCtx *dc,  // engine
	NodeID src_id,          // source node (walk stops here)
	NodeID dst_id           // destination node (walk starts here)
) {
	// reconstruct the path by walking parent pointers from dst back to src,
	// one finalized record at a time.
	NodeID cur = dst_id;
	Path *p = Path_New(8);

	while(cur != src_id) {
		uint32_t idx = NodeMap_find(&dc->record_idx, cur);
		ASSERT(idx != 0);
		DijkstraLabel *label = dc->records + (idx - 1);

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

// free a Dijkstra engine and all its scratch/iterators.
void DijkstraCtx_Free
(
	DijkstraCtx *dc  // engine
) {
	if(dc == NULL) {
		return;
	}

	DijkstraHeap_free(&dc->heap);
	NodeMap_free(&dc->record_idx);
	arr_free(dc->records);
	rm_free(dc->iters);
	rm_free(dc);
}

//------------------------------------------------------------------------------
// Dijkstra
//------------------------------------------------------------------------------

bool Dijkstra_ShortestPath
(
	Path **path,                     // [output] src -> dst path
	double *weight,                  // [output] total path weight
	const Graph *g,                  // graph to traverse
	NodeID src_id,                   // source node
	NodeID dst_id,                   // destination node
	GRAPH_EDGE_DIR dir,              // traverse direction
	const RelationID *relationIDs,   // edge type(s) to traverse
	const Tensor *relationMatrices,  // relation matrix per relationIDs entry
	int relationCount,               // length of relationIDs
	AttributeID weight_prop          // weight attribute id
) {
	ASSERT(g      != NULL);
	ASSERT(path   != NULL);
	ASSERT(weight != NULL);

	// convenience wrapper for the single-pair, single-shot case: build a
	// context, run one search, reconstruct the path, tear it down. callers that
	// issue many searches over the same graph (Yen's spur searches, the
	// bidirectional all-shortest DAG) instead hold a DijkstraCtx and call
	// DijkstraCtx_Run repeatedly, which is what the reuse machinery is for.
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
