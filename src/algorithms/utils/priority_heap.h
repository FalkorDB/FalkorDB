/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "RG.h"
#include "../../util/rmalloc.h"
#include "../../graph/graph.h"

//------------------------------------------------------------------------------
// NodeWeightHeap: min-heap of (node, weight) candidates
//------------------------------------------------------------------------------

// heap entry: a candidate (node, weight) pair waiting to be finalized.
// duplicate/stale entries for the same node are allowed (lazy deletion);
// callers skip them at pop time via their own label's 'finalized' flag.
// 'weight' is the search priority at the time this entry was queued --
// for a plain shortest-path search that's the true distance, for A* it's
// f = g_score + h; either way the true cost lives on the caller's label,
// not here.
typedef struct {
	NodeID node;
	double weight;
} NodeWeightItem;

// min-heap of NodeWeightItem, ordered by ascending weight.
//
// unlike a generic heap (items stored via a runtime-sized memcpy/void* cmp
// callback), this is fixed to NodeWeightItem: element moves are plain
// struct assignments the compiler can inline as two register moves, and
// the weight comparison is inlined rather than called through a function
// pointer. entries are stored by value (no per-push allocation) since
// NodeWeightItem is a small POD.
typedef struct {
	NodeWeightItem *items;  // contiguous buffer of 'cap' slots
	uint32_t count;         // number of items currently held
	uint32_t cap;           // number of slots currently allocated
} NodeWeightHeap;

#define NODE_WEIGHT_HEAP_DEFAULT_CAP 64

static inline void NodeWeightHeap_init
(
	NodeWeightHeap *hp
) {
	hp->cap   = NODE_WEIGHT_HEAP_DEFAULT_CAP ;
	hp->count = 0 ;
	hp->items = rm_calloc (hp->cap, sizeof (NodeWeightItem)) ;
}

// move 'item' up from 'idx', treating 'idx' as an empty hole until the
// final resting place is found, then write 'item' there once (one struct
// assignment per level, instead of a swap's three)
static inline void _node_weight_heap_sift_up
(
	NodeWeightHeap *hp,
	uint32_t idx,
	NodeWeightItem item
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
// _node_weight_heap_sift_up
static inline void _node_weight_heap_sift_down
(
	NodeWeightHeap *hp,
	uint32_t idx,
	NodeWeightItem item
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

static inline void NodeWeightHeap_offer
(
	NodeWeightHeap *hp,
	NodeWeightItem item
) {
	if(hp->count == hp->cap) {
		hp->cap *= 2;
		hp->items = rm_realloc(hp->items, (size_t)hp->cap * sizeof(NodeWeightItem));
	}

	_node_weight_heap_sift_up(hp, hp->count, item);
	hp->count++;
}

static inline bool NodeWeightHeap_poll
(
	NodeWeightHeap *hp,
	NodeWeightItem *out
) {
	if(hp->count == 0) {
		return false;
	}

	*out = hp->items[0];

	hp->count--;
	if(hp->count > 0) {
		_node_weight_heap_sift_down(hp, 0, hp->items[hp->count]);
	}

	return true;
}

// reset the heap to empty without releasing its buffer, so it can be reused by
// a subsequent search.
static inline void NodeWeightHeap_clear
(
	NodeWeightHeap *hp
) {
	hp->count = 0;
}

static inline void NodeWeightHeap_free
(
	NodeWeightHeap *hp
) {
	rm_free(hp->items);
}
