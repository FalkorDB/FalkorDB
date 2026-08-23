/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "RG.h"
#include "../../value.h"
#include "../../util/rmalloc.h"
#include "../../graph/graph.h"

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

static inline void NodeMap_init
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

static inline void _node_map_grow
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
static inline uint32_t *NodeMap_findOrInsert
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
static inline uint32_t NodeMap_find
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

static inline void NodeMap_free
(
	NodeMap *m
) {
	rm_free(m->slots);
}
