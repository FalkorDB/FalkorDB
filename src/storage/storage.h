/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// the storage component is responsible for managing data loading and offloading
// from and to disk via tidesdb
//
// FalkorDB instance manages a single tidesdb database where each graph uses
// one of more column families
//
// at the moment FalkorDB offloads entire attribute-sets to disk
// node attribute-sets are mapped to disk via the key: <node_id> e.g.
// '280982'
//
// storage exposes three main functions:
// 1. offload items to disk
// 2. load items from disk
// 3. delete items from disk
#pragma once

#include "db.h"
#include "../graph/entities/graph_entity.h"
#include "../graph/entities/attribute_set.h"

#define KEY_SIZE sizeof (uint64_t)  // key byte size

// initialize storage
// returns 0 on success
int Storage_init(void) ;

// serializes and persists an array of items to a TidesDB column family
// each item is stored under its corresponding ID
// the caller is responsible for ensuring that `items`, `sizes`, and `ids`
// are all valid pointers to arrays of at least `n_items` elements
//
// return 0 on success, or a negative error code on failure
//
// note: items are stored in the order they appear
// no deduplication is performed if an ID already exists
// its value will be overwritten
int Storage_save
(
	tidesdb_column_family_t *cf,  // tidesdb column family
	const void * const *items,    // array of pointers to the items to store
	const size_t *sizes,          // array of byte sizes for each item in items
	const uint64_t *ids,          // array of unique 64-bit IDs
								  // under which items will be stored
	size_t n_items                // number of items to save
);

// loads and deserializes an array of items from a TidesDB column family by ID
// retrieves each item corresponding to the given IDs and writes pointers to
// the allocated data into `items`
// the caller is responsible for freeing each
// non-NULL pointer in `items` after use
//
// return 0 on success, or a negative error code on failure
//
// note: caller must free each non-NULL pointer written into items
// if any ID is not found, the corresponding items[i] is set to NULL
int Storage_load
(
	tidesdb_column_family_t *cf,  // tidesdb column family
	void **items,                 // [output] array of pointers
								  // must contain at least n_items pointers

	size_t *sizes,                // [optional] [output] array of byte sizes
								  // for each item
								  // must contain at least n_items elements

	const uint64_t *ids,          // array of unique 64-bit IDs to look up
	size_t n_items                // number of items to load
);

// delete items from tidesdb
// returns 0 on success
int Storage_delete
(
	tidesdb_column_family_t *cf,  // tidesdb column family
	const uint64_t *ids,          // array of unique 64-bit IDs
	size_t n_ids                  // number of IDs to delete
);

// finalize tidesdb
// returns 0 on success
int Storage_finalize (void) ;
