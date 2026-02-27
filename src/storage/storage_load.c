/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "storage.h"
#include "../util/rmalloc.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern tidesdb_t *db ;

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
) {
	// validate arguments
	ASSERT (db      != NULL) ;
	ASSERT (cf      != NULL) ;
	ASSERT (ids     != NULL) ;
	ASSERT (items   != NULL) ;
	ASSERT (n_items > 0) ;

	//--------------------------------------------------------------------------
	// initialize outputs
	//--------------------------------------------------------------------------

	memset (items, 0, sizeof (void*) * n_items) ;

	if (sizes != NULL) {
		memset (sizes, 0, sizeof (size_t) * n_items) ;
	}

	int res = 0 ;
	tidesdb_txn_t *txn = NULL ;
	
	// create transaction
	res = tidesdb_txn_begin (db, &txn) ;
	if (res != 0) {
		RedisModule_Log (NULL, "warning", "failed to start transaction: %d",
				res) ;
		return res ;
	}

	//--------------------------------------------------------------------------
	// fetch batch
	//--------------------------------------------------------------------------

	for (size_t i = 0 ; i < n_items ; i++) {
		uint64_t id = ids[i] ;

		// get item
		size_t item_size = 0 ;
		res = tidesdb_txn_get (txn, cf, (const uint8_t*) &id, KEY_SIZE,
				(uint8_t**)(items + i), &item_size) ;

		if (res != 0) {
			RedisModule_Log (NULL, "warning",
					"failed to load itemt for id: %" PRIu64 ","
					"err code: %d", id, res) ;
			break ;
		}

		if (sizes != NULL) {
			sizes[i] = item_size ;
		}
	}

	if (txn != NULL) {
		tidesdb_txn_free (txn) ;
	}

	return res ;
}

