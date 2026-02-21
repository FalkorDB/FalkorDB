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

#define NO_TTL           -1     // item never expires
#define TRANSACTION_CAP  10000  // maximum number of puts per transaction batch

extern tidesdb_t *db ;

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
) {
	// validate arguments
	ASSERT (db      != NULL) ;
	ASSERT (cf      != NULL) ;
	ASSERT (ids     != NULL) ;
	ASSERT (sizes   != NULL) ;
	ASSERT (items   != NULL) ;
	ASSERT (n_items > 0) ;

	int    res = 0 ;
	char   key[KEY_SIZE] ;        // 8 bytes entity id
	size_t item_idx = 0 ;         // current item
	size_t total_offloaded = 0 ;  // number of bytes offloaded

	// number of transactions
	size_t n_txn = (n_items + TRANSACTION_CAP - 1) / TRANSACTION_CAP ;
	tidesdb_txn_t *txn = NULL ;

	//--------------------------------------------------------------------------
	// create transaction
	//--------------------------------------------------------------------------

	res = tidesdb_txn_begin (db, &txn) ;
	if (res != 0) {
		RedisModule_Log (NULL, "warning", "failed to start transaction: %d",
				res) ;
		return res ;
	}

	for (size_t txn_id = 0 ; txn_id < n_txn ; txn_id++) {
		// transaction size
		size_t remaining = n_items - item_idx ;
		size_t txn_size  = (remaining > TRANSACTION_CAP) ?
			TRANSACTION_CAP : remaining ;

		for (size_t i = 0; i < txn_size ; i++, item_idx++) {
			size_t      n    = sizes [item_idx] ;
			uint64_t    id   = ids   [item_idx] ;
			const void *item = items [item_idx] ;

			ASSERT (n > 0) ;
			ASSERT (item != NULL) ;

			// key format: <entity_id>
			COMPUTE_KEY (key, id) ;

			//------------------------------------------------------------------
			// put item into transaction
			//------------------------------------------------------------------

			res = tidesdb_txn_put (txn, cf, (const uint8_t *)key, KEY_SIZE,
					(const uint8_t *)item, n, NO_TTL) ;

			if (res != 0) {
				RedisModule_Log (NULL, "warning",
						"failed to put item into transaction: %d", res) ;
				goto cleanup ;
			}

			total_offloaded += n ;
		}

		//----------------------------------------------------------------------
		// commit transaction
		//----------------------------------------------------------------------

		res = tidesdb_txn_commit (txn);
		if (res != 0) {
			RedisModule_Log (NULL, "warning",
					"failed to commit transaction: %d", res) ;
			goto cleanup ;
		}

		res = tidesdb_txn_reset (txn, TDB_ISOLATION_READ_COMMITTED) ;
		if (res != 0) {
			RedisModule_Log (NULL, "warning", "failed to reset transaction: %d",
					res) ;
			goto cleanup ;
		}
	}

	// report number of bytes committed
	RedisModule_Log (NULL, "debug", "committing %zu bytes", total_offloaded) ;

cleanup:
	if (txn != NULL) {
		tidesdb_txn_free (txn) ;
	}
	return res ;
}

