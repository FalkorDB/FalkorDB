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

// delete items from tidesdb
// returns 0 on success
int Storage_delete
(
	tidesdb_column_family_t *cf,  // tidesdb column family
	const uint64_t *ids,          // array of unique 64-bit IDs
	size_t n_ids                  // number of IDs to delete
) {
	ASSERT (cf    != NULL) ;
	ASSERT (ids   != NULL) ;
	ASSERT (n_ids > 0) ;

	int res = 0 ;
	tidesdb_txn_t *txn = NULL ;

	// create transaction
	res = tidesdb_txn_begin (db, &txn) ;
	if (res != 0) {
		RedisModule_Log (NULL, "warning", "failed to start transaction") ;
		return res ;
	}

	//--------------------------------------------------------------------------
	// delete batch
	//--------------------------------------------------------------------------

	for (size_t i = 0 ; i < n_ids ; i++) {
		uint64_t id = ids[i] ;

		res = tidesdb_txn_delete (txn, cf, (const uint8_t*) &id, KEY_SIZE) ;

		if (res != 0) {
			RedisModule_Log (NULL, "warning",
					"failed to delete item with id: %" PRIu64 ", err code: %d",
					id, res) ;
		}
	}

	res = tidesdb_txn_commit (txn) ;
	if (res != 0) {
		RedisModule_Log (NULL, "warning",
				"failed to commit tidesdb transaction, err code: %d", res) ;

	}

	if (tidesdb_txn_free (txn) != 0) {
		RedisModule_Log (NULL, "warning",
				"failed to free tidesdb transaction, err code: %d", res) ;

	}

	return res ;
}

