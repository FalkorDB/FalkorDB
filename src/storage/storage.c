/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "storage.h"

#include <stdio.h>
#include <stdlib.h>

#define TIDESDB_ROOT_DIR "./tidesdb"  // TODO: should be user configurable
#define NO_TTL -1  // tidesdb key's TTL
#define KEY_SIZE (sizeof (char) + sizeof (EntityID))  // tidesdb key byte size
#define TRANSACTION_CAP 10000 // tidesdb transaction max size 10K

tidesdb_t *db = NULL ;

// compute tidesdb key for entity
// key = 'N'/'E'<entity_id>
static inline void compute_key
(
	char *key,          // [output] tidesdb key
	GraphEntityType t,  // entity type node/edge
	EntityID id         // entity id
) {
	// set prefix
	key[0] = (t == GETYPE_NODE) ? 'N' : 'E' ;

	// set entity id
	*((uint64_t*) (key+1)) = id ;
}

// create tidesdb
int Storage_init(void) {
	ASSERT (db == NULL) ;

	// tidesdb config
	tidesdb_config_t config = tidesdb_default_config () ;

	config.db_path   = TIDESDB_ROOT_DIR ;
	config.log_level = TDB_LOG_ERROR ;

	// try to open the database
	if (tidesdb_open (&config, &db) != 0) {
		RedisModule_Log (NULL, "warning", "failed to open tidesdb") ;
		return -1 ;
	}

	// TODO: clear all data from tidesdb

	return 0 ;
}

// offloads attribute sets to tidesdb
int Storage_putAttributes
(
	tidesdb_column_family_t *cf,  // tidesdb column family
	const AttributeSet *sets,     // array of attribute sets
	size_t n_sets,                // number of sets to offload
	GraphEntityType t,            // Node or Edge type
	const EntityID *ids           // array of entity IDs
) {
	// validate arguments
	ASSERT (t      == GETYPE_NODE || t == GETYPE_EDGE) ;
	ASSERT (cf     != NULL) ;
	ASSERT (ids    != NULL) ;
	ASSERT (sets   != NULL) ;
	ASSERT (n_sets > 0) ;

	//--------------------------------------------------------------------------
	// compute tidesdb key
	//--------------------------------------------------------------------------

	int    res = 0 ;
	char   key[KEY_SIZE] ;        // prefix char followed by 8 bytes entity id
	size_t set_idx = 0 ;          // current attribute-set
	size_t total_offloaded = 0 ;  // number of bytes offloaded

	// number of transactions
	uint n_txn = (n_sets + TRANSACTION_CAP - 1) / TRANSACTION_CAP ;
	tidesdb_txn_t *txn = NULL ;

	for (size_t txn_id = 0 ; txn_id < n_txn ; txn_id++) {
		//----------------------------------------------------------------------
		// create transaction
		//----------------------------------------------------------------------

		res = tidesdb_txn_begin (db, &txn) ;
		if (res != 0) {
			RedisModule_Log (NULL, "warning", "failed to start transaction") ;
			goto cleanup ;
		}

		// transaction size
		size_t remaining = n_sets - set_idx ;
		size_t txn_size  = (remaining > TRANSACTION_CAP) ?
			TRANSACTION_CAP : remaining ;

		for (size_t i = 0; i < txn_size ; i++, set_idx++) {
			EntityID     id  = ids[set_idx] ;
			AttributeSet set = sets[set_idx] ;

			ASSERT (id != INVALID_ENTITY_ID) ;

			// key format: N/E<entity_id>
			compute_key (key, t, id) ;

			//------------------------------------------------------------------
			// put attribute-set into transaction
			//------------------------------------------------------------------

			size_t n = AttributeSet_ByteSize (set) ;
			res = tidesdb_txn_put (txn, cf, (const uint8_t *)key, KEY_SIZE,
					(const uint8_t *)set, n, NO_TTL) ;

			if (res != 0) {
				RedisModule_Log (NULL, "warning",
						"failed to put attribute-set into transaction") ;
				goto cleanup ;
			}

			total_offloaded += n ;
		}

		// report number of bytes offloaded
		RedisModule_Log (NULL, "debug", "offloading %zu bytes",
				total_offloaded) ;
		total_offloaded = 0 ;

		//----------------------------------------------------------------------
		// commit transaction
		//----------------------------------------------------------------------

		res = tidesdb_txn_commit (txn);
		tidesdb_txn_free (txn) ;
		txn = NULL ;

		if (res != 0) {
			RedisModule_Log (NULL, "warning", "failed to commit transaction") ;
			goto cleanup ;
		}
	}

cleanup:
	if (txn != NULL) {
		tidesdb_txn_free (txn) ;
	}
	return res ;
}

int Storage_finalize (void) {
	ASSERT (db != NULL) ;

	int res = tidesdb_close (db) ;
	if (res != 0) {
		RedisModule_Log (NULL, "warning", "failed to close tidesdb") ;
	}

	return res ;
}

