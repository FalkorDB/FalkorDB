/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "storage.h"
#include "../util/rmalloc.h"

#include <stdio.h>
#include <stdlib.h>

#define TIDESDB_ROOT_DIR "./tidesdb"  // TODO: should be user configurable
#define NO_TTL           -1                 // key TTL
#define KEY_SIZE         sizeof (EntityID)  // key byte size
#define TRANSACTION_CAP  10000              // txn capacity

tidesdb_t *db = NULL ;

// compute tidesdb key for entity
// key = <entity_id>
static inline void compute_key
(
	char *key,          // [output] tidesdb key
	EntityID id         // entity id
) {
	// set entity id
	*((uint64_t*) (key)) = id ;
}

// create tidesdb
// returns 0 on success
int Storage_init(void) {
	ASSERT (db == NULL) ;

	int res = tidesdb_init (RedisModule_Alloc, RedisModule_Calloc,
			RedisModule_Realloc, RedisModule_Free) ;

	if (res != 0) {
		return res ;	
	}

	//--------------------------------------------------------------------------
	// open the database
	//--------------------------------------------------------------------------

	tidesdb_config_t config = tidesdb_default_config () ;

	config.db_path   = TIDESDB_ROOT_DIR ;
	config.log_level = TDB_LOG_ERROR ;

	res = tidesdb_open (&config, &db) ;
	if (res != 0) {
		RedisModule_Log (NULL, "warning", "failed to open tidesdb") ;
		return res ;
	}

	//--------------------------------------------------------------------------
	// clear tidesdb database
	//--------------------------------------------------------------------------

	int count    = 0 ;
	char **names = NULL ;

	if (tidesdb_list_column_families (db, &names, &count) == 0) {
		for (int i = 0 ; i < count ; i++) {
			res = tidesdb_drop_column_family (db, names[i]) ;
			if (res != 0) {
				RedisModule_Log (NULL, "warning", "failed to remove column %s",
						names[i]) ;
				break ;
			}
		}
	}

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	if (names != NULL) {
		for (int i = 0 ; i < count ; i++) {
			rm_free (names[i]) ;
		}
		rm_free (names) ;
	}

	return res ;
}

// offloads attribute sets to tidesdb
// returns 0 on success
int Storage_putAttributes
(
	tidesdb_column_family_t *cf,  // tidesdb column family
	const AttributeSet *sets,     // array of attribute sets
	size_t n_sets,                // number of sets to offload
	const EntityID *ids           // array of entity IDs
) {
	// validate arguments
	ASSERT (db     != NULL) ;
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

			ASSERT (set != NULL) ;
			ASSERT (id  != INVALID_ENTITY_ID) ;

			// key format: <entity_id>
			compute_key (key, id) ;

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

// loads attribute sets from tidesdb
// returns 0 on success
int Storage_loadAttributes
(
	tidesdb_column_family_t *cf,  // tidesdb column family
	void **sets,                  // array of attribute sets
	size_t n_sets,                // number of sets to load
	const EntityID *ids           // array of entity IDs
) {
	// validate arguments
	ASSERT (db     != NULL) ;
	ASSERT (cf     != NULL) ;
	ASSERT (ids    != NULL) ;
	ASSERT (sets   != NULL) ;
	ASSERT (n_sets > 0) ;

	int res = 0 ;
	tidesdb_txn_t *txn = NULL ;
	char key[KEY_SIZE] ;        // prefix char followed by 8 bytes entity id

	// create transaction
	res = tidesdb_txn_begin (db, &txn) ;
	if (res != 0) {
		RedisModule_Log (NULL, "warning", "failed to start transaction") ;
		return res ;
	}

	//--------------------------------------------------------------------------
	// fetch batch
	//--------------------------------------------------------------------------

	for (size_t i = 0 ; i < n_sets ; i++) {
		EntityID id = ids[i] ;
		ASSERT (id != INVALID_ENTITY_ID) ;

		// key format: <entity_id>
		compute_key (key, id) ;

		// get attribute-set
		size_t value_size ;
		res = tidesdb_txn_get (txn, cf, (const uint8_t*) key, KEY_SIZE,
				(uint8_t**)(sets + i), &value_size) ;

		if (res != 0) {
			RedisModule_Log (NULL, "warning",
					"failed to get attribute-set for entity id: %llu,"
					"err code: %d", id, res) ;
			// setting attribute-set to NULL
			sets[i] = NULL ;
		}
	}

	if (txn != NULL) {
		tidesdb_txn_free (txn) ;
	}

	return res ;
}

// finalize tidesdb
// returns 0 on success
int Storage_finalize (void) {
	ASSERT (db != NULL) ;

	int res = tidesdb_close (db) ;
	if (res != 0) {
		RedisModule_Log (NULL, "warning", "failed to close tidesdb") ;
	}

	return res ;
}

