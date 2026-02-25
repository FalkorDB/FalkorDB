/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "storage.h"
#include "../util/memory.h"
#include "../util/rmalloc.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define TIDESDB_ROOT_DIR "./tidesdb"  // TODO: should be user configurable

tidesdb_t *db = NULL ;

// create tidesdb
// returns 0 on success
int Storage_init(void) {
	// expecting a one time initialization
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

	char *path = NULL ;
	res = asprintf (&path, "%s_%d", TIDESDB_ROOT_DIR, getpid ()) ;
	if (res == -1) {
		return res ;
	}

	config.db_path          = path ;
	config.log_level        = TDB_LOG_ERROR ;
	config.max_memory_usage = get_host_available_memory () * 0.1 ;

	res = tidesdb_open (&config, &db) ;
	free (path) ;

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
			RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
					"tidesdb removing column %s", names[i]) ;

			res = tidesdb_drop_column_family (db, names[i]) ;
			if (res != 0) {
				RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_WARNING,
						"failed to remove column %s", names[i]) ;
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

// delete attribute sets from tidesdb
// returns 0 on success
int Storage_deleteAttributes
(
	tidesdb_column_family_t *cf,  // tidesdb column family
	const EntityID *ids,          // array of entity IDs
	size_t n_ids                  // number of IDs
) {
	ASSERT (cf    != NULL) ;
	ASSERT (ids   != NULL) ;
	ASSERT (n_ids > 0) ;

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
	// delete batch
	//--------------------------------------------------------------------------

	for (size_t i = 0 ; i < n_ids ; i++) {
		EntityID id = ids[i] ;
		ASSERT (id != INVALID_ENTITY_ID) ;

		// key format: <entity_id>
		COMPUTE_KEY (key, id) ;

		res = tidesdb_txn_delete (txn, cf, (const uint8_t*) key, KEY_SIZE) ;

		if (res != 0) {
			RedisModule_Log (NULL, "warning",
					"failed to delete attribute-set for entity id: %" PRIu64 ","
					"err code: %d", id, res) ;
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

