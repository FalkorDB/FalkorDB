/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../../../udf/utils.h"
#include "../../../../redismodule.h"

// returns false if the UDF section is malformed or a library fails to load
bool AUXLoadUDF_latest
(
	RedisModuleIO *io
) {
	// decode UDFs
	// format:
	// number of UDFs
	// [
	//    library's name
	//    library's script
	// ]

	ASSERT (io != NULL) ;

	// write the library count
	uint64_t n = RedisModule_LoadUnsigned (io) ;

	for (uint64_t i = 0; i < n; i++) {
		// abort on a short read; the count may be valid while the payload is
		// truncated, so re-check before trusting each entry
		if (RedisModule_IsIOError (io)) {
			return false ;
		}

		size_t lib_len    = 0 ;
		size_t script_len = 0 ;
		const char *lib    = RedisModule_LoadStringBuffer (io, &lib_len) ;
		const char *script = RedisModule_LoadStringBuffer (io, &script_len) ;

		// a short read yields NULL buffers and latches the IO error; bail out
		// before decrementing the (zero) lengths or handing NULL to UDF_Load
		if (RedisModule_IsIOError (io) || lib == NULL || script == NULL) {
			if (lib    != NULL) RedisModule_Free ((void*)lib) ;
			if (script != NULL) RedisModule_Free ((void*)script) ;
			return false ;
		}

		// both strings are encoded with their null terminator and must be
		// non-empty
		if (lib_len < 2 || lib [lib_len - 1] != '\0' ||
			script_len < 2 || script [script_len - 1] != '\0') {
			RedisModule_Free ((void*)lib) ;
			RedisModule_Free ((void*)script) ;
			RedisModule_Log (NULL, "warning",
					"Failed loading graph UDFs: malformed library entry") ;
			return false ;
		}

		// do not count null terminator
		lib_len-- ;
		script_len-- ;

		// the RDB is the source of truth: replace a library this process
		// already holds (UDFs survive a keyspace flush, e.g. on a replica's
		// second full sync or DEBUG RELOAD)
		char *err = NULL ;
		bool res = UDF_Load (script, script_len, lib, lib_len, true, &err) ;
		if (!res) {
			RedisModule_Log (NULL, "warning",
					"Failed loading graph UDF library '%s': %s",
					lib, err != NULL ? err : "unknown error") ;
		}
		if (err != NULL) {
			free (err) ;
		}

		RedisModule_Free ((void*)lib) ;
		RedisModule_Free ((void*)script) ;

		if (!res) {
			return false ;
		}
	}

	return true ;
}

