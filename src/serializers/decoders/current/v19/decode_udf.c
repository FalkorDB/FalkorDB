/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../../../udf/utils.h"
#include "../../../../redismodule.h"

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
		size_t lib_len ;
		size_t script_len ;
		const char *lib    = RedisModule_LoadStringBuffer (io, &lib_len) ;
		const char *script = RedisModule_LoadStringBuffer (io, &script_len) ; 

		// `n` is read off the wire and is not bounded here; bounding it belongs
		// with the rest of the decoder surface in #2376. Stopping on the first
		// failed read is what keeps an absurd count from spinning: a truncated
		// payload fails immediately rather than looping `n` times.
		//
		// a malformed payload can also hand us a zero-length buffer;
		// decrementing it unconditionally underflows size_t to SIZE_MAX, which
		// is then read as a length
		if (RedisModule_IsIOError (io) || lib == NULL || script == NULL ||
			lib_len == 0 || script_len == 0) {
			RedisModule_LogIOError (io, "warning",
					"UDF: malformed library entry %llu of %llu, aborting load",
					(unsigned long long)i, (unsigned long long)n) ;

			if (lib    != NULL) RedisModule_Free ((void*)lib) ;
			if (script != NULL) RedisModule_Free ((void*)script) ;

			return false ;
		}

		// do not count null terminator
		lib_len-- ;
		script_len-- ;

		char *err = NULL ;
		bool res = UDF_Load (script, script_len, lib, lib_len, false, &err) ;

		RedisModule_Free ((void*)lib) ;
		RedisModule_Free ((void*)script) ;

		// an ASSERT here compiles out in release builds, so a library that
		// failed to load was skipped silently and the server came up reporting
		// success with the library missing; calls to it then failed with no
		// indication why
		if (!res) {
			RedisModule_LogIOError (io, "warning",
					"UDF: failed to load library: %s",
					(err != NULL) ? err : "unknown error") ;

			if (err != NULL) free (err) ;

			return false ;
		}

	}

	return true ;
}
