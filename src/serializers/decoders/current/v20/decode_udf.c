/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../../../udf/utils.h"
#include "../../../../redismodule.h"

void AUXLoadUDF_latest
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
			return ;
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
			return ;
		}

		// do not count null terminator
		lib_len-- ;
		script_len-- ;

		bool res = UDF_Load (script, script_len, lib, lib_len, false, NULL) ;
		ASSERT (res == true) ;

		RedisModule_Free ((void*)lib) ;
		RedisModule_Free ((void*)script) ;
	}
}

