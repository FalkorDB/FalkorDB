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

	for (unsigned int i = 0; i < n; i++) {
		size_t lib_len ;
		size_t script_len ;
		const char *lib    = RedisModule_LoadStringBuffer (io, &lib_len) ;
		const char *script = RedisModule_LoadStringBuffer (io, &script_len) ; 

		// do not count null terminator
		lib_len-- ;
		script_len-- ;

		bool res = UDF_Load (script, script_len, lib, lib_len, false, NULL) ;
		ASSERT (res == true) ;

		RedisModule_Free ((void*)lib) ;
		RedisModule_Free ((void*)script) ;
	}
}

