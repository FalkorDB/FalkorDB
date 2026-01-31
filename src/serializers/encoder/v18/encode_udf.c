/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../../redismodule.h"
#include "../../../udf/repository.h"

// persist all UDF libraries to the AUX field (Redis persistence)
// format (version 18):
//   <unsigned int> number_of_libraries
//   for each library:
//       <string buffer> library_name
//       <string buffer> library_script
//
// notes:
//   - both library name and script must be NUL-terminated strings
//   - this function defines the v18 on-disk format; changes require a new
//     versioned save/load pair to preserve backward compatibility
void AUXSaveUDF_latest
(
	RedisModuleIO *io
) {
	// encode UDFs
	// format:
	// number of UDFs
	// [
	//    library's name
	//    library's script
	// ]

	ASSERT (io != NULL) ;

	// get the number of registered UDF libraries
	unsigned int n = UDF_RepoLibsCount();

	// write the library count
	RedisModule_SaveUnsigned (io, n) ;

	for (unsigned int i = 0; i < n; i++) {
		
		const char *lib ;
		const char *script ;

		UDF_RepoGetLibIdx (i, &lib, NULL, &script) ;

		ASSERT (lib    != NULL) ;
		ASSERT (script != NULL) ;

		// write library name
		RedisModule_SaveStringBuffer (io, lib, strlen (lib) + 1) ;

		// write library script
		RedisModule_SaveStringBuffer (io, script, strlen (script) + 1) ;
	}
}

