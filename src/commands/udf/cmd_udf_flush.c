/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "../../udf/utils.h"
#include "../../udf/repository.h"

// remove all registered UDF libraries from the repository
// deletes in reverse order (last → first) to avoid index shifting
// this is an internal helper; errors will trigger ASSERT failures
static void _UDF_Flush (void) {
	// get the number of UDF libraries
	int n = UDF_RepoLibsCount () ;

	for (int i = n-1; i >= 0; i--) {
		const char *lib = NULL ;
		UDF_RepoGetLibIdx (i, &lib, NULL, NULL) ;
		ASSERT (lib != NULL) ;

		char *err = NULL ;
		bool removed = UDF_Delete (lib, NULL, &err) ;
		ASSERT (err     == NULL) ;
		ASSERT (removed == true) ;
	}
}

// syntax:
//   GRAPH.UDF FLUSH
//
// behavior:
//   - removes all UDF libraries from the repository
//   - replies with "OK" on success
//
// returns:
//   REDISMODULE_OK (always)
int Graph_UDF_Flush
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command arguments (must be empty)
	int argc                   // number of arguments
) {
	ASSERT (ctx  != NULL) ;
	ASSERT (argv != NULL) ;

	// GRAPH.UDF FLUSH takes no arguments
	if (argc != 0) {
		RedisModule_WrongArity (ctx) ;
		return REDISMODULE_OK ;
	}

	_UDF_Flush() ;

	RedisModule_ReplyWithSimpleString (ctx, "OK") ;
	return REDISMODULE_OK ;
}

