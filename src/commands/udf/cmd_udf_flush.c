/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "../../udf/utils.h"
#include "../../udf/repository.h"

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
	int argc,                  // number of arguments
	bool *success              // rather or not operation succeeded
) {
	ASSERT (ctx     != NULL) ;
	ASSERT (argv    != NULL) ;
	ASSERT (success != NULL) ;

	*success = false ;  // default to false

	// GRAPH.UDF FLUSH takes no arguments
	if (argc != 0) {
		RedisModule_WrongArity (ctx) ;
		return REDISMODULE_OK ;
	}

	UDF_Flush() ;

	*success = true ;
	RedisModule_ReplyWithSimpleString (ctx, "OK") ;
	return REDISMODULE_OK ;
}

