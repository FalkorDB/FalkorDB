/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "../../util/arr.h"
#include "../../udf/utils.h"
#include "../../udf/repository.h"
#include "../../arithmetic/func_desc.h"

// GRAPH.UDF DELETE <library>
// GRAPH.UDF DELETE utils
int Graph_UDF_Delete
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command args
	int argc,                  // number of arguments
	bool *success              // rather or not operation succeeded
) {
	ASSERT (ctx     != NULL) ;
	ASSERT (argv    != NULL) ;
	ASSERT (success != NULL) ;

	*success = false ; // default to false

	// expecting a single argument
	if (argc != 1) {
		RedisModule_WrongArity (ctx) ;
		return REDISMODULE_OK ;
	}

	const char *lib = RedisModule_StringPtrLen (argv[0], NULL) ;

	char *err = NULL;
	if (UDF_Delete (lib, NULL, &err)) {
		*success = true ;
		RedisModule_ReplyWithSimpleString (ctx, "OK") ;
	} else {
		ASSERT (err != NULL) ;
		RedisModule_ReplyWithError (ctx, err) ;
		free (err) ;
	}

	return REDISMODULE_OK ;
}

