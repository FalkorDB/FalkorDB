/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "quickjs.h"
#include "../../util/arr.h"
#include "../../udf/utils.h"
#include "../../udf/udf_ctx.h"
#include "../../udf/repository.h"
#include "../../arithmetic/func_desc.h"
#include "../../arithmetic/udf_funcs/udf_funcs.h"

// GRAPH.UDF LOAD [REPLACE] <LIBNAME> <SCRIPT>
//
// examples:
//   GRAPH.UDF LOAD mylib "function add(a, b) { return a + b; }"
//   GRAPH.UDF LOAD REPLACE mylib "function add(a, b) { return a + b; }"
//
// behavior:
//   - loads a new library named <LIBNAME> with the given JavaScript code
//   - if REPLACE is specified, an existing library is overwritten
//   - returns "OK" on success, or an error reply otherwise
int Graph_UDF_Load
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command args
	int argc                   // number of arguments
) {
	ASSERT (ctx  != NULL) ;
	ASSERT (argv != NULL) ;

	// expect 2 or 3 arguments: [REPLACE] <LIBNAME> <SCRIPT>
	if (argc < 2 || argc > 3) {
		RedisModule_WrongArity (ctx) ;
		return REDISMODULE_OK ;
	}

	//--------------------------------------------------------------------------
	// extract arguments
	//--------------------------------------------------------------------------
	
	size_t script_len ;
	const char *script = RedisModule_StringPtrLen (argv[argc-1], &script_len) ;

	size_t lib_len ;
	const char *lib = RedisModule_StringPtrLen (argv[argc-2], &lib_len) ;

	bool replace = false ;

	ASSERT (lib    != NULL) ;
	ASSERT (script != NULL) ;

	if (argc == 3) {
		const char *replace_str = RedisModule_StringPtrLen (argv[0], NULL) ;
		if (strcasecmp (replace_str, "replace") == 0) {
			replace = true ;
		} else {
			// unknown argument
			RedisModule_ReplyWithErrorFormat (ctx, "Unknown option given: %s",
					replace_str) ;
			return REDISMODULE_OK ;
		}
	}

	//--------------------------------------------------------------------------
	// trivial validations
	//--------------------------------------------------------------------------

	if (lib_len == 0) {
		RedisModule_ReplyWithError (ctx, "empty lib name") ;
		return REDISMODULE_OK ;
	}

	if (script_len == 0) {
		RedisModule_ReplyWithError (ctx, "empty script") ;
		return REDISMODULE_OK ;
	}

	//--------------------------------------------------------------------------
	// load library
	//--------------------------------------------------------------------------

	char *err = NULL ;
	if (!UDF_Load (script, script_len, lib, lib_len, replace, &err)) {
		ASSERT (err != NULL) ;

		RedisModule_ReplyWithError (ctx, err) ;
		free (err) ;

		return REDISMODULE_OK ;
	}

	RedisModule_ReplyWithSimpleString (ctx, "OK") ;

	return REDISMODULE_OK ;
}

