/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "../../util/arr.h"
#include "../../udf/repository.h"

// reply with a library definition in RESP3 format
// the reply is a map containing:
//   "library_name" -> <name>
//   "functions"    -> [ <function1>, <function2>, ... ]
//   "library_code" -> <script>   (only if script is not NULL)
//
// notes:
//   - all input strings must remain valid for the duration of the call
//   - the caller owns memory management; this function does not copy/free
static void _emit_lib
(
	RedisModuleCtx *ctx,     // [in] Redis module context
	const char *name,        // [in] library name (non-NULL)
	const char **functions,  // [in] array of library function names
	const char *script       // [in] library script, optional (may be NULL)
) {

	ASSERT (ctx       != NULL) ;
	ASSERT (name      != NULL) ;
	ASSERT (functions != NULL) ;

	// number of entries in the map reply
	int n_fields = 2 ;  // name + functions
	if (script != NULL) n_fields++ ;

	RedisModule_ReplyWithMap (ctx, n_fields) ;

	// library_name
	RedisModule_ReplyWithSimpleString (ctx, "library_name") ;
	RedisModule_ReplyWithSimpleString (ctx, name) ;

	// functions
	int func_count = array_len (functions) ;
	RedisModule_ReplyWithSimpleString (ctx, "functions") ;
	RedisModule_ReplyWithArray (ctx, func_count) ;
	for (int i = 0; i < func_count; i++) {
		RedisModule_ReplyWithSimpleString (ctx, functions[i]) ;
	}

	// library_code (optional)
	if (script != NULL) {
		RedisModule_ReplyWithSimpleString (ctx, "library_code") ;
		RedisModule_ReplyWithStringBuffer (ctx, script, strlen (script)) ;
	}
}

// GRAPH.UDF LIST [LIBRARYNAME] [WITHCODE]
//
// examples:
//   GRAPH.UDF LIST
//   GRAPH.UDF LIST utils
//   GRAPH.UDF LIST utils WITHCODE
//
// response (RESP3 map per library):
//   1) "library_name" -> <lib name>
//   2) "functions"    -> [ <func1>, <func2>, ... ]
//   3) "library_code" -> <script>   (only if WITHCODE is specified)
//
// notes:
//   - `functions` and `script` pointers returned by repo functions must remain
//     valid until the reply is constructed.
//   - This function does not free or copy any library data.
//
int Graph_UDF_List 
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command args
	int argc                   // number of arguments
) {
	ASSERT (ctx  != NULL) ;
	ASSERT (argv != NULL) ;

	// expect at most two arguments: [LIBNAME] [WITHCODE]
	if (argc > 2) {
		RedisModule_WrongArity (ctx) ;
		return REDISMODULE_OK ;
	}

	const char *lib = NULL  ;  // requested library name, or NULL for all
	bool with_code  = false ;

	if (argc == 1) {
		// either lib name or "withcode"
		const char *arg = RedisModule_StringPtrLen (argv[0], NULL) ;

		if (strcasecmp (arg, "withcode") == 0) {
			with_code = true ;
		} else {
			lib = arg ;
		}
	}

	else if (argc == 2) {
		// first arg is a library name
		// second arg should be "withcode"

		lib = RedisModule_StringPtrLen (argv[0], NULL) ;

		const char *arg = RedisModule_StringPtrLen (argv[1], NULL) ;
		if (strcasecmp (arg, "withcode") != 0) {
			RedisModule_ReplyWithErrorFormat (ctx, "Unknown option given: %s",
					arg) ;
			return REDISMODULE_OK ;
		} 

		with_code = true ;
	}

	// prepare optional outputs
	const char *script     = NULL ;
	const char **_script   = (with_code) ? &script : NULL ;
	const char **functions = NULL ;

	// user specified lib
	if (lib != NULL) {
		if (UDF_RepoGetLib (lib, &functions, _script)) {
			RedisModule_ReplyWithArray (ctx, 1) ;
			_emit_lib (ctx, lib, functions, script) ;
		} else {
			// unknown lib
			RedisModule_ReplyWithArray (ctx, 0) ;
		}
	}

	// all libs
	else {
		unsigned int n = UDF_RepoLibsCount() ;
		RedisModule_ReplyWithArray (ctx, n) ;
		for (unsigned int i = 0; i < n; i++) {
			UDF_RepoGetLibIdx (i, &lib, &functions, _script);
			_emit_lib (ctx, lib, functions, script) ;
		}
	}

	return REDISMODULE_OK ;
}

