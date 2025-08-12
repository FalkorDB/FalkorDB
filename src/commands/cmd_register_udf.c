/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "quickjs.h"
#include "../udf/utils.h"
#include "../udf/udf_ctx.h"
#include "../udf/repository.h"

// GRAPH.UDF <FUNC_NAME> <FUNCTION>
// GRAPH.UDF "greet" "function greet(name) { console.log ('Hello ' + name); }"
int Graph_RegisterUDF
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT (ctx  != NULL) ;
	ASSERT (argv != NULL) ;

	if (argc < 3) {
		RedisModule_WrongArity (ctx) ;
		return REDISMODULE_OK ;
	}

	// get function name and script from arguments
	size_t script_len ;
	const char *func_name = RedisModule_StringPtrLen (argv[1], NULL) ;
	const char *script    = RedisModule_StringPtrLen (argv[2], &script_len) ;

	// make sure UDF isn't already registered
	if (UDF_RepoGetScript (func_name) != NULL) {
		RedisModule_ReplyWithErrorFormat (ctx,
				"Failed to register '%s', function already registered",
				func_name) ;

		return REDISMODULE_OK ;
	}

	//--------------------------------------------------------------------------
	// evaluate script
	//--------------------------------------------------------------------------

	JSContext *js_ctx = UDFCtx_GetJSContext () ;

    JSValue val = JS_Eval (js_ctx, script, strlen (script), "<input>",
			JS_EVAL_TYPE_GLOBAL) ;

    // handle exceptions
    if (JS_IsException (val)) {
        JSValue exc = JS_GetException(js_ctx);
        const char *msg = JS_ToCString(js_ctx, exc);
        JS_FreeCString(js_ctx, msg);
        JS_FreeValue(js_ctx, exc);

		RedisModule_ReplyWithErrorFormat (ctx,
				"Failed to evaluate UDF '%s', Exception: %s",
				func_name, msg) ;

		return REDISMODULE_OK ;
    }

    JS_FreeValue(js_ctx, val);

	// make sure script contains function
	if (!UDF_ContainsFunction (func_name, js_ctx)) {
		// TODO: note script is alreay in context
		//       either use a dedicated context for first load
		//       or find a way to remove script from context
		RedisModule_ReplyWithErrorFormat (ctx,
				"Script doesn't seems to contain function '%s'", func_name) ;
		return REDISMODULE_OK ;
	}

	//--------------------------------------------------------------------------
	// register UDF
	//--------------------------------------------------------------------------

	// try to register UDF
	if (!UDF_RepoSetScript (func_name, script)) {
		RedisModule_ReplyWithErrorFormat (ctx,
				"Failed to register '%s', function might be already registered",
				func_name) ;
	}

	RedisModule_ReplyWithSimpleString (ctx, "OK") ;

	return REDISMODULE_OK ;
}

