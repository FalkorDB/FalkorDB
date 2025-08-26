/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "quickjs.h"
#include "../udf/utils.h"
#include "../udf/udf_ctx.h"
#include "../udf/repository.h"

// GRAPH.UDF LOAD <script>
// GRAPH.UDF LOAD "function greet(name) { console.log ('Hello ' + name); }"
int Graph_UDF_Load
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT (ctx  != NULL) ;
	ASSERT (argv != NULL) ;

	if (argc != 1) {
		RedisModule_WrongArity (ctx) ;
		return REDISMODULE_OK ;
	}

	// get script from arguments
	size_t script_len ;
	const char *script = RedisModule_StringPtrLen (argv[0], &script_len) ;

	// check for empty script
	// GRAPH.UDF LOAD ""
	if (script_len == 0) {
		// empty string
		RedisModule_ReplyWithError (ctx, "empty script") ;
		return REDISMODULE_OK ;
	}

	// validate script wasn't already loaded
	// compute script SHA

	if (UDF_RepoGetScript (func_name) != NULL) {
		RedisModule_ReplyWithErrorFormat (ctx,
				"Failed to register '%s', function already registered",
				func_name) ;

		return REDISMODULE_OK ;
	}

	// load script into a dedicated JavaScript context
	// validate:
	// 1. script loads
	// 2. functions do not already exists
	//
	// if scripts passes validations add script to repository
	// and add it to the thread JavaScript context

	//--------------------------------------------------------------------------
	// create dedicated js runtime
	//--------------------------------------------------------------------------

	JSRuntime *js_rt  = NULL ;
	JSContext *js_ctx = NULL ;
	JSValue global    = JS_NULL ;
    JSValue val       = JS_NULL ;

	js_rt = JS_NewRuntime() ;
	ASSERT (js_rt != NULL) ;

	JS_SetMaxStackSize (js_rt, 1024 * 1024) ; // 1 MB stack limit

	// create js context
	js_ctx = JS_NewContext(js_rt) ;
	ASSERT (js_ctx != NULL) ;

	// evalute script
    val = JS_Eval (js_ctx, script, script_len, "<input>", JS_EVAL_TYPE_GLOBAL) ;

    // report exception
    if (JS_IsException (val)) {
        JSValue exc = JS_GetException (js_ctx) ;
        const char *msg = JS_ToCString (js_ctx, exc) ;

		RedisModule_ReplyWithErrorFormat (ctx,
				"Failed to evaluate UDF script, Exception: %s", msg) ;

        JS_FreeCString (js_ctx, msg) ;
        JS_FreeValue   (js_ctx, exc) ;

		goto cleanup ;
    }

	//--------------------------------------------------------------------------
	// list each top level function(s)
	//--------------------------------------------------------------------------

	global = JS_GetGlobalObject (js_ctx) ;

	// get all property keys (own only, enumerable or not)
	uint32_t len ;
	JSPropertyEnum *props ;

	int res = JS_GetOwnPropertyNames (js_ctx, &props, &len, global,
			JS_GPN_STRING_MASK | JS_GPN_SYMBOL_MASK) ;
	ASSERT (res == 0) ;

	// true if UDF is conflicting with an already loaded function
	bool func_conflict = false ;  

	for (uint32_t i = 0; i < len && !func_conflict; i++) {
		JSAtom atom = props[i].atom ;

		JSValue val = JS_GetProperty (js_ctx, global, atom) ;

		// is this a user defined function ?
		if (!JS_IsFunction (js_ctx, val) && UDF_IsUserFunction (js_ctx, val)) {
			JS_FreeValue (js_ctx, val) ;
			continue ;
		}

		// function name
		const char *func_name = JS_AtomToCString (js_ctx, atom) ;

		// skip anonymous function
		if (!func_name) {
			JS_FreeValue (js_ctx, val) ;
			continue ;
		}

		printf ("Function: %s \n", func_name) ;

		// make sure UDF isn't already registered
		if (AR_GetFunc (func_name,  false)) {
			RedisModule_ReplyWithErrorFormat (ctx,
					"can't load UDF, function: %s already exists", func_name) ;
			func_conflict = true ;
		}

		JS_FreeCString (js_ctx, func_name) ;
		JS_FreeValue (js_ctx, val) ;
	}

	if (func_conflict) {
		goto cleanup ;	
	}

	// UDF passed validations
	// register script & each function

	RedisModule_ReplyWithSimpleString (ctx, "OK") ;

cleanup:

	if (!JS_IsNull (global)) {
		JS_FreeValue (js_ctx, global) ;
	}

	if (js_ctx != NULL) {
		js_free (js_ctx, props) ;
	}

	return REDISMODULE_OK ;

	//--------------------------------------------------------------------------
	// evaluate script
	//--------------------------------------------------------------------------

//	JSContext *js_ctx = UDFCtx_GetJSContext () ;
//
//    JSValue val = JS_Eval (js_ctx, script, strlen (script), "<input>",
//			JS_EVAL_TYPE_GLOBAL) ;
//
//    // handle exceptions
//    if (JS_IsException (val)) {
//        JSValue exc = JS_GetException(js_ctx);
//        const char *msg = JS_ToCString(js_ctx, exc);
//        JS_FreeCString(js_ctx, msg);
//        JS_FreeValue(js_ctx, exc);
//
//		RedisModule_ReplyWithErrorFormat (ctx,
//				"Failed to evaluate UDF '%s', Exception: %s",
//				func_name, msg) ;
//
//		return REDISMODULE_OK ;
//    }
//
//    JS_FreeValue(js_ctx, val);
//
//	// make sure script contains function
//	if (!UDF_ContainsFunction (func_name, js_ctx)) {
//		// TODO: note script is alreay in context
//		//       either use a dedicated context for first load
//		//       or find a way to remove script from context
//		RedisModule_ReplyWithErrorFormat (ctx,
//				"Script doesn't seems to contain function '%s'", func_name) ;
//		return REDISMODULE_OK ;
//	}
//
//	//--------------------------------------------------------------------------
//	// register UDF
//	//--------------------------------------------------------------------------
//
//	// try to register UDF
//	if (!UDF_RepoSetScript (func_name, script)) {
//		RedisModule_ReplyWithErrorFormat (ctx,
//				"Failed to register '%s', function might be already registered",
//				func_name) ;
//	}
//
//	RedisModule_ReplyWithSimpleString (ctx, "OK") ;
//
//	return REDISMODULE_OK ;
}

// GRAPH.UDF * command handler
// sub commands:
// GRAPH.UDF LOAD <script>
// ...
// todo: introduce additional sub commands
int Graph_UDF
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT (ctx  != NULL) ;
	ASSERT (argv != NULL) ;

	if (argc < 2) {
		RedisModule_WrongArity (ctx) ;
		return REDISMODULE_OK ;
	}

	RedisModuleString *rm_sub_cmd = argv[1] ;
	const char *sub_cmd = RedisModule_StringPtrLen (rm_sub_cmd, NULL) ;

	if (strcasecmp (sub_cmd, "load")) {
		return Graph_UDF_Load (ctx, argv+2, argc-2) ;
	} else {
		RedisModule_ReplyWithErrorFormat (ctx,
				"Unknown GRAPH.UDF sub command %s", sub_cmd) ;
	}

	return REDISMODULE_OK ;
}

