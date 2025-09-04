/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "quickjs.h"
#include "../util/arr.h"
#include "../udf/utils.h"
#include "../udf/udf_ctx.h"
#include "../udf/repository.h"
#include "../arithmetic/func_desc.h"
#include "../arithmetic/udf_funcs/udf_funcs.h"

#include <openssl/sha.h>

static int _UDF_Unload
(
	RedisModuleCtx *ctx,                     // redis module context
	unsigned char digest[SHA_DIGEST_LENGTH]  // script sha1 digest
) {
	ASSERT (ctx != NULL) ;

	char *script = NULL ;
	if (!UDF_RepoRemoveScript (digest, &script)) {
		char sha1[SHA_DIGEST_LENGTH*2+1] ;
		sha1_to_hex (digest, sha1) ;

		RedisModule_ReplyWithErrorFormat (ctx,
				"Script with hash %s doesn't exists", sha1) ;

		return 0 ;
	}
	size_t script_len = strlen (script) ;

	// unregister script's functions
	JSRuntime      *js_rt  = NULL ;
	JSContext      *js_ctx = NULL ;
	JSPropertyEnum *props  = NULL ;

	JSValue val    = JS_NULL ;
	JSValue global = JS_NULL ;

	js_rt = JS_NewRuntime() ;
	ASSERT (js_rt != NULL) ;

	JS_SetMaxStackSize (js_rt, 1024 * 1024) ; // 1 MB stack limit

	// create js context
	js_ctx = JS_NewContext(js_rt) ;
	ASSERT (js_ctx != NULL) ;

	// evalute script
	val = JS_Eval (js_ctx, script, script_len, "<input>", JS_EVAL_TYPE_GLOBAL) ;
	rm_free (script) ;

    // report exception
    ASSERT (!JS_IsException (val)) ;

	//--------------------------------------------------------------------------
	// unregister each of the script's functions
	//--------------------------------------------------------------------------

	global = JS_GetGlobalObject (js_ctx) ;

	// get all property keys (own only, enumerable or not)
	uint32_t len ;

	int res = JS_GetOwnPropertyNames (js_ctx, &props, &len, global,
			JS_GPN_STRING_MASK | JS_GPN_SYMBOL_MASK) ;
	ASSERT (res == 0) ;

	for (uint32_t i = 0; i < len; i++) {
		JSAtom atom = props[i].atom ;

		JSValue val = JS_GetProperty (js_ctx, global, atom) ;

		// is this a user defined function ?
		if (!JS_IsFunction (js_ctx, val) || !UDF_IsUserFunction (js_ctx, val)) {
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

		JS_FreeCString (js_ctx, func_name) ;
		JS_FreeValue (js_ctx, val) ;
	}

	if (props != NULL) {
		js_free (js_ctx, props) ;
	}

	if (!JS_IsNull (global)) {
		JS_FreeValue (js_ctx, global) ;
	}

	if (js_ctx != NULL) {
		JS_FreeContext (js_ctx) ;
	}

	if (js_rt != NULL) {
		JS_FreeRuntime (js_rt) ;
	}

	return 1 ;
}

// GRAPH.UDF LOAD |REPLACE| <lib> <script>
// GRAPH.UDF LOAD hello_world "function greet(name) { console.log ('Hello ' + name); }"
int Graph_UDF_Load
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command args
	int argc                   // number of arguments
) {
	ASSERT (ctx  != NULL) ;
	ASSERT (argv != NULL) ;

	// expecting two or three arguments: optional REPLACE, lib name and script
	if (argc < 2 || argc > 3) {
		RedisModule_WrongArity (ctx) ;
		return REDISMODULE_OK ;
	}

	// get arguments

	size_t script_len ;
	const char *script = RedisModule_StringPtrLen (argv[argc-1], &script_len) ;

	size_t lib_len ;
	const char *lib = RedisModule_StringPtrLen (argv[argc-2], &lib_len) ;

	bool replace = false ;

	ASSERT (lib    != NULL) ;
	ASSERT (script != NULL) ;

	// check for empty script
	if (script_len == 0) {
		// empty string
		RedisModule_ReplyWithError (ctx, "empty script") ;
		return REDISMODULE_OK ;
	}

	// check for empty lib name
	if (lib_len == 0) {
		// empty string
		RedisModule_ReplyWithError (ctx, "empty lib name") ;
		return REDISMODULE_OK ;
	}

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

	// fail in case script alreay exists and replace is false
	bool script_exists = UDF_RepoContainsScript (lib, NULL) ;
	if (script_exists && replace == false) {
		RedisModule_ReplyWithError (ctx,
				"Failed to register UDF script, already registered") ;

		return REDISMODULE_OK ;
	}

	// load script into a dedicated JavaScript context
	// validate:
	// 1. script loads
	// 2. functions do not already exists (in case replace is false)
	//
	// if scripts passes validations add script to repository
	// and add it to the thread JavaScript context

	//--------------------------------------------------------------------------
	// create dedicated js runtime
	//--------------------------------------------------------------------------

	JSRuntime      *js_rt  = NULL ;
	JSContext      *js_ctx = NULL ;
	JSPropertyEnum *props  = NULL ;

	JSValue val    = JS_NULL ;
	JSValue global = JS_NULL ;

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

	int res = JS_GetOwnPropertyNames (js_ctx, &props, &len, global,
			JS_GPN_STRING_MASK | JS_GPN_SYMBOL_MASK) ;
	ASSERT (res == 0) ;

	// true if UDF is conflicting with an already loaded function
	bool func_conflict = false ;  

	for (uint32_t i = 0; i < len && !func_conflict; i++) {
		JSAtom atom = props[i].atom ;

		JSValue val = JS_GetProperty (js_ctx, global, atom) ;

		// is this a user defined function ?
		if (!JS_IsFunction (js_ctx, val) || !UDF_IsUserFunction (js_ctx, val)) {
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

	if (script_exists && replace) {
		if (!_UDF_Unload (ctx, digest)) {
			RedisModule_ReplyWithError (ctx, "Failed to replace script") ;
			goto cleanup ;
		}
	}

	res = UDF_RepoRegisterScript (script) ;
	ASSERT (res == true) ;

	//--------------------------------------------------------------------------
	// register functions
	//--------------------------------------------------------------------------

	for (uint32_t i = 0; i < len; i++) {
		JSAtom atom = props[i].atom ;

		JSValue val = JS_GetProperty (js_ctx, global, atom) ;

		// is this a user defined function ?
		if (!JS_IsFunction (js_ctx, val) || !UDF_IsUserFunction (js_ctx, val)) {
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

		SIType *types = array_new (SIType, 1) ;
		array_append (types, SI_ALL) ;

		AR_FuncDesc *func_desc = AR_FuncDescNew (func_name, AR_UDF, 0,
				VAR_ARG_LEN, types, SI_ALL, false, false) ;

		AR_SetUDF  (func_desc) ;  // mark function as UDF
		AR_RegFunc (func_desc) ;  // register function

		JS_FreeCString (js_ctx, func_name) ;
		JS_FreeValue (js_ctx, val) ;
	}

	// reply with sha1
	char sha1[SHA_DIGEST_LENGTH*2+1] ;
	sha1_to_hex (digest, sha1) ;
	RedisModule_ReplyWithSimpleString (ctx, sha1) ;

cleanup:

	if (props != NULL) {
		js_free (js_ctx, props) ;
	}

	if (!JS_IsNull (global)) {
		JS_FreeValue (js_ctx, global) ;
	}

	if (js_ctx != NULL) {
		JS_FreeContext (js_ctx) ;
	}

	if (js_rt != NULL) {
		JS_FreeRuntime (js_rt) ;
	}

	return REDISMODULE_OK ;
}

// GRAPH.UDF UNLOAD <sha1>
// GRAPH.UDF UNLOAD fe8dae7c4c140c76d8532c947ecbd779cbade835
int Graph_UDF_Unload
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command args
	int argc                   // number of arguments
) {
	ASSERT (ctx  != NULL) ;
	ASSERT (argv != NULL) ;

	// expecting a single argument
	if (argc != 1) {
		RedisModule_WrongArity (ctx) ;
		return REDISMODULE_OK ;
	}

	size_t sha1_len;
	const char *sha1 = RedisModule_StringPtrLen (argv[0], &sha1_len) ;

	// transition from hex string represention to sha1 digest
	unsigned char digest[SHA_DIGEST_LENGTH] ;
	if (hex_to_sha1 (sha1, digest) != 0) {
		RedisModule_ReplyWithErrorFormat (ctx, "invalid sha1 hash %s", sha1) ;
		return REDISMODULE_OK ;
	}

	if (_UDF_Unload(ctx, digest)) {
		RedisModule_ReplyWithSimpleString (ctx, "OK") ;
	}

	return REDISMODULE_OK ;
}

// GRAPH.UDF * command handler
// sub commands:
// GRAPH.UDF LOAD |REPLACE| <lib> <script>
// GRAPH.UDF UNLOAD <lib>
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

	if (strcasecmp (sub_cmd, "load") == 0) {
		return Graph_UDF_Load (ctx, argv+2, argc-2) ;
	}

	else if (strcasecmp (sub_cmd, "unload") == 0) {
		return Graph_UDF_Unload (ctx, argv+2, argc-2) ;
	}

	else {
		RedisModule_ReplyWithErrorFormat (ctx,
				"Unknown GRAPH.UDF sub command %s", sub_cmd) ;
	}

	return REDISMODULE_OK ;
}

