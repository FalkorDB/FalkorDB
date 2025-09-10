/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "udf_ctx.h"
#include "functions.h"
#include "repository.h"
#include "../arithmetic/func_desc.h"

extern const char *UDF_LIB ;  // global register library name

// dryrun register UDF
static JSValue validate_register_udf
(
	JSContext *js_ctx,      // JavaScript context
	JSValueConst this_val,  // 'this' value passed by the caller
	int argc,               // number of arguments
	JSValueConst *argv      // function arguments
) {
	ASSERT (argv   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	// user called: register('func_name', function);

	if (argc != 2) {
		// raise a TypeError to the JS caller
		return JS_ThrowTypeError (js_ctx, "register() expects 2 arguments") ;
	}

	if (!JS_IsString (argv[0])) {
		return JS_ThrowTypeError (js_ctx,
				"first argument must be a string (function name)") ;
	}

	if (!JS_IsFunction (js_ctx, argv[1])) {
		return JS_ThrowTypeError (js_ctx,
				"second argument must be a function") ;
	}

	JSValue res;
	const char *func_name = JS_ToCString(js_ctx, argv[0]) ;

	// fail if UDF is a registered function
	if (AR_FuncExists (func_name)) {
		res = JS_ThrowTypeError (js_ctx, "function: '%s' already registered",
				func_name) ;

		goto cleanup ;
	}

	res = JS_NewBool (js_ctx, true) ;

cleanup:
	JS_FreeCString (js_ctx, func_name) ;

	return res ;
}

// local register UDF
static JSValue local_register_udf
(
	JSContext *js_ctx,      // JavaScript context
	JSValueConst this_val,  // 'this' value passed by the caller
	int argc,               // number of arguments
	JSValueConst *argv      // function arguments
) {
	ASSERT (argv   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	// user called: register('func_name', function);

	if (argc != 2) {
		// raise a TypeError to the JS caller
		return JS_ThrowTypeError (js_ctx, "register() expects 2 arguments") ;
	}

	if (!JS_IsString (argv[0])) {
		return JS_ThrowTypeError (js_ctx,
				"first argument must be a string (function name)") ;
	}

	const char *func_name = JS_ToCString(js_ctx, argv[0]) ;

	JSValueConst func = argv[1] ;
	if (!JS_IsFunction (js_ctx, func)) {
		return JS_ThrowTypeError (js_ctx,
				"second argument must be a function") ;
	}

	// register function with TLS UDF context
	UDFCtx_RegisterFunction (JS_DupValue (js_ctx, func), func_name) ;

cleanup:
	JS_FreeCString (js_ctx, func_name) ;

	return JS_NewBool (js_ctx, true) ;
}

// register UDF
static JSValue global_register_udf
(
	JSContext *js_ctx,      // JavaScript context
	JSValueConst this_val,  // 'this' value passed by the caller
	int argc,               // number of arguments
	JSValueConst *argv      // function arguments
) {
	ASSERT (argv   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	// user called: register('func_name', function);

	if (argc != 2) {
		// raise a TypeError to the JS caller
		return JS_ThrowTypeError (js_ctx, "register() expects 2 arguments") ;
	}

	if (!JS_IsString (argv[0])) {
		return JS_ThrowTypeError (js_ctx,
				"first argument must be a string (function name)") ;
	}

	if (!JS_IsFunction (js_ctx, argv[1])) {
		return JS_ThrowTypeError (js_ctx,
				"second argument must be a function") ;
	}

	JSValue res;
	const char *func_name = JS_ToCString(js_ctx, argv[0]) ;

	if (!UDF_RepoRegisterFunc (UDF_LIB, func_name)) {
		res = JS_ThrowTypeError (js_ctx, "function: '%s' already registered",
				func_name) ;

		goto cleanup ;
	}

	res = JS_NewBool (js_ctx, true) ;

cleanup:
	JS_FreeCString (js_ctx, func_name) ;

	return res ;
}

// register proxy functions
void UDF_RegisterFunctions
(
	JSContext *js_ctx,              // javascript context
	UDF_JSCtxRegisterFuncMode mode  // type of 'register' function
) {
	ASSERT (js_ctx != NULL) ;

	JSValue f;

	switch (mode) {
		case UDF_FUNC_REG_MODE_VALIDATE:
			// the 'register' funcion only perform validations
			// nothing gets registered
			f = JS_NewCFunction (js_ctx, validate_register_udf, "register", 0) ;
			break ;

		case UDF_FUNC_REG_MODE_LOCAL:
			f = JS_NewCFunction (js_ctx, local_register_udf,    "register", 0) ;
			break ;

		case UDF_FUNC_REG_MODE_GLOBAL:
			// the 'register' function adds UDF function to the UDF repository
			f = JS_NewCFunction (js_ctx, global_register_udf,   "register", 0) ;
			break;

		default:
			assert("unknown UDF_JSCtxRegisterFuncMode mode" && false) ;
			break;
	}
	ASSERT (!JS_IsException (f)) ;

	// expose it in the global object
	JSValue global = JS_GetGlobalObject (js_ctx) ;

	JS_SetPropertyStr (js_ctx, global, "register", f) ;

	JS_FreeValue (js_ctx, global) ;
}

