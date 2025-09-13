/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "udf_ctx.h"
#include "classes.h"
#include "repository.h"
#include "../query_ctx.h"
#include "../arithmetic/func_desc.h"

extern const char *UDF_LIB ;  // global register library name

// register the falkor object with the js-context
void ctx_register_falkor_object
(
	JSContext *js_ctx
) {
	ASSERT (js_ctx != NULL) ;

    // create a plain namespace object: const falkor = {};
    JSValue falkor_obj = JS_NewObject (js_ctx) ;

    // expose the namespace globally as "falkor"
    JSValue global_obj = JS_GetGlobalObject (js_ctx) ;
    JS_SetPropertyStr (js_ctx, global_obj, "falkor", falkor_obj) ;

    // free what we got from GetGlobalObject
    JS_FreeValue (js_ctx, global_obj) ;
}

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

	// user called: falkor.register('func_name', function);

	if (argc != 2) {
		// raise a TypeError to the JS caller
		return JS_ThrowTypeError (js_ctx, "falkor.register() expects 2 arguments") ;
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

	// user called: falkor.register('func_name', function);

	if (argc != 2) {
		// raise a TypeError to the JS caller
		return JS_ThrowTypeError (js_ctx, "falkor.register() expects 2 arguments") ;
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

	// user called: falkor.register('func_name', function);

	if (argc != 2) {
		// raise a TypeError to the JS caller
		return JS_ThrowTypeError (js_ctx, "falkor.register() expects 2 arguments") ;
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

// set falkor.register function
void falkor_set_register_impl
(
	JSContext *js_ctx,              // javascript context
	UDF_JSCtxRegisterFuncMode mode  // type of 'register' function
) {
	ASSERT(js_ctx != NULL);

    // get global.falkor
    JSValue global_obj = JS_GetGlobalObject (js_ctx) ;
    JSValue falkor_obj = JS_GetPropertyStr  (js_ctx, global_obj, "falkor") ;

	ASSERT (JS_IsObject (falkor_obj));

    // create the C function object depending on mode
    JSValue func_obj = JS_UNDEFINED;
    switch (mode) {
        case UDF_FUNC_REG_MODE_VALIDATE:
            func_obj = JS_NewCFunction (js_ctx, validate_register_udf, "register", 1) ;
            break ;

        case UDF_FUNC_REG_MODE_LOCAL:
            func_obj = JS_NewCFunction (js_ctx, local_register_udf, "register", 1) ;
            break ;

        case UDF_FUNC_REG_MODE_GLOBAL:
            func_obj = JS_NewCFunction (js_ctx, global_register_udf, "register", 1) ;
            break ;

        default:
            assert (false && "unknown mode") ;
    }

	ASSERT (JS_IsFunction (js_ctx, func_obj)) ;

    // define property with explicit flags (writable, configurable)
    int def_res = JS_DefinePropertyValueStr (js_ctx, falkor_obj, "register",
			func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;

	ASSERT (def_res >= 0) ;

    // clean up
    JS_FreeValue (js_ctx, global_obj) ;
    JS_FreeValue (js_ctx, falkor_obj) ;
}

