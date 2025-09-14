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

// global UDF library name used when registering functions globally
extern const char *UDF_LIB ;

//------------------------------------------------------------------------------
// falkor.register implementations
//------------------------------------------------------------------------------

// validation-only implementation of `falkor.register`
// ensures the function signature is correct and not already defined
// but does not persist the function
// JS call: falkor.register("func_name", function);
static JSValue validate_register_udf
(
	JSContext *js_ctx,      // JavaScript context
	JSValueConst this_val,  // 'this' value passed by the caller
	int argc,               // number of arguments
	JSValueConst *argv      // function arguments
) {
	ASSERT (argv   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	if (argc != 2) {
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

// thread-local implementation of `falkor.register`
// functions are stored in the TLS UDF context
// JS call: falkor.register("func_name", function);
static JSValue local_register_udf
(
	JSContext *js_ctx,      // JavaScript context
	JSValueConst this_val,  // 'this' value passed by the caller
	int argc,               // number of arguments
	JSValueConst *argv      // function arguments
) {
	ASSERT (argv   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	if (argc != 2) {
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

	// register function in TLS UDF context
	UDFCtx_RegisterFunction (JS_DupValue (js_ctx, func), func_name) ;

cleanup:
	JS_FreeCString (js_ctx, func_name) ;

	return JS_NewBool (js_ctx, true) ;
}

// global implementation of `falkor.register`
// functions are persisted in the global repository
// JS call: falkor.register("func_name", function);
static JSValue global_register_udf
(
	JSContext *js_ctx,      // JavaScript context
	JSValueConst this_val,  // 'this' value passed by the caller
	int argc,               // number of arguments
	JSValueConst *argv      // function arguments
) {
	ASSERT (argv   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	if (argc != 2) {
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

//------------------------------------------------------------------------------
// falkor object setup
//------------------------------------------------------------------------------

// register the global falkor object in the given QuickJS context
void UDF_RegisterFalkorObject
(
	JSContext *js_ctx  // the QuickJS context in which to register the object
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

// set the implementation of the `falkor.register` function
void UDF_SetFalkorRegisterImpl
(
	JSContext *js_ctx,              // the QuickJS context
	UDF_JSCtxRegisterFuncMode mode  // the registration mode
) {
	ASSERT(js_ctx != NULL);

    // get global.falkor
    JSValue global_obj = JS_GetGlobalObject (js_ctx) ;
    JSValue falkor_obj = JS_GetPropertyStr  (js_ctx, global_obj, "falkor") ;

	ASSERT (JS_IsObject (falkor_obj));

	// pick implementation
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

