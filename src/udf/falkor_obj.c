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

	//--------------------------------------------------------------------------
	// fail if UDF is a registered function
	//--------------------------------------------------------------------------

	// check both UDF repository & general functions repo
	if (UDF_RepoContainsFunc (UDF_LIB, func_name)) {
		res = JS_ThrowTypeError (js_ctx, "function: '%s.%s' already registered",
				UDF_LIB, func_name) ;

		goto cleanup ;
	}

	char *fullname = NULL ;
	asprintf (&fullname, "%s.%s", UDF_LIB, func_name) ;
	if (AR_FuncExists (fullname)) {
		res = JS_ThrowTypeError (js_ctx, "function: '%s' already registered",
				fullname) ;

		goto cleanup ;
	}

	res = JS_NewBool (js_ctx, true) ;

cleanup:
	if (fullname != NULL) {
		free (fullname) ;
	}
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

	JSValue res ;
	const char *func_name = JS_ToCString (js_ctx, argv[0]) ;

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

// similar to console.log
// prints argument to stdout
static JSValue falkor_log
(
	JSContext *ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
    for (int i = 0 ; i < argc ; i++) {
        const char *str ;
        JSValue val = argv[i] ;

        // check if the value is an object (and not null) to use JSON stringify
        if (JS_IsObject (val) && !JS_IsNull (val) && !JS_IsFunction (ctx, val)) {
            // JS_JSONStringify(ctx, value, replacer, space0)
            // using a space value of 2 for "pretty-printing"
            JSValue json_str_val =
				JS_JSONStringify(ctx, val, JS_UNDEFINED, JS_NewInt32(ctx, 2));

            if (JS_IsException (json_str_val)) {
                // this usually happens with circular references
                str = "[Circular or Un-stringifiable Object]" ;
            } else {
                str = JS_ToCString (ctx, json_str_val) ;
                JS_FreeValue (ctx, json_str_val) ;
            }
        } else {
            // For strings, numbers, booleans, null, and undefined
            str = JS_ToCString (ctx, val) ;
        }

        if (!str) {
            return JS_EXCEPTION ;
        }

        printf ("%s%s", str, (i < argc - 1) ? " " : "") ;

        // only free the string if it was successfully allocated by QuickJS
        // (avoid freeing the static fallback string used in the exception check)
        if (str[0] != '[' || str[1] != 'C') {
            JS_FreeCString (ctx, str) ;
        }
    }

    printf ("\n") ;

	// ensure the user sees the word immediately
    fflush (stdout) ;

    return JS_UNDEFINED ;
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

	// register falkor.log
	JSValue func_obj = JS_NewCFunction (js_ctx, falkor_log, "log", 1) ;

    int def_res = JS_DefinePropertyValueStr (js_ctx, falkor_obj, "log",
			func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;
	ASSERT (def_res >= 0) ;

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

