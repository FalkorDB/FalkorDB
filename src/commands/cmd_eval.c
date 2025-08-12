/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "quickjs.h"
#include "../value.h"
#include "../datatypes/datatypes.h"

#include <stdio.h>
#include <stdlib.h>

// forward declarations
static void ReplyVal (RedisModuleCtx *rm_ctx, JSContext *js_ctx, JSValue val) ;

#include "quickjs.h"
#include <stdio.h>
#include <string.h>

static void ReplyException
(
	RedisModuleCtx *rm_ctx,
    JSContext *js_ctx,
	JSValue val
) {
	ASSERT (JS_IsException (val)) ;

	// get the actual exception object
	JSValue exception = JS_GetException(js_ctx);

	// if it's an Error object, you can get its .message
	if (JS_IsError (js_ctx, exception)) {
		JSValue message_val = JS_GetPropertyStr (js_ctx, exception, "message") ;

		const char *message_str = JS_ToCString (js_ctx, message_val) ;
		if (message_str) {
			printf("Exception: %s\n", message_str);
			RedisModule_ReplyWithError (rm_ctx, message_str) ;
			JS_FreeCString (js_ctx, message_str) ;
		}
		JS_FreeValue (js_ctx, message_val) ;
	} else {
		// non-Error exceptions, just try to stringify
		const char *exception_str = JS_ToCString (js_ctx, exception) ;
		if (exception_str) {
			printf ("Exception: %s\n", exception_str) ;
			RedisModule_ReplyWithError (rm_ctx, exception_str) ;
			JS_FreeCString (js_ctx, exception_str) ;
		}
	}
	JS_FreeValue (js_ctx, exception) ;
}

static void ReplyArray
(
	RedisModuleCtx *rm_ctx,
    JSContext *js_ctx,
	JSValue arr
) {
	uint32_t len = 0 ;
	JSValue len_val = JS_GetPropertyStr (js_ctx, arr, "length") ;
	JS_ToUint32 (js_ctx, &len, len_val) ;
	JS_FreeValue (js_ctx, len_val) ;

	RedisModule_ReplyWithArray (rm_ctx, len) ;

	for (uint32_t i = 0; i < len; i++) {
		JSValue elem = JS_GetPropertyUint32 (js_ctx, arr, i) ;
		ReplyVal (rm_ctx, js_ctx, elem) ;
		JS_FreeValue (js_ctx, elem) ;
	}
}

static void ReplyObj
(
	RedisModuleCtx *rm_ctx,
    JSContext *js_ctx,
	JSValue obj
) {
    int i ;
    uint32_t len ;
    JSPropertyEnum *props ;

    // get enumerable own properties (keys)
    if (JS_GetOwnPropertyNames (js_ctx, &props, &len, obj,
                               JS_GPN_STRING_MASK | JS_GPN_ENUM_ONLY) < 0) {
        return ; // error
    }

	// TODO: validate map keys are all strings

	RedisModule_ReplyWithMap(rm_ctx, len) ;

    for (i = 0; i < (int)len; i++) {
        const char *key_str = JS_AtomToCString (js_ctx, props[i].atom) ;
        ASSERT (key_str) ;
		RedisModule_ReplyWithCString (rm_ctx, key_str) ;

        // get the property value
        JSValue val = JS_GetProperty (js_ctx, obj, props[i].atom) ;
		ReplyVal (rm_ctx, js_ctx, val) ;

        // free value and key
        JS_FreeValue   (js_ctx, val)     ;
        JS_FreeCString (js_ctx, key_str) ;
    }

    // free the property enum array
    js_free (js_ctx, props) ;
}

static void ReplyVal
(
	RedisModuleCtx *rm_ctx,
    JSContext *js_ctx,
	JSValue val
) {
	// supported value types:
	//
	// JavaScript native types:
	// array
	// map
	// numeric
	// string
	// null
	//
	// FalkorDB types:
	// edge
	// node
	//

	int tag = JS_VALUE_GET_TAG (val) ;

    //JS_TAG_FIRST       = -9, /* first negative tag */
    //JS_TAG_BIG_INT     = -9,
    //JS_TAG_SYMBOL      = -8,
    //JS_TAG_STRING      = -7,
    //JS_TAG_STRING_ROPE = -6,
    //JS_TAG_MODULE      = -3, /* used internally */
    //JS_TAG_FUNCTION_BYTECODE = -2, /* used internally */
    //JS_TAG_OBJECT      = -1,

    //JS_TAG_INT         = 0,
    //JS_TAG_BOOL        = 1,
    //JS_TAG_NULL        = 2,
    //JS_TAG_UNDEFINED   = 3,
    //JS_TAG_UNINITIALIZED = 4,
    //JS_TAG_CATCH_OFFSET = 5,
    //JS_TAG_EXCEPTION   = 6,
    //JS_TAG_SHORT_BIG_INT = 7,
    //JS_TAG_FLOAT64     = 8,
    /* any larger tag is FLOAT64 if JS_NAN_BOXING */

	switch (tag) {
		case JS_TAG_INT: {
			int32_t i ;
			JS_ToInt32 (js_ctx, &i, val) ;
			printf ("int: %d\n", i) ;
			RedisModule_ReplyWithLongLong (rm_ctx, i) ;
			break ;
		}

		case JS_TAG_FLOAT64: {
			double f ;
			JS_ToFloat64 (js_ctx, &f, val) ;
			printf ("float: %f\n", f) ;
			RedisModule_ReplyWithDouble (rm_ctx, f) ;
			break;
		}

		case JS_TAG_STRING: {
			const char *str = JS_ToCString (js_ctx, val) ;
			printf ("string: %s\n", str) ;
			RedisModule_ReplyWithCString (rm_ctx, str) ;
			JS_FreeCString (js_ctx, str) ;
			break ;
		}

		case JS_TAG_BOOL: {
			int b = JS_ToBool (js_ctx, val) ;
			printf ("bool: %s\n", b ? "true" : "false") ;
			RedisModule_ReplyWithBool (rm_ctx, b) ;
			break ;
		}

		case JS_TAG_NULL: {
			printf("null\n");
			RedisModule_ReplyWithNull (rm_ctx) ;
			break ;
		}

		case JS_TAG_OBJECT: {
			if (JS_IsArray (js_ctx, val)) {
				printf("array\n");
				ReplyArray (rm_ctx, js_ctx, val) ;
			} else if (JS_IsObject(val)) {
				printf("plain object\n");
				ReplyObj (rm_ctx, js_ctx, val) ;
			} else {
				printf ("unknown class: %d\n", JS_GetClassID (val)) ;
			}
			break ;
		}

		case JS_TAG_EXCEPTION: {
			printf("exception object\n");
			ReplyException (rm_ctx, js_ctx, val) ;
			break ;
		}

		default:
			printf("unknown tag: %d\n", tag);
	}
}

// Native C function
static JSValue js_add
(
	JSContext *ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
    int32_t a, b;
    // Get arguments from JS
    if (JS_ToInt32(ctx, &a, argv[0]) ||
        JS_ToInt32(ctx, &b, argv[1])) {
        return JS_ThrowTypeError(ctx, "Expected two integers");
    }
    int32_t result = a + b;
    // Return result to JS
    return JS_NewInt32(ctx, result);
}

// Native print function
static JSValue js_print
(
	JSContext *ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
    for (int i = 0; i < argc; i++) {
        const char *str = JS_ToCString(ctx, argv[i]);
        if (!str)
            return JS_EXCEPTION;
        printf("%s", str);
        JS_FreeCString(ctx, str);
        if (i < argc - 1) printf(" ");
    }
    printf("\n");
    return JS_UNDEFINED;
}

int Graph_Eval
(
	RedisModuleCtx *rm_ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT (rm_ctx != NULL) ;
	ASSERT (argv   != NULL) ;

	if (argc < 2) {
		RedisModule_WrongArity (rm_ctx) ;
		return REDISMODULE_OK ;
	}

	size_t script_len ;
	const char *script = RedisModule_StringPtrLen (argv[1], &script_len) ;
	printf ("script: %s\n", script) ;

    // 1. Create runtime and context
    JSRuntime *rt = JS_NewRuntime() ;
    if (!rt) {
        fprintf(stderr, "Failed to create JSRuntime\n");
        return REDISMODULE_OK;
    }

    // Optional: limit memory/time to sandbox
    JS_SetMaxStackSize(rt, 1024 * 1024); // 1 MB stack limit
    JSContext *js_ctx = JS_NewContext(rt);
    if (!js_ctx) {
        fprintf(stderr, "Failed to create JSContext\n");
        JS_FreeRuntime(rt);
        return REDISMODULE_OK;
    }

    // 2. Create a global object reference
    JSValue global_obj = JS_GetGlobalObject(js_ctx);

    // 3. Bind C functions into the global object
    JS_SetPropertyStr(js_ctx, global_obj, "add",
        JS_NewCFunction(js_ctx, js_add, "add", 2));

    JS_SetPropertyStr(js_ctx, global_obj, "print",
        JS_NewCFunction(js_ctx, js_print, "print", 1));

    JS_FreeValue(js_ctx, global_obj); // we don't need it anymore

    // 4. JavaScript code
    //const char *script =
    //    "let result = add(3, 4);\n"
    //    "print('The sum is', result);\n";

    // 5. Evaluate JS code
    JSValue val = JS_Eval (js_ctx, script, script_len, "<input>",
			JS_EVAL_TYPE_GLOBAL) ;

    // 6. Handle exceptions
    if (JS_IsException(val)) {
        JSValue exc = JS_GetException(js_ctx);
        const char *msg = JS_ToCString(js_ctx, exc);
        fprintf(stderr, "Exception: %s\n", msg);
        JS_FreeCString(js_ctx, msg);
        JS_FreeValue(js_ctx, exc);
    }

	// replay to caller
	ReplyVal (rm_ctx, js_ctx, val) ;

    JS_FreeValue(js_ctx, val);

    // 7. Cleanup
    JS_FreeContext(js_ctx);
    JS_FreeRuntime(rt);

	return REDISMODULE_OK;
}

