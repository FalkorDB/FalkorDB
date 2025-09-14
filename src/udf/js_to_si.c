/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "utils.h"
#include "quickjs.h"
#include "../value.h"
#include "../errors/errors.h"
#include "../datatypes/datatypes.h"

//------------------------------------------------------------------------------
// external class IDs
//------------------------------------------------------------------------------

extern JSClassID js_node_class_id;        // JS Node class
extern JSClassID js_edge_class_id;        // JS Edge class
extern JSClassID js_path_class_id;        // JS Path class
extern JSClassID js_attributes_class_id;  // JS Attributes class

// forward declarations
SIValue UDF_JSToSIValue (JSContext *js_ctx, JSValue val) ;

// report a QuickJS exception to FalkorDB's error context
// extracts message and stack trace if available
static void _report_exception
(
	JSContext *js_ctx  // java script context
) {
    // extract exception object
    JSValue exception = JS_GetException(js_ctx);

    // if it's an Error object, you can get "message" and "stack"
    JSValue stack   = JS_GetPropertyStr(js_ctx, exception, "stack");
    JSValue message = JS_GetPropertyStr(js_ctx, exception, "message");

    const char *msg_str   = JS_ToCString(js_ctx, message);
    const char *stack_str = JS_ToCString(js_ctx, stack);

	ErrorCtx_SetError ("UDF Exception: %s\n, Stack: %s",
			msg_str   ? msg_str   : "<no message>",
			stack_str ? stack_str : "<no stack>") ;

    // cleanup
    JS_FreeCString(js_ctx, msg_str);
    JS_FreeCString(js_ctx, stack_str);
    JS_FreeValue(js_ctx, message);
    JS_FreeValue(js_ctx, stack);
    JS_FreeValue(js_ctx, exception);
}

// convert a JS Array into an SI Array
static SIValue _JSArrayToSIValue
(
    JSContext *js_ctx,
	JSValue js_arr
) {
	uint32_t len = 0 ;
	JSValue len_val = JS_GetPropertyStr (js_ctx, js_arr, "length") ;
	JS_ToUint32 (js_ctx, &len, len_val) ;
	JS_FreeValue (js_ctx, len_val) ;

	SIValue arr = SI_Array (len) ;

	for (uint32_t i = 0; i < len; i++) {
		JSValue elem = JS_GetPropertyUint32 (js_ctx, js_arr, i) ;
		SIArray_Append (&arr, UDF_JSToSIValue (js_ctx, elem)) ;
		JS_FreeValue (js_ctx, elem) ;
	}

	return arr ;
}

// convert a plain JS Object into an SI_Map
// only enumerable string-keyed properties are processed
static SIValue _JSObjToSIValue
(
    JSContext *js_ctx,
	JSValue obj
) {
    int i ;
    uint32_t len ;
    JSPropertyEnum *props ;

    // get enumerable own properties (keys)
    if (JS_GetOwnPropertyNames (js_ctx, &props, &len, obj,
                               JS_GPN_STRING_MASK | JS_GPN_ENUM_ONLY) < 0) {
        return SI_NullVal () ;
    }

	// TODO: validate map keys are all strings

	SIValue map = SI_Map (len) ;

    for (i = 0; i < (int)len; i++) {
        const char *key_str = JS_AtomToCString (js_ctx, props[i].atom) ;
        ASSERT (key_str) ;
		SIValue key = SI_DuplicateStringVal (key_str) ;

        // get the property value
        JSValue js_val = JS_GetProperty (js_ctx, obj, props[i].atom) ;
		SIValue val = UDF_JSToSIValue(js_ctx, js_val);

		Map_AddNoClone (&map, key, val) ;

        // free value and key
        JS_FreeValue (js_ctx, js_val) ;
		JS_FreeCString (js_ctx, key_str) ;
    }

    // free the property enum array
    js_free (js_ctx, props) ;

	return map ;
}

// convert a QuickJS value into an SIValue
//
//  supported conversions:
//    - JS primitives: number, string, boolean, null
//    - JS BigInt     -> SI Long
//    - JS Array      -> SI Array
//    - JS Object     -> SI Map
//    - JS Attributes -> SI Map of properties
//    - Exceptions    -> logged, return Null
//
//  unsupported values:
//    - JS Symbols -> error
//    - undefined  -> null
SIValue UDF_JSToSIValue
(
	JSContext *js_ctx,
	JSValue val
) {
	SIValue ret = SI_NullVal () ;
	int tag = JS_VALUE_GET_TAG (val) ;

	switch (tag) {

		case JS_TAG_SHORT_BIG_INT: {
			int64_t out;
			if (JS_ToBigInt64 (js_ctx, &out, val) == 0) {
				ret = SI_LongVal (out) ;
				break ;
			} else {
				ErrorCtx_SetError ("JS failed to return BitInt64") ;
				ret = SI_NullVal () ;
				break ;
			}
		}

		case JS_TAG_INT: {
			int64_t i ;
			JS_ToInt64 (js_ctx, &i, val) ;
			ret = SI_LongVal (i) ;
			break ;
		}

		case JS_TAG_FLOAT64: {
			double f ;
			JS_ToFloat64 (js_ctx, &f, val) ;
			ret = SI_DoubleVal (f) ;
			break ;
		}

		case JS_TAG_STRING: {
			const char *str = JS_ToCString (js_ctx, val) ;
			ret = SI_DuplicateStringVal (str) ;
			JS_FreeCString (js_ctx, str) ;
			break ;
		}

		case JS_TAG_BOOL: {
			int b = JS_ToBool (js_ctx, val) ;
			ret = SI_BoolVal (b) ;
			break ;
		}

		case JS_TAG_NULL: {
			ret = SI_NullVal () ;
			break ;
		}

		case JS_TAG_OBJECT: {
			if (JS_IsArray (js_ctx, val)) {
				ret = _JSArrayToSIValue (js_ctx, val) ;
			} else if (JS_IsObject (val)) {
				// handle registered Falkor classes
				JSClassID cid = JS_GetClassID (val) ;

				if (cid == js_node_class_id) {
					Node *n = JS_GetOpaque (val, js_node_class_id) ;
					ret = SI_Node (n) ;
				}

				else if (cid == js_edge_class_id) {
					Edge *e = JS_GetOpaque (val, js_edge_class_id) ;
					ret = SI_Edge (e) ;
				}

				else if (cid == js_path_class_id) {
					Path *p = JS_GetOpaque (val, js_path_class_id) ;
					ret = SIPath_New (p) ;
				}

				else if (cid == js_attributes_class_id) {
					GraphEntity *e = JS_GetOpaque (val, js_attributes_class_id) ;
					ret = GraphEntity_Properties (e) ;
				}

				else if (cid == 18) { // JS_CLASS_REGEXP
					const char *str = JS_ToCString (js_ctx, val) ;
					ret = SI_DuplicateStringVal (str) ;
					JS_FreeCString (js_ctx, str) ;
				}

				else  {
					ret = _JSObjToSIValue(js_ctx, val) ;
				}
			}
			break ;
		}

		case JS_TAG_EXCEPTION: {
			_report_exception (js_ctx) ;
			ret = SI_NullVal () ;
			break ;
		}

		case JS_TAG_SYMBOL: {
			ErrorCtx_SetError ("JS Symbols can not be returned") ;
			ret = SI_NullVal () ;
			break ;
		}

		case JS_TAG_UNDEFINED: {
			ret = SI_NullVal () ;
			break ;
		}

		default:
			ASSERT (false && "unknown tag") ;
			ret = SI_NullVal () ;
			break ;
	}

	return ret ;
}

