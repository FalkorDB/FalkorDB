/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "quickjs.h"
#include "../value.h"
#include "../datatypes/datatypes.h"

// forward declarations
SIValue UDF_JSToSIValue (JSContext *js_ctx, JSValue val) ;

// returns true if func is a user defined function
// native functions e.g. console.log typically contains "[native code]"
static bool _is_user_function
(
	JSContext *js_ctx,  // java script context
	JSValue func        // function
) {
    if (!JS_IsFunction (js_ctx, func)) {
        return false ;
	}

    // get the function's source code to check if it's user-defined
    JSValue str = JS_ToString (js_ctx, func) ;
    if (JS_IsException (str)) {
        JS_FreeValue (js_ctx, str) ;
        return false ;
    }

    const char *src = JS_ToCString (js_ctx, str) ;
    bool result = false ;

    if (src) {
        // built-in functions in QuickJS typically appear as: "function name() { [native code] }"
        if (strstr (src, "[native code]") == NULL) {
            result = true ; // user-defined
        }
        JS_FreeCString (js_ctx, src) ;
    }

    JS_FreeValue (js_ctx, str) ;
    return result ;
}

JSValue UDF_GetFunction
(
	const char *func_name,
	JSContext *js_ctx
) {
    JSValue global_obj = JS_GetGlobalObject (js_ctx) ;

    JSPropertyEnum *props ;
    uint32_t len ;

    if (JS_GetOwnPropertyNames (js_ctx, &props, &len, global_obj,
				JS_GPN_STRING_MASK | JS_GPN_ENUM_ONLY) < 0) {
        JS_FreeValue (js_ctx, global_obj) ;
        return JS_NULL ;
    }

	JSValue ret = JS_NULL ;

    for (uint32_t i = 0; i < len; i++) {
        JSAtom atom = props[i].atom ;
        JSValue val = JS_GetProperty (js_ctx, global_obj, atom) ;

        if (_is_user_function (js_ctx, val)) {
            const char *name = JS_AtomToCString (js_ctx, atom) ;
            JS_FreeCString (js_ctx, name) ;

			if ((strcmp (name, func_name) == 0)) {
				JS_FreeAtom (js_ctx, atom) ;
				ret = val ;
				break ;
			}
        }

		JS_FreeValue (js_ctx, val)  ;
		JS_FreeAtom  (js_ctx, atom) ;
    }

    js_free (js_ctx, props) ;
    JS_FreeValue (js_ctx, global_obj) ;

	return ret ;
}

bool UDF_ContainsFunction
(
	const char *func_name,
	JSContext *js_ctx
) {
	return (!JS_IsNull (UDF_GetFunction (func_name, js_ctx))) ;
}

// convert an SIValue to a JavaScript object
JSValue UDF_SIValueToJS
(
	JSContext *js_ctx,
	SIValue val
) {
	JSValue js_val ;

	SIType t = SI_TYPE (val) ;
	switch (t) {

		case T_MAP:
		{
			js_val = JS_NewObject (js_ctx) ;
			uint n = Map_KeyCount (val) ;
			for (int i = 0; i < n; i++) {
				SIValue key;
				SIValue value;

				Map_GetIdx (val, i, &key, &value) ;
				JS_SetPropertyStr (js_ctx, js_val, key.stringval,
						UDF_SIValueToJS (js_ctx, value)) ;
			}
			break ;
		}

		case T_NODE:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_EDGE :
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_ARRAY:
		{
			js_val = JS_NewArray (js_ctx) ;
			int n = SIArray_Length (val) ;
			for (int i = 0; i < n; i++) {
				JS_SetPropertyUint32 (js_ctx, js_val, i,
						UDF_SIValueToJS (js_ctx, SIArray_Get (val, i))) ;
			}
			break ;
		}

		case T_PATH:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_DATETIME:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_LOCALDATETIME:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_DATE:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_TIME:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_LOCALTIME:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_DURATION:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_STRING:
		{
			js_val = JS_NewString (js_ctx, val.stringval) ;
			break ;
		}

		case T_BOOL:
		{
			js_val = JS_NewBool (js_ctx, val.longval) ;
			break ;
		}

		case T_INT64:
		{
			js_val = JS_NewInt32 (js_ctx, val.longval) ;
			break ;
		}

		case T_DOUBLE:
		{
			js_val = JS_NewFloat64 (js_ctx, val.doubleval) ;
			break ;
		}

		case T_NULL:
		{
			js_val = JS_NULL ;
			break ;
		}

		case T_PTR: 
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_POINT:
		{
			js_val = JS_NewObject (js_ctx) ;
			JS_SetPropertyStr (js_ctx, js_val, "latitude", JS_NewFloat64 (js_ctx, Point_lat (val))) ;
			JS_SetPropertyStr (js_ctx, js_val, "longitude", JS_NewFloat64 (js_ctx, Point_lon (val))) ;
			break ;
		}

		case T_VECTOR_F32:
		{
			js_val = JS_NewArray (js_ctx) ;
			int n = SIVector_Dim (val) ;
			float *elements = SIVector_Elements (val) ;
			for (int i = 0; i < n; i++) {
				JS_SetPropertyUint32 (js_ctx, js_val, i,
						JS_NewFloat64(js_ctx, elements[i])) ;
			}
			break ;
		}

		case T_INTERN_STRING: 
		{
			js_val = JS_NewString (js_ctx, val.stringval) ;
			break ;
		}
		
		default:
		{
			assert (false && "Unknown SIValue type") ;
			break ;
		}
	}

	return js_val ;
}

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
        JS_FreeValue (js_ctx, js_val)  ;
    }

    // free the property enum array
    js_free (js_ctx, props) ;

	return map ;
}

// convert a JS object to an SIValue
SIValue UDF_JSToSIValue
(
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

	int tag = JS_VALUE_GET_TAG (val) ;

	switch (tag) {

		case JS_TAG_INT: {
			int32_t i ;
			JS_ToInt32 (js_ctx, &i, val) ;
			printf ("int: %d\n", i) ;
			return SI_LongVal (i) ;
		}

		case JS_TAG_FLOAT64: {
			double f ;
			JS_ToFloat64 (js_ctx, &f, val) ;
			printf ("float: %f\n", f) ;
			return SI_DoubleVal (f) ;
		}

		case JS_TAG_STRING: {
			const char *str = JS_ToCString (js_ctx, val) ;
			printf ("string: %s\n", str) ;
			return SI_DuplicateStringVal (str) ;
		}

		case JS_TAG_BOOL: {
			int b = JS_ToBool (js_ctx, val) ;
			printf ("bool: %s\n", b ? "true" : "false") ;
			return SI_BoolVal (b) ;
		}

		case JS_TAG_NULL: {
			printf ("null\n") ;
			return SI_NullVal () ;
		}

		case JS_TAG_OBJECT: {
			if (JS_IsArray (js_ctx, val)) {
				printf("array\n");
				return _JSArrayToSIValue (js_ctx, val) ;
			} else if (JS_IsObject(val)) {
				printf("plain object\n");
				return _JSObjToSIValue(js_ctx, val) ;
			} else {
				ASSERT (false && "unknown class") ;
				return SI_NullVal () ;
			}
			break ;
		}

		case JS_TAG_EXCEPTION: {
			printf("exception object\n");
			return SI_NullVal () ;
		}

		default:
			ASSERT (false && "unknown tag") ;
			return SI_NullVal () ;
	}

	ASSERT (false && "not supposed to reach this point") ;
	return SI_NullVal () ;
}

