/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "quickjs.h"
#include "../value.h"
#include "classes.h"
#include "functions.h"
#include "node_class.h"
#include "edge_class.h"
#include "path_class.h"
#include "repository.h"
#include "../util/arr.h"
#include "../errors/errors.h"
#include "../datatypes/datatypes.h"
#include "../arithmetic/func_desc.h"

extern JSClassID js_node_class_id;        // JS Node class
extern JSClassID js_edge_class_id;        // JS Edge class
extern JSClassID js_path_class_id;        // JS Path class
extern JSClassID js_attributes_class_id;  // JS Attributes class

const char *UDF_LIB = NULL ;              // global register library name

// forward declarations
SIValue UDF_JSToSIValue (JSContext *js_ctx, JSValue val) ;

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

    printf("Exception: %s\n", msg_str ? msg_str : "<no message>");
    printf("Stack: %s\n", stack_str ? stack_str : "<no stack>");

	ErrorCtx_SetError ("UDF Exception: %s\n, Stack: %s",
			msg_str   ? msg_str   : "<no message>",
			stack_str ? stack_str : "<no stack>") ;

    // free temporary values
    JS_FreeCString(js_ctx, msg_str);
    JS_FreeCString(js_ctx, stack_str);
    JS_FreeValue(js_ctx, message);
    JS_FreeValue(js_ctx, stack);

    // finally free the exception object
    JS_FreeValue(js_ctx, exception);
}

bool UDF_Delete
(
	const char *lib,      // UDF library
	const char **script,  // [optional] [output] library script
	char **err            // [output] error message
) {
	ASSERT (lib != NULL) ;
	ASSERT (err != NULL) ;

	*err = NULL ;

	const char **functions ;

	// locate library
	if (!UDF_RepoGetLib (lib, &functions, NULL)) {
		asprintf (err, "Library %s doesn't exists", lib) ;
		return false ;
	}

	// remove library's functions from global functions repo
	bool removed ;
	int n = array_len (functions) ;
	for (int i = 0; i < n; i++) {
		removed = AR_FuncRemove (functions[i], NULL) ;
		ASSERT (removed == true) ;
	}

	// remove library from UDF repo
	removed = UDF_RepoRemoveLib (lib, script) ;
	ASSERT (removed == true) ;

	return true ;
}

// load and register a UDF library
//   1. validates the provided script in a temporary JS context
//   2. ensures the library does not already exist (unless REPLACE is set)
//   3. on success, registers the library and re-evaluates it to capture functions
//
// arguments:
//   script     - JavaScript source code for the library
//   script_len - length of the source code
//   lib        - library name
//   lib_len    - length of the library name
//   replace    - whether to overwrite an existing library
//   err[out]   - on failure, set to an allocated error string (must be freed)
//
// returns:
//   true on success, false on failure (err will be set).

bool UDF_Load
(
	const char *script,  // lib's script
	size_t script_len,   // script's length
	const char *lib,     // library name
	size_t lib_len,      // library name length
	bool replace,        // replace flag
	char **err           // [optional] error msg
) {
	ASSERT (lib    != NULL) ;
	ASSERT (script != NULL) ;

	ASSERT (lib_len    > 0) ;
	ASSERT (script_len > 0) ;

	bool res = true ;

	if (err != NULL) *err = NULL ;

	// fail in case script already exists and replace is false
	bool lib_exists = UDF_RepoContainsLib (lib, NULL) ;

	// replace only if the library exists
	replace = (replace && lib_exists) ;

	if (lib_exists && replace == false) {
		if (err != NULL) {
			asprintf (err,
					"Failed to register, UDF Library '%s' already registered",
					lib) ;
		}
		return false ;
	}

	//--------------------------------------------------------------------------
	// remove previous version of the lib
	//--------------------------------------------------------------------------
	
	const char *prev_script = NULL ;
	if (replace) {
		// back up prev version script
		// we'll use this script to restore the previous version in case
		// the new library fails to load
		bool deleted = UDF_Delete (lib, &prev_script, err) ;

		ASSERT (*err         == NULL) ;
		ASSERT (deleted      == true) ;
		ASSERT (prev_script  != NULL) ;
	}

	// set global library name
	UDF_LIB = lib ;

	// load script into a dedicated JavaScript context
	// validate:
	// 1. script loads
	// 2. functions do not already exists (in case replace is false)
	//
	// if scripts passes validations add library to repository

	//--------------------------------------------------------------------------
	// create dedicated js runtime
	//--------------------------------------------------------------------------

	JSRuntime *js_rt  = JS_NewRuntime() ;
	JS_SetMaxStackSize (js_rt, 1024 * 1024) ; // 1 MB stack limit

	// create js context
	JSContext *js_ctx = JS_NewContext(js_rt) ;

	// provide validation-only register() hook
	UDF_RegisterFunctions (js_ctx, UDF_FUNC_REG_MODE_VALIDATE) ;

	JSValue val = JS_Eval (js_ctx, script, script_len, "<input>",
			JS_EVAL_TYPE_GLOBAL) ;

    // report exception
    if (JS_IsException (val)) {
		res = false ;

        JSValue exc = JS_GetException (js_ctx) ;
        const char *msg = JS_ToCString (js_ctx, exc) ;

		if (err) {
			asprintf (err, "Failed to evaluate UDF library '%s', Exception: %s",
					lib, msg);
		}

        JS_FreeCString (js_ctx, msg) ;
        JS_FreeValue   (js_ctx, exc) ;
		JS_FreeValue   (js_ctx, val) ;

		goto cleanup ;
    }
	JS_FreeValue (js_ctx, val) ;

	//--------------------------------------------------------------------------
	// UDF passed validations, register library
	//--------------------------------------------------------------------------

	// re-run script in registration mode
	// create a new js context
	JS_FreeContext (js_ctx) ;

	js_ctx = JS_NewContext(js_rt) ;
	ASSERT (js_ctx != NULL) ;

	// provide global functions registration register() hook
	UDF_RegisterFunctions (js_ctx, UDF_FUNC_REG_MODE_GLOBAL) ;

	res = UDF_RepoRegisterLib (lib, script) ;
	ASSERT (res == true) ;

	// re-evaluate the script this time with the 'register' function actually
	// adding UDF functions to the UDF repository
	val = JS_Eval (js_ctx, script, script_len, "<input>", JS_EVAL_TYPE_GLOBAL) ;
	ASSERT (!JS_IsException (val)) ;
	JS_FreeValue (js_ctx, val) ;

	// all done expose the library
	UDF_RepoExposeLib (lib) ;

cleanup:
	if (res == false && replace == true) {
		// we've failed to replace the library
		// restore previous version
		bool restore = UDF_Load (prev_script, strlen(prev_script), lib, lib_len,
				false, NULL) ;
		ASSERT (restore) ;
	}

	UDF_LIB = NULL ;  

	if (js_ctx != NULL) {
		JS_FreeContext (js_ctx) ;
	}

	if (js_rt != NULL) {
		JS_FreeRuntime (js_rt) ;
	}

	return res ;
}

// returns true if func is a user defined function
// native functions e.g. console.log typically contains "[native code]"
bool UDF_IsUserFunction
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
			js_val = js_create_node (js_ctx, val.ptrval) ;
			break ;
		}

		case T_EDGE :
		{
			js_val = js_create_edge (js_ctx, val.ptrval) ;
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
			js_val = js_create_path (js_ctx, val.ptrval) ;
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
			js_val = JS_NewInt64 (js_ctx, val.longval) ;
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

		case JS_TAG_SHORT_BIG_INT: {
			int64_t out;
			if (JS_ToBigInt64 (js_ctx, &out, val) == 0) {
				return SI_LongVal (out) ;
			} else {
				ErrorCtx_SetError ("JS failed to return BitInt64") ;
				return SI_NullVal () ;
			}
		}

		case JS_TAG_INT: {
			int64_t i ;
			JS_ToInt64 (js_ctx, &i, val) ;
			return SI_LongVal (i) ;
		}

		case JS_TAG_FLOAT64: {
			double f ;
			JS_ToFloat64 (js_ctx, &f, val) ;
			return SI_DoubleVal (f) ;
		}

		case JS_TAG_STRING: {
			const char *str = JS_ToCString (js_ctx, val) ;
			return SI_DuplicateStringVal (str) ;
		}

		case JS_TAG_BOOL: {
			int b = JS_ToBool (js_ctx, val) ;
			return SI_BoolVal (b) ;
		}

		case JS_TAG_NULL: {
			return SI_NullVal () ;
		}

		case JS_TAG_OBJECT: {
			if (JS_IsArray (js_ctx, val)) {
				return _JSArrayToSIValue (js_ctx, val) ;
			} else if (JS_IsObject (val)) {
				// registered classes
				JSClassID cid = JS_GetClassID (val) ;

				if (cid == js_node_class_id) {
					Node *n = JS_GetOpaque (val, js_node_class_id) ;
					return SI_Node (n) ;
				}

				else if (cid == js_edge_class_id) {
					Edge *e = JS_GetOpaque (val, js_edge_class_id) ;
					return SI_Edge (e) ;
				}

				else if (cid == js_path_class_id) {
					Path *p = JS_GetOpaque (val, js_path_class_id) ;
					return SIPath_New (p) ;
				}

				else if (cid == js_attributes_class_id) {
					GraphEntity *e = JS_GetOpaque (val, js_attributes_class_id) ;
					return GraphEntity_Properties (e) ;
				}

				else if (cid == 18 ) { // JS_CLASS_REGEXP
					const char *str = JS_ToCString (js_ctx, val) ;
					return SI_DuplicateStringVal (str) ;
				}

				else  {
					return _JSObjToSIValue(js_ctx, val) ;
				}
			}
			break ;
		}

		case JS_TAG_EXCEPTION: {
			printf ("exception object\n") ;
			_report_exception (js_ctx) ;
			return SI_NullVal () ;
		}

		case JS_TAG_SYMBOL: {
			ErrorCtx_SetError ("JS Symbols can not be returned") ;
			return SI_NullVal () ;
		}

		case JS_TAG_UNDEFINED: {
			printf ("undefined\n") ;
			return SI_NullVal () ;
		}

		default:
			ASSERT (false && "unknown tag") ;
			return SI_NullVal () ;
	}

	ASSERT (false && "not supposed to reach this point") ;
	return SI_NullVal () ;
}

