/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "quickjs.h"
#include "classes.h"
#include "repository.h"
#include "../util/arr.h"
#include "../errors/errors.h"
#include "../arithmetic/func_desc.h"

extern JSClassID js_node_class_id;        // JS Node class
extern JSClassID js_edge_class_id;        // JS Edge class
extern JSClassID js_path_class_id;        // JS Path class
extern JSClassID js_attributes_class_id;  // JS Attributes class

const char *UDF_LIB = NULL ;              // global register library name

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
		asprintf (err, "Library %s does not exist", lib) ;
		return false ;
	}

	// remove library's functions from global functions repo
	bool removed ;
	int n = array_len (functions) ;
	for (int i = 0; i < n; i++) {
		removed = AR_FuncRemoveUDF (functions[i]) ;
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

	JSRuntime *js_rt = JS_NewRuntime() ;
	UDF_RT_RegisterClasses (js_rt) ;
	JS_SetMaxStackSize (js_rt, 1024 * 1024) ; // 1 MB stack limit

	// create js context
	JSContext *js_ctx = JS_NewContext(js_rt) ;
	UDF_CTX_RegisterClasses (js_ctx) ;

	// provide validation-only register() hook
	falkor_set_register_impl (js_ctx, UDF_FUNC_REG_MODE_VALIDATE) ;

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
	UDF_CTX_RegisterClasses (js_ctx) ;

	// provide global functions registration register() hook
	falkor_set_register_impl (js_ctx, UDF_FUNC_REG_MODE_GLOBAL) ;

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

