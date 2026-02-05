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

// 8 MB stack limit - sufficient for most UDF operations
#define JS_RUNTIME_STACK_SIZE (8 * 1024 * 1024)

// 1024 MB memory limit - prevents resource exhaustion from malicious scripts
#define JS_RUNTIME_MEMORY_LIMIT (1024 * 1024 * 1024)

extern JSClassID js_node_class_id;        // JS Node class
extern JSClassID js_edge_class_id;        // JS Edge class
extern JSClassID js_path_class_id;        // JS Path class
extern JSClassID js_attributes_class_id;  // JS Attributes class

const char *UDF_LIB = NULL ;              // global register library name

// allocate and return a new JavaScript runtime for UDF operations
// each call creates an independent runtime
// the caller owns the runtime and is responsible for freeing it via
// JS_FreeRuntime() once no longer needed
// returns: pointer to a newly created JSRuntime
JSRuntime *UDF_GetJSRuntime(void) {
	JSRuntime *js_rt = JS_NewRuntime () ;
	ASSERT (js_rt != NULL) ;

	UDF_RT_RegisterClasses (js_rt) ;
	JS_SetMaxStackSize (js_rt, JS_RUNTIME_STACK_SIZE) ;
	JS_SetMemoryLimit (js_rt, JS_RUNTIME_MEMORY_LIMIT) ;

	return js_rt ;
}

// create a JavaScript context dedicated to validating UDF scripts
// the validation context is used only for syntax checking and static analysis
// of UDF libraries before they are registered
// it should not expose database
// bindings or allow execution of UDFs against live data
// returns pointer to a JSContext configured for validation operations
JSContext *UDF_GetValidationJSContext
(
	JSRuntime *js_rt // the JSRuntime from which to create the context
) {
	ASSERT (js_rt != NULL) ;

	JSContext *js_ctx = JS_NewContext (js_rt) ;
	ASSERT (js_ctx != NULL) ;

	UDF_RegisterGraphObject  (js_ctx) ;
	UDF_SetGraphTraverseImpl (js_ctx, UDF_FUNC_REG_MODE_VALIDATE) ;

	// provide validation-only register() hook
	UDF_RegisterFalkorObject  (js_ctx) ;
	UDF_SetFalkorRegisterImpl (js_ctx, UDF_FUNC_REG_MODE_VALIDATE) ;

	return js_ctx ;
}

// create a JavaScript context dedicated to UDF registration
// the registration context is used when loading a UDF library into the system
// it should expose APIs required to declare UDFs
// but not execution-time bindings
// once registration completes, the context can be discarded
// returns pointer to a JSContext configured for UDF registration
JSContext *UDF_GetRegistrationJSContext
(
	JSRuntime *js_rt  // the JSRuntime from which to create the context
) {
	ASSERT (js_rt != NULL) ;

	JSContext *js_ctx = JS_NewContext(js_rt) ;
	ASSERT (js_ctx != NULL) ;

	UDF_RegisterGraphObject  (js_ctx) ;
	UDF_SetGraphTraverseImpl (js_ctx, UDF_FUNC_REG_MODE_GLOBAL) ;

	UDF_RegisterFalkorObject  (js_ctx) ;
	UDF_SetFalkorRegisterImpl (js_ctx, UDF_FUNC_REG_MODE_GLOBAL) ;

	return js_ctx ;
}

// create a JavaScript context dedicated to executing UDFs
// the execution context is used when queries invoke registered UDFs
// it should provide the runtime environment necessary for execution
// including bindings for type conversion, database value access
// and error propagation
// returns pointer to a JSContext configured for UDF execution
JSContext *UDF_GetExecutionJSContext
(
	JSRuntime *js_rt  // the JSRuntime from which to create the context
) {
	ASSERT (js_rt != NULL) ;

	JSContext *js_ctx = JS_NewContext(js_rt) ;
	ASSERT (js_ctx != NULL) ;

	UDF_CTX_RegisterClasses   (js_ctx) ;
	UDF_SetGraphTraverseImpl  (js_ctx, UDF_FUNC_REG_MODE_LOCAL) ;
	UDF_SetFalkorRegisterImpl (js_ctx, UDF_FUNC_REG_MODE_LOCAL) ;

	return js_ctx ;
}

// remove a UDF library and all of its registered functions
//
// this function performs the following steps:
//   1. Verify the library exists in the UDF repository
//   2. Remove all functions defined by that library from the global function
//      registry
//   3. Remove the library itself from the repository
//
// returns:
//   true  - if the library and its functions were successfully removed
//   false - if the library does not exist. In this case, *err is set
//
// Notes:
//   - unexpected internal errors (e.g. failure to remove a function) will
//     trigger assertions
//   - this function does not free the memory of *script; ownership is passed
//     to the caller
bool UDF_Delete
(
	const char *lib,  // the name of the UDF library to delete

	char **script,    // optional output pointer
					  // if not NULL, set to the original JS source
					  // caller owns the returned string

	char **err        // output pointer for an error message
					  // on error, set to a heap-allocated string describing
					  // the issue, caller must free the string using free()
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
		// concat lib and function name
		char *udf;
		asprintf (&udf, "%s.%s", lib, functions[i]) ;

		removed = AR_FuncRemoveUDF (udf) ;
		ASSERT (removed == true) ;

		free (udf) ;
	}

	// remove library from UDF repo
	removed = UDF_RepoRemoveLib (lib, script) ;
	ASSERT (removed == true) ;

	return true ;
}

// remove all registered UDF libraries from the repository
// deletes in reverse order (last → first) to avoid index shifting
// this is an internal helper; errors will trigger ASSERT failures
void UDF_Flush (void) {
	// get the number of UDF libraries
	int n = UDF_RepoLibsCount () ;

	for (int i = n-1; i >= 0; i--) {
		const char *lib = NULL ;
		UDF_RepoGetLibIdx (i, &lib, NULL, NULL) ;
		ASSERT (lib != NULL) ;

		char *err = NULL ;
		bool removed = UDF_Delete (lib, NULL, &err) ;
		ASSERT (err     == NULL) ;
		ASSERT (removed == true) ;
	}
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
	
	char *prev_script = NULL ;
	if (replace) {
		// back up prev version script
		// we'll use this script to restore the previous version in case
		// the new library fails to load
		bool deleted = UDF_Delete (lib, &prev_script, err) ;

		ASSERT (*err        == NULL) ;
		ASSERT (deleted     == true) ;
		ASSERT (prev_script != NULL) ;
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

	JSRuntime *js_rt  = UDF_GetJSRuntime () ;
	JSContext *js_ctx = UDF_GetValidationJSContext (js_rt) ;

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

	res = UDF_RepoRegisterLib (lib, script) ;
	ASSERT (res == true) ;

	// re-run script in registration mode
	// create a new js context
	JS_FreeContext (js_ctx) ;
	js_ctx = UDF_GetRegistrationJSContext (js_rt) ;

	// re-evaluate the script this time with the 'register' function actually
	// adding UDF functions to the UDF repository
	val = JS_Eval (js_ctx, script, script_len, "<input>", JS_EVAL_TYPE_GLOBAL) ;

	// although we've passed validation we can still fail registering the lib
	// this can happen if the scripts tried to register the same function
	// multiple times e.g. falkor.register('a', A); falkor.register('a', B);
	if (JS_IsException (val)) {
		res = false ;
		UDF_RepoRemoveLib (lib, NULL) ;
		JS_FreeValue (js_ctx, val) ;

		if (err) {
			asprintf (err, "Failed to register UDF library: '%s'", lib);
		}

		goto cleanup ;
	}

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

	if (prev_script != NULL) {
		rm_free (prev_script) ;
	}

	JS_FreeContext (js_ctx) ;
	JS_FreeRuntime (js_rt) ;

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

