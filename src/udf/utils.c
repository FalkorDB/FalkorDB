/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "utils.h"
#include "quickjs.h"
#include "classes.h"
#include "repository.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../configuration/config.h"
#include "../arithmetic/func_desc.h"

#include <time.h>

extern JSClassID js_node_class_id;        // JS Node class
extern JSClassID js_edge_class_id;        // JS Edge class
extern JSClassID js_path_class_id;        // JS Path class
extern JSClassID js_attributes_class_id;  // JS Attributes class

const char *UDF_LIB = NULL ;              // global register library name

// milliseconds since the epoch, for JS evaluation deadlines
static int64_t _UDF_NowMs (void) {
	struct timespec ts ;
	clock_gettime (CLOCK_MONOTONIC, &ts) ;
	return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000 ;
}

// QuickJS calls this periodically; returning non-zero aborts the running
// script with an "interrupted" exception
static int _UDF_InterruptHandler
(
	JSRuntime *js_rt,  // runtime being interrupted
	void *opaque       // deadline, milliseconds
) {
	return (_UDF_NowMs () > *(int64_t*)opaque) ? 1 : 0 ;
}

// arm an interrupt handler bounding a single JS evaluation on js_rt
//
// memory is capped by JS_SetMemoryLimit, but nothing caps CPU: a script
// containing an unterminating loop pins its thread forever, and enough of them
// exhaust the pool. The budget is the configured query timeout, bounded by
// UDF_JS_TIMEOUT_CAP_MS so that a timeout of 0, which means "no limit" for
// queries, still cannot let a script hold a thread indefinitely
//
// returns the deadline, to be passed to UDF_DisarmJSDeadline
int64_t *UDF_ArmJSDeadline
(
	JSRuntime *js_rt  // runtime to bound
) {
	ASSERT (js_rt != NULL) ;

	uint64_t timeout = 0 ;  // unlimited
	Config_Option_get (Config_TIMEOUT_DEFAULT, &timeout) ;

	int64_t budget = (timeout > 0 && timeout < UDF_JS_TIMEOUT_CAP_MS)
		? (int64_t)timeout
		: (int64_t)UDF_JS_TIMEOUT_CAP_MS ;

	int64_t *deadline_ms = rm_malloc (sizeof (int64_t)) ;
	*deadline_ms = _UDF_NowMs () + budget ;

	JS_SetInterruptHandler (js_rt, _UDF_InterruptHandler, deadline_ms) ;

	return deadline_ms ;
}

// clear the interrupt handler once the evaluation completes, so a deadline set
// for one evaluation cannot fire during the next
void UDF_DisarmJSDeadline
(
	JSRuntime *js_rt,     // runtime to release
	int64_t *deadline_ms  // deadline returned by UDF_ArmJSDeadline
) {
	JS_SetInterruptHandler (js_rt, NULL, NULL) ;
	rm_free (deadline_ms) ;
}

// allocate and return a new JavaScript runtime for UDF operations
// each call creates an independent runtime
// the caller owns the runtime and is responsible for freeing it via
// JS_FreeRuntime() once no longer needed
// returns: pointer to a newly created JSRuntime
JSRuntime *UDF_GetJSRuntime(void) {
	JSRuntime *js_rt = JS_NewRuntime () ;
	ASSERT (js_rt != NULL) ;

	UDF_RT_RegisterClasses (js_rt) ;

	size_t heap_size  = -1 ;  // unlimited
	size_t stack_size = 0 ;   // unlimited

	// read JS heap and stack limits from config
	Config_Option_get (Config_JS_HEAP_SIZE,  &heap_size)  ;
	Config_Option_get (Config_JS_STACK_SIZE, &stack_size) ;

	JS_SetMemoryLimit  (js_rt, heap_size)  ;
	JS_SetMaxStackSize (js_rt, stack_size) ;

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
	UDF_SetGraphAPI (js_ctx, UDF_FUNC_REG_MODE_VALIDATE) ;

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
	UDF_SetGraphAPI (js_ctx, UDF_FUNC_REG_MODE_GLOBAL) ;

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

	JSContext *js_ctx = JS_NewContext (js_rt) ;
	ASSERT (js_ctx != NULL) ;

	UDF_CTX_RegisterClasses (js_ctx) ;
	UDF_SetGraphAPI (js_ctx, UDF_FUNC_REG_MODE_LOCAL) ;
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
	int n = arr_len (functions) ;
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

	int64_t *deadline = UDF_ArmJSDeadline (js_rt) ;
	JSValue val = JS_Eval (js_ctx, script, script_len, "<input>",
			JS_EVAL_TYPE_GLOBAL) ;
	UDF_DisarmJSDeadline (js_rt, deadline) ;

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
	deadline = UDF_ArmJSDeadline (js_rt) ;
	val = JS_Eval (js_ctx, script, script_len, "<input>", JS_EVAL_TYPE_GLOBAL) ;
	UDF_DisarmJSDeadline (js_rt, deadline) ;

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
		//
		// this re-evaluates the old script, so it is subject to the same
		// deadline as any other load and can fail. An ASSERT here compiles out
		// in release, which would leave the library removed and the caller told
		// only that the replacement failed
		char *restore_err = NULL ;
		bool restore = UDF_Load (prev_script, strlen(prev_script), lib, lib_len,
				false, &restore_err) ;

		if (!restore) {
			RedisModule_Log (NULL, "warning",
					"UDF: failed to restore library '%s' after a failed "
					"replace, the library is no longer loaded: %s", lib,
					(restore_err != NULL) ? restore_err : "unknown error") ;

			// say that the old library is gone, not just that the replacement
			// failed, since the two leave the server in very different states
			if (err != NULL) {
				char *combined = NULL ;
				asprintf (&combined,
						"%s. Restoring the previous version also failed: %s. "
						"Library '%s' is no longer loaded",
						(*err != NULL) ? *err : "Failed to replace UDF library",
						(restore_err != NULL) ? restore_err : "unknown error",
						lib) ;

				if (*err != NULL) free (*err) ;
				*err = combined ;
			}
		}

		if (restore_err != NULL) free (restore_err) ;
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

