/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "../value.h"

// allocate and return a new JavaScript runtime for UDF operations
// each call creates an independent runtime
// the caller owns the runtime and is responsible for freeing it via
// JS_FreeRuntime() once no longer needed
// returns: pointer to a newly created JSRuntime
JSRuntime *UDF_GetJSRuntime(void);

// create a JavaScript context dedicated to validating UDF scripts
// the validation context is used only for syntax checking and static analysis
// of UDF libraries before they are registered
// it should not expose database
// bindings or allow execution of UDFs against live data
// returns pointer to a JSContext configured for validation operations
JSContext *UDF_GetValidationJSContext
(
	JSRuntime *js_rt // the JSRuntime from which to create the context
) ;

// create a JavaScript context dedicated to UDF registration
// the registration context is used when loading a UDF library into the system
// it should expose APIs required to declare UDFs
// but not execution-time bindings
// once registration completes, the context can be discarded
// returns pointer to a JSContext configured for UDF registration
JSContext *UDF_GetRegistrationJSContext
(
	JSRuntime *js_rt  // the JSRuntime from which to create the context
) ;

// create a JavaScript context dedicated to executing UDFs
// the execution context is used when queries invoke registered UDFs
// it should provide the runtime environment necessary for execution
// including bindings for type conversion, database value access
// and error propagation
// returns pointer to a JSContext configured for UDF execution
JSContext *UDF_GetExecutionJSContext
(
	JSRuntime *js_rt  // the JSRuntime from which to create the context
) ;

// remove a UDF library and all of its registered functions
//
// this function performs the following steps:
//   1. Verify the library exists in the UDF repository
//   2. Remove all functions defined by that library from the global function
//      registry
//   3. Remove the library itself from the repository
//
// returns
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
	const char *lib,      // the name of the UDF library to delete

	const char **script,  // optional output pointer
						  // if not NULL, set to the original JS source
						  // caller owns the returned string

	char **err            // output pointer for an error message
						  // on error, set to a heap-allocated string describing
						  // the issue, caller must free the string using free()
);

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
);

// returns true if func is a user defined function
// native functions e.g. console.log typically contains "[native code]"
bool UDF_IsUserFunction
(
	JSContext *js_ctx,  // JavaScript context
	JSValue func        // function
);

// convert a FalkorDB SIValue to a QuickJS value
//
// this function maps internal database types to their JavaScript equivalents
// for composite types (maps, arrays, paths, nodes, edges), new JS objects are
// allocated and populated recursively
//
// Memory ownership:
//   - returns a newly created JSValue with a strong reference
//   - the caller is responsible for freeing the value with JS_FreeValue()
//     once it is no longer needed
//
// returns: a JSValue representing the given SIValue in unimplemented cases
// the function aborts with an assertion failure
JSValue UDF_SIValueToJS
(
	JSContext *js_ctx,
	SIValue val
);

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
);

