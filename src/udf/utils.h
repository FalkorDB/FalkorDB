/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "../value.h"

// returns true if func is a user defined function
// native functions e.g. console.log typically contains "[native code]"
bool UDF_IsUserFunction
(
	JSContext *js_ctx,  // JavaScript context
	JSValue func        // function
);

JSValue UDF_GetFunction
(
	const char *func_name,
	JSContext *js_ctx
);

bool UDF_ContainsFunction
(
	const char *func_name,
	JSContext *js_ctx
);

// convert an SIValue to a JavaScript object
JSValue UDF_SIValueToJS
(
	JSContext *js_ctx,
	SIValue val
);

// convert a JS object to an SIValue
SIValue UDF_JSToSIValue
(
	JSContext *js_ctx,
	JSValue val
);

