/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "../value.h"

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

