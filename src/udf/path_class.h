/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "../datatypes/path/path.h"

// create a JSValue of type Path
// create a JavaScript Path object from a FalkorDB Path
// wraps a native FalkorDB Path into a QuickJS JSValue instance
// return JSValue representing the Path in QuickJS
JSValue UDF_CreatePath
(
	JSContext *js_ctx,  // JavaScript context
	const Path *path    // pointer to the native FalkorDB Path
);

// register the Path class with a QuickJS runtime
// associates the Path class definition with the given QuickJS runtime
void UDF_RegisterPathClass
(
	JSRuntime *js_runtime  // JavaScript runtime
);

// register the Path class with a QuickJS context
// makes the Path class available within the provided QuickJS context
void UDF_RegisterPathProto
(
	JSContext *js_ctx  // JavaScript context
);

