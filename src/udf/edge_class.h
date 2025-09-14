/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "../graph/entities/edge.h"

//------------------------------------------------------------------------------
// edge class binding
//------------------------------------------------------------------------------

// create a JavaScript Edge object from a FalkorDB Edge
// wraps a native FalkorDB Edge into a QuickJS JSValue instance
// return JSValue representing the Edge in QuickJS
JSValue UDF_CreateEdge
(
	JSContext *js_ctx,  // JavaScript context
	const Edge *edge    // pointer to the native FalkorDB Edge
);

// register the Edge class with a JavaScript runtime
// associates the Edge class definition with the given QuickJS runtime
void UDF_RegisterEdgeClass
(
	JSRuntime *js_runtime  // the QuickJS runtime in which to register the class
);

// register the Edge class with a JavaScript context
// makes the Edge class available within the provided QuickJS context
void UDF_RegisterEdgeProto
(
	JSContext *js_ctx  // JavaScript context
);

