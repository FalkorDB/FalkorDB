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

// create a new JavaScript object of class `Edge` wrapping a FalkorDB edge
// return a JSValue representing the edge object, or JS_EXCEPTION on error.
// note The returned JSValue is owned by QuickJS; caller must free it with
// JS_FreeValue when no longer needed
JSValue UDF_CreateEdge
(
	JSContext *js_ctx,  // the QuickJS context in which to create the object.
	const Edge *edge    // the FalkorDB edge to wrap (must not be NULL)
);

// register the `Edge` class with the given QuickJS runtime
void UDF_RegisterEdgeClass
(
	JSRuntime *js_runtime  // the QuickJS runtime in which to register the class
);

// register the `Edge` class with the given QuickJS context
void UDF_RegisterEdgeProto
(
	JSContext *js_ctx  // the QuickJS context in which to register the class
);

