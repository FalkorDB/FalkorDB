/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "../graph/entities/edge.h"

// create a JSValue of type Edge
JSValue js_create_edge
(
	JSContext *js_ctx,
	const Edge *edge
);

// create a JSValue of type Edge
void register_edge_class
(
	JSRuntime *js_runtime, 
	JSContext *js_ctx
);

