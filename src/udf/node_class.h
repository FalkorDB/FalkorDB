/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "../graph/entities/node.h"

// create a JavaScript Node object from a FalkorDB Node
// wraps a native FalkorDB Node into a QuickJS JSValue instance
// return JSValue representing the Node in QuickJS
JSValue UDF_CreateNode
(
	JSContext *js_ctx,  // JavaScript context
	const Node *node    // pointer to the native FalkorDB Node
);

// register the Node class with a JavaScript runtime
// associates the Node class definition with the given QuickJS runtime
void UDF_RegisterNodeClass
(
	JSRuntime *js_runtime  // JavaScript runtime
);

// register the Node class with a JavaScript context
// makes the Node class available within the provided QuickJS context
void UDF_RegisterNodeProto
(
	JSContext *js_ctx  // JavaScript context
);

