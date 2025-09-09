/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "../graph/entities/node.h"

// create a JSValue of type Node
JSValue js_create_node
(
	JSContext *js_ctx,
	const Node *node
);

// register the node class with the js-runtime
void rt_register_node_class
(
	JSRuntime *js_runtime
);

// register the node class with the js-context
void ctx_register_node_class
(
	JSContext *js_ctx
);

