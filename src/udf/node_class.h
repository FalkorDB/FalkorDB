/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "../graph/entities/node.h"

// create a JSValue of type Node
static JSValue js_create_node
(
	JSContext *js_ctx,
	const Node *node
);

