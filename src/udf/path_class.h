/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "../datatypes/path/path.h"

// create a JSValue of type Path
JSValue js_create_path
(
	JSContext *js_ctx,
	const Path *path
);

// register the path class with the js-runtime
void rt_register_path_class
(
	JSRuntime *js_runtime
);

// register the path class with the js-context
void ctx_register_path_class
(
	JSContext *js_ctx
);

