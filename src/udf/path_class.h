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

// create a JSValue of type Path
void register_path_class
(
	JSRuntime *js_runtime, 
	JSContext *js_ctx
);

