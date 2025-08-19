/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"

static JSClassID js_node_class_id;        // JS node class
static JSClassID js_edge_class_id;        // JS edge class
static JSClassID js_attributes_class_id;  // JS attributes class

// init classes
void UDF_InitClasses(void) ;

// register all classes
void UDF_RegisterClasses
(
	JSRuntime *js_runtime,
	JSContext *js_ctx
);

