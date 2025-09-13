/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "node_class.h"
#include "edge_class.h"
#include "path_class.h"
#include "falkor_obj.h"
#include "attributes_class.h"

// init classes
void UDF_InitClasses(void) ;

// register all classes
void UDF_RT_RegisterClasses
(
	JSRuntime *js_runtime
);

// register all classes
void UDF_CTX_RegisterClasses
(
	JSContext *js_ctx
);

