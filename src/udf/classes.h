/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "node_class.h"
#include "edge_class.h"
#include "path_class.h"
#include "attributes_class.h"

#include "graph_obj.h"
#include "falkor_obj.h"

// initialize all QuickJS classes required by the UDF subsystem
// this should be called once during application startup
// before any QuickJS runtime or context is created
void UDF_InitClasses(void) ;

// register all FalkorDB classes with the given QuickJS runtime
void UDF_RT_RegisterClasses
(
	JSRuntime *js_rt  // the QuickJS runtime in which to register classes
);

// register all FalkorDB classes with the given QuickJS context
void UDF_CTX_RegisterClasses
(
	JSContext *js_ctx  // the QuickJS context in which to register classes.
);

