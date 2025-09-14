/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"

// retrieve the attributes of a given entity
// returns a JSValue containing the entity's attributes
JSValue UDF_EntityGetAttributes
(
	JSContext *js_ctx,      // the QuickJS context
	JSValueConst this_val   // the JavaScript object representing the entity
);

// register the `Attributes` class with the provided QuickJS runtime
void UDF_RegisterAttributesClass
(
	JSRuntime *js_runtime  // the QuickJS runtime in which to register the class
);

