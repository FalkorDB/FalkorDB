/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"

JSValue js_entity_get_attributes
(
	JSContext *js_ctx,
	JSValueConst this_val
);

