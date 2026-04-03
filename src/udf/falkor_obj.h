/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "utils.h"
#include "quickjs.h"

//------------------------------------------------------------------------------
// UDF Falkor object binding
//------------------------------------------------------------------------------

// register the global falkor object in the given QuickJS context
void UDF_RegisterFalkorObject
(
	JSContext *js_ctx  // the QuickJS context in which to register the object
);

// set the implementation of the `falkor.register` function
void UDF_SetFalkorRegisterImpl
(
	JSContext *js_ctx,              // the QuickJS context
	UDF_JSCtxRegisterFuncMode mode  // the registration mode
);

