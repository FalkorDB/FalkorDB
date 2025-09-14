/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"

//------------------------------------------------------------------------------
// UDF Falkor object binding
//------------------------------------------------------------------------------

// modes for registering UDF functions within a QuickJS context
typedef enum {
	UDF_FUNC_REG_MODE_VALIDATE,  // only validate UDF register functions
	UDF_FUNC_REG_MODE_LOCAL,     // register UDF functions with TLS
	UDF_FUNC_REG_MODE_GLOBAL     // register UDF functions in global functions repository
} UDF_JSCtxRegisterFuncMode ;

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

