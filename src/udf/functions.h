/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"

typedef enum {
	UDF_FUNC_REG_MODE_VALIDATE,  // only validate UDF register functions
	UDF_FUNC_REG_MODE_LOCAL,     // register UDF functions with TLS
	UDF_FUNC_REG_MODE_GLOBAL     // register UDF functions in global functions repository
} UDF_JSCtxRegisterFuncMode ;

// register proxy functions
void UDF_RegisterFunctions
(
	JSContext *js_ctx,              // javascript context
	UDF_JSCtxRegisterFuncMode mode  // type of 'register' function
);

