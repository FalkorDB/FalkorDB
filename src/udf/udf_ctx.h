/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"

// instantiate the thread-local UDFCtx on module load
bool UDFCtx_Init(void);

// retrive thread's javascript context
JSContext *UDFCtx_GetJSContext(void) ;

// register a UDF function with TLS UDF context
void UDFCtx_RegisterFunction
(
	JSValueConst func,     // JS function
	const char *func_name  // function name
);

// get UDF function
JSValueConst *UDFCtx_GetFunction
(
	const char *func_name  // function to retrieve
);

// free UDF context
void UDFCtx_Free(void) ;

