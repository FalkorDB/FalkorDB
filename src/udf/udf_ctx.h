/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"

// instantiate the thread-local UDFCtx on module load
bool UDFCtx_Init(void);

// retrive thread's javascript runtime
JSRuntime *UDFCtx_GetJSRuntime(void) ;

// retrive thread's javascript context
JSContext *UDFCtx_GetJSContext(void) ;

// free UDF context
void UDFCtx_Free(void) ;

