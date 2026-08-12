/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"

// instantiate the thread-local UDFCtx on module load
bool UDFCtx_Init(void);

// get number of libraries in TLS UDF context
uint16_t UDFCtx_LibCount(void);

// retrive thread's javascript runtime
JSRuntime *UDFCtx_GetJSRuntime(void) ;

// retrive thread's javascript context
JSContext *UDFCtx_GetJSContext(void) ;

// make sure the UDF context is up to date
void UDFCtx_Update(void) ;

// register a new UDF library with TLS UDF context
void UDFCtx_RegisterLibrary
(
	const char *lib_name  // library name
);

// register a UDF function with TLS UDF context
void UDFCtx_RegisterFunction
(
	JSValueConst func,     // JS function
	const char *func_name  // function name
);

// get UDF function
JSValueConst *UDFCtx_GetFunction
(
	const char *lib_name,  // lib to search function in
	const char *func_name  // function to retrieve
);

// free UDF context
// records why `lib_name` failed to load; takes ownership of `err`
void UDFCtx_SetLibraryError
(
	const char *lib_name,  // library that failed
	char *err              // reason, ownership transferred
) ;

// returns why `lib_name` failed to load into this thread's context, or NULL
const char *UDFCtx_GetLoadError
(
	const char *lib_name  // library the caller was looking up
) ;

void UDFCtx_Free(void) ;

