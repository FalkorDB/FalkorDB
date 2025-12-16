/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "udf_ctx.h"
#include "classes.h"
#include "repository.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"

#include <pthread.h>

static pthread_key_t _tlsUDFCtx;  // thread local storage UDF context key

// UDF function desc
typedef struct {
	char *name;         // function name
	JSValueConst func;  // javascript UDF function 
} UDFFunc ;

// UDF library desc
typedef struct {
	const char *name;  // library name
	UDFFunc *funcs;    // library's functions
} UDFLib ;

// TLS UDF context
typedef struct {
	JSRuntime *js_rt;   // javascript runtime
    JSContext *js_ctx;  // javascript context
	UDF_RepoVersion v;  // UDF repository version
	UDFLib *libs;       // list of UDF libraries
} UDFCtx ;

// get library
static UDFLib *_UDFCtx_GetLib
(
	UDFCtx *ctx,          // UDF context
	const char *lib_name  // library to get
) {
	ASSERT (ctx      != NULL) ;
	ASSERT (lib_name != NULL) ;

	int16_t n = array_len (ctx->libs) ;
	for (int16_t i = 0; i < n; i++) {
		UDFLib *l = ctx->libs + i ;
		if (strcmp (l->name, lib_name) == 0) {
			return l ;
		}
	}

	return NULL ;
}

// get function
static UDFFunc *_UDFCtx_GetFunc
(
	const UDFLib *lib,     // lib to search function in
	const char *func_name  // function name to retrieve
) {
	ASSERT (lib       != NULL) ;
	ASSERT (func_name != NULL) ;

	int16_t n = array_len (lib->funcs) ;
	for (int16_t i = 0; i < n; i++) {
		UDFFunc *f = lib->funcs + i ;
		if (strcmp (f->name, func_name) == 0) {
			return f ;
		}
	}

	return NULL ;
}

// clear libraries
static void _UDFCtx_ClearLibs
(
	UDFCtx *ctx  // UDF context
) {
	ASSERT (ctx != NULL) ;

	int16_t n = array_len (ctx->libs) ;
	for (int16_t i = 0; i < n; i++) {
		UDFLib *l = ctx->libs + i ;
		UDFFunc *funcs = l->funcs ;

		int16_t m = array_len (funcs) ;
		for (int16_t j = 0; j < m; j++) {
			UDFFunc *f = funcs + j ;
			JS_FreeValue (ctx->js_ctx, f->func) ;
			rm_free (f->name) ;
		}

		array_free (l->funcs) ;
	}

	array_clear (ctx->libs) ;
}

// retrieve UDFCtx from TLS
static UDFCtx *_UDFCtx_GetCtx(void) {
	UDFCtx *ctx = pthread_getspecific (_tlsUDFCtx) ;

	if (unlikely (ctx == NULL)) {
		ctx = rm_calloc (1, sizeof(UDFCtx)) ;

		ctx->libs = array_new (UDFLib, 0) ;

		// create js runtime & context
		ctx->js_rt  = UDF_GetJSRuntime() ;
		ctx->js_ctx = UDF_GetExecutionJSContext (ctx->js_rt) ;

		// set context in TLS
		pthread_setspecific (_tlsUDFCtx, ctx) ;

		// populate JS context
		UDF_RepoPopulateJSContext (ctx->js_ctx, &ctx->v) ;
	}

	return ctx;
}

// TLS destructor callback
static void UDFCtx_FreeTLSData
(
	void *data
) {
	if (data == NULL) {
		return ;
	}

	UDFCtx *ctx = (UDFCtx*) data ;

	//--------------------------------------------------------------------------
	// free UDF context
	//--------------------------------------------------------------------------

	_UDFCtx_ClearLibs (ctx) ;

	if (ctx->js_ctx != NULL) {
		JS_FreeContext (ctx->js_ctx) ;
	}

	if (ctx->js_rt) {
		JS_FreeRuntime (ctx->js_rt) ;
	}

	rm_free (ctx) ;
}

// instantiate the thread-local UDFCtx on module load
bool UDFCtx_Init(void) {
	return (pthread_key_create (&_tlsUDFCtx, UDFCtx_FreeTLSData) == 0) ;
}

// get number of libraries in TLS UDF context
uint16_t UDFCtx_LibCount(void) {
	UDFCtx *ctx = _UDFCtx_GetCtx () ;
	return array_len (ctx->libs) ;
}

// retrive thread's javascript runtime
JSRuntime *UDFCtx_GetJSRuntime(void) {
	UDFCtx *ctx = _UDFCtx_GetCtx () ;

	ASSERT (ctx        != NULL) ;
	ASSERT (ctx->v     == UDF_RepoGetVersion ()) ;
	ASSERT (ctx->js_rt != NULL) ;

	return ctx->js_rt ;
}

// retrive thread's javascript context
JSContext *UDFCtx_GetJSContext(void) {
	UDFCtx *ctx = _UDFCtx_GetCtx () ;

	ASSERT (ctx         != NULL) ;
	ASSERT (ctx->v      == UDF_RepoGetVersion ()) ;
	ASSERT (ctx->js_rt  != NULL) ;
	ASSERT (ctx->js_ctx != NULL) ;

	return ctx->js_ctx ;
}

// make sure the UDF context is up to date
void UDFCtx_Update(void) {
	UDFCtx *ctx = _UDFCtx_GetCtx () ;

	ASSERT (ctx         != NULL) ;
	ASSERT (ctx->js_rt  != NULL) ;
	ASSERT (ctx->js_ctx != NULL) ;

	// validate UDF context version against UDF repository version
	if (unlikely (ctx->v < UDF_RepoGetVersion ())) {
		// free registered functions
		_UDFCtx_ClearLibs (ctx) ;

		// UDF context is outdated, reconstruct JSContext
		JS_FreeContext (ctx->js_ctx) ;  // free outdated js context

		// create js context
		ctx->js_ctx = UDF_GetExecutionJSContext (ctx->js_rt) ;

		UDF_RepoPopulateJSContext (ctx->js_ctx, &ctx->v) ;
	}
}

// register a new UDF library with TLS UDF context
void UDFCtx_RegisterLibrary
(
	const char *lib_name  // library name
) {
	ASSERT (lib_name != NULL) ;
	UDFCtx *ctx = _UDFCtx_GetCtx () ;

	// make sure lib doesn't already exists
	ASSERT (_UDFCtx_GetLib (ctx, lib_name) == NULL) ;

	// add new library
	UDFLib l = {.name = lib_name, .funcs = array_new (UDFFunc, 0)} ;
	array_append (ctx->libs, l) ;
}

// register a UDF function with TLS UDF context
void UDFCtx_RegisterFunction
(
	JSValueConst func,     // JS function
	const char *func_name  // function name
) {
	UDFCtx *ctx = _UDFCtx_GetCtx () ;

	// functions are added to the last library
	int n = array_len(ctx->libs) ;
	ASSERT (n > 0) ;

	UDFLib  *l = ctx->libs + (n - 1) ;
	UDFFunc *f = _UDFCtx_GetFunc (l, func_name) ;

	if (unlikely (f != NULL)) {
		// replace existing function pointer
		JS_FreeValue (ctx->js_ctx, f->func) ;
		f->func = func ;
	} else {
		// add a new function
		UDFFunc _f = (UDFFunc) {.func = func, .name = (rm_strdup (func_name))} ;
		array_append (l->funcs, _f) ;
	}
}

// get UDF function
JSValueConst *UDFCtx_GetFunction
(
	const char *lib_name,  // lib to search function in
	const char *func_name  // function to retrieve
) {
	ASSERT (lib_name  != NULL) ;
	ASSERT (func_name != NULL) ;

	UDFCtx *ctx = _UDFCtx_GetCtx () ;
	ASSERT (ctx != NULL) ;

	UDFLib *lib = _UDFCtx_GetLib (ctx, lib_name) ;
	if (lib == NULL) {
		return NULL ;
	}

	UDFFunc *f = _UDFCtx_GetFunc (lib, func_name) ;
	return (f != NULL) ? &f->func : NULL ;
}

