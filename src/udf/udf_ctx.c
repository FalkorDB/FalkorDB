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
	JSValueConst func;  // javascript UDF function 
	const char *name;   // function name
} UDFFunc ;

// TLS UDF context
typedef struct {
	JSRuntime *js_rt;   // javascript runtime
    JSContext *js_ctx;  // javascript context
	UDF_RepoVersion v;  // UDF repository version
	UDFFunc *funcs;     // list of UDF functions
} UDFCtx ;

// instantiate the thread-local UDFCtx on module load
bool UDFCtx_Init(void) {
	return (pthread_key_create(&_tlsUDFCtx, NULL) == 0);
}

static void _UDFCtx_ClearFuncs
(
	UDFCtx *ctx
) {
	ASSERT (ctx != NULL) ;

	int n = array_len (ctx->funcs) ;
	for (int i = 0; i < n; i++) {
		UDFFunc *f = ctx->funcs + i ;
		JS_FreeValue (ctx->js_ctx, f->func) ;
	}

	array_clear (ctx->funcs) ;
}

// retrieve UDFCtx from TLS
static UDFCtx *_UDFCtx_GetCtx(void) {
	UDFCtx *ctx = pthread_getspecific(_tlsUDFCtx);

	if (ctx == NULL) {
		ctx = rm_calloc (1, sizeof(UDFCtx)) ;

		ctx->funcs = array_new (UDFFunc, 0) ;

		// create js runtime & context
		ctx->js_rt  = UDF_GetJSRuntime() ;
		ctx->js_ctx = UDF_GetExecutionJSContext (ctx->js_rt) ;

		// set context in TLS
		pthread_setspecific (_tlsUDFCtx, ctx) ;

		// populate JS context
		UDF_RepoPopulateJSContext (ctx->js_ctx, &ctx->v) ;
	}

	// validate UDF context version against UDF repository version
	if (unlikely (ctx->v < UDF_RepoGetVersion())) {
		// free registered functions
		_UDFCtx_ClearFuncs (ctx) ;

		// UDF context is outdated, reconstruct JSContext
		JS_FreeContext (ctx->js_ctx) ;  // free outdated js context

		// create js context
		ctx->js_ctx = UDF_GetExecutionJSContext (ctx->js_rt) ;

		UDF_RepoPopulateJSContext (ctx->js_ctx, &ctx->v) ;
	}

	return ctx;
}

// retrive thread's javascript runtime
JSRuntime *UDFCtx_GetJSRuntime(void) {
	UDFCtx *ctx = _UDFCtx_GetCtx () ;

	ASSERT (ctx         != NULL) ;
	ASSERT (ctx->js_rt  != NULL) ;

	return ctx->js_rt ;
}

// retrive thread's javascript context
JSContext *UDFCtx_GetJSContext(void) {
	UDFCtx *ctx = _UDFCtx_GetCtx () ;

	ASSERT (ctx         != NULL) ;
	ASSERT (ctx->js_rt  != NULL) ;
	ASSERT (ctx->js_ctx != NULL) ;

	return ctx->js_ctx ;
}

// register a UDF function with TLS UDF context
void UDFCtx_RegisterFunction
(
	JSValueConst func,     // JS function
	const char *func_name  // function name
) {
	UDFCtx *ctx = _UDFCtx_GetCtx () ;

#ifdef RG_DEBUG
	// make sure function isn't already registered
	int n = array_len (ctx->funcs) ;
	for (int i = 0; i < n; i++) {
		UDFFunc *f = ctx->funcs + i ;
		if (strcmp (f->name, func_name) == 0) {
			ASSERT (false && "duplicated UDF") ; 
		}
	}
#endif

	UDFFunc f = (UDFFunc) {.func = func, .name = (rm_strdup (func_name))} ;
	array_append (ctx->funcs, f) ;
}

// get UDF function
JSValueConst *UDFCtx_GetFunction
(
	const char *func_name  // function to retrieve
) {
	ASSERT (func_name != NULL) ;

	UDFCtx *ctx = _UDFCtx_GetCtx () ;
	int n = array_len (ctx->funcs) ;

	// search for function
	for (int i = 0; i < n; i++) {
		UDFFunc *f = ctx->funcs + i ;
		if (strcmp (f->name, func_name) == 0) {
			return &(f->func) ;
		}
	}

	// couldn't find function
	return NULL ;
}

// free UDF context
void UDFCtx_Free(void) {
	UDFCtx *ctx = _UDFCtx_GetCtx () ;

	if (ctx == NULL) {
		return ;
	}

    JS_FreeContext (ctx->js_ctx) ;
    JS_FreeRuntime (ctx->js_rt) ;

	rm_free (ctx) ;

	// NULL-set the context
	pthread_setspecific (_tlsUDFCtx, NULL) ;
}

