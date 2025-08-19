/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "udf_ctx.h"
#include "classes.h"
#include "../util/rmalloc.h"
#include <pthread.h>

#include <pthread.h>

static pthread_key_t _tlsUDFCtx;  // thread local storage UDF context key

typedef struct {
	JSRuntime *js_rt;   // javascript runtime
    JSContext *js_ctx;  // javascript context
} UDFCtx ;

// instantiate the thread-local UDFCtx on module load
bool UDFCtx_Init(void) {
	return (pthread_key_create(&_tlsUDFCtx, NULL) == 0);
}

// retrieve UDFCtx from TLS
static inline UDFCtx *_UDFCtx_GetCtx(void) {
	UDFCtx *ctx = pthread_getspecific(_tlsUDFCtx);

	if (ctx == NULL) {
		ctx = rm_calloc (1, sizeof(UDFCtx)) ;

		// create js runtime
		ctx->js_rt = JS_NewRuntime() ;
		JS_SetMaxStackSize (ctx->js_rt, 1024 * 1024) ; // 1 MB stack limit

		// create js context
		ctx->js_ctx = JS_NewContext(ctx->js_rt) ;

		// register all classes
		UDF_RegisterClasses (ctx->js_rt, ctx->js_ctx) ;

		pthread_setspecific (_tlsUDFCtx, ctx) ;
	}

	return ctx;
}

// retrive thread's javascript runtime
JSRuntime *UDFCtx_GetJSRuntime(void) {
	UDFCtx *ctx = _UDFCtx_GetCtx () ;

	ASSERT (ctx        != NULL) ;
	ASSERT (ctx->js_rt != NULL) ;

	return ctx->js_rt ;
}

// retrive thread's javascript context
JSContext *UDFCtx_GetJSContext(void) {
	UDFCtx *ctx = _UDFCtx_GetCtx () ;

	ASSERT (ctx         != NULL) ;
	ASSERT (ctx->js_ctx != NULL) ;

	return ctx->js_ctx ;
}

// free UDF context
void UDFCtx_Free(void) {
	UDFCtx *ctx = _UDFCtx_GetCtx() ;

	if (ctx == NULL) {
		return ;
	}

    JS_FreeContext (ctx->js_ctx) ;
    JS_FreeRuntime (ctx->js_rt) ;

	rm_free (ctx) ;

	// NULL-set the context
	pthread_setspecific (_tlsUDFCtx, NULL) ;
}

