/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "errors.h"
#include "util/arr.h"
#include "query_ctx.h"
#include "util/rax_extensions.h"
#include "../deps/libcypher-parser/lib/src/operators.h"

pthread_key_t _tlsErrorCtx; // Error-handling context held in thread-local storage.

//------------------------------------------------------------------------------
// Error context initialization
//------------------------------------------------------------------------------

bool ErrorCtx_Init(void) {
	int res = pthread_key_create(&_tlsErrorCtx, NULL);
	ASSERT(res == 0);

	return (res == 0);
}

ErrorCtx *ErrorCtx_Get(void) {
	ErrorCtx *ctx = pthread_getspecific(_tlsErrorCtx);

	if(ctx == NULL) {
		ctx = rm_calloc(1, sizeof(ErrorCtx));
		int res = pthread_setspecific(_tlsErrorCtx, ctx);
		ASSERT(res == 0);
	}

	return ctx;
}

void ErrorCtx_Clear(void) {
	ErrorCtx *ctx = ErrorCtx_Get();
	ASSERT(ctx != NULL);

	if(ctx->error != NULL) {
		free(ctx->error);
		ctx->error = NULL;
	}

	if(ctx->breakpoint != NULL) {
		rm_free(ctx->breakpoint);
		ctx->breakpoint = NULL;
	}
}

//------------------------------------------------------------------------------
// Error setting and emitting
//------------------------------------------------------------------------------

static void _ErrorCtx_SetError(const char *err_fmt, va_list args) {
	ErrorCtx *ctx = ErrorCtx_Get();
	ASSERT(ctx != NULL);

	// An error is already set - free it
	if(ctx->error != NULL) free(ctx->error);

	int rc __attribute__((unused));
	rc = vasprintf(&ctx->error, err_fmt, args);
}

void ErrorCtx_SetError(const char *err_fmt, ...) {
	// Set the new error
	va_list valist;
	va_start(valist, err_fmt);
	_ErrorCtx_SetError(err_fmt, valist);
	va_end(valist);
}

/* An error was encountered during evaluation, and has already been set in the ErrorCtx.
 * If an exception handler has been set, exit this routine and return to
 * the point on the stack where the handler was instantiated. */
void ErrorCtx_RaiseRuntimeException(const char *err_fmt, ...) {
	ErrorCtx *ctx = ErrorCtx_Get();
	ASSERT(ctx != NULL);

	// set error if specified
	if(err_fmt != NULL) {
		va_list valist;
		va_start(valist, err_fmt);
		_ErrorCtx_SetError(err_fmt, valist);
		va_end(valist);
	}

	jmp_buf *env = ctx->breakpoint;
	// If the exception handler hasn't been set, this function returns to the caller,
	// which will manage its own freeing and error reporting.
	if(env != NULL) longjmp(*env, 1);
}

// Reply to caller with error
void ErrorCtx_EmitException(void) {
	ErrorCtx *ctx = ErrorCtx_Get();
	ASSERT(ctx != NULL);

	if(ctx->error != NULL) {
		bolt_client_t *bolt_client = QueryCtx_GetBoltClient();
		if(bolt_client != NULL) {
			bolt_client_reply_for(bolt_client, BST_RUN, BST_FAILURE, 1);
			bolt_reply_map(bolt_client, 2);
			bolt_reply_string(bolt_client, "code", 4);
			bolt_reply_string(bolt_client, "SyntaxError", 11);
			bolt_reply_string(bolt_client, "message", 7);
			bolt_reply_string(bolt_client, ctx->error, strlen(ctx->error));
			bolt_client_end_message(bolt_client);
			bolt_client_finish_write(bolt_client);
		} else {
			RedisModuleCtx *rm_ctx = QueryCtx_GetRedisModuleCtx();
			RedisModule_ReplyWithError(rm_ctx, ctx->error);
		}
	}

	// clear error context once error emitted
	ErrorCtx_Clear();
}

// Returns true if error is set
inline bool ErrorCtx_EncounteredError(void) {
	ErrorCtx *ctx = ErrorCtx_Get();
	ASSERT(ctx != NULL);

	return ctx->error != NULL;
}

//------------------------------------------------------------------------------
// Specific error scenarios
//------------------------------------------------------------------------------

void Error_InvalidFilterPlacement(rax *entitiesRax) {
	ASSERT(entitiesRax != NULL);

	// Something is wrong - could not find a matching op where all references are solved.
	raxIterator it;
	raxStart(&it, entitiesRax);
	// Retrieve the first key in the rax.
	raxSeek(&it, "^", NULL, 0);
	raxNext(&it);
	// Build invalid entity string on the stack to add null terminator.
	char invalid_entity[it.key_len + 1];
	memcpy(invalid_entity, it.key, it.key_len);
	invalid_entity[it.key_len] = 0;
	// Emit compile-time error.
	ErrorCtx_SetError(EMSG_UNABLE_TO_RESOLVE_FILTER_ALIAS, invalid_entity);
	raxFree(entitiesRax);
}

void Error_SITypeMismatch
(
	SIType actual,
	SIType expected
) {
	size_t bufferLen = MULTIPLE_TYPE_STRING_BUFFER_SIZE;
	char buf[bufferLen];

	SIType_ToMultipleTypeString(expected, buf, bufferLen);
	ErrorCtx_SetError(EMSG_TYPE_MISMATCH, buf, SIType_ToString(actual));
}

void Error_UnsupportedASTNodeType(const cypher_astnode_t *node) {
	ASSERT(node != NULL);

	cypher_astnode_type_t type = cypher_astnode_type(node);
	const char *type_str = cypher_astnode_typestr(type);
	ErrorCtx_SetError(EMSG_FALKORDB_SUPPORT, type_str);
}

void Error_UnsupportedASTOperator(const cypher_operator_t *op) {
	ASSERT(op != NULL);

	ErrorCtx_SetError(EMSG_FALKORDB_SUPPORT, op->str);
}

inline void Error_InvalidPropertyValue(void) {
	ErrorCtx_SetError(EMSG_INVALID_PROPERTY_VALUE);
}

void Error_DivisionByZero(void) {
	ErrorCtx_SetError(EMSG_DIVISION_BY_ZERO);
}

