/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../globals.h"
#include "../redismodule.h"
#include "../enterprise_api.h"
#include "../graph/graphcontext.h"

// offloaded-graph-stub type, exported by the enterprise module via its
// shared API - resolved lazily; stays NULL when that module isn't loaded
// (community edition)
static GraphStubType_Get_t GraphStubType_Get = NULL;

// state threaded through the keyspace scan used to pick out stub keys
typedef struct {
	RedisModuleCtx   *ctx;
	RedisModuleType  *stub_type;
	uint64_t         n;
} StubScanCtx;

static void _collect_stub_name
(
	RedisModuleCtx *ctx,
	RedisModuleString *keyname,
	RedisModuleKey *key,
	void *privdata
) {
	StubScanCtx *sctx = privdata;

	if(RedisModule_ModuleTypeGetType(key) != sctx->stub_type) {
		return;
	}

	size_t len;
	const char *name = RedisModule_StringPtrLen(keyname, &len);
	RedisModule_ReplyWithStringBuffer(sctx->ctx, name, len);
	sctx->n++;
}

int Graph_List
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT (ctx != NULL) ;

	if (argc != 1) {
		return RedisModule_WrongArity (ctx) ;
	}

	KeySpaceGraphIterator it ;
	Globals_ScanGraphs (&it) ;
	RedisModule_ReplyWithArray (ctx, REDISMODULE_POSTPONED_LEN) ;

	// reply with each live graph name
	uint64_t     n   = Globals_GraphsCount () ;
	GraphContext *gc = NULL ;

	while ((gc = GraphIterator_Next (&it)) != NULL) {
		const char *name = GraphContext_GetName (gc) ;
		RedisModule_ReplyWithStringBuffer (ctx, name, strlen (name)) ;
		GraphContext_DecreaseRefCount (gc) ;
	}

	// merge in offloaded graphs (stubs), if the enterprise module that
	// owns them is loaded - stubs aren't registered in the global graph
	// registry, so they're only discoverable via a keyspace scan
	if (GraphStubType_Get == NULL) {
		GraphStubType_Get = RedisModule_GetSharedAPI (ctx, "GraphStubType_Get") ;
	}

	if (GraphStubType_Get != NULL) {
		StubScanCtx sctx = {
			.ctx       = ctx,
			.stub_type = GraphStubType_Get (),
			.n         = 0
		};

		RedisModuleScanCursor *cursor = RedisModule_ScanCursorCreate () ;
		while (RedisModule_Scan (ctx, cursor, _collect_stub_name, &sctx)) ;
		RedisModule_ScanCursorDestroy (cursor) ;

		n += sctx.n ;
	}

	RedisModule_ReplySetArrayLength (ctx, n) ;

	return REDISMODULE_OK ;
}

