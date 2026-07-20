/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "cmd_context.h"
#include "graph/graph.h"
#include "graph/graphcontext.h"
#include "query_ctx.h"
#include "resultset/resultset.h"
#include "../enterprise_api.h"

// graphContext type as it is registered at Redis
extern RedisModuleType *GraphContextRedisModuleType;

// offloaded-graph-stub type, exported by the enterprise module via its
// shared API - resolved lazily; stays NULL when that module isn't loaded
// (community edition)
static GraphStubType_Get_t GraphStubType_Get = NULL;

// delete graph, removing the key from Redis and
// freeing every resource allocated by the graph
int Graph_Delete
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	if(argc != 2) {
		return RedisModule_WrongArity(ctx);
	}

	int res = REDISMODULE_OK;
	bool deleted = false;
	RedisModuleString *key_name = argv[1];

	// remove graph from keyspace
	RedisModuleKey *key = RedisModule_OpenKey(ctx, key_name, REDISMODULE_WRITE);
	if(key != NULL) {
		RedisModuleType *type = RedisModule_ModuleTypeGetType(key);

		if(GraphStubType_Get == NULL) {
			GraphStubType_Get = RedisModule_GetSharedAPI(ctx, "GraphStubType_Get");
		}

		// a stub (offloaded graph) can be deleted outright, without first
		// loading it - it is not registered in the global graph registry,
		// so no ref count / untracking is involved
		if(type == GraphContextRedisModuleType ||
		   (GraphStubType_Get != NULL && type == GraphStubType_Get())) {
			deleted = true;
			RedisModule_DeleteKey(key);  // untrack graph & decreases graph ref count
			RedisModule_ReplyWithSimpleString(ctx, "OK");
			// delete commands should always modify slaves
			RedisModule_ReplicateVerbatim(ctx);
		}
		RedisModule_CloseKey(key);  // close key handle
	}

	// unable to delete graph
	if(!deleted) {
		res = REDISMODULE_ERR;
		RedisModule_ReplyWithError(ctx, "ERR Invalid graph operation on empty key");
	}

	return res;
}

