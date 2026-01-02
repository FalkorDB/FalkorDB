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
#include "keyspace_events.h"

// graphContext type as it is registered at Redis
extern RedisModuleType *GraphContextRedisModuleType;

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
		if(RedisModule_ModuleTypeGetType(key) == GraphContextRedisModuleType) {
			deleted = true;
			RedisModule_DeleteKey(key);  // untrack graph & decreases graph ref count
			
			// emit keyspace notification for graph deletion
			KeyspaceEvent_EmitGraphDeleted(ctx, key_name);
			
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

