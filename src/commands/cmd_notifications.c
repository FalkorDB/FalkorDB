/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "cmd_notifications.h"
#include "../redismodule.h"
#include "../util/rmalloc.h"
#include <string.h>

// fire a keyspace notification on the main thread
// graph_name is an rm_strdup'd string that will be freed after use
static void _NotifyKeyspaceEvent
(
	char *graph_name,
	const char *event
) {
	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
	RedisModuleString *key = RedisModule_CreateString(ctx,
			graph_name, strlen(graph_name));
	RedisModule_NotifyKeyspaceEvent(ctx,
			REDISMODULE_NOTIFY_MODULE,
			event,
			key);
	RedisModule_FreeString(ctx, key);
	RedisModule_FreeThreadSafeContext(ctx);
	rm_free(graph_name);
}

static void _GraphModifiedCB(void *user_data) {
	_NotifyKeyspaceEvent(user_data, "graph.modified");
}

static void _GraphCopyToCB(void *user_data) {
	_NotifyKeyspaceEvent(user_data, "graph.copy_to");
}

void Notify_Keyspace_GraphModified
(
	const char *graph_name
) {
	RedisModule_EventLoopAddOneShot(_GraphModifiedCB, rm_strdup(graph_name));
}

void Notify_Keyspace_GraphCopyTo
(
	const char *graph_name
) {
	RedisModule_EventLoopAddOneShot(_GraphCopyToCB, rm_strdup(graph_name));
}
