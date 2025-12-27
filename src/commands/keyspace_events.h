/*
 * Copyright FalkorDB Ltd. 2024 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../redismodule.h"
#include "../resultset/resultset_statistics.h"

// Emit Redis keyspace notification for graph operations
// These notifications allow other Redis modules and applications to react to graph changes
// Notifications only fire when Redis keyspace notifications are enabled
// e.g., CONFIG SET notify-keyspace-events AKE

// Emit notifications based on result statistics
void KeyspaceEvent_EmitStats
(
	RedisModuleCtx *ctx,           // Redis module context
	RedisModuleString *key,        // Graph key
	const ResultSetStatistics *stats  // Result statistics
);

// Emit graph deletion notification
void KeyspaceEvent_EmitGraphDeleted
(
	RedisModuleCtx *ctx,    // Redis module context
	RedisModuleString *key  // Graph key
);

// Emit index creation notification
void KeyspaceEvent_EmitIndexCreated
(
	RedisModuleCtx *ctx,    // Redis module context
	RedisModuleString *key  // Graph key
);

// Emit index drop notification
void KeyspaceEvent_EmitIndexDropped
(
	RedisModuleCtx *ctx,    // Redis module context
	RedisModuleString *key  // Graph key
);

// Emit constraint creation notification
void KeyspaceEvent_EmitConstraintCreated
(
	RedisModuleCtx *ctx,    // Redis module context
	RedisModuleString *key  // Graph key
);

// Emit constraint drop notification
void KeyspaceEvent_EmitConstraintDropped
(
	RedisModuleCtx *ctx,    // Redis module context
	RedisModuleString *key  // Graph key
);
