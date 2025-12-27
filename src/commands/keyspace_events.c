/*
 * Copyright FalkorDB Ltd. 2024 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "keyspace_events.h"
#include "RG.h"

// Helper function to emit a keyspace event
static inline void _emit_event
(
	RedisModuleCtx *ctx,
	const char *event,
	RedisModuleString *key
) {
	ASSERT(ctx   != NULL);
	ASSERT(event != NULL);
	ASSERT(key   != NULL);

	// Emit keyspace notification
	// REDISMODULE_NOTIFY_MODULE corresponds to the 'd' flag in notify-keyspace-events
	RedisModule_NotifyKeyspaceEvent(ctx, REDISMODULE_NOTIFY_MODULE, event, key);
}

void KeyspaceEvent_EmitStats
(
	RedisModuleCtx *ctx,
	RedisModuleString *key,
	const ResultSetStatistics *stats
) {
	ASSERT(ctx   != NULL);
	ASSERT(key   != NULL);
	ASSERT(stats != NULL);

	// Check if any modifications occurred
	if(!ResultSetStat_IndicateModification(stats)) {
		return;
	}

	// Handle index operations
	if(stats->indices_created > 0) {
		_emit_event(ctx, "graph.index.create", key);
		return;  // Index operations don't combine with other operations
	}

	if(stats->indices_deleted > 0) {
		_emit_event(ctx, "graph.index.drop", key);
		return;  // Index operations don't combine with other operations
	}

	// Emit general graph.query event for any write operation
	_emit_event(ctx, "graph.query", key);

	// Emit specific events based on operation types
	if(stats->nodes_created > 0) {
		_emit_event(ctx, "graph.node.create", key);
	}

	if(stats->nodes_deleted > 0) {
		_emit_event(ctx, "graph.node.delete", key);
	}

	if(stats->relationships_created > 0) {
		_emit_event(ctx, "graph.edge.create", key);
	}

	if(stats->relationships_deleted > 0) {
		_emit_event(ctx, "graph.edge.delete", key);
	}

	// Property updates indicate node/edge updates
	if(stats->properties_set > 0 || stats->properties_removed > 0) {
		// If properties were set/removed, it's an update operation
		// We can't distinguish between node and edge property updates from stats alone
		// So we emit both events when there are any property modifications
		// This is a conservative approach that may emit false positives
		_emit_event(ctx, "graph.node.update", key);
		_emit_event(ctx, "graph.edge.update", key);
	}

	// Label operations also indicate updates
	if(stats->labels_added > 0 || stats->labels_removed > 0) {
		_emit_event(ctx, "graph.node.update", key);
	}
}

void KeyspaceEvent_EmitGraphDeleted
(
	RedisModuleCtx *ctx,
	RedisModuleString *key
) {
	ASSERT(ctx != NULL);
	ASSERT(key != NULL);

	_emit_event(ctx, "graph.delete", key);
}

void KeyspaceEvent_EmitIndexCreated
(
	RedisModuleCtx *ctx,
	RedisModuleString *key
) {
	ASSERT(ctx != NULL);
	ASSERT(key != NULL);

	_emit_event(ctx, "graph.index.create", key);
}

void KeyspaceEvent_EmitIndexDropped
(
	RedisModuleCtx *ctx,
	RedisModuleString *key
) {
	ASSERT(ctx != NULL);
	ASSERT(key != NULL);

	_emit_event(ctx, "graph.index.drop", key);
}

void KeyspaceEvent_EmitConstraintCreated
(
	RedisModuleCtx *ctx,
	RedisModuleString *key
) {
	ASSERT(ctx != NULL);
	ASSERT(key != NULL);

	_emit_event(ctx, "graph.constraint.create", key);
}

void KeyspaceEvent_EmitConstraintDropped
(
	RedisModuleCtx *ctx,
	RedisModuleString *key
) {
	ASSERT(ctx != NULL);
	ASSERT(key != NULL);

	_emit_event(ctx, "graph.constraint.drop", key);
}
