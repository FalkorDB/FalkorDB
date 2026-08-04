/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../redismodule.h"

// invoked when a command replicated from the master, or replayed from the
// local AOF/RDB at startup, (GRAPH.QUERY / GRAPH.PROFILE / GRAPH.EFFECT)
// fails to apply cleanly on this instance
//
// since a command is only ever persisted (to the replication stream or to
// the AOF) after it succeeded locally, a failure here means this instance's
// dataset no longer matches the state that command was recorded against
//
// if we're currently loading from AOF/RDB (RedisModule_GetContextFlags()
// has REDISMODULE_CTX_FLAGS_LOADING set), a resync can't fix anything: the
// divergence is baked into this instance's own local persisted state, not
// the live replication link, and attempting REPLICAOF mid-load is both
// premature (the replication subsystem isn't running yet) and pointless
// (the rest of the file keeps replaying regardless of what's scheduled
// here). In that case this logs the divergence and terminates immediately
//
// otherwise, logs the divergence, then forces a full resync with the
// master: REPLICAOF NO ONE discards this replica's cached master state
// (replication ID + offset), so the following REPLICAOF back to the same
// master cannot be satisfied with a partial resync (PSYNC CONTINUE) and
// the master must send a FULLRESYNC, which replaces this instance's entire
// dataset
//
// if the master's address can't be determined, or either REPLICAOF call
// fails, falls back to logging and terminating the process so this replica
// stops serving diverged data
void DivergenceGuard_OnFailure
(
	RedisModuleCtx *ctx,     // redis module context
	const char *graph_name,  // name of the graph that diverged
	const char *cmd_name,    // GRAPH.QUERY / GRAPH.PROFILE / GRAPH.EFFECT
	const char *detail       // human readable failure detail
);
