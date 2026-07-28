/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../redismodule.h"

// invoked when a command replicated from the master
// (GRAPH.QUERY / GRAPH.PROFILE / GRAPH.EFFECT) fails to apply cleanly on
// this replica
//
// since the master only ever replicates a command after it succeeded
// locally, a failure here means this replica's dataset has diverged from
// the master
//
// logs the divergence, then forces a full resync with the master:
// REPLICAOF NO ONE discards this replica's cached master state (replication
// ID + offset), so the following REPLICAOF back to the same master cannot
// be satisfied with a partial resync (PSYNC CONTINUE) and the master must
// send a FULLRESYNC, which replaces this instance's entire dataset
//
// if the master's address can't be determined, or either REPLICAOF call
// fails, falls back to logging and terminating the process so this replica
// stops serving diverged data
void ReplicationGuard_OnFailure
(
	RedisModuleCtx *ctx,     // redis module context
	const char *graph_name,  // name of the graph that diverged
	const char *cmd_name,    // GRAPH.QUERY / GRAPH.PROFILE / GRAPH.EFFECT
	const char *detail       // human readable failure detail
);
