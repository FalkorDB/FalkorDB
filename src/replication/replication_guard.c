/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "replication_guard.h"
#include "../util/rmalloc.h"

#include <stdlib.h>

// data handed off to the deferred resync timer
typedef struct {
	char *graph_name;
	char *cmd_name;
} _ResyncCtx;

static _ResyncCtx *_ResyncCtx_New
(
	const char *graph_name,
	const char *cmd_name
) {
	_ResyncCtx *rc = rm_malloc (sizeof (_ResyncCtx)) ;

	rc->graph_name = rm_strdup (graph_name) ;
	rc->cmd_name   = rm_strdup (cmd_name) ;

	return rc ;
}

static void _ResyncCtx_Free
(
	_ResyncCtx *rc
) {
	rm_free (rc->graph_name) ;
	rm_free (rc->cmd_name) ;
	rm_free (rc) ;
}

// determine the current master's host and port
// on success populates *host (caller frees with rm_free) and *port
static bool _GetMasterAddress
(
	RedisModuleCtx *ctx,
	char **host,
	long long *port
) {
	RedisModuleServerInfoData *info =
		RedisModule_GetServerInfo (ctx, "Replication") ;
	if (info == NULL) {
		return false ;
	}

	const char *master_host =
		RedisModule_ServerInfoGetFieldC (info, "master_host") ;

	int err = REDISMODULE_OK ;
	long long master_port =
		RedisModule_ServerInfoGetFieldSigned (info, "master_port", &err) ;

	bool ok = (master_host != NULL && err == REDISMODULE_OK) ;
	if (ok) {
		*host = rm_strdup (master_host) ;
		*port = master_port ;
	}

	RedisModule_FreeServerInfo (ctx, info) ;

	return ok ;
}

// issue REPLICAOF <arg1> <arg2>, returns true on success
static bool _ReplicaOf
(
	RedisModuleCtx *ctx,
	const char *arg1,
	const char *arg2
) {
	RedisModuleCallReply *reply =
		RedisModule_Call (ctx, "REPLICAOF", "cc", arg1, arg2) ;

	bool ok = (reply != NULL &&
			RedisModule_CallReplyType (reply) != REDISMODULE_REPLY_ERROR) ;

	if (reply != NULL) {
		RedisModule_FreeCallReply (reply) ;
	}

	return ok ;
}

// runs on the Redis main thread's event loop, strictly after the command
// that detected the divergence has fully returned
//
// this can't run nested inside that command's call stack: REPLICAOF NO ONE
// disconnects (and briefly caches, then frees) server.master, which is the
// very client object whose execution is still unwinding on the stack above
// us if we're called synchronously - deferring via a 0ms timer guarantees
// we only run once that stack is gone
static void _ForceFullResync
(
	RedisModuleCtx *ctx,
	void *data
) {
	_ResyncCtx *rc = data ;

	char     *host = NULL ;
	long long port = 0 ;

	if (!_GetMasterAddress (ctx, &host, &port)) {
		RedisModule_Log (ctx, "warning",
				"Unable to determine master address for graph '%s', "
				"shutting down instead of forcing a full resync",
				rc->graph_name) ;
		_ResyncCtx_Free (rc) ;
		exit (1) ;
	}

	char port_str[32] ;
	snprintf (port_str, sizeof (port_str), "%lld", port) ;

	//--------------------------------------------------------------------------
	// discard cached master state, so the reconnect below can't be
	// satisfied with a partial resync (PSYNC CONTINUE) against the same,
	// now-diverged, dataset
	//--------------------------------------------------------------------------

	if (!_ReplicaOf (ctx, "NO", "ONE")) {
		RedisModule_Log (ctx, "warning",
				"REPLICAOF NO ONE failed for graph '%s', shutting down "
				"instead of forcing a full resync", rc->graph_name) ;
		rm_free (host) ;
		_ResyncCtx_Free (rc) ;
		exit (1) ;
	}

	//--------------------------------------------------------------------------
	// reattach to the same master; cached state was just discarded so this
	// reconnect is guaranteed a FULLRESYNC, not a PSYNC CONTINUE
	//--------------------------------------------------------------------------

	if (!_ReplicaOf (ctx, host, port_str)) {
		RedisModule_Log (ctx, "warning",
				"Failed to reattach to master %s:%s for graph '%s', "
				"shutting down", host, port_str, rc->graph_name) ;
		rm_free (host) ;
		_ResyncCtx_Free (rc) ;
		exit (1) ;
	}

	RedisModule_Log (ctx, "notice",
			"Forced full resync with master %s:%s initiated after "
			"divergence detected on graph '%s'",
			host, port_str, rc->graph_name) ;

	rm_free (host) ;
	_ResyncCtx_Free (rc) ;
}

void ReplicationGuard_OnFailure
(
	RedisModuleCtx *ctx,
	const char *graph_name,
	const char *cmd_name,
	const char *detail
) {
	RedisModule_Log (ctx, "warning",
			"Replica diverged from master applying %s on graph '%s': %s. "
			"Scheduling a forced full resync with master.",
			cmd_name, graph_name, detail) ;

	// defer the actual REPLICAOF sequence, see _ForceFullResync
	_ResyncCtx *rc = _ResyncCtx_New (graph_name, cmd_name) ;
	RedisModule_CreateTimer (ctx, 0, _ForceFullResync, rc) ;
}
