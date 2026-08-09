/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../query_ctx.h"
#include "../errors/errors.h"
#include "../effects/effects.h"
#include "../graph/graphcontext_retrieve.h"
#include "../replication/divergence_guard.h"

// GRAPH.EFFECT command handler
int Graph_Effect
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command arguments
	int argc                   // number of arguments
) {
	// clear any stale error left in this thread's TLS by a
	// previously executed command
	ErrorCtx_Clear () ;

	// GRAPH.EFFECT <key> <effects>
	if (argc != 3) {
		return RedisModule_WrongArity (ctx) ;
	}

	// get graph context - never fails due to mere contention with another
	// load (see GraphContext_RetrieveOrForce); reaching gc == NULL means a
	// genuine, unrecoverable failure: a missing/corrupt dump, OOM, or a
	// concurrent GRAPH.OFFLOAD of this graph still in flight (the one case
	// that cannot be bypassed, since there is nothing yet to load)
	GraphContext *gc = NULL ;
	GraphContext_RetrieveOrForce (ctx, argv[1], false, true, &gc) ;

	if (gc == NULL) {
		// applying an effect only happens while replicating a write from
		// the master; silently dropping it here would let this replica's
		// data diverge from the master without either side noticing.
		// Crashing is the safer failure mode: a replication resync
		// (partial or full) on restart brings this replica back to a
		// correct, consistent state. Use RELEASE_ASSERT, not ASSERT - the
		// latter is a no-op in release builds, which would fall through to
		// an undiagnosable NULL-deref crash a few lines below instead.
		RedisModule_Log (ctx, REDISMODULE_LOGLEVEL_WARNING,
				"GRAPH.EFFECT: failed to retrieve graph: %s - crashing to "
				"force a replication resync rather than risk silent "
				"master/replica divergence",
				RedisModule_StringPtrLen (argv[1], NULL)) ;
	}

	RELEASE_ASSERT (gc != NULL) ;

	// set the GraphCtx on the (lazily created) thread-local QueryCtx - some
	// effect handlers (e.g. index creation) reach code paths that resolve
	// their graph via the ambient QueryCtx rather than an explicit
	// parameter, same as any regular query execution would provide
	QueryCtx_SetGraphCtx (gc) ;

	// lock graph for writing
	GraphContext_AcquireWriteLock (gc) ;

	// update graph sync policy
	Graph *g = GraphContext_GetGraph (gc) ;
	MATRIX_POLICY policy = Graph_SetMatrixPolicy (g, SYNC_POLICY_RESIZE) ;

	//--------------------------------------------------------------------------
	// process effects
	//--------------------------------------------------------------------------

	size_t l = 0 ;  // effects buffer length
	const char *effects_buff = RedisModule_StringPtrLen (argv[2], &l) ;

	// apply effects
	bool ok = Effects_Apply (gc, effects_buff, l) ;

	// restore graph sync policy
	Graph_SetMatrixPolicy (g, policy) ;

	// release write lock
	GraphContext_ReleaseLock (gc) ;

	const char *graph_name = GraphContext_GetName (gc) ;

	if (!ok) {
		// replica has diverged from the master, don't propagate this
		// effect any further down a replication sub-chain
		DivergenceGuard_OnFailure (ctx, graph_name, "GRAPH.EFFECT",
				"failed to apply effects, see preceding log entries") ;
		RedisModule_ReplyWithError (ctx, "ERR graph diverged from master") ;
		goto cleanup ;
	}

	// replicate effect
	RedisModule_ReplicateVerbatim (ctx) ;

	// reply back to caller
	RedisModule_ReplyWithSimpleString (ctx, "OK") ;

cleanup:
	// release GraphContext
	GraphContext_DecreaseRefCount (gc) ;

	QueryCtx_Free () ;
	ErrorCtx_Clear () ;
	return REDISMODULE_OK ;
}

