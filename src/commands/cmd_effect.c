/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../effects/effects.h"
#include "../graph/graphcontext_retrieve.h"

// GRAPH.EFFECT command handler
int Graph_Effect
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command arguments
	int argc                   // number of arguments
) {
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
	Effects_Apply (gc, effects_buff, l) ;

	// restore graph sync policy
	Graph_SetMatrixPolicy (g, policy) ;

	// release write lock
	GraphContext_ReleaseLock (gc) ;

	// release GraphContext
	GraphContext_DecreaseRefCount (gc) ;

	// replicate effect
	RedisModule_ReplicateVerbatim (ctx) ;

	// reply back to caller
	RedisModule_ReplyWithSimpleString (ctx, "OK") ;

	return REDISMODULE_OK ;
}

