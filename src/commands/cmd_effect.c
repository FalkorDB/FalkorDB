/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../effects/effects.h"
#include "../graph/graphcontext.h"
#include "../replication/divergence_guard.h"

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

	// get graph context
	GraphContext *gc = NULL ;
	GraphContext_Retrieve (ctx, argv[1], false, true, true, &gc) ;
	ASSERT (gc != NULL) ;

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

	// release GraphContext
	GraphContext_DecreaseRefCount (gc) ;

	if (!ok) {
		// replica has diverged from the master, don't propagate this
		// effect any further down a replication sub-chain
		DivergenceGuard_OnFailure (ctx, graph_name, "GRAPH.EFFECT",
				"failed to apply effects, see preceding log entries") ;
		RedisModule_ReplyWithError (ctx, "ERR graph diverged from master") ;
		return REDISMODULE_OK ;
	}

	// replicate effect
	RedisModule_ReplicateVerbatim (ctx) ;

	// reply back to caller
	RedisModule_ReplyWithSimpleString (ctx, "OK") ;

	return REDISMODULE_OK ;
}

