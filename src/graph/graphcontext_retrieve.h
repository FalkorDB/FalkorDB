/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "graphcontext.h"

// Like GraphContext_Retrieve (see graphcontext.h), always attempting to load
// the graph from disk if it is offloaded, but with different handling of
// concurrent loads: if another thread is already loading this stub,
// (`handler`, `arg`) is parked instead of GraphRetrieve_LOADING being
// returned immediately. Once that in-flight load resolves, `handler(arg)`
// is resubmitted to the thread pool - as if freshly dispatched - so the
// caller must not touch `arg` again until then. This lets the resumed call
// re-discover the load's outcome itself (already loaded vs. still a stub)
// rather than being told.
// May be called from any thread.
GraphRetrieveStatus GraphContext_RetrieveOrQueue
(
	RedisModuleCtx *ctx,         // Redis module context
	RedisModuleString *graphID,  // key identifying the graph
	bool readOnly,               // if true, opens the key in read mode
	bool shouldCreate,           // create new graph if the key is absent
	void (*handler) (void *),    // resubmitted if the graph is loading
	void *arg,                   // passed to `handler` if parked
	GraphContext **gc            // out: graph context on success
) ;

// Like GraphContext_Retrieve (see graphcontext.h), always attempting to load
// the graph from disk if it is offloaded, but never failing due to another
// load being in flight: an independent load is attempted regardless (see
// bypass_claim in the enterprise repo's graph_load.h), so
// GraphRetrieve_FAILED is only ever returned for a genuine failure (missing
// dump, OOM, corruption, etc.), never for mere contention with another
// load. An in-flight offload is never bypassed, since there is nothing yet
// to load in that case, so this may still fail while a concurrent offload
// of the same graph is in progress.
// For use by callers that cannot tolerate a transient retrieval failure -
// e.g. GRAPH.EFFECT, where a failure risks master/replica divergence.
// May be called from any thread.
GraphRetrieveStatus GraphContext_RetrieveOrForce
(
	RedisModuleCtx *ctx,         // Redis module context
	RedisModuleString *graphID,  // key identifying the graph
	bool readOnly,               // if true, opens the key in read mode
	bool shouldCreate,           // create new graph if the key is absent
	GraphContext **gc            // out: graph context on success
) ;

