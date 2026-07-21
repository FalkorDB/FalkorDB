/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../commands/cmd_context.h"

#include <stdbool.h>

//------------------------------------------------------------------------------
// Per-graph coordination for concurrent loads of an offloaded stub
//
// The first thread to reach a given stub becomes its load "owner" and is the
// one that actually performs the load (i.e. calls graph_load). Every other
// thread that reaches the same stub while the owner's load is still in
// flight parks itself here (keyed by graph name) instead of failing
// outright, and is woken once the owner's load resolves.
//
// Owner-election and parking share a single lock, so there is no gap in
// which a thread can observe "someone else owns this load" and then, before
// registering as a waiter, have the owner finish and drain an empty list —
// the two operations are atomic with respect to each other.
//
// Waiters are parked CommandCtx pointers paired with a caller-supplied
// callback, invoked once, exactly one time, when the wait ends.
//------------------------------------------------------------------------------

typedef enum {
	GraphLoadQueue_OWNER,   // no load is currently in flight for this graph;
	                        // caller must perform the load itself and, once
	                        // it resolves, call GraphLoadQueue_Drain exactly
	                        // once with the outcome
	GraphLoadQueue_PARKED,  // another thread already owns the in-flight load;
	                        // `waiter` was parked, `cb` will be invoked
	                        // exactly once with the outcome once the owner
	                        // calls GraphLoadQueue_Drain
	GraphLoadQueue_FULL,    // another thread owns the in-flight load and the
	                        // per-graph wait list is already at capacity;
	                        // `waiter` was NOT parked, caller must handle it
} GraphLoadQueueStatus ;

// invoked once for a parked waiter when the in-flight load it was waiting on
// completes; `success` reflects whether the load succeeded
typedef void (*GraphLoadWaiterCB)
(
	CommandCtx *waiter,
	bool        success
) ;

// attempt to become the load owner for `graph_name`, or park `waiter` behind
// whichever thread already owns it
GraphLoadQueueStatus GraphLoadQueue_AcquireOrWait
(
	const char        *graph_name,
	CommandCtx        *waiter,
	GraphLoadWaiterCB  cb
) ;

// called exactly once, by the thread for which GraphLoadQueue_AcquireOrWait
// returned GraphLoadQueue_OWNER, once its load attempt resolves; wakes every
// waiter parked on `graph_name` with `success`, then clears bookkeeping for
// `graph_name` so the next load attempt starts a fresh round
void GraphLoadQueue_Drain
(
	const char *graph_name,
	bool        success
) ;

// release every resource held by this module: any graph names and waiter
// lists still tracked (e.g. loads that never drained because the server
// shut down mid-flight), and the registry itself
// each remaining waiter's callback is still invoked, with success=false -
// same as a normal failed drain, this replies with an error and frees the
// CommandCtx; there is simply no real load outcome left to report
// not thread-safe with concurrent GraphLoadQueue_AcquireOrWait / _Drain
// calls - intended for server shutdown only
void GraphLoadQueue_Free (void) ;
