/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

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
// Waiters are parked as a generic (handler, arg) pair - e.g. a command's
// top-level entry point paired with its CommandCtx - so any caller that may
// get parked behind an in-flight load can be resumed the same way: by
// resubmitting `handler(arg)` to the thread pool, as if freshly dispatched.
// Each parked waiter is resumed exactly once.
//------------------------------------------------------------------------------

typedef enum {
	GraphLoadQueue_OWNER,   // no load is currently in flight for this graph;
	                        // caller must perform the load itself and, once
	                        // it resolves, call GraphLoadQueue_Drain exactly
	                        // once
	GraphLoadQueue_PARKED,  // another thread already owns the in-flight load;
	                        // (`handler`, `arg`) was parked and will be
	                        // resubmitted to the thread pool exactly once,
	                        // once the owner calls GraphLoadQueue_Drain
	GraphLoadQueue_FULL,    // another thread owns the in-flight load and the
	                        // per-graph wait list is already at capacity;
	                        // (`handler`, `arg`) was NOT parked, caller must
	                        // handle it
} GraphLoadQueueStatus ;

// attempt to become the load owner for `graph_name`, or park (`handler`,
// `arg`) behind whichever thread already owns it
GraphLoadQueueStatus GraphLoadQueue_AcquireOrWait
(
	const char *graph_name,
	void      (*handler) (void *),  // resubmitted to the thread pool on drain
	void       *arg                 // passed to `handler` on drain
) ;

// called exactly once, by the thread for which GraphLoadQueue_AcquireOrWait
// returned GraphLoadQueue_OWNER, once its load attempt resolves; resubmits
// every waiter parked on `graph_name` to the thread pool (regardless of the
// load's outcome - each resumed handler re-discovers that outcome itself,
// e.g. by re-attempting the retrieval it was parked behind), then clears
// bookkeeping for `graph_name` so the next load attempt starts a fresh round
void GraphLoadQueue_Drain
(
	const char *graph_name
) ;

// release every resource held by this module: any graph names and waiter
// lists still tracked (e.g. loads that never drained because the server
// shut down mid-flight), and the registry itself
// each remaining waiter is still resubmitted to the thread pool, same as a
// normal drain
// not thread-safe with concurrent GraphLoadQueue_AcquireOrWait / _Drain
// calls - intended for server shutdown only
void GraphLoadQueue_Free (void) ;

