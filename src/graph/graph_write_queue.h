/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "graphcontext.h"

//------------------------------------------------------------------------------
// per-graph single-writer election
//------------------------------------------------------------------------------
//
// every command that mutates a graph must flow through this election so that,
// for any single graph, exactly ONE writer lifecycle touches it at a time.
// this is the invariant the schema machinery (the writelocked reentrancy flag,
// the writer_tid copy-on-write pending arrays, FindOrAddSchema) silently relies
// on: it is NOT enforced by those low-level pieces, it is enforced here.
//
// a writer submits a WriteTask via GraphContext_SubmitWrite. one thread wins the
// election (the atomic write_in_progress flag) and drains the per-graph queue,
// running each task's fn(payload) in FIFO order; losers simply enqueue and
// return, their task executed for them by the elected writer.

// a unit of serialized write work
typedef void (*WriteTaskFn)(void *payload);

typedef struct {
	WriteTaskFn fn;   // task executor
	void *payload;    // task argument; fn takes ownership
} WriteTask;

// initialize a graph's write-election state (write-in-progress flag + queue)
void GraphContext_WriteQueueInit
(
	GraphContext *gc  // graph context
);

// free a graph's write-election state; the pending queue must be empty
void GraphContext_WriteQueueFree
(
	GraphContext *gc  // graph context
);

// submit a write task to the graph's single-writer election: enqueue it, then
// try to become the writer and drain the queue. if this thread is elected it
// runs the task (and any others queued) inline; otherwise the current writer
// runs it. returns false if the task could not be enqueued (queue full), in
// which case it is NOT run and its payload remains the caller's responsibility
bool GraphContext_SubmitWrite
(
	GraphContext *gc,  // graph context
	WriteTaskFn fn,    // task executor
	void *payload      // task argument
);
