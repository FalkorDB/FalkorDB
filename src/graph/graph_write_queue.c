/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "globals.h"
#include "../util/arr.h"
#include "../util/uuid.h"
#include "../query_ctx.h"
#include "graphcontext.h"
#include "graph_write_queue.h"
#include "../redismodule.h"
#include "../util/rwlock.h"
#include "../util/rmalloc.h"
#include "graph_memoryUsage.h"
#include "../util/thpool/pool.h"
#include "../errors/errors.h"
#include "../errors/error_msgs.h"
#include "../constraint/constraint.h"
#include "../index/cch_index.h"
#include "../util/identifier_limits.h"
#include "../commands/execution_ctx.h"
#include "../serializers/graphcontext_type.h"

#include <time.h>
#include <pthread.h>
#include <sys/param.h>
#include <stdatomic.h>

#include "graphcontext_struct.h"

//------------------------------------------------------------------------------
// write-election state lifecycle
//------------------------------------------------------------------------------

void GraphContext_WriteQueueInit
(
	GraphContext *gc  // graph context
) {
	ASSERT (gc != NULL) ;

	// write-in-progress flag starts false
	atomic_init (&gc->write_in_progress, false) ;

	// create graph's pending write queue (holds WriteTask*)
	gc->pending_write_queue = CircularBuffer_New (sizeof (void*), 1024) ;
}

void GraphContext_WriteQueueFree
(
	GraphContext *gc  // graph context
) {
	ASSERT (gc != NULL) ;

	if (gc->pending_write_queue != NULL) {
		ASSERT (CircularBuffer_Empty (gc->pending_write_queue)) ;
		CircularBuffer_Free (gc->pending_write_queue, NULL) ;
		gc->pending_write_queue = NULL ;
	}
}

//------------------------------------------------------------------------------
// election primitives
//------------------------------------------------------------------------------

// attempt to acquire exclusive write access to the given graph
// returns true if the calling thread successfully acquired write ownership
// returns false if another write is already in progress
bool GraphContext_TimeTryEnterWrite
(
	GraphContext *gc,  // graph context
	uint timeout_ms    // maximum time in milliseconds to wait for the lock:
					   // - timeout_ms = 0 : non-blocking attempt (try-lock)
					   // - timeout_ms > 0 : block up to timeout_ms milliseconds
) {
	ASSERT (gc != NULL) ;

	bool expected = false ;

    // atomically set to true only if current value is false
	bool acquired = atomic_compare_exchange_strong (&gc->write_in_progress,
			&expected, true) ;

	if (acquired == true) {
		return true ;
	}

	// failed to acquire, poll until acquired or the timeout elapses
	if (timeout_ms > 0) {
		// poll against an absolute monotonic deadline (not a decremented sleep)
		// so the timeout stays honest if nanosleep wakes early on a signal
		struct timespec deadline ;
		clock_gettime (CLOCK_MONOTONIC, &deadline) ;
		deadline.tv_sec  += timeout_ms / 1000 ;
		deadline.tv_nsec += (long)(timeout_ms % 1000) * 1000000L ;
		if (deadline.tv_nsec >= 1000000000L) {
			deadline.tv_sec++ ;
			deadline.tv_nsec -= 1000000000L ;
		}

		// 1ms poll interval — fine enough to grab the flag promptly once it frees
		struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000000 } ;

		while (true) {
			nanosleep (&ts, NULL) ;  // may wake early on signal; deadline guards us

			expected = false ;  // reset, CAS clobbers it on failure
			acquired = atomic_compare_exchange_strong (&gc->write_in_progress,
					&expected, true) ;

			if (acquired == true) {
				return true ;
			}

			struct timespec now ;
			clock_gettime (CLOCK_MONOTONIC, &now) ;
			if (now.tv_sec > deadline.tv_sec ||
				(now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec)) {
				break ;  // timeout elapsed
			}
		}
	}

	return false ;
}

// release exclusive write access to the graph
// this should be called by a thread that previously acquired write ownership
// via GraphContext_TimeTryEnterWrite, it clears the write-in-progress flag
void GraphContext_ExitWrite
(
	GraphContext *gc  // graph context
) {
	ASSERT (gc != NULL) ;

	atomic_store (&gc->write_in_progress, false) ;
}

// enqueue a write task for deferred execution on the specified graph
// returns true if the task was successfully enqueued
// false if the enqueue operation failed (e.g., due to allocation failure)
bool GraphContext_EnqueueWriteQuery
(
	GraphContext *gc,  // graph context
	void *query_ctx    // write task
) {
	ASSERT (gc        != NULL) ;
	ASSERT (query_ctx != NULL) ;

	return (CircularBuffer_Add (gc->pending_write_queue, &query_ctx) != 0) ;
}

// dequeue the next pending write task for the specified graph
// returns a task pointer if a task was dequeued,
// or NULL if the pending write queue is empty
void *GraphContext_DequeueWriteQuery
(
	GraphContext *gc  // graph context
) {
	ASSERT (gc != NULL) ;

	void *item = NULL ;
	CircularBuffer_Read (gc->pending_write_queue, &item) ;

	return item ;
}

//------------------------------------------------------------------------------
// writer loop
//------------------------------------------------------------------------------

// drain all pending write tasks on the calling thread
// the caller must already hold the writer token (GraphContext_TimeTryEnterWrite)
// the writer only releases write access once the queue is truly empty
static void enter_writer_loop
(
	GraphContext *gc
) {
	while (true) {
		// drain the queue
		WriteTask *task ;
		while ((task = (WriteTask *)GraphContext_DequeueWriteQuery (gc))) {
			task->fn (task->payload) ;
			rm_free (task) ;
		}

		// release write access
		GraphContext_ExitWrite (gc) ;

		// race condition handling: after releasing write access, another thread
		// may have enqueued a task
		// we must check the queue again and attempt
		// to reacquire write access
		// if we succeed, continue processing
		// if we fail, another thread is now the writer and will handle the queue
		if (GraphContext_WriteQueueEmpty    (gc) ||
			!GraphContext_TimeTryEnterWrite (gc, 0)) {
			// either the queue is empty
			// or another thread became a writer
			break ;
		}
	}
}

//------------------------------------------------------------------------------
// submit
//------------------------------------------------------------------------------

bool GraphContext_SubmitWrite
(
	GraphContext *gc,  // graph context
	WriteTaskFn fn,    // task executor
	void *payload      // task argument
) {
	ASSERT (gc != NULL && fn != NULL) ;

	WriteTask *task = rm_malloc (sizeof (WriteTask)) ;
	task->fn      = fn ;
	task->payload = payload ;

	// increase graph ref count, guard against the graph context being freed too
	// early as the writer needs access to the graph's pending queue and flag
	GraphContext_IncreaseRefCount (gc) ;

	if (!GraphContext_EnqueueWriteQuery (gc, task)) {
		// queue full — task not submitted
		GraphContext_DecreaseRefCount (gc) ;
		rm_free (task) ;
		return false ;
	}

	// try to acquire exclusive write access to the graph
	if (GraphContext_TimeTryEnterWrite (gc, 0)) {
		// this thread is the writer: drain the queue (incl. the task above)
		enter_writer_loop (gc) ;
	}

	// counter to GraphContext_IncreaseRefCount above
	GraphContext_DecreaseRefCount (gc) ;

	return true ;
}

//------------------------------------------------------------------------------
// async drain
//------------------------------------------------------------------------------

// worker-pool task: elect a writer and drain pending write tasks on `gc`
// (dispatched by GraphContext_AsyncDrainWriteQueries; releases the reference
// taken there)
static void _drain_write_queue_task
(
	void *arg
) {
	GraphContext *gc = (GraphContext *)arg ;

	// become the writer and drain; if another thread is already the writer it
	// drains the queue itself, so there is nothing to do
	if (GraphContext_TimeTryEnterWrite (gc, 0)) {
		enter_writer_loop (gc) ;
	}

	GraphContext_DecreaseRefCount (gc) ;  // counter to the ref in the dispatcher
}

// asynchronously drain pending write tasks queue
void GraphContext_AsyncDrainWriteQueries
(
	GraphContext *gc  // graph context
) {
	ASSERT (gc != NULL) ;

	// exit if the queue is empty
	if (GraphContext_WriteQueueEmpty (gc)) {
		return ;
	}

	// keep gc alive until the drain task runs
	GraphContext_IncreaseRefCount (gc) ;

	// force=true: never dropped for a full queue, so the only failure is an
	// allocation error (returns non-zero); undo the ref so gc isn't leaked
	if (ThreadPool_AddWork (_drain_write_queue_task, gc, true) != 0) {
		GraphContext_DecreaseRefCount (gc) ;  // couldn't enqueue; decrease ref
	}
}

// checks if the graph's pending write queue is empty
bool GraphContext_WriteQueueEmpty
(
	const GraphContext *gc  // graph context
) {
	return CircularBuffer_Empty(gc->pending_write_queue);
}
