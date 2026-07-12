/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../value.h"
#include "../redismodule.h"
#include "../commands/commands.h"
#include "../datatypes/array.h"
#include "../graph/graphcontext.h"
#include "../util/datablock/datablock.h"

#include <time.h>

// forward declaration
static void defrag_array (RedisModuleDefragCtx *ctx, SIValue *arr) ;

// max time a single defrag step may block the Redis main thread, shared across
// the election wait, the rwlock wait and the scan
#define DEFRAG_BUDGET_MS 50

// set `deadline` to now + `ms` on the monotonic clock
static void _deadline_in
(
	struct timespec *deadline,
	int ms
) {
	clock_gettime (CLOCK_MONOTONIC, deadline) ;
	deadline->tv_sec  += ms / 1000 ;
	deadline->tv_nsec += (long)(ms % 1000) * 1000000L ;

	if (deadline->tv_nsec >= 1000000000L) {
		deadline->tv_sec++ ;
		deadline->tv_nsec -= 1000000000L ;
	}
}

// milliseconds remaining until `deadline` on the monotonic clock (0 if passed)
static int _ms_until
(
	const struct timespec *deadline
) {
	struct timespec now ;
	clock_gettime (CLOCK_MONOTONIC, &now) ;
	long ms = (deadline->tv_sec  - now.tv_sec ) * 1000L +
	          (deadline->tv_nsec - now.tv_nsec) / 1000000L ;
	return ms > 0 ? (int)ms : 0 ;
}

#define STAGE_SHIFT 56
#define OFFSET_MASK 0x00FFFFFFFFFFFFFFULL

// defrage stage
typedef enum {
	DEFRAG_NODES = 0,
	DEFRAG_EDGES = 1,
	DEFRAG_DONE  = 2
} defrag_stage;

// get the stage & offset from which we've left off
static void _load_stage
(
	RedisModuleDefragCtx *ctx,
	defrag_stage *stage,
	uint64_t *offset
) {
	ASSERT (ctx    != NULL) ;
	ASSERT (stage  != NULL) ;
	ASSERT (offset != NULL) ;

	// start stage
	*offset = 0 ;
	*stage  = DEFRAG_NODES ;

	// attempt to get cursor
	unsigned long raw ;
	if (RedisModule_DefragCursorGet (ctx, &raw) != REDISMODULE_OK) {
		return ;
	}

	uint64_t cursor = (uint64_t)raw ;
	*stage  = (defrag_stage)(cursor >> STAGE_SHIFT) ;  // high byte
	*offset = cursor & OFFSET_MASK ;  // low 7 bytes

	// sanity
	ASSERT (*stage >= DEFRAG_NODES && *stage < DEFRAG_DONE) ;
}

// save stage + offset in a single 64-bit value
// (stage in high byte, offset low 56 bits)
static void _save_stage
(
	RedisModuleDefragCtx *ctx,
	defrag_stage stage,
	uint64_t offset
) {
	ASSERT (ctx != NULL) ;
	ASSERT (stage >= DEFRAG_NODES && stage < DEFRAG_DONE) ;

	// mask offset to 56 bits to be compatible with load function
	uint64_t cursor = (((uint64_t)stage) << STAGE_SHIFT) | (offset & OFFSET_MASK);
	unsigned long long raw = (unsigned long long)cursor;

	RedisModule_DefragCursorSet (ctx, raw) ;
}

// defrag an AttributeSet
// ensure any heap pointers are moved using RedisModule_DefragAlloc
static void defrag_attributeset
(
	RedisModuleDefragCtx *ctx,
	AttributeSet *set
) {
	void *moved = NULL ;
	AttributeSet _set = *set ;

	// defrag set
	moved = RedisModule_DefragAlloc (ctx, _set) ;
	if (moved != NULL) {
		*set = moved ;
		_set = *set  ;
	}

	// defrag attributes
	AttributeSet_Defrag (_set, ctx) ;
}

// defrag entities (both nodes and edges)
static int defrag_entities
(
	RedisModuleDefragCtx *ctx,
	defrag_stage stage,
	const Graph *g,
	GraphContext *gc,
	DataBlockIterator *it,
	const struct timespec *deadline  // main-thread budget deadline
) {
	uint64_t counter = 0 ;
	AttributeSet *set = NULL ;

	// get current entity attribute-set
	while ((set = (AttributeSet*)(DataBlockIterator_Next (it, NULL))) != NULL) {
		// entity has no attributes, skip
		if (*set != NULL) {
			defrag_attributeset (ctx, set) ;
		}

		counter++ ;

		// stop if we've spent our main-thread budget — checked every entity
		// as a single entity can carry a large attribute-set
        if ((counter % 64 == 0) && RedisModule_DefragShouldStop (ctx)) {
			// only pause if NOT at the end
			if (!DataBlockIterator_Depleted (it)) {
				// save current stage and offset
				_save_stage (ctx, stage, DataBlockIterator_Position (it)) ;

				return 1;
			}
			// else: fall through, loop will terminate, return 0
        }
	}

	// iterator exhausted, no more work
	return 0 ;
}

static int defrag_edges
(
	RedisModuleDefragCtx *ctx,
	GraphContext *gc,
	uint64_t offset
) {
	int res = 1 ;  // there's more work to be done
	Graph *g = GraphContext_GetGraph (gc) ;

	// defrag runs on the main thread; bound this step to DEFRAG_BUDGET_MS
	// shared across the election wait, the rwlock wait and the scan
	// yield if it elapses
	struct timespec deadline ;
	_deadline_in (&deadline, DEFRAG_BUDGET_MS) ;

	// block out any new writer (wait up to the remaining budget)
	if (!GraphContext_TimeTryEnterWrite (gc, _ms_until (&deadline))) {
		return res ;
	}

	// then acquire the rwlock to block out readers too (remaining budget)
	if (GraphContext_TimeAcquireWriteLock (gc, _ms_until (&deadline)) != 0) {
		goto release_writer ;
	}

	DataBlockIterator *it = Graph_ScanEdges (g) ;
	DataBlockIterator_Seek (it, offset) ;  // seek iterator to offset

	res = defrag_entities (ctx, DEFRAG_EDGES, g, gc, it, &deadline) ;
	DataBlockIterator_Free (it) ;

	GraphContext_ReleaseLock (gc) ;

release_writer:
	// a writer may have queued while we held the election; defrag doesn't drain
	// the queue, so hand it to a writer thread (avoids orphaning the query)
	GraphContext_ExitWrite (gc) ;
	GraphContext_AsyncDrainWriteQueries (gc) ;

	// clean up
	return res ;
}

static int defrag_nodes
(
	RedisModuleDefragCtx *ctx,
	GraphContext *gc,
	uint64_t offset
) {
	int res = 1 ;  // there's more work to be done
	Graph *g = GraphContext_GetGraph (gc) ;

	// defrag runs on the main thread; bound this step to DEFRAG_BUDGET_MS, shared
	// across the election wait, the rwlock wait and the scan. yield if it elapses
	struct timespec deadline ;
	_deadline_in (&deadline, DEFRAG_BUDGET_MS) ;

	// block out any new writer (wait up to the remaining budget)
	if (!GraphContext_TimeTryEnterWrite (gc, _ms_until (&deadline))) {
		return res ;  // there's more work to be done
	}

	// then acquire the rwlock to block out readers too (remaining budget)
	if (GraphContext_TimeAcquireWriteLock (gc, _ms_until (&deadline)) != 0) {
		goto release_writer ;
	}

	DataBlockIterator *it = Graph_ScanNodes (g) ;
	DataBlockIterator_Seek (it, offset) ;  // seek iterator to offset

	res = defrag_entities (ctx, DEFRAG_NODES, g, gc, it, &deadline) ;
	DataBlockIterator_Free (it) ;

	GraphContext_ReleaseLock (gc) ;

release_writer:
	// a writer may have queued while we held the election; defrag doesn't drain
	// the queue, so hand it to a writer thread (avoids orphaning the query)
	GraphContext_ExitWrite (gc) ;
	GraphContext_AsyncDrainWriteQueries (gc) ;

	// clean up
	return res ;
}

// graph context type defrag call back
// invoked by redis active defrag
int _GraphContextType_Defrag
(
	RedisModuleDefragCtx *ctx,
	RedisModuleString *key,
	void **value
) {
	ASSERT (ctx   != NULL) ;
	ASSERT (key   != NULL) ;
	ASSERT (value != NULL) ;

	RedisModule_Log (NULL, "notice", "Defrag key: %s",
		RedisModule_StringPtrLen(key, NULL)) ;

	GraphContext *gc = *((GraphContext**)(value)) ;

	//--------------------------------------------------------------------------
	// determine stage
	//--------------------------------------------------------------------------

	uint64_t offset = 0 ;
	defrag_stage stage = DEFRAG_NODES ;

	_load_stage (ctx, &stage, &offset) ;

	RedisModule_Log (NULL, "notice", "defrag stage: %d, defrag offset: %"PRIu64,
			stage, offset) ;

	int res = 0 ;

	while (res == 0 && stage < DEFRAG_DONE) {
		switch (stage) {
			case DEFRAG_NODES:
				res = defrag_nodes (ctx, gc, offset) ;
				break ;

			case DEFRAG_EDGES:
				res = defrag_edges (ctx, gc, offset) ;
				break ;

			default:
				ASSERT (false && "unexpected defrag stage") ;
		}

		// are we done we current stage ?
		if (res == 0) {
			// advance to next stage and reset offset to zero
			// (will be picked up by subsequent _load_stage)
			stage++ ;
			offset = 0;
		}
	}

    return res ;
}

