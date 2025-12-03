/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../value.h"
#include "../redismodule.h"
#include "../datatypes/array.h"
#include "../graph/graphcontext.h"
#include "../util/datablock/datablock.h"

// forward declaration
static void defrag_array (RedisModuleDefragCtx *ctx, SIValue *arr) ;

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
	uint16_t n = AttributeSet_Count (_set) ;

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
	DataBlockIterator *it
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

		// check if we should stop
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
	Graph *g = GraphContext_GetGraph (gc) ;

	// try to obtain exclusive access to the graph
	int timeout_ms = 50 ;
	if (Graph_TimeAcquireWriteLock (g, timeout_ms) != 0) {
		// failed to acquire write lock, check if we're out of time
		RedisModule_Log (NULL, "notice",
				"Graph defrag, failed to acquire write lock") ;
		return 1 ;  // there's more work to be done
	}

	DataBlockIterator *it = Graph_ScanEdges (g) ;
	DataBlockIterator_Seek (it, offset) ;  // seek iterator to offset

	int res = defrag_entities (ctx, DEFRAG_EDGES, g, gc, it) ;

	Graph_ReleaseLock (g) ;

	// clean up
	DataBlockIterator_Free (it) ;
	return res ;
}

static int defrag_nodes
(
	RedisModuleDefragCtx *ctx,
	GraphContext *gc,
	uint64_t offset
) {
	Graph *g = GraphContext_GetGraph (gc) ;

	// try to obtain exclusive access to the graph
	int timeout_ms = 50 ;
	if (Graph_TimeAcquireWriteLock (g, timeout_ms) != 0) {
		// failed to acquire write lock, check if we're out of time
		RedisModule_Log (NULL, "notice",
				"Graph defrag, failed to acquire write lock") ;
			return 1 ;  // there's more work to be done
	}

	DataBlockIterator *it = Graph_ScanNodes (g) ;
	DataBlockIterator_Seek (it, offset) ;  // seek iterator to offset

	int res = defrag_entities (ctx, DEFRAG_NODES, g, gc, it) ;

	Graph_ReleaseLock (g) ;

	// clean up
	DataBlockIterator_Free (it) ;
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

