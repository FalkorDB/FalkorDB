/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../../globals.h"
#include "../../util/rmalloc.h"
#include "../../storage/storage.h"
#include "../../util/simple_timer.h"
#include "../../graph/graphcontext.h"
#include "../../util/memory_consumption.h"
#include "../../util/datablock/datablock.h"
#include "../../graph/entities/graph_entity.h"

#include <stdint.h>
#define MAX_BATCH_SIZE 1024  // max number of attribute-sets to offload
#define DEADLINE 40          // task should run up to 40ms

// task context
typedef struct {
	uint64_t graph_idx;  // current processed graph
	EntityID entity_id;  // current processed entity id
	GraphEntityType t;   // current processed entity type Node/Edge
} OffloadTaskCtx;

void *CronTask_newOffloadEntities
(
	void *pdata  // [optional] task context
) {
	OffloadTaskCtx *ctx = rm_calloc (1, sizeof (OffloadTaskCtx)) ;
	ASSERT (ctx != NULL) ;

	// default values
	ctx->t = GETYPE_NODE ;
	ctx->entity_id = 0 ;
	ctx->graph_idx = 0 ;

	if (pdata != NULL) {
		// create a new offload task continuing from where the previous one
		// (pdata) had leftoff
		OffloadTaskCtx *prev_ctx = (OffloadTaskCtx*)pdata ;
		memcpy (ctx, prev_ctx, sizeof (OffloadTaskCtx)) ;
	}

	// trivial validations
	ASSERT (ctx->t == GETYPE_NODE || ctx->t == GETYPE_EDGE) ;
	ASSERT (ctx->entity_id != INVALID_ENTITY_ID) ;

	return ctx ;
}

// offload graph to disk
// currently only attribute-sets are offloaded
// data is offloaded in batches of size MAX_BATCH_SIZE
//
// this operation is bounded by time, once the time quota is over
// the function returns, not before updating the tasks' context
// for future resumt
static void _offloadGraph
(
	OffloadTaskCtx *ctx,      // task context
	GraphContext *gc,         // graph context
	EntityID *ids,            // pre allocated array of entity IDs
	AttributeSet *sets,       // pre allocated array of attribute sets
	simple_timer_t stopwatch  // timer
) {
	ASSERT (ids  != NULL) ;
	ASSERT (ctx  != NULL) ;
	ASSERT (sets != NULL) ;

	// acquire READ lock
	Graph *g = GraphContext_GetGraph (gc) ;
	Graph_AcquireReadLock (g) ;

	//----------------------------------------------------------------------
	// attach entity iterator
	//----------------------------------------------------------------------

	DataBlock *datablock = (ctx->t == GETYPE_NODE) ? g->nodes : g->edges ;

	DataBlockIterator *it = NULL ;
	it = DataBlock_Scan (datablock) ;
	ASSERT (it != NULL) ;
	DataBlockIterator_Seek (it, ctx->entity_id) ;

	// as long as we've got processing time
	while (TIMER_GET_ELAPSED_MILLISECONDS (stopwatch) < DEADLINE) {

		//----------------------------------------------------------------------
		// collect attribute sets
		//----------------------------------------------------------------------

		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
				"Collecting attribute-sets") ;

		// TODO: optimize ids + count
		size_t count = 0 ;
		for (; count < MAX_BATCH_SIZE ; count++) {
			AttributeSet *set =
				(AttributeSet*)DataBlockIterator_NextSkipOffloaded (it,
						ids + count) ;

			if (set == NULL) {
				break ;
			}

			// empty attribute set, e.g. CREATE ()
			if (*set == NULL) {
				count-- ;
				continue ;
			}

			sets[count] = *set ;
		}

		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG, "Collected %zu sets",
				count) ;

		// update context
		ctx->entity_id += count ;

		//----------------------------------------------------------------------
		// offload attribute sets
		//----------------------------------------------------------------------

		if (count > 0) {
			RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
					"Offloading collected sets to disk") ;

			if (Storage_putAttributes (datablock->cf, sets, count, ids) == 0) {
				// attribute sets been offloaded to disk
				// mark datablock entries as offloaded
				DataBlock_MarkOffloaded (datablock, ids, count) ;

				// attribute sets been offloaded to disk and the datablock
				// marked each one as offloaded
				// now we're ready to free attribute-sets
				// to do that we must hold the graph's write lock

				Graph_ReleaseLock (g) ;
				Graph_AcquireWriteLock (g) ;

				RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
						"Freeing sets") ;

				// "shallow" free attribute-sets
				int j = count ;
				while (j > 0) {
					rm_free (sets[--j]) ;
				}

				Graph_ReleaseLock (g) ;
				Graph_AcquireReadLock (g) ;
			}
		}

		if (count < MAX_BATCH_SIZE) {
			// datablock iterator depleted

			RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
					"Datablock iterator depleted") ;

			if (ctx->t == GETYPE_NODE) {
				// next iteration should scan edges
				RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
						"Move to edges") ;

				ctx->t = GETYPE_EDGE ;
				ctx->entity_id = 0 ;

				DataBlockIterator_Free (it) ;

				datablock = g->edges ;
				it = DataBlock_Scan (datablock) ;
				DataBlockIterator_Seek (it, ctx->entity_id) ;
			} else {
				RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
						"Done offloading graph, move on to the next one") ;
				// both node & edge iterators are depleted
				// reset context and move on to the next graph
				ASSERT (ctx->t == GETYPE_EDGE) ;
				ctx->t = GETYPE_NODE ;
				ctx->entity_id = 0 ;
				ctx->graph_idx++ ;
				break ;  // break out of the main while loop
			}
		}
	}

	Graph_ReleaseLock (g) ;

	if (it != NULL) {
		DataBlockIterator_Free (it) ;
	}
}

// cron task entry point
// offloads all graphs in the keyspace one by one to disk
// the task is bounded by time (DEADLINE) after which it will exit
// and be rescheduled for future runs
//
// the task will quickly return if memory consumption is below a specified limit
// e.g. memory consumption < 65% quickly return otherwise start offloading
bool CronTask_offloadEntities
(
	void *pdata  // task context
) {
	// quick return if keyspace does not contains any graphs
	if (unlikely (Globals_GraphsCount () == 0)) {
		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
			"No graphs to process quick return") ;
		printf ("No graphs to process quick return\n") ;
		return false ;
	}

	size_t rss = get_current_rss () ;
	rss /= (1024 * 1024) ;
	RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
			"rss: %zu", rss) ;
	printf ("rss: %zumb\n", rss) ;

	// TODO: introduce config
	if (rss <= 500) {
		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
			"Memory consumption too low, skipping data offloading") ;
		printf ("Memory consumption too low, skipping data offloading\n") ;
		return false ;
	}

	RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
			"Offload data task start") ;

	OffloadTaskCtx *ctx = (OffloadTaskCtx*)pdata ;

	// start stopwatch
	simple_timer_t stopwatch ;
	simple_tic (stopwatch) ;

	//--------------------------------------------------------------------------
	// allocate workspace
	//--------------------------------------------------------------------------

	// TODO: consider moving into task's context and allocate just once
	EntityID     *ids  = rm_calloc (MAX_BATCH_SIZE, sizeof (EntityID)) ;
	AttributeSet *sets = rm_calloc (MAX_BATCH_SIZE, sizeof (AttributeSet)) ;

	// return if allocation fails
	if (ids == NULL || sets == NULL) {
		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
			"Failed to allocate workspace, quickly return") ;
		if (ids  != NULL) rm_free (ids)  ;
		if (sets != NULL) rm_free (sets) ;
		return true ;
	}

	// initialize graphs iterator and pick up from where we've left
	KeySpaceGraphIterator it ;
	Globals_ScanGraphs (&it) ;
	GraphIterator_Seek (&it, ctx->graph_idx) ;

	GraphContext *gc = NULL ;

	// as long as we've got processing time
	while (TIMER_GET_ELAPSED_MILLISECONDS (stopwatch) < DEADLINE) {
		// get next graph to offload attribute sets from
		gc = GraphIterator_Next (&it) ;

		// iterator depleted
		if (gc == NULL) {
			RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
					"Finished scanning through all graphs in keyspace, "
					"resetting graphs iterator") ;

			// reset context
			ctx->t         = GETYPE_NODE ;
			ctx->graph_idx = 0 ;
			ctx->entity_id = 0 ;

			// seek to the very beginning
			GraphIterator_Seek (&it, ctx->graph_idx) ;
			continue ;
		}

		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
				"Offloading graph %s", gc->graph_name) ;

		// offload current graph
		_offloadGraph (ctx, gc, ids, sets, stopwatch) ;

		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
				"Done processing graph %s, either out of time or finished "
				"processing all attribute-sets", gc->graph_name) ;

		// either we've offloaded the entire graph or we're out of time
		// regardless decrease the graph's ref count
		GraphContext_DecreaseRefCount (gc) ;
	}

	RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG, "Time quota expired") ;

	// clean up
	rm_free (ids) ;
	rm_free (sets) ;

	// indicate if there's additional work to be done
	return true ;
}

