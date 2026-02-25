/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../../globals.h"
#include "../../util/memory.h"
#include "../../util/rmalloc.h"
#include "../../storage/storage.h"
#include "../../util/simple_timer.h"
#include "../../graph/graphcontext.h"
#include "../../util/datablock/datablock.h"
#include "../../graph/entities/graph_entity.h"

#include <stdint.h>

#define DEADLINE 1000            // maximum task runtime in ms before yielding
#define MAX_BATCH_SIZE 64000   // maximum number of attribute-sets collected
                               // and offloaded per batch

#define BYTES_TO_MB(bytes) (bytes) / (1024LL * 1024LL)

// task context
typedef struct {
	uint64_t graph_idx;  // index of the graph currently being processed
	EntityID entity_id;  // ID of the next entity to process (resume point)
	GraphEntityType t;   // current processed entity type Node/Edge
	EntityID *ids;       // array of entity IDs
	AttributeSet *sets;  // array of attribute sets
	size_t *sizes;       // array of attribute-set byte sizes
} OffloadTaskCtx;

// create a new offload entities context
void *OffloadEntities_new (void) {
	//--------------------------------------------------------------------------
	// allocate workspace
	//--------------------------------------------------------------------------

	EntityID     *ids   = rm_calloc (MAX_BATCH_SIZE, sizeof (EntityID)) ;
	AttributeSet *sets  = rm_calloc (MAX_BATCH_SIZE, sizeof (AttributeSet)) ;
	size_t       *sizes = rm_calloc (MAX_BATCH_SIZE, sizeof (size_t)) ;

	// return if allocation fails
	if (ids == NULL || sets == NULL || sizes == NULL) {
		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
			"Failed to allocate workspace") ;

		if (ids   != NULL) rm_free (ids)   ;
		if (sets  != NULL) rm_free (sets)  ;
		if (sizes != NULL) rm_free (sizes) ;
		return NULL ;
	}

	// create the context object
	OffloadTaskCtx *ctx = rm_calloc (1, sizeof (OffloadTaskCtx)) ;

	if (ctx == NULL) {
		rm_free (ids)   ;
		rm_free (sets)  ;
		rm_free (sizes) ;
		return NULL ;
	}

	// initialize context
	ctx->t     = GETYPE_NODE ;
	ctx->ids   = ids ;
	ctx->sets  = sets ;
	ctx->sizes = sizes ;

	return ctx ;
}

// free offload entities context
void OffloadEntities_free
(
	OffloadTaskCtx **ctx
) {
	ASSERT (ctx != NULL && *ctx != NULL) ;

	rm_free ((*ctx)->ids)   ;
	rm_free ((*ctx)->sets)  ;
	rm_free ((*ctx)->sizes) ;

	rm_free (*ctx) ;

	*ctx = NULL ;
}

// offload graph to disk
// only attribute-sets are offloaded
// data is offloaded in batches of size MAX_BATCH_SIZE
//
// this operation is bounded by time, once the time quota is over
// the function returns, not before updating the tasks' context
// for future resume
static size_t _offloadGraph
(
	OffloadTaskCtx *ctx,      // task context
	GraphContext *gc,         // graph context
	simple_timer_t stopwatch  // timer
) {
	printf ("Offloading graph %s\n", gc->graph_name) ;

	size_t total_size = 0 ;  // number of bytes offloaded
	Graph *g = GraphContext_GetGraph (gc) ;

	// acquire READ lock
	Graph_AcquireReadLock (g) ;

	//----------------------------------------------------------------------
	// attach entity iterator
	//----------------------------------------------------------------------

	DataBlock *datablock = (ctx->t == GETYPE_NODE) ? g->nodes : g->edges ;
	if (!DataBlock_HasStorage (datablock)) {
		Graph_ReleaseLock (g) ;
		printf ("Graph %s doesn't has a backing storage, skipping graph\n",
				gc->graph_name) ;
		return total_size ;
	}

	DataBlockIterator *it = NULL ;
	it = DataBlock_Scan (datablock) ;
	if (it == NULL) {
		Graph_ReleaseLock (g) ;
		return total_size ;
	}

	printf ("Resuming scan from entity ID: %" PRIu64 "\n", ctx->entity_id) ;
	DataBlockIterator_Seek (it, ctx->entity_id) ;

	// as long as we've got processing time
	while (TIMER_GET_ELAPSED_MILLISECONDS (stopwatch) < DEADLINE) {

		//----------------------------------------------------------------------
		// collect attribute sets
		//----------------------------------------------------------------------

		size_t read  = 0 ;  // number of items scanned
		size_t write = 0 ;  // number of items collected

		while (write < MAX_BATCH_SIZE) {
			read++ ;
			AttributeSet *set =
				(AttributeSet*)DataBlockIterator_NextSkipOffloaded (it,
						ctx->ids + write) ;

			// iterator depleted
			if (set == NULL) {
				break ;
			}

			// empty attribute set, e.g. CREATE ()
			if (*set == NULL) {
				continue ;
			}

			ctx->sets  [write] = *set ;
			ctx->sizes [write] = AttributeSet_ByteSize (*set) ;
			write++ ;
		}

		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG, "Collected %zu sets",
				write) ;
		printf ("Collected %zu sets\n", write) ;

		// update context
		ctx->entity_id += read ;
		size_t count = write ;

		//----------------------------------------------------------------------
		// offload attribute sets
		//----------------------------------------------------------------------

		if (count > 0) {
			RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
					"Offloading collected sets to disk") ;
			printf ("Offloading collected sets to disk\n") ;

			if (Storage_save (datablock->cf, (const void * const *)ctx->sets,
						ctx->sizes, ctx->ids, count) == 0) {
				// attribute sets been offloaded to disk
				// mark datablock entries as offloaded
				DataBlock_MarkOffloaded (datablock, ctx->ids, count) ;

				// attribute sets been offloaded to disk and the datablock
				// marked each one as offloaded
				// now we're ready to free attribute-sets
				// to do that we must hold the graph's write lock

				// release read lock
				Graph_ReleaseLock (g) ;

				// acquire & release write lock
				// this guarantees noone holds pointers to offloaded sets
				Graph_AcquireWriteLock (g) ;
				Graph_ReleaseLock (g) ;

				RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
						"Freeing sets") ;
				printf ("Freeing sets\n") ;

				// "shallow" free attribute-sets
				int j = count -1 ;
				while (j >= 0) {
					total_size += ctx->sizes[j] ;
					rm_free (ctx->sets[j]) ;
					j-- ;
				}

				Graph_AcquireReadLock (g) ;
			}
		}

		if (count < MAX_BATCH_SIZE) {
			// datablock iterator depleted
			RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
					"Datablock iterator depleted") ;
			printf ("Datablock iterator depleted\n") ;

			DataBlockIterator_Free (&it) ;

			if (ctx->t == GETYPE_NODE) {
				// next iteration should scan edges
				RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
						"Move to edges") ;
				printf ("Move to edges\n") ;

				ctx->t = GETYPE_EDGE ;
				ctx->entity_id = 0 ;

				datablock = g->edges ;
				it = DataBlock_Scan (datablock) ;
				DataBlockIterator_Seek (it, ctx->entity_id) ;
			} else {
				RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
						"Done offloading graph, move on to the next one") ;
				printf ("Done offloading graph, move on to the next one\n") ;
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
		DataBlockIterator_Free (&it) ;
	}

	// return number of bytes offloaded
	return total_size ;
}

// cron task entry point
// offloads all graphs in the keyspace one by one to disk
// the task is bounded by time (DEADLINE) after which it will exit
// and be rescheduled for future runs
//
// the task will quickly return if memory consumption is below a specified limit
// e.g. memory consumption < 65% quickly return otherwise start offloading
// returns true to increase task frequency, false to slowdown
bool OffloadEntities
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

	// initialize graphs iterator and pick up from where we've left
	KeySpaceGraphIterator it ;
	Globals_ScanGraphs (&it) ;
	GraphIterator_Seek (&it, ctx->graph_idx) ;

	bool it_reset = false ; // true if graph iterator had gone a full cycle
	GraphContext *gc = NULL ;

	// as long as we've got processing time
	while (TIMER_GET_ELAPSED_MILLISECONDS (stopwatch) < DEADLINE) {
		// get next graph to offload attribute sets from
		gc = GraphIterator_Next (&it) ;

		// iterator depleted
		if (gc == NULL) {
			if (it_reset) {
				RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
					"offload task scanned through the entire keyspace") ;
				printf ("offload task scanned through the entire keyspace\n") ;
				break ;
			}

			RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG,
					"Finished scanning through all graphs in keyspace, "
					"resetting graphs iterator") ;

			it_reset = true ;

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
		size_t bytes_offloaded = _offloadGraph (ctx, gc, stopwatch) ;
		printf ("Graph %s offloaded: %.2f MB\n",
				gc->graph_name, (double)(BYTES_TO_MB (bytes_offloaded))) ;

		// either we've offloaded the entire graph or we're out of time
		// regardless decrease the graph's ref count
		GraphContext_DecreaseRefCount (gc) ;
	}

	RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_DEBUG, "Time quota expired") ;

	// indicate if there's additional work to be done
	size_t new_rss = (get_current_rss () / (1024 * 1024)) ;

	// spped up if we've managed to decrease memory consumption
	// but we're still above the threshold
	bool speedup = (new_rss < rss) && (new_rss > 500) ;
	return speedup ;
}

