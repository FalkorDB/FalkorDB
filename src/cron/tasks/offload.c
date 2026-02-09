/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// task context
typedef struct {
	uint32_t graph_idx;  // current processed graph
	EntityID entity_id;  // current processed entity id
	GraphEntityType t;   // current processed entity type Node/Edge
} OffloadTaskCtx;

void *CronTask_newOffloadEntities
(
	void *pdata  // [optional] task context
) {
	OffloadTaskCtx *ctx = rm_calloc (sizeof (OffloadTaskCtx)) ;
	ASSERT (ctx != NULL) ;

	// default values
	ctx->t = GETYPE_NODE ;
	ctx->entity_id = 0 ;
	ctx->graph_idx = 0 ;

	if (pdata != NULL) {
		// create a new offload task continuing from where the previous one
		// (pdata) had leftoff
		OffloadTaskCtx *prev_ctx = (OffloadTaskCtx*) ;
		memcpy (ctx, prev_ctx, sizeof (OffloadTaskCtx)) ;
	}

	// trival validations
	ASSERT (ctx->t == GETYPE_NODE || ctx->t == GETYPE_EDGE) ;
	ASSERT (ctx->entity_id != INVALID_ENTITY_ID) ;

	return ctx ;
}

bool CronTask_offloadEntities
(
	void *pdata  // task context
) {
	OffloadTaskCtx *ctx    = (OffloadTaskCtx*)pdata ;
	RedisModuleCtx *rm_ctx = RedisModule_GetThreadSafeContext (NULL) ;

	// start stopwatch
	double deadline = 3;  // 3ms
	simple_timer_t stopwatch ;
	simple_tic (stopwatch) ;

	uint64_t max_sets_count = 1000 ;  //  max number of attribute-sets to offload

	EntityID *ids  = rm_calloc (sizeof (EntityID) * max_sets_count) ;
	if (ids == NULL) {
		return true ;
	}

	AttributeSet *sets = rm_calloc (sizeof (AttributeSet) * max_sets_count) ;
	if (sets == NULL) {
		rm_free (ids) ;
		return true ;
	}

	KeySpaceGraphIterator it ;
	Globals_ScanGraphs (&it) ;

	// pick up from where we've left
	GraphIterator_Seek (&it, ctx->graph_idx) ;

	GraphContext *gc = NULL ;

	// as long as we've got processing time
	while (TIMER_GET_ELAPSED_MILLISECONDS (stopwatch) < deadline) {
		// get next graph to offload attribute sets from
		gc = GraphIterator_Next (&it) ;

		// iterator depleted
		if ((gc) == NULL) {
			// reset graph index
			ctx->graph_idx = 0 ;
			GraphIterator_Seek (&it, ctx->graph_idx) ;
			continue ;
		}

		Graph *g = GraphContext_GetGraph (gc) ;

		// acquire READ lock
		Graph_AcquireReadLock (g) ;

		//----------------------------------------------------------------------
		// attach entity iterator
		//----------------------------------------------------------------------

		DataBlock datablock = (ctx->t == GETYPE_NODE) ? g->nodes : g->edges ;
		DataBlockIterator *it = DataBlock_Scan (datablock) ;
		ASSERT (it != NULL) ;

		// continue from where we've left off
		DataBlockIterator_Seek (it, ctx->entity_id) ;

		size_t i = 0 ;
		AttributeSet set = NULL ;
		for (; i < max_sets_count ; i++) {
			sets[i] = DataBlockIterator_Next (it, ids + i) ;
			if (sets[i] == NULL) {
				// depleted
				if (ctx->t == GETYPE_NODE) {
					// next iteration scans edges
					ctx->t = GETYPE_EDGE ;
					ctx->entity_id = 0 ;
				}
				break ;
			}
		}

		//----------------------------------------------------------------------
		// offload attribute sets
		//----------------------------------------------------------------------

		if (i > 0) {
			if (Storage_putAttributes (gc->cf, sets, i, ctx->t, ids) == 0) {
				// attribute sets been offloaded to disk
				// mark datablock entries as offloaded
				DataBlock_MarkOffloaded (dataBlock, ids, i) ;

				// attribute sets been offloaded to disk and the datablock
				// marked each one as offloaded
				// now we're ready to free attribute-sets
				// to do that we must hold the graph's write lock

				Graph_ReleaseLock (g) ;
				Graph_AcquireWriteLock (g) ;

				// "shallow" free attribute-sets
				while (i > 0) {
					rm_free (sets[--i]) ;
				}
			}
		}

		Graph_ReleaseLock (g) ;
		GraphContext_DecreaseRefCount (gc) ;
	}

	if (gil_acquired) {
		RedisModule_ThreadSafeContextUnlock (rm_ctx) ;
	}

	RedisModule_FreeThreadSafeContext (rm_ctx) ;

	// set next iteration graph index
	ctx->graph_idx = (gc == NULL) ? 0 : ctx->graph_idx;

	// indicate if there's additional work to be done
	return (gc != NULL) ;
}

