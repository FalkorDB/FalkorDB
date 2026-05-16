/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "indexer.h"
#include "../redismodule.h"
#include "../util/rmalloc.h"
#include <assert.h>
#include <pthread.h>

// lock indexer task queue
#define INDEXER_LOCK_QUEUE()                            \
	do {                                                \
		int res = pthread_mutex_lock (&indexer->m) ;    \
		ASSERT (res == 0) ;                             \
	} while (0) ;

// unlock indexer task queue
#define INDEXER_UNLOCK_QUEUE()                          \
	do {                                                \
		int res = pthread_mutex_unlock (&indexer->m) ;  \
		ASSERT (res == 0) ;                             \
	} while (0) ;

// operations performed by indexer
typedef enum {
	INDEXER_IDX_DROP,            // drop index
	INDEXER_IDX_POPULATE,        // populate index
	INDEXER_CONSTRAINT_DROP,     // drop index
	INDEXER_CONSTRAINT_ENFORCE,  // populate index
	INDEXER_EXIT,                // fake task which will cause indexer to exit
} IndexerOp;

// indexer task
typedef struct {
	IndexerOp op;  // type of task
	void *pdata;   // task private data
} IndexerTask;

// index population context
typedef struct {
	Schema *s;         // schema containing the index
	Index idx;         // index to populate
	GraphContext *gc;  // graph holding entities to index
} IndexPopulateCtx;

// index drop context
typedef struct {
	Index idx;         // index to populate
	GraphContext *gc;  // graph holding entities to index
} IndexDropCtx;

// constraint enforce context
typedef struct {
	GraphContext *gc;  // graph object
	Constraint c;      // constraint to enforce
} ConstraintEnforceCtx;

// constraint drop context
typedef struct {
	GraphContext *gc;  // graph object
	Constraint c;      // constraint to enforce
} ConstraintDropCtx;

typedef struct {
	pthread_t t;         // worker thread handel
	pthread_mutex_t m;   // queue mutex (also guards condition variable)
	pthread_cond_t c;    // conditional variable
	IndexerTask * q;     // task queue
} Indexer;

// forward declarations
static void _indexer_PopTask(IndexerTask *task);

static Indexer *indexer = NULL;

// clear indexer's tasks
static void _Indexer_ClearTasks(void) {
	INDEXER_LOCK_QUEUE () ;
	arr_clear (indexer->q) ;
	INDEXER_UNLOCK_QUEUE () ;
}

// index populate task handler
static void _indexer_idx_populate
(
	IndexPopulateCtx *ctx
) {
	Index idx = ctx->idx;
	GraphContext *gc = ctx->gc;
	Graph *g = GraphContext_GetGraph (gc) ;

	// populate index
	Index_Populate (idx, g) ;

	// we're required to hold both GIL and write lock
	// as Schema_ActivateIndex might drop an index
	RedisModuleCtx *rm_ctx = RedisModule_GetThreadSafeContext(NULL);
	RedisModule_ThreadSafeContextLock(rm_ctx);
	Graph_AcquireWriteLock (g) ;

	// index populated, try to enable
	Index_Enable(idx);

	if(Index_Enabled(idx)) {
		Schema_ActivateIndex(ctx->s);
	}

	// release locks
	Graph_ReleaseLock (g) ;
	RedisModule_ThreadSafeContextUnlock(rm_ctx);
	RedisModule_FreeThreadSafeContext(rm_ctx);

	// decrease graph reference count
	GraphContext_DecreaseRefCount(ctx->gc);

	rm_free(ctx);
}

// index drop task handler
static void _indexer_idx_drop
(
	IndexDropCtx *ctx
) {
	RedisModuleCtx *rm_ctx = RedisModule_GetThreadSafeContext(NULL);
	RedisModule_ThreadSafeContextLock(rm_ctx);

	// TODO: expecting index pending_changes count to be either 0 or 1
	Index_Free(ctx->idx);

	RedisModule_ThreadSafeContextUnlock(rm_ctx);
	RedisModule_FreeThreadSafeContext(rm_ctx);

	// decrease graph reference count
	GraphContext_DecreaseRefCount(ctx->gc);

	rm_free(ctx);
}

// constraint enforce task handler
static void _indexer_enforce_constraint
(
	ConstraintEnforceCtx *ctx
) {
	Constraint c = ctx->c;
	GraphContext *gc = ctx->gc;
	Graph *g = GraphContext_GetGraph(gc);

	// unique constraint uses index to enforce the constraint
	// if the index is not enabled, we'll delay the enforcement
	// until the index is ready
	if(Constraint_GetType(c) == CT_UNIQUE) {
		Index idx = Constraint_GetPrivateData(c);
		// if index is not enabled and the constraint is not marked for deletion
		// postpone the enforcement
		if(!Index_Enabled(idx) && Constraint_PendingChanges(c) == 1) {
			Indexer_EnforceConstraint(c, gc);
			goto cleanup;
		}
	}

	// try to enforce constraint on all relevent entities
	if(Constraint_GetEntityType(c) == GETYPE_NODE) {
		Constraint_EnforceNodes(c, g);
	} else {
		Constraint_EnforceEdges(c, g);
	}

	// replicate constraint if active
	// upon constraint creation
	// it is possible for the primary shard to replicate the constraint via RDB
	// in which case the constraint wouldn't be included
	// as only active constraints are encoded within RDBs
	// to make sure the constraint is introduced to the replica we re-issue it
	// once the constraint becomes active
	if(Constraint_GetStatus(c) == CT_ACTIVE) {
		// lock before calling replicate
		RedisModuleCtx *rm_ctx = RedisModule_GetThreadSafeContext(NULL);
		RedisModule_ThreadSafeContextLock(rm_ctx);

		Constraint_Replicate(rm_ctx, c, (const struct GraphContext*)gc);

		// unlock and free
		RedisModule_ThreadSafeContextUnlock(rm_ctx);
		RedisModule_FreeThreadSafeContext(rm_ctx);
	}

	// decrease number of pending changes
	Constraint_DecPendingChanges(c);

cleanup:
	// decrease graph reference count
	GraphContext_DecreaseRefCount(gc);

	rm_free(ctx);
}

// constraint drop task handler
static void _indexer_drop_constraint
(
	ConstraintDropCtx *ctx
) {
	Constraint_Free(&ctx->c);

	// decrease graph reference count
	GraphContext_DecreaseRefCount(ctx->gc);

	rm_free(ctx);
}

// populate index
// this function executes on the indexer's worker thread
static void *_indexer_run
(
	void *arg
) {
	while(true) {
		// pop an item from queue
		// if queue is empty thread will be put to sleep
		IndexerTask ctx;
		_indexer_PopTask(&ctx);

		switch(ctx.op) {
			case INDEXER_IDX_POPULATE:
			{
				IndexPopulateCtx *pdata = (IndexPopulateCtx*)ctx.pdata;
				_indexer_idx_populate(pdata);
				break;
			}

			case INDEXER_IDX_DROP:
			{
				IndexDropCtx *pdata = (IndexDropCtx*)ctx.pdata;
				_indexer_idx_drop(pdata);
				break;
			}

			case INDEXER_CONSTRAINT_ENFORCE:
			{
				ConstraintEnforceCtx *pdata = (ConstraintEnforceCtx*)ctx.pdata;
				_indexer_enforce_constraint(pdata);
				break;
			}

			case INDEXER_CONSTRAINT_DROP:
			{
				ConstraintDropCtx *pdata = (ConstraintDropCtx*)ctx.pdata;
				_indexer_drop_constraint(pdata);
				break;
			}

			case INDEXER_EXIT:
			{
				return NULL ;
			}

			default:
				assert(false && "unknown indexer operation");
				break;
		}
	}

	return NULL;
}

// add a new task to indexer queue
void _indexer_AddTask
(
	IndexerOp op,
	void *pdata
) {
	// add task to queue
	IndexerTask	task = {.op = op, .pdata = pdata} ;

	// update the predicate and signal the condition variable while holding
	// the mutex that guards both. signaling inside the critical section
	// avoids the classic lost-wakeup race in which a consumer evaluates the
	// predicate (empty queue) and is preempted before it begins waiting,
	// while the producer signals a condition variable nobody is waiting on.
	INDEXER_LOCK_QUEUE () ;
	arr_append (indexer->q, task) ;
	pthread_cond_signal (&indexer->c) ;
	INDEXER_UNLOCK_QUEUE () ;
}

// pops a task from queue
// if queue is empty caller will be waiting on conditional variable
static void _indexer_PopTask
(
	IndexerTask *task
) {
	ASSERT(task != NULL);

	// lock queue
	INDEXER_LOCK_QUEUE () ;

	// wait for work; use a while-loop to tolerate spurious wakeups and to
	// re-check the predicate after the mutex is re-acquired
	while (arr_len (indexer->q) == 0) {
		pthread_cond_wait (&indexer->c, &indexer->m) ;
	}

	*task = indexer->q[0] ;
	arr_del (indexer->q, 0) ;

	INDEXER_UNLOCK_QUEUE () ;
}

// initialize indexer
// create indexer worker thread and task queue
bool Indexer_Init(void) {
	ASSERT(indexer == NULL);

	// track which primitives were successfully initialized so that the
	// cleanup path only destroys what was actually created. destroying an
	// uninitialized pthread primitive is undefined behaviour and can
	// corrupt pthread internal state for subsequent Init calls.
	bool m_init    = false;  // queue mutex initialized
	bool c_init    = false;  // condition variable initialized
	bool attr_init = false;  // pthread attr initialized

	pthread_attr_t attr;

	indexer = rm_calloc(1, sizeof(Indexer));

	// create queue mutex
	if(pthread_mutex_init(&indexer->m, NULL) != 0) {
		goto cleanup;
	}
	m_init = true;

	// create conditional var
	if(pthread_cond_init(&indexer->c, NULL) != 0) {
		goto cleanup;
	}
	c_init = true;

	// create task queue
	indexer->q = arr_new (IndexerTask, 0) ;

	// create worker thread
	if(pthread_attr_init(&attr) != 0) {
		goto cleanup;
	}
	attr_init = true;

	if(pthread_create(&indexer->t, &attr, _indexer_run, NULL) != 0) {
		goto cleanup;
	}

	pthread_attr_destroy(&attr);

	return true;

cleanup:
	if(attr_init) {
		pthread_attr_destroy(&attr);
	}

	if(c_init) {
		pthread_cond_destroy(&indexer->c);
	}

	if(m_init) {
		pthread_mutex_destroy(&indexer->m);
	}

	if(indexer->q != NULL) {
		arr_free (indexer->q) ;
	}

	rm_free(indexer);
	indexer = NULL;

	return false;
}

// populates index asynchronously
// this function simply place the population request onto a queue
// eventually the indexer working thread will pick it up and populate the index
void Indexer_PopulateIndex
(
	GraphContext *gc, // graph to operate on
	Schema *s,        // schema containing the idx
	Index idx         // index to populate
) {
	ASSERT(s       != NULL);
	ASSERT(gc      != NULL);
	ASSERT(idx     != NULL);
	ASSERT(indexer != NULL);
	ASSERT(Index_Enabled(idx) == false);

	// create work item
	IndexPopulateCtx *ctx = rm_malloc(sizeof(IndexPopulateCtx));
	ctx->s   = s;
	ctx->gc  = gc;
	ctx->idx = idx;

	// increase graph reference count
	// count will be reduced once this task is perfomed
	// this is done to handle the case where a graph has pending index
	// population tasks and it is being asked to be deleted
	GraphContext_IncreaseRefCount(gc);

	// place task into queue
	_indexer_AddTask(INDEXER_IDX_POPULATE, ctx);
}

// drops index asynchronously
// this function simply place the drop request onto a queue
// eventually the indexer working thread will pick it up and drop the index
void Indexer_DropIndex
(
	Index idx,        // index to drop
	GraphContext *gc  // graph context
) {
	ASSERT(idx     != NULL);
	ASSERT(indexer != NULL);

	// create work item
	IndexDropCtx *ctx = rm_malloc(sizeof(IndexDropCtx));
	ctx->gc  = gc;
	ctx->idx = idx;

	// increase graph reference count
	// count will be reduced once this task is perfomed
	// this is done to handle the case where a graph has pending index
	// population tasks and it is being asked to be deleted
	GraphContext_IncreaseRefCount(gc);

	// place task into queue
	_indexer_AddTask(INDEXER_IDX_DROP, ctx);
}

// enforces constraint
// adds the task for enforcing the given constraint to the indexer
void Indexer_EnforceConstraint
(
	Constraint c,     // constraint to enforce
	GraphContext *gc  // graph context
) {
	ASSERT(c       != NULL);
	ASSERT(gc      != NULL);
	ASSERT(indexer != NULL);

	ConstraintEnforceCtx *ctx = rm_malloc(sizeof(ConstraintEnforceCtx));
	ctx->c  = c;
	ctx->gc = gc;

	// increase graph reference count
	// count will be reduced once this task is perfomed
	// this is done to handle the case where a graph has pending constraint
	// enforcement tasks and it is being asked to be deleted
	GraphContext_IncreaseRefCount(gc);

	_indexer_AddTask(INDEXER_CONSTRAINT_ENFORCE, ctx);
}

// drops constraint asynchronously
// this function simply place the drop constraint request onto the queue
// eventually the indexer working thread will pick it up and drop the constraint
void Indexer_DropConstraint
(
	Constraint c,     // constraint to drop
	GraphContext *gc  // graph context
) {
	ASSERT(c       != NULL);
	ASSERT(indexer != NULL);

	ConstraintDropCtx *ctx = rm_malloc(sizeof(ConstraintDropCtx));
	ctx->c  = c;
	ctx->gc = gc;

	// increase graph reference count
	// count will be reduced once this task is perfomed
	// this is done to handle the case where a graph has pending index
	// population tasks and it is being asked to be deleted
	GraphContext_IncreaseRefCount(gc);

	// place task into queue
	_indexer_AddTask(INDEXER_CONSTRAINT_DROP, ctx);
}

// stop and free indexer
void Indexer_Stop(void) {
	// add fake task to cause indexer thread to exit
	_Indexer_ClearTasks () ;
	_indexer_AddTask (INDEXER_EXIT, NULL) ;
	
	// wait for indexer thread to exit
	pthread_join (indexer->t, NULL) ;

	// free indexer
	arr_free (indexer->q) ;
	pthread_cond_destroy (&indexer->c) ;
	pthread_mutex_destroy (&indexer->m) ;

	rm_free (indexer) ;
	indexer = NULL;
}

