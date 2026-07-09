/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "globals.h"
#include "../util/arr.h"
#include "../util/uuid.h"
#include "../query_ctx.h"
#include "graphcontext.h"
#include "../redismodule.h"
#include "../util/rwlock.h"
#include "../util/rmalloc.h"
#include "graph_memoryUsage.h"
#include "../util/thpool/pool.h"
#include "../constraint/constraint.h"
#include "../util/identifier_limits.h"
#include "../commands/execution_ctx.h"
#include "../serializers/graphcontext_type.h"

#include <time.h>
#include <pthread.h>
#include <sys/param.h>
#include <stdatomic.h>

// telemetry stream format
#define TELEMETRY_FORMAT "telemetry{%s}"

// import the GraphContext struct
#include "graphcontext_struct.h"

// forward declarations
static void _DeleteTelemetryStream(RedisModuleCtx *ctx, const GraphContext *gc);

// increase graph context ref count by 1
inline void GraphContext_IncreaseRefCount
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;
	__atomic_fetch_add (&gc->ref_count, 1, __ATOMIC_RELAXED) ;
}

// decrease graph context ref count by 1
inline void GraphContext_DecreaseRefCount
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	// if the reference count is 0
	// the graph has been marked for deletion and no queries are active
	// free the graph
	if (__atomic_sub_fetch (&gc->ref_count, 1, __ATOMIC_RELAXED) == 0) {
		bool async_delete ;
		Config_Option_get (Config_ASYNC_DELETE, &async_delete) ;

		if (async_delete) {
			// Async delete
			// add deletion task to pool using force mode
			// we can't lose this task in-case pool's queue is full
			ThreadPool_AddWork ((void (*)(void *))GraphContext_Free, gc, 1) ;
		} else {
			// Sync delete
			GraphContext_Free (gc) ;
		}
	}
}

// return graph context reference count
int GraphContext_RefCount
(
	const GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	return gc->ref_count ;
}

//------------------------------------------------------------------------------
// pending array accessors
//------------------------------------------------------------------------------

static Schema **_GetNodeSchemas
(
	GraphContext *gc
) {
	if (unlikely (gc->_node_schemas != NULL &&
				  pthread_equal (gc->writer_tid, pthread_self ()) != 0)) {
		return gc->_node_schemas ;
	}

	return gc->node_schemas ;
}

static Schema **_GetRelationSchemas
(
	GraphContext *gc
) {
	if (unlikely (gc->_relation_schemas != NULL &&
				  pthread_equal (gc->writer_tid, pthread_self ()) != 0)) {
		return gc->_relation_schemas ;
	}

	return gc->relation_schemas ;
}

static char **_GetAttributes
(
	GraphContext *gc
) {
	if (unlikely (gc->_attributes != NULL &&
				  pthread_equal (gc->writer_tid, pthread_self ()) != 0)) {
		return gc->_attributes ;
	}

	return gc->attributes ;
}

static void _CreateRWLocks
(
	GraphContext *gc
) {
	// create a read write lock which favors writes
	//
	// consider the following locking sequence:
	// T0 read lock  (acquired)
	// T1 write lock (waiting)
	// T2 read lock  (acquired if lock favor reads, waiting if favor writes)
	//
	// we don't want to cause write starvation as this can impact overall
	// system performance

	// specify prefer write in lock creation attributes
	int res = 0 ;
	UNUSED (res) ;

	pthread_rwlockattr_t attr ;
	res = pthread_rwlockattr_init (&attr) ;
	ASSERT (res == 0) ;

#ifdef PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP
	int pref = PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP ;
	res = pthread_rwlockattr_setkind_np (&attr, pref) ;
	ASSERT (res == 0) ;
#endif

	res = pthread_rwlock_init (&gc->rwlock, &attr) ;
	ASSERT (res == 0) ;
}

// update graph context hash
static void _GraphContext_UpdateHash
(
	GraphContext *gc
) {
	// update graph hash by hashing newly added schema additions
	//
	// Using the current graph hash as a seed, by doing so we avoid
	// hashing the entire graph schema on each change, while guaranteeing the
	// exact same hash across a cluster: same graph hash on both
	// primary and replica shards

	ASSERT (gc->_attributes       != NULL ||
			gc->_node_schemas     != NULL ||
			gc->_relation_schemas != NULL) ;

	uint new_count   ;  // number of elements in pending array
	uint prev_count  ;  // number of elements in current array
	const char *name ;  // name of attribute / schema

	// update graph's hash with new attributes / schemas
	XXH32_state_t *state = XXH32_createState () ;
	XXH32_reset (state, gc->hash) ;

	//--------------------------------------------------------------------------
	// hash attributes
	//--------------------------------------------------------------------------

	if (gc->_attributes != NULL) {
		new_count  = arr_len (gc->_attributes) ;
		prev_count = arr_len (gc->attributes)  ;

		for (uint i = prev_count ; i < new_count ; i++) {
			const char *name = gc->_attributes [i] ;
			XXH32_update (state, name, strlen (name)) ;
		}
	}

	//--------------------------------------------------------------------------
	// hash node schemas
	//--------------------------------------------------------------------------

	if (gc->_node_schemas != NULL) {
		new_count  = arr_len (gc->_node_schemas) ;
		prev_count = arr_len (gc->node_schemas)  ;

		for (uint i = prev_count ; i < new_count ; i++) {
			Schema *s = gc->_node_schemas [i] ;
			const char *name = Schema_GetName (s) ; 
			XXH32_update (state, name, strlen (name)) ;
		}
	}

	//--------------------------------------------------------------------------
	// hash relation schemas
	//--------------------------------------------------------------------------

	if (gc->_relation_schemas != NULL) {
		new_count  = arr_len (gc->_relation_schemas) ;
		prev_count = arr_len (gc->relation_schemas)  ;

		for (uint i = prev_count ; i < new_count ; i++) {
			Schema *s = gc->_relation_schemas [i] ;
			const char *name = Schema_GetName (s) ; 
			XXH32_update (state, name, strlen (name)) ;
		}
	}

	// finalize hash
	gc->hash = XXH32_digest (state) ;
	XXH32_freeState (state) ;
}

// commit graph's pending schema changes
static void _GraphContext_CommitPendings
(
	GraphContext *gc  // graph context
) {
	ASSERT (gc != NULL) ;

	// only writer thread is allowed to commit pending changes
	if (pthread_equal (gc->writer_tid, pthread_self ()) == 0) {
		return ;
	}

	if (gc->_attributes       == NULL &&
		gc->_node_schemas     == NULL &&
		gc->_relation_schemas == NULL) {
		// no changes
		ASSERT (gc->writer_tid == (pthread_t)0) ;
		return ;
	}

	_GraphContext_UpdateHash (gc) ;

	//--------------------------------------------------------------------------
	// commit pending attributes
	//--------------------------------------------------------------------------

	if (gc->_attributes != NULL) {
		ASSERT (arr_len (gc->_attributes) >= arr_len (gc->attributes)) ;

		if (arr_len (gc->_attributes) == arr_len (gc->attributes)) {
			// no new attributes
			// undo occurred since pending == commited
			arr_free (gc->_attributes) ;
		} else {
			// introduce new attributes
			arr_free (gc->attributes) ;
			gc->attributes = gc->_attributes ;
		}
		gc->_attributes = NULL ;
	}

	//--------------------------------------------------------------------------
	// commit pending node schemas
	//--------------------------------------------------------------------------

	if (gc->_node_schemas != NULL) {
		ASSERT (arr_len (gc->_node_schemas) >= arr_len (gc->node_schemas)) ;

		if (arr_len (gc->_node_schemas) == arr_len (gc->node_schemas)) {
			// no new node schemas
			arr_free (gc->_node_schemas) ;
		} else {
			// introduce new node schemas
			arr_free (gc->node_schemas) ;
			gc->node_schemas = gc->_node_schemas ;
		}
		gc->_node_schemas = NULL ;
	}

	//--------------------------------------------------------------------------
	// commit pending relationship schemas
	//--------------------------------------------------------------------------

	if (gc->_relation_schemas != NULL) {
		ASSERT (arr_len (gc->_relation_schemas) >= arr_len (gc->relation_schemas)) ;

		if (arr_len (gc->_relation_schemas) == arr_len (gc->relation_schemas)) {
			// no new relationship schemas
			arr_free (gc->_relation_schemas) ;
		} else {
			// introduce new relationship schemas
			arr_free (gc->relation_schemas) ;
			gc->relation_schemas = gc->_relation_schemas ;
		}
		gc->_relation_schemas = NULL ;
	}

	// reset tid to 0
	gc->writer_tid = (pthread_t)0 ;

	// commit graph's pending schema changes
	Graph_CommitPendingsMatrices (gc->g) ;
}

//------------------------------------------------------------------------------
// GraphContext API
//------------------------------------------------------------------------------

// creates and initializes a graph context struct
GraphContext *GraphContext_New
(
	const char *graph_name
) {
	GraphContext *gc = rm_calloc (1, sizeof (GraphContext)) ;

	gc->hash        = 0 ;  // initial graph hash
	gc->slowlog     = SlowLog_New () ;
	gc->queries_log = QueriesLog_New () ;
	gc->ref_count   = 0 ;  // no refences

	gc->index_count      = 0 ;  // no indicies
	gc->encoding_context = GraphEncodeContext_New () ;
	gc->decoding_context = GraphDecodeContext_New () ;

	// initialize a read-write lock scoped to the individual graph
	_CreateRWLocks (gc) ;
	gc->writelocked = false ;

	// initial graph's write in progress atomic flag to false
	atomic_init (&gc->write_in_progress, false) ;

	// create graph's pending write queries queue
	gc->pending_write_queue = CircularBuffer_New (sizeof (void*), 1024) ;

	// read NODE_CREATION_BUFFER size from configuration
	// this value controls how much extra room we're willing to spend for:
	// 1. graph entity storage
	// 2. matrices dimensions
	size_t node_cap ;
	size_t edge_cap ;
	bool rc = Config_Option_get (Config_NODE_CREATION_BUFFER, &node_cap) ;
	assert (rc) ;
	edge_cap = node_cap ;

	gc->g = Graph_New (node_cap, edge_cap) ;
	gc->graph_name = rm_strdup (graph_name) ;
	gc->telemetry_stream = RedisModule_CreateStringPrintf (NULL,
			TELEMETRY_FORMAT, gc->graph_name) ;

	// build the execution plans cache
	uint64_t cache_size ;
	Config_Option_get (Config_CACHE_SIZE, &cache_size) ;
	gc->cache = Cache_New (cache_size, (CacheEntryFreeFunc)ExecutionCtx_Free,
						  (CacheEntryCopyFunc)ExecutionCtx_Clone) ;

	Graph_SetMatrixPolicy (gc->g, SYNC_POLICY_FLUSH_RESIZE) ;

	return gc ;
}

//------------------------------------------------------------------------------
// Synchronization functions
//------------------------------------------------------------------------------

// acquires a READ lock on the graph context
void GraphContext_AcquireReadLock
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	int res = pthread_rwlock_rdlock (&gc->rwlock) ;
	ASSERT (res == 0) ;

	gc->writelocked = false ;
}

// acquires a WRITE lock on the graph context
void GraphContext_AcquireWriteLock
(
	GraphContext *gc  // graph context
) {
	ASSERT (gc != NULL) ;

	if (gc->writelocked == true) {
		return ;
	}

	pthread_rwlock_wrlock (&gc->rwlock) ;
	gc->writelocked = true ;
}

// acquire the graph context write lock with a timeout
// attempts to acquire the write lock on the given graphcontext
// if the lock is not acquired immediately the function will block until either
// the lock becomes available or the timeout elapses
//
// returns:
// - 0 on success (lock acquired)
// - ETIMEDOUT if the timeout expired before acquiring the lock
// - EBUSY if called with timeout_ms == 0 and the lock could not be acquired
// - other nonzero error codes may be returned for unexpected failures
int GraphContext_TimeAcquireWriteLock
(
	GraphContext *gc,  // graph to lock
	int timeout_ms     // maximum time in milliseconds to wait for the lock:
                       // - timeout_ms < 0 : block until the lock is acquired
                       // - timeout_ms = 0 : non-blocking attempt (try-lock)
                       // - timeout_ms > 0 : wait up to timeout_ms milliseconds
) {
	ASSERT (gc != NULL) ;
	if (gc->writelocked == true) {
		return 0 ;
	}

	int res = rwlock_timedwrlock (&gc->rwlock, timeout_ms) ;
	gc->writelocked = (res == 0) ;

	return res ;
}

void GraphContext_ReleaseReadLock
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;
	ASSERT (gc->writelocked == false) ;

	// set default synchronization behavior
	Graph_SetMatrixPolicy (gc->g, SYNC_POLICY_FLUSH_RESIZE) ;

	pthread_rwlock_unlock (&gc->rwlock) ;
}

// releases the lock currently held on the graph context
// must be called exactly once for every successful acquire call
void GraphContext_ReleaseLock
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	_GraphContext_CommitPendings (gc) ;

	// set writelocked to false BEFORE unlocking
	// if this is a reader thread no harm done,
	// if this is a writer thread the writer is about to unlock so once again
	// no harm done, if we set `writelocked` to false after unlocking
	// it is possible for a reader thread to be considered as writer
	// performing illegal access to underline graph
	// consider a context switch after unlocking `rwlock`
	// but before setting `writelocked` to false
	gc->writelocked = false ;

	// set default synchronization behavior
	Graph_SetMatrixPolicy (gc->g, SYNC_POLICY_FLUSH_RESIZE) ;

	pthread_rwlock_unlock (&gc->rwlock) ;
}

// returns rather or not graph is locked for writing
bool GraphContext_IsWriteLocked
(
	const GraphContext *gc
) {
	ASSERT (gc != NULL) ;
	return gc->writelocked ;
}

// mark graph key as "dirty" for Redis to pick up on
void GraphContext_MarkWriter
(
	RedisModuleCtx *ctx,
	GraphContext *gc
) {
	RedisModuleString *graphID =
		RedisModule_CreateString (ctx, gc->graph_name, strlen (gc->graph_name)) ;

	// reopen only if key exists (do not re-create) make sure key still exists
	RedisModuleKey *key = RedisModule_OpenKey (ctx, graphID, REDISMODULE_READ) ;
	if (RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_EMPTY) {
		goto cleanup ;
	}
	RedisModule_CloseKey (key) ;

	// mark as writer
	key = RedisModule_OpenKey (ctx, graphID, REDISMODULE_WRITE) ;
	RedisModule_CloseKey (key) ;

cleanup:
	RedisModule_FreeString (ctx, graphID) ;
}

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
		return acquired ;
	}

	// failed to acquire, sleep and retry
	if (timeout_ms > 0) {
		int remaining_ms = timeout_ms ;
		int sleep_for_ms = MIN (5, timeout_ms) ;

		// sleep for 5ms
		struct timespec ts = {
			.tv_sec = 0,
			.tv_nsec = sleep_for_ms * 1000000  // 5 ms
		};

		while (remaining_ms > 0) {
			// sleep and retry
			nanosleep (&ts, NULL) ;

			expected = false ;  // reset, CAS clobbers it on failure
			acquired = atomic_compare_exchange_strong (&gc->write_in_progress,
					&expected, true) ;

			if (acquired == true) {
				return acquired ;
			}

			remaining_ms -= sleep_for_ms ;
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

// enqueue a write query for deferred execution on the specified graph
// returns true if the query was successfully enqueued
// false if the enqueue operation failed (e.g., due to allocation failure)
bool GraphContext_EnqueueWriteQuery
(
	GraphContext *gc,  // graph context
	void *query_ctx    // query context
) {
	ASSERT (gc        != NULL) ;
	ASSERT (query_ctx != NULL) ;

	return (CircularBuffer_Add (gc->pending_write_queue, &query_ctx) != 0) ;
}

// dequeue the next pending write query for the specified graph
// returns a query context pointer if a query was dequeued,
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

// checks if the graph's pending write queue is empty
bool GraphContext_WriteQueueEmpty
(
	const GraphContext *gc  // graph context
) {
	return CircularBuffer_Empty(gc->pending_write_queue);
}

const char *GraphContext_GetName
(
	const GraphContext *gc
) {
	ASSERT (gc != NULL) ;
	return gc->graph_name ;
}

// get graph context's telemetry stream name
const RedisModuleString *GraphContext_GetTelemetryStreamName
(
	const GraphContext *gc
) {
	ASSERT (gc != NULL) ;
	ASSERT (gc->telemetry_stream != NULL) ;

	return gc->telemetry_stream ;
}

void GraphContext_FreeTelemetryStreamName
(
	GraphContext *gc,
	RedisModuleCtx *ctx
) {
	ASSERT (gc != NULL) ;

	if (gc->telemetry_stream != NULL) {
		RedisModule_FreeString (ctx, gc->telemetry_stream) ;
		gc->telemetry_stream = NULL ;
	}
}

// rename a graph context
void GraphContext_Rename
(
	RedisModuleCtx *ctx,  // redis module context
	GraphContext *gc,     // graph context to rename
	const char *name      // new name
) {
	rm_free(gc->graph_name);
	gc->graph_name = rm_strdup(name);

	// drop old telemetry stream
	_DeleteTelemetryStream(ctx, gc);

	// recreate telemetry stream name
	RedisModule_FreeString(ctx, gc->telemetry_stream);
	gc->telemetry_stream = RedisModule_CreateStringPrintf(NULL,
			TELEMETRY_FORMAT, gc->graph_name);
}

XXH32_hash_t GraphContext_GetHash
(
	const GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	return gc->hash ;
}

// get graph from graph context
Graph *GraphContext_GetGraph
(
	const GraphContext *gc
) {
	ASSERT(gc != NULL);

	return gc->g;
}

// returns the graph's current RAM footprint in bytes
// samples attributes and indices for an accurate estimate
uint64_t GraphContext_MemoryUsage
(
	const GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	GraphContext *_gc = (GraphContext *)gc ;

	MemoryUsageResult result = {0} ;
	result.node_attr_by_label_sz = arr_new (size_t, 0) ;
	result.edge_attr_by_type_sz  = arr_new (size_t, 0) ;

	GraphContext_AcquireReadLock (_gc) ;
	GraphContext_EstimateMemoryUsage (_gc, 1000, &result) ;
	GraphContext_ReleaseReadLock (_gc) ;

	arr_free (result.node_attr_by_label_sz) ;
	arr_free (result.edge_attr_by_type_sz) ;

	return (uint64_t)result.total_graph_sz_mb * (1 << 20) ;
}

//------------------------------------------------------------------------------
// Schema API
//------------------------------------------------------------------------------

// returns the number of schemas of the given type visible
// schemas are stored in insertion order
unsigned short GraphContext_SchemaCount
(
	GraphContext *gc,
	SchemaType t
) {
	ASSERT (gc != NULL) ;

	if (t == SCHEMA_NODE) {
		return arr_len (_GetNodeSchemas (gc)) ;
	} else {
		return arr_len (_GetRelationSchemas (gc)) ;
	}
}

// checks if graph has constraints
bool GraphContext_HasConstraints
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	Schema **schemas = _GetNodeSchemas (gc) ;
	uint n = arr_len (schemas) ;

	for (uint i = 0 ; i < n ; i++) {
		Schema *s = schemas [i] ;
		if (Schema_HasConstraints (s)) {
			return true ;
		}
	}

	schemas = _GetRelationSchemas (gc) ;
	n = arr_len (schemas) ;

	for (uint i = 0 ; i < n ; i++) {
		Schema *s = schemas [i] ;
		if (Schema_HasConstraints (s)) {
			return true ;
		}
	}

	return false ;
}

// enable all constraints
void GraphContext_EnableConstrains
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	Schema **schemas = _GetNodeSchemas (gc) ;
	uint n = arr_len (schemas) ;

	for (uint i = 0; i < n; i ++) {
		Schema *s = schemas [i] ;
		for (uint j = 0; j < arr_len (s->constraints) ; j ++) {
			Constraint_Enable (s->constraints [j]) ;
		}
	}

	schemas = _GetRelationSchemas (gc) ;
	n = arr_len (schemas) ;

	for (uint i = 0; i < n; i ++) {
		Schema *s = schemas [i] ;
		for (uint j = 0; j < arr_len (s->constraints); j ++) {
			Constraint_Enable (s->constraints [j]) ;
		}
	}
}

// disable all constraints
void GraphContext_DisableConstrains
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	Schema **schemas = _GetNodeSchemas (gc) ;
	uint n = arr_len (schemas) ;

	for (uint i = 0; i < n; i ++) {
		Schema *s = schemas [i] ;
		for (uint j = 0; j < arr_len (s->constraints) ; j ++) {
			Constraint_Disable (s->constraints [j]) ;
		}
	}

	schemas = _GetRelationSchemas (gc) ;
	n = arr_len (schemas) ;

	for (uint i = 0; i < n; i ++) {
		Schema *s = schemas [i] ;
		for (uint j = 0; j < arr_len (s->constraints); j ++) {
			Constraint_Disable (s->constraints [j]) ;
		}
	}
}

// retrieve the specific schema for the provided ID
Schema *GraphContext_GetSchemaByID
(
	GraphContext *gc,
	int id,
	SchemaType t
) {
	if (id == GRAPH_NO_LABEL) {
		return NULL ;
	}

	Schema **schemas = (t == SCHEMA_NODE) ?
		_GetNodeSchemas (gc) :
		_GetRelationSchemas (gc) ;

	ASSERT (id < arr_len (schemas)) ;
	return schemas [id] ;
}

// returns the Schema matching 'name' for the given entity type
//
// TODO: optimize — currently O(n); consider indexing by name via rax
Schema *GraphContext_GetSchema
(
	GraphContext *gc,    // graph context
	const char   *name,  // label or relation-type name to look up
	SchemaType    t      // SCHEMA_NODE or SCHEMA_EDGE
) {
	ASSERT (gc   != NULL) ;
	ASSERT (name != NULL) ;

	Schema *res = NULL ;

	// choose the appropriate schema array given the entity type
	Schema **schemas = (t == SCHEMA_NODE) ?
		_GetNodeSchemas (gc) : 
		_GetRelationSchemas (gc) ;

	uint32_t n = arr_len (schemas) ;
	for (uint32_t i = 0; i < n; i++) {
		Schema *s = schemas [i] ;
		ASSERT (s != NULL) ;

		if (strcmp (name, Schema_GetName (s)) == 0) {
			res = s ;
			break ;
		}
	}

	return res ;
}

// tries to located schema, in case schema doesn't exists
// registers a new schema and its backing matrix for the given type:
// allocates a label matrix (node) or relation-type matrix (edge) in the graph
// then appends the schema to the corresponding schema array
Schema *GraphContext_FindOrAddSchema
(
	GraphContext *gc,  // graph context
	const char *name,  // schema name
	SchemaType t,      // SCHEMA_NODE or SCHEMA_EDGE
	bool *created      // true if schema was created
) {
	ASSERT (gc   != NULL) ;
	ASSERT (name != NULL) ;

	// quick return if schema already exists
	Schema *s = GraphContext_GetSchema (gc, name, t) ;
	if (s != NULL) {
		if (created != NULL) {
			*created = false ;
		}
		return s ;
	}

	// create schema
	ASSERT (gc->writer_tid == (pthread_t) 0 ||
			pthread_equal (gc->writer_tid, pthread_self ())) ;

	if (t == SCHEMA_NODE) {
		LabelID id = Graph_AddLabel (gc->g) ;
		s = Schema_New (SCHEMA_NODE, id, name) ;

		if (gc->_node_schemas == NULL) {
			gc->writer_tid = pthread_self () ;
			arr_clone (gc->_node_schemas, gc->node_schemas) ;
		}
		arr_append (gc->_node_schemas, s) ;
	} else {
		RelationID id = Graph_AddRelationType (gc->g) ;
		s = Schema_New (SCHEMA_EDGE, id, name) ;

		if (gc->_relation_schemas == NULL) {
			gc->writer_tid = pthread_self () ;
			arr_clone (gc->_relation_schemas, gc->relation_schemas) ;
		}
		arr_append (gc->_relation_schemas, s) ;
	}

	ASSERT (pthread_equal (gc->writer_tid, pthread_self ())) ;

	if (created != NULL) {
		*created = true ;
	}

	return s ;
}

// removes the schema at index 'id', frees it, and removes its backing
// matrix from the graph
//
// after removal the schema array is compacted: every schema with index > id
// shifts down by one
// callers must invalidate any cached schema IDs
void GraphContext_RemoveSchema
(
	GraphContext *gc,  // graph context
	int id,            // schema ID to remove
	SchemaType t       // SCHEMA_NODE or SCHEMA_EDGE
) {
	ASSERT (gc != NULL) ;
	ASSERT (id >= 0 && id < GraphContext_SchemaCount (gc, t)) ;
	ASSERT (pthread_equal (gc->writer_tid, pthread_self ())) ;

	Graph *g = GraphContext_GetGraph (gc) ;

	Schema ***schemas = (t == SCHEMA_NODE) ?
		&gc->_node_schemas :
		&gc->_relation_schemas ;

	ASSERT (schemas != NULL) ;
	ASSERT (arr_len (*schemas) -1 == id) ;

	Schema *s = (*schemas) [id] ;
	ASSERT (Schema_GetID (s) == id) ;

	Schema_Free (s) ;

	*schemas = arr_del (*schemas, id) ;

	if (t == SCHEMA_NODE) {
		Graph_RemoveLabel (g, id) ;
	} else {
		Graph_RemoveRelation (g, id) ;
	}
}

// returns the relation type string for a given edge object
const char *GraphContext_GetEdgeRelationType
(
	GraphContext *gc,
	Edge *e
) {
	ASSERT (e  != NULL) ;
	ASSERT (gc != NULL) ;

	RelationID id = Edge_GetRelationID (e) ;
	ASSERT (id != GRAPH_NO_RELATION) ;
	ASSERT (id < GraphContext_SchemaCount (gc, SCHEMA_EDGE)) ;

	Schema *s = _GetRelationSchemas (gc) [id] ;
	ASSERT (s != NULL) ;

	return Schema_GetName (s) ;
}

// returns number of unique attribute keys
uint GraphContext_AttributeCount
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	return arr_len (_GetAttributes (gc)) ;
}

// returns an attribute ID given a string, creating one if not found
AttributeID GraphContext_FindOrAddAttribute
(
	GraphContext *gc,       // graph context
	const char *attribute,  // attribute name
	bool *created           // [optional] rather or not attribute was created
) {
	ASSERT (gc        != NULL) ;
	ASSERT (attribute != NULL) ;

	if (created) {
		*created = false ;
	}

	// see if attribute already exists
	AttributeID id = GraphContext_GetAttributeID (gc, attribute) ;
	if (id != ATTRIBUTE_ID_NONE) {
		return id ;
	}

	//--------------------------------------------------------------------------
	// Create new attribute locally
	//--------------------------------------------------------------------------

	ASSERT (gc->writer_tid == (pthread_t) 0 ||
			pthread_equal (gc->writer_tid, pthread_self ())) ;

	// should only happen if an old rdb with an overlong name is loaded
	ASSERT (strnlen (attribute, MAX_IDENTIFIER_LEN) <= MAX_IDENTIFIER_LEN) ;

	// attribute missing
	// add it as a pending attribute
	if (gc->_attributes == NULL) {
		gc->writer_tid = pthread_self () ;
		arr_clone (gc->_attributes, gc->attributes) ;
	}

	id = arr_len (gc->_attributes) ;

	arr_append (gc->_attributes, rm_strdup (attribute)) ;
	if (created) {
		*created = true ;
	}

	ASSERT (pthread_equal (gc->writer_tid, pthread_self ())) ;

	return id ;
}

// returns attribute name given an ID
const char *GraphContext_GetAttributeName
(
	GraphContext *gc,
	AttributeID id
) {
	ASSERT (gc != NULL) ;

	// pick attributes array
	char **attributes = _GetAttributes(gc) ;

	ASSERT (id < arr_len (attributes)) ;

	return attributes [id] ;
}

// returns an attribute ID given a string
// or ATTRIBUTE_ID_NONE if attribute doesn't exist
AttributeID GraphContext_GetAttributeID
(
	GraphContext *gc,
	const char *attribute
) {
	// pick attributes array
	char **attributes = _GetAttributes (gc) ;

	// look up the attribute ID
	AttributeID id = ATTRIBUTE_ID_NONE ;
	uint16_t n = arr_len (attributes) ;

	for (uint16_t i = 0 ; i < n ; i++) {
		if (strcmp (attribute, attributes [i]) == 0) {
			id = i ;
			break  ;
		}
	}

	return id ;
}

// removes an attribute from the graph
void GraphContext_RemoveAttribute
(
	GraphContext *gc,
	AttributeID id
) {
	ASSERT (gc != NULL) ;
	ASSERT (gc->_attributes != NULL) ;
	ASSERT (id == arr_len (gc->_attributes) - 1) ;
	ASSERT (pthread_equal (gc->writer_tid, pthread_self ())) ;

	rm_free (gc->_attributes [id]) ;
	arr_del (gc->_attributes, id) ;
}

//------------------------------------------------------------------------------
// Index API
//------------------------------------------------------------------------------

// returns true if the passed graph context has indices, false otherwise.
bool GraphContext_HasIndices
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	const bool has_node_indices = GraphContext_NodeIndexCount (gc) ;
	const bool has_edge_indices = GraphContext_EdgeIndexCount (gc) ;

	return has_node_indices || has_edge_indices ;
}

static uint16_t _count_indices_from_schemas
(
	Schema **schemas
) {
	uint16_t count = 0;

	const uint16_t length = arr_len (schemas) ;
	for (uint16_t i = 0 ; i < length ; i++) {
		Schema *s = schemas [i] ;
		ASSERT (s != NULL) ;

		if (Schema_HasIndices (s)) {
			count++ ;
		}
	}

	return count ;
}

// returns the number of node indices within the passed graph context.
uint16_t GraphContext_NodeIndexCount
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	Schema **schemas = _GetNodeSchemas (gc) ;
	return _count_indices_from_schemas (schemas) ;
}

// returns the number of edge indices within the passed graph context.
uint16_t GraphContext_EdgeIndexCount
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	Schema **schemas = _GetRelationSchemas (gc) ;
	return _count_indices_from_schemas (schemas) ;
}

// attempt to retrieve an index on the given label and attribute IDs
Index GraphContext_GetIndexByID
(
	GraphContext *gc,            // graph context
	int lbl_id,                  // label / rel-type ID
	const AttributeID *attrs,    // attributes
	uint n,                      // attributes count
	IndexFieldType t,            // all index attributes must be of this type
	GraphEntityType entity_type  // schema type NODE / EDGE
) {
	// validations
	ASSERT (gc != NULL) ;
	ASSERT ((attrs == NULL && n == 0) || (attrs != NULL && n > 0)) ;

	// retrieve the schema for given id
	SchemaType st = (entity_type == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE;
	Schema *s = GraphContext_GetSchemaByID (gc, lbl_id, st) ;
	if (s == NULL) {
		return NULL ;
	}

	return Schema_GetIndex (s, attrs, n, t, false) ;
}

// attempt to retrieve an index on the given label and attribute
Index GraphContext_GetIndex
(
	GraphContext *gc,
	const char *label,
	AttributeID *attrs,
	uint n,
	IndexFieldType type,
	SchemaType schema_type
) {
	ASSERT (gc    != NULL) ;
	ASSERT (label != NULL) ;

	// Retrieve the schema for this label
	Schema *s = GraphContext_GetSchema (gc, label, schema_type) ;
	if (s == NULL) {
		return NULL ;
	}

	return Schema_GetIndex (s, attrs, n, type, false) ;
}

int GraphContext_DeleteIndex
(
	GraphContext *gc,
	SchemaType schema_type,
	const char *label,
	const char *field,
	IndexFieldType t
) {
	ASSERT (gc    != NULL) ;
	ASSERT (label != NULL) ;
	ASSERT (field != NULL) ;

	// retrieve the schema for this label
	int res = INDEX_FAIL ;
	Schema *s = GraphContext_GetSchema (gc, label, schema_type) ;

	if (s != NULL) {
		res = Schema_RemoveIndex (s, field, t) ;
		if (res == INDEX_OK) {
			// update resultset statistics
			ResultSet *result_set = QueryCtx_GetResultSet () ;
			ResultSet_IndexDeleted (result_set, res) ;
		}
	}

	return res ;
}

// remove a single node from all indices that refer to it
static void _DeleteNodeFromIndices
(
	GraphContext *gc,  // graph context
	Node *n,           // node to remove from index
	LabelID *lbls,     // [optional] node labels to remove from index
	uint label_count   // [optional] number of labels
) {
	ASSERT (n    != NULL) ;
	ASSERT (gc   != NULL) ;
	ASSERT (lbls != NULL) ;

	Schema   *s      = NULL;
	EntityID node_id = ENTITY_GET_ID (n) ;

	for (uint i = 0 ; i < label_count ; i++) {
		LabelID label_id = lbls [i] ;
		ASSERT (Graph_IsNodeLabeled (gc->g, ENTITY_GET_ID (n), label_id)) ;

		s = GraphContext_GetSchemaByID (gc, label_id, SCHEMA_NODE) ;
		ASSERT (s != NULL) ;

		// update any indices this entity is represented in
		Schema_RemoveNodeFromIndex (s, n) ;
	}
}

// remove a single node from all indices that refer to it
void GraphContext_DeleteNodeFromIndices
(
	GraphContext *gc,  // graph context
	Node *n,           // node to remove from index
	LabelID *lbls,     // [optional] node labels to remove from index
	uint label_count   // [optional] number of labels
) {
	ASSERT (n    != NULL) ;
	ASSERT (gc   != NULL) ;
	ASSERT (lbls != NULL || label_count == 0) ;

	EntityID node_id = ENTITY_GET_ID (n) ;

	if (lbls == NULL) {
		// retrieve node labels
		NODE_GET_LABELS (gc->g, n, label_count) ;
		_DeleteNodeFromIndices (gc, n, labels, label_count) ;
	} else {
		_DeleteNodeFromIndices (gc, n, lbls, label_count) ;
	}
}

// remove a single edge from all indices that refer to it
void GraphContext_DeleteEdgeFromIndices
(
	GraphContext *gc,  // graph context
	Edge *e            // edge to remove from index
) {
	ASSERT (e  != NULL) ;
	ASSERT (gc != NULL) ;

	Schema *s = NULL;
	Graph  *g = gc->g;

	RelationID relation_id = Edge_GetRelationID (e) ;

	s = GraphContext_GetSchemaByID (gc, relation_id, SCHEMA_EDGE) ;
	ASSERT (s != NULL) ;

	// update any indices this entity is represented in
	Schema_RemoveEdgeFromIndex (s, e) ;
}

// add node to any relevant index
void GraphContext_AddNodeToIndices
(
	GraphContext *gc,  // graph context
	Node *n            // node to add to index
) {
	ASSERT (n  != NULL) ;
	ASSERT (gc != NULL) ;

	Schema   *s      = NULL ;
	Graph    *g      = gc->g ;
	EntityID node_id = ENTITY_GET_ID (n) ;

	// retrieve node labels
	uint label_count ;
	NODE_GET_LABELS (g, n, label_count) ;

	for (uint i = 0; i < label_count; i++) {
		LabelID label_id = labels [i] ;
		s = GraphContext_GetSchemaByID (gc, label_id, SCHEMA_NODE) ;
		ASSERT (s != NULL) ;
		Schema_AddNodeToIndex (s, n) ;
	}
}

// add edge to any relevant index
void GraphContext_AddEdgeToIndices
(
	GraphContext *gc,  // graph context
	Edge *e            // edge to add to index
) {
	ASSERT (e  != NULL) ;
	ASSERT (gc != NULL) ;

	Schema *s = NULL ;
	Graph  *g = gc->g ;

	RelationID relation_id = Edge_GetRelationID (e) ;

	s = GraphContext_GetSchemaByID (gc, relation_id, SCHEMA_EDGE) ;
	ASSERT (s != NULL) ;

	Schema_AddEdgeToIndex (s, e) ;
}

//------------------------------------------------------------------------------
// Functions for globally tracking GraphContexts
//------------------------------------------------------------------------------

// register a new GraphContext for module-level tracking
void GraphContext_RegisterWithModule
(
	GraphContext *gc
) {
	Globals_AddGraph (gc) ;
}

// retrive GraphContext from the global array
// graph isn't registered, NULL is returned
// graph's references count isn't increased!
// this is OK as long as only a single thread has access to the graph
GraphContext *GraphContext_UnsafeGetGraphContext
(
	const char *graph_name
) {
	KeySpaceGraphIterator it;
	Globals_ScanGraphs(&it);

	GraphContext *gc = NULL;

	while((gc = GraphIterator_Next(&it)) != NULL) {
		bool match = (strcmp(gc->graph_name, graph_name) == 0);
		GraphContext_DecreaseRefCount(gc);
		if(match == true) {
			break;
		}
	}

	return gc;
}

//------------------------------------------------------------------------------
// Slowlog API
//------------------------------------------------------------------------------

// Return slowlog associated with graph context.
SlowLog *GraphContext_GetSlowLog
(
	const GraphContext *gc
) {
	ASSERT (gc) ;

	return gc->slowlog ;
}

//------------------------------------------------------------------------------
// Queries API
//------------------------------------------------------------------------------

void GraphContext_LogQuery
(
	const GraphContext *gc,     // graph context
	uint64_t received,          // query received timestamp
	double wait_duration,       // waiting time
	double execution_duration,  // executing time
	double report_duration,     // reporting time
	bool parameterized,         // uses parameters
	bool utilized_cache,        // utilized cache
	bool write,                 // write query
	bool timeout,               // timeout query
	uint params_len,            // length of parameters
	const char *query           // query string
) {
	ASSERT (gc    != NULL) ;
	ASSERT (query != NULL) ;

	QueriesLog_AddQuery (gc->queries_log, received, wait_duration,
			execution_duration, report_duration, parameterized, utilized_cache,
			write, timeout, params_len, query);
}

//------------------------------------------------------------------------------
// Cache API
//------------------------------------------------------------------------------

// Return cache associated with graph context and current thread id.
Cache *GraphContext_GetCache
(
	const GraphContext *gc
) {
	ASSERT (gc != NULL) ;
	return gc->cache ;
}

//------------------------------------------------------------------------------
// Encoding context
//------------------------------------------------------------------------------

GraphEncodeContext *GraphContext_GetEncodingCtx
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;
	return gc->encoding_context ;
}

GraphDecodeContext *GraphContext_GetDecodingCtx
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;
	return gc->decoding_context ;
}

//------------------------------------------------------------------------------
// QueriesLog
//------------------------------------------------------------------------------

QueriesLog GraphContext_GetQueriesLog
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	return gc->queries_log ;
}

//------------------------------------------------------------------------------
// Free routine
//------------------------------------------------------------------------------

// delete graph's telemetry stream
static void _DeleteTelemetryStream
(
	RedisModuleCtx *ctx,    // redis module context
	const GraphContext *gc  // graph context
) {
	ASSERT(gc  != NULL);
	ASSERT(ctx != NULL);

	RedisModuleKey *key = RedisModule_OpenKey(ctx, gc->telemetry_stream,
			REDISMODULE_WRITE);
	RedisModule_DeleteKey(key);
	RedisModule_CloseKey(key);
}

// free all data associated with graph
void GraphContext_Free
(
	GraphContext *gc
) {
	uint len;

	ASSERT (gc->ref_count == 0) ;

	if (gc->decoding_context == NULL ||
			GraphDecodeContext_Finished (gc->decoding_context)) {
		Graph_Free (gc->g) ;
	} else {
		Graph_PartialFree (gc->g) ;
	}

	// Redis main thread is 0
	RedisModuleCtx *ctx = NULL;
	extern pthread_t MAIN_THREAD_ID;  // redis main thread ID
	bool main_thread = (pthread_equal(pthread_self(), MAIN_THREAD_ID) != 0);
	bool should_lock = !main_thread && RedisModule_GetThreadSafeContext != NULL;

	if(should_lock) {
		ctx = RedisModule_GetThreadSafeContext(NULL);
		// GIL need to be acquire because RediSearch change Redis data structure
		RedisModule_ThreadSafeContextLock(ctx);
	}

	//--------------------------------------------------------------------------
	// delete graph telemetry stream
	//--------------------------------------------------------------------------

	if(gc->telemetry_stream != NULL) {
		bool should_create = (ctx == NULL);
		if(should_create) {
			ctx = RedisModule_GetThreadSafeContext(NULL);
		}
		_DeleteTelemetryStream(ctx, gc);
		RedisModule_FreeString(ctx, gc->telemetry_stream);
		if (should_create) {
			RedisModule_FreeThreadSafeContext(ctx);
			ctx = NULL;
		}
	}

	//--------------------------------------------------------------------------
	// free node schemas
	//--------------------------------------------------------------------------

	if (gc->_node_schemas     != NULL ||
		gc->_relation_schemas != NULL ||
		gc->_attributes       != NULL) {
		// should not happen
		// unless a graph wasn't fully loaded
		// and its virtual keys are being deleted
		// TODO: should be logged?

		ASSERT (gc->attributes       == NULL) ;
		ASSERT (gc->node_schemas     == NULL) ;
		ASSERT (gc->relation_schemas == NULL) ;

		gc->attributes       = gc->_attributes ;
		gc->node_schemas     = gc->_node_schemas ;
		gc->relation_schemas = gc->_relation_schemas ;
	}

	arr_free_cb (gc->node_schemas, Schema_Free) ;
	gc->node_schemas = NULL ;

	arr_free_cb (gc->relation_schemas, Schema_Free) ;
	gc->relation_schemas = NULL ;

	if (should_lock) {
		RedisModule_ThreadSafeContextUnlock (ctx) ;
		RedisModule_FreeThreadSafeContext (ctx) ;
	}

	arr_free_cb (gc->attributes, rm_free) ;
	gc->attributes = NULL ;

	//--------------------------------------------------------------------------
	// free queries log
	//--------------------------------------------------------------------------

	QueriesLog_Free (gc->queries_log) ;

	if (gc->slowlog) {
		SlowLog_Free (gc->slowlog) ;
	}

	//--------------------------------------------------------------------------
	// clear cache
	//--------------------------------------------------------------------------

	if (gc->cache) {
		Cache_Free (gc->cache) ;
	}

	//--------------------------------------------------------------------------
	// free pending write queue
	//--------------------------------------------------------------------------

	if (gc->pending_write_queue != NULL) {
		ASSERT (CircularBuffer_Empty (gc->pending_write_queue)) ;
		CircularBuffer_Free (gc->pending_write_queue, NULL) ;
	}

	GraphEncodeContext_Free (gc->encoding_context) ;
	GraphDecodeContext_Free (gc->decoding_context) ;

	if (gc->writelocked) {
		pthread_rwlock_unlock (&gc->rwlock) ;
	}

	int res = pthread_rwlock_destroy (&gc->rwlock) ;
	ASSERT (res == 0) ;

	rm_free (gc->graph_name) ;
	rm_free (gc) ;
}

