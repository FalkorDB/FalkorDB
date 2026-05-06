/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "globals.h"
#include "graphcontext.h"
#include "../util/arr.h"
#include "../util/uuid.h"
#include "../query_ctx.h"
#include "../redismodule.h"
#include "../util/rmalloc.h"
#include "../util/thpool/pool.h"
#include "../constraint/constraint.h"
#include "../serializers/graphcontext_type.h"
#include "../commands/execution_ctx.h"

#include <pthread.h>
#include <sys/param.h>
#include <stdatomic.h>

// telemetry stream format
#define TELEMETRY_FORMAT "telemetry{%s}"

// import the GraphContext struct
#include "graphcontext_struct.h"

extern uint aux_field_counter;
// GraphContext type as it is registered at Redis.
extern RedisModuleType *GraphContextRedisModuleType;

// forward declarations
static void _GraphContext_Free(void *arg);
static void _GraphContext_UpdateVersion(GraphContext *gc, const char *str);
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
			ThreadPool_AddWork (_GraphContext_Free, gc, 1) ;
		} else {
			// Sync delete
			_GraphContext_Free (gc) ;
		}
	}
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

	gc->version          = 0 ;  // initial graph version
	gc->slowlog          = SlowLog_New () ;
	gc->queries_log      = QueriesLog_New () ;
	gc->ref_count        = 0 ;  // no refences
	gc->attributes       = raxNew () ;
	gc->index_count      = 0 ;  // no indicies
	gc->string_mapping   = arr_new (char *, 64) ;
	gc->encoding_context = GraphEncodeContext_New () ;
	gc->decoding_context = GraphDecodeContext_New () ;

	// initial graph's write in progress atomic flag to false
	atomic_init (&gc->write_in_progress, false) ;

	// create graph's pending write queries queue
	gc->pending_write_queue = CircularBuffer_New (sizeof (void*), 1024) ;

	// read NODE_CREATION_BUFFER size from configuration
	// this value controls how much extra room we're willing to spend for:
	// 1. graph entity storage
	// 2. matrices dimensions
	size_t node_cap;
	size_t edge_cap;
	bool rc = Config_Option_get(Config_NODE_CREATION_BUFFER, &node_cap);
	assert(rc);
	edge_cap = node_cap;

	gc->g = Graph_New (node_cap, edge_cap) ;
	gc->graph_name = rm_strdup (graph_name) ;
	gc->telemetry_stream = RedisModule_CreateStringPrintf (NULL,
			TELEMETRY_FORMAT, gc->graph_name) ;

	// allocate the default space for schemas and indices
	gc->node_schemas = arr_new (Schema *, GRAPH_DEFAULT_LABEL_CAP) ;
	gc->relation_schemas = arr_new (Schema *, GRAPH_DEFAULT_RELATION_TYPE_CAP) ;

	// initialize the read-write lock to protect access to the graph's schema
	int init = pthread_rwlock_init (&gc->_schema_rwlock, NULL) ;
	assert (init == 0) ;

	// build the execution plans cache
	uint64_t cache_size ;
	Config_Option_get (Config_CACHE_SIZE, &cache_size) ;
	gc->cache = Cache_New (cache_size, (CacheEntryFreeFunc)ExecutionCtx_Free,
						  (CacheEntryCopyFunc)ExecutionCtx_Clone) ;

	Graph_SetMatrixPolicy (gc->g, SYNC_POLICY_FLUSH_RESIZE) ;

	return gc ;
}

// _GraphContext_Create tries to get a graph context
// and if it does not exists, create a new one
// the try-get-create flow is done when module global lock is acquired
// to enforce consistency while BGSave is called
static GraphContext *_GraphContext_Create
(
	RedisModuleCtx *ctx,
	const char *graph_name
) {
	// create and initialize a graph context
	GraphContext *gc = GraphContext_New(graph_name);
	RedisModuleString *graphID = RedisModule_CreateString(ctx, graph_name,
			strlen(graph_name));

	RedisModuleKey *key = RedisModule_OpenKey(ctx, graphID, REDISMODULE_WRITE);

	// set value in key
	RedisModule_ModuleTypeSetValue(key, GraphContextRedisModuleType, gc);

	// register graph context for BGSave
	GraphContext_RegisterWithModule(gc);

	RedisModule_FreeString(ctx, graphID);
	RedisModule_CloseKey(key);

	return gc;
}

// retrive the graph context according to the graph name
// readOnly is the access mode to the graph key
GraphContext *GraphContext_Retrieve
(
	RedisModuleCtx *ctx,
	RedisModuleString *graphID,
	bool readOnly,
	bool shouldCreate
) {
	// check if we're still replicating, if so don't allow access to the graph
	if (aux_field_counter > 0) {
		// the whole module is currently replicating, emit an error
		RedisModule_ReplyWithError (ctx,
				"ERR FalkorDB module is currently replicating") ;
		return NULL ;
	}

	GraphContext *gc = NULL ;
	int rwFlag = readOnly ? REDISMODULE_READ : REDISMODULE_WRITE ;

	RedisModuleKey *key = RedisModule_OpenKey (ctx, graphID, rwFlag) ;
	if (RedisModule_KeyType (key) == REDISMODULE_KEYTYPE_EMPTY) {
		if (shouldCreate) {
			// key doesn't exist, create it
			const char *graphName = RedisModule_StringPtrLen (graphID, NULL) ;
			gc = _GraphContext_Create (ctx, graphName) ;
		} else {
			// key does not exist and won't be created, emit an error.
			RedisModule_ReplyWithError (ctx,
					"ERR Invalid graph operation on empty key") ;
		}
	} else if (RedisModule_ModuleTypeGetType (key) == GraphContextRedisModuleType) {
		gc = RedisModule_ModuleTypeGetValue (key) ;
	} else {
		// key exists but is not a graph, emit an error
		RedisModule_ReplyWithError (ctx, REDISMODULE_ERRORMSG_WRONGTYPE) ;
	}

	RedisModule_CloseKey (key) ;

	if (gc) {
		GraphContext_IncreaseRefCount (gc) ;
	}

	return gc ;
}

// decrease graph context reference count
// graph context will be free once reference count reaches 0
void GraphContext_Release
(
	GraphContext *gc // graph context to release
) {
	ASSERT (gc != NULL) ;
	GraphContext_DecreaseRefCount (gc) ;
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

void GraphContext_LockForCommit
(
	RedisModuleCtx *ctx,
	GraphContext *gc
) {
	// aquire GIL
	RedisModule_ThreadSafeContextLock (ctx) ;

	// acquire graph write lock
	Graph_AcquireWriteLock (gc->g) ;
}

void GraphContext_UnlockCommit
(
	RedisModuleCtx *ctx,
	GraphContext *gc
) {
	// release graph R/W lock
	Graph_ReleaseLock(gc->g);

	// unlock GIL
	RedisModule_ThreadSafeContextUnlock(ctx);
}

// attempt to acquire exclusive write access to the given graph
// returns true if the calling thread successfully acquired write ownership
// returns false if another write is already in progress
bool GraphContext_TryEnterWrite
(
	GraphContext *gc  // graph context
) {
	ASSERT(gc != NULL);

	bool expected = false;

    // atomically set to true only if current value is false
    return atomic_compare_exchange_strong(&gc->write_in_progress, &expected,
			true);
}

// release exclusive write access to the graph
// this should be called by a thread that previously acquired write ownership
// via GraphContext_TryEnterWrite, it clears the write-in-progress flag
void GraphContext_ExitWrite
(
	GraphContext *gc  // graph context
) {
	ASSERT(gc != NULL);

	atomic_store(&gc->write_in_progress, false);
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

XXH32_hash_t GraphContext_GetVersion
(
	const GraphContext *gc
) {
	ASSERT(gc != NULL);

	return gc->version;
}

// get graph from graph context
Graph *GraphContext_GetGraph
(
	const GraphContext *gc
) {
	ASSERT(gc != NULL);
	
	return gc->g;
}

// Update graph context version
static void _GraphContext_UpdateVersion
(
	GraphContext *gc,
	const char *str
) {
	ASSERT (gc  != NULL) ;
	ASSERT (str != NULL) ;

	/* Update graph version by hashing 'str' representing the current
	 * addition to the graph schema: (Label, Relationship-type, Attribute)
	 *
	 * Using the current graph version as a seed, by doing so we avoid
	 * hashing the entire graph schema on each change, while guaranteeing the
	 * exact same version across a cluster: same graph version on both
	 * primary and replica shards. */

	XXH32_state_t *state = XXH32_createState();
	XXH32_reset(state, gc->version);
	XXH32_update(state, str, strlen(str));
	gc->version = XXH32_digest(state);
	XXH32_freeState(state);
}

//------------------------------------------------------------------------------
// Schema API
//------------------------------------------------------------------------------

static inline void _GraphContext_ReadLockSchemas
(
	GraphContext *gc
) {
	int res = pthread_rwlock_rdlock (&gc->_schema_rwlock) ;
	ASSERT (res == 0) ;
}

static inline void _GraphContext_WriteLockSchemas
(
	GraphContext *gc
) {
	int res = pthread_rwlock_wrlock (&gc->_schema_rwlock) ;
	ASSERT (res == 0) ;
}

static inline void _GraphContext_UnLockSchemas
(
	GraphContext *gc
) {
	int res = pthread_rwlock_unlock (&((GraphContext *)gc)->_schema_rwlock) ;
	ASSERT (res == 0) ;
}

// find the ID associated with a label for schema and matrix access
static int _GraphContext_GetLabelID
(
	GraphContext *gc,
	const char *label,
	SchemaType t
) {
	_GraphContext_ReadLockSchemas (gc) ;

	// choose the appropriate schema array given the entity type
	Schema **schemas = (t == SCHEMA_NODE) ?
		gc->node_schemas :
		gc->relation_schemas ;

	// TODO: optimize lookup
	uint32_t l = arr_len (schemas) ;
	int result = GRAPH_NO_LABEL ;

	for (uint32_t i = 0; i < l; i++) {
		if (!strcmp (label, schemas[i]->name)) {
			result = i ;
			break ;
		}
	}

	_GraphContext_UnLockSchemas (gc) ;

	return result ;
}

unsigned short GraphContext_SchemaCount
(
	GraphContext *gc,
	SchemaType t
) {
	ASSERT (gc) ;

	_GraphContext_ReadLockSchemas (gc) ;

	unsigned short count = (t == SCHEMA_NODE) ?
		arr_len (gc->node_schemas) :
		arr_len (gc->relation_schemas) ;

	_GraphContext_UnLockSchemas (gc) ;

	return count ;
}

// checks if graph has constraints
bool GraphContext_HasConstraints
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	bool has_constraints = false ;

	_GraphContext_ReadLockSchemas (gc) ;

	uint n = arr_len (gc->node_schemas) ;
	for (uint i = 0 ; i < n ; i++) {
		Schema *s = gc->node_schemas [i] ;
		if (Schema_HasConstraints (s)) {
			has_constraints = true ;
			goto cleanup ;
		}
	}

	n = arr_len (gc->relation_schemas) ;
	for (uint i = 0 ; i < n ; i++) {
		Schema *s = gc->relation_schemas [i] ;
		if (Schema_HasConstraints (s)) {
			has_constraints = true ;
			goto cleanup ;
		}
	}

cleanup:

	_GraphContext_UnLockSchemas (gc) ;
	return has_constraints ;
}

// enable all constraints
void GraphContext_EnableConstrains
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	_GraphContext_ReadLockSchemas (gc) ;

	uint n = arr_len (gc->node_schemas) ;
	for (uint i = 0; i < n; i ++) {
		Schema *s = gc->node_schemas [i] ;
		for (uint j = 0; j < arr_len (s->constraints) ; j ++) {
			Constraint_Enable (s->constraints [j]) ;
		}
	}

	n = arr_len (gc->relation_schemas) ;
	for (uint i = 0; i < n; i ++) {
		Schema *s = gc->relation_schemas[i];
		for (uint j = 0; j < arr_len(s->constraints); j ++) {
			Constraint_Enable (s->constraints [j]) ;
		}
	}

	_GraphContext_UnLockSchemas (gc) ;
}

// disable all constraints
void GraphContext_DisableConstrains
(
	GraphContext *gc
) {
	ASSERT (gc != NULL) ;

	_GraphContext_ReadLockSchemas (gc) ;

	uint n = arr_len (gc->node_schemas) ;
	for (uint i = 0; i < n; i ++) {
		Schema *s = gc->node_schemas [i] ;
		for (uint j = 0; j < arr_len (s->constraints) ; j ++) {
			Constraint_Disable (s->constraints [j]) ;
		}
	}

	n = arr_len (gc->relation_schemas) ;
	for (uint i = 0; i < n; i ++) {
		Schema *s = gc->relation_schemas [i] ;
		for (uint j = 0; j < arr_len (s->constraints); j ++) {
			Constraint_Disable (s->constraints [j]) ;
		}
	}

	_GraphContext_UnLockSchemas (gc) ;
}

Schema *GraphContext_GetSchemaByID
(
	GraphContext *gc,
	int id,
	SchemaType t
) {
	if (id == GRAPH_NO_LABEL) {
		return NULL ;
	}

	_GraphContext_ReadLockSchemas (gc) ;

	Schema **schemas = (t == SCHEMA_NODE) ?
		gc->node_schemas :
		gc->relation_schemas ;
	Schema *s = schemas [id] ;

	_GraphContext_UnLockSchemas (gc) ;

	return s ;
}

Schema *GraphContext_GetSchema
(
	GraphContext *gc,
	const char *label,
	SchemaType t
) {
	ASSERT (gc    != NULL) ;
	ASSERT (label != NULL) ;

	int id = _GraphContext_GetLabelID (gc, label, t) ;
	Schema *s = GraphContext_GetSchemaByID (gc, id, t) ;

	return s ;
}

Schema *GraphContext_AddSchema
(
	GraphContext *gc,
	const char *label,
	SchemaType t
) {
	ASSERT (gc    != NULL) ;
	ASSERT (label != NULL) ;

	int id ;
	Schema *schema ;

	if(t == SCHEMA_NODE) {
		id = Graph_AddLabel (gc->g) ;
		schema = Schema_New (SCHEMA_NODE, id, label) ;

		_GraphContext_WriteLockSchemas (gc) ;
		arr_append (gc->node_schemas, schema) ;
		_GraphContext_UnLockSchemas (gc) ;
	} else {
		id = Graph_AddRelationType (gc->g) ;
		schema = Schema_New (SCHEMA_EDGE, id, label) ;

		_GraphContext_WriteLockSchemas (gc) ;
		arr_append (gc->relation_schemas, schema) ;
		_GraphContext_UnLockSchemas (gc) ;
	}

	// new schema added, update graph version
	_GraphContext_UpdateVersion (gc, label) ;

	return schema ;
}

void GraphContext_RemoveSchema
(
	GraphContext *gc,
	int schema_id,
	SchemaType t
) {
	_GraphContext_WriteLockSchemas (gc) ;

	if(t == SCHEMA_NODE) {
		Schema *schema = gc->node_schemas [schema_id] ;
		Schema_Free (schema) ;
		gc->node_schemas = arr_del (gc->node_schemas, schema_id) ;
	} else {
		Schema *schema = gc->relation_schemas [schema_id] ;
		Schema_Free (schema) ;
		gc->relation_schemas = arr_del (gc->relation_schemas, schema_id) ;
	}

	_GraphContext_UnLockSchemas (gc) ;
}

// returns the relation type string for a given edge object
const char *GraphContext_GetEdgeRelationType
(
	GraphContext *gc,
	Edge *e
) {
	int reltype_id = Edge_GetRelationID (e) ;
	ASSERT (reltype_id != GRAPH_NO_RELATION) ;

	_GraphContext_ReadLockSchemas (gc) ;

	const char *rel_name = gc->relation_schemas [reltype_id]->name ;

	_GraphContext_UnLockSchemas (gc) ;

	return rel_name ;
}

// returns number of unique attribute keys
uint GraphContext_AttributeCount
(
	GraphContext *gc
) {
	_GraphContext_ReadLockSchemas (gc) ;

	uint size = arr_len (gc->string_mapping) ;

	_GraphContext_UnLockSchemas (gc) ;

	return size ;
}

// returns an attribute ID given a string, creating one if not found
AttributeID GraphContext_FindOrAddAttribute
(
	GraphContext *gc,
	const char *attribute,
	bool *created
) {
	ASSERT(gc);

	bool created_flag = false;
	unsigned char *attr = (unsigned char*)attribute;
	uint l = strlen(attribute);

	// acquire a read lock for looking up the attribute
	_GraphContext_ReadLockSchemas (gc) ;

	// see if attribute already exists
	void *attribute_id = raxFind (gc->attributes, attr, l) ;

	if (attribute_id == raxNotFound) {
		// we are writing to the shared GraphContext
		// release the held lock and re-acquire as a writer
		_GraphContext_UnLockSchemas    (gc) ;
		_GraphContext_WriteLockSchemas (gc) ;

		// lookup the attribute again now that we are in a critical region
		attribute_id = raxFind (gc->attributes, attr, l) ;

		// if set by another thread, use the retrieved value
		if (attribute_id == raxNotFound) {
			// otherwise, it will be assigned an ID
			// equal to the current mapping size
			attribute_id = (void *) raxSize (gc->attributes) ;
			// insert the new attribute key and ID
			raxInsert (gc->attributes, attr, l, attribute_id, NULL) ;
			arr_append (gc->string_mapping, rm_strdup(attribute)) ;
			created_flag = true ;

			// new attribute been added, update graph version
			_GraphContext_UpdateVersion (gc, attribute) ;
		}
	}

	// release the lock
	_GraphContext_UnLockSchemas (gc) ;

	if (created) {
		*created = created_flag ;
	}

	return (uintptr_t)attribute_id ;
}

// returns an attribute string given an ID
const char *GraphContext_GetAttributeString
(
	GraphContext *gc,
	AttributeID id
) {
	ASSERT (gc != NULL) ;
	ASSERT (id >= 0 && id < arr_len (gc->string_mapping)) ;

	_GraphContext_ReadLockSchemas (gc) ;

	const char *name = gc->string_mapping [id] ;

	_GraphContext_UnLockSchemas (gc) ;

	return name ;
}

// returns an attribute ID given a string
// or ATTRIBUTE_ID_NONE if attribute doesn't exist
AttributeID GraphContext_GetAttributeID
(
	GraphContext *gc,
	const char *attribute
) {
	// acquire a read lock for looking up the attribute
	_GraphContext_ReadLockSchemas (gc) ;

	// look up the attribute ID
	void *id = raxFind (gc->attributes, (unsigned char *) attribute,
			strlen (attribute)) ;

	// release the lock
	_GraphContext_UnLockSchemas (gc) ;

	if (id == raxNotFound) {
		return ATTRIBUTE_ID_NONE ;
	}

	return (uintptr_t) id ;
}

// removes an attribute from the graph
void GraphContext_RemoveAttribute
(
	GraphContext *gc,
	AttributeID id
) {
	ASSERT (gc) ;
	ASSERT (id == arr_len (gc->string_mapping) - 1) ;

	_GraphContext_WriteLockSchemas (gc) ;

	const char *attribute = gc->string_mapping [id] ;
	int ret = raxRemove (gc->attributes,  (unsigned char *) attribute,
			strlen (attribute), NULL) ;
	ASSERT (ret == 1) ;

	rm_free (gc->string_mapping [id]) ;
	gc->string_mapping = arr_del (gc->string_mapping, id) ;
	_GraphContext_UnLockSchemas (gc) ;
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
	const Schema** schemas
) {
	ASSERT (schemas) ;
	uint16_t count = 0;

	const uint16_t length = arr_len (schemas) ;
	for (uint16_t i = 0; i < length; ++i) {
		const Schema *s = schemas [i] ;
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
	ASSERT (gc) ;

	_GraphContext_ReadLockSchemas (gc) ;

	uint16_t n =
		_count_indices_from_schemas ((const Schema**)gc->node_schemas) ;

	_GraphContext_UnLockSchemas (gc) ;

	return n ;
}

// returns the number of edge indices within the passed graph context.
uint16_t GraphContext_EdgeIndexCount
(
	GraphContext *gc
) {
	ASSERT (gc) ;

	_GraphContext_ReadLockSchemas (gc) ;

	uint16_t n =
		_count_indices_from_schemas ((const Schema**)gc->relation_schemas) ;

	_GraphContext_UnLockSchemas (gc) ;

	return n ;
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
	ASSERT(n  != NULL);
	ASSERT(gc != NULL);
	ASSERT(lbls != NULL);

	Schema   *s      = NULL;
	EntityID node_id = ENTITY_GET_ID(n);

	for(uint i = 0; i < label_count; i++) {
		int label_id = lbls[i];
		ASSERT(Graph_IsNodeLabeled(gc->g, ENTITY_GET_ID(n), label_id));
		s = GraphContext_GetSchemaByID(gc, label_id, SCHEMA_NODE);
		ASSERT(s != NULL);

		// update any indices this entity is represented in
		Schema_RemoveNodeFromIndex(s, n);
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
	ASSERT (n  != NULL) ;
	ASSERT (gc != NULL) ;
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
	Schema *s = NULL;
	Graph  *g = gc->g;

	int relation_id = Edge_GetRelationID(e);

	s = GraphContext_GetSchemaByID(gc, relation_id, SCHEMA_EDGE);
	ASSERT(s != NULL);

	// update any indices this entity is represented in
	Schema_RemoveEdgeFromIndex(s, e);
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
		int label_id = labels[i] ;
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
	Schema *s = NULL;
	Graph  *g = gc->g;

	int relation_id = Edge_GetRelationID(e);

	s = GraphContext_GetSchemaByID(gc, relation_id, SCHEMA_EDGE);
	ASSERT(s != NULL);

	Schema_AddEdgeToIndex(s, e);
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
static void _GraphContext_Free
(
	void *arg
) {
	GraphContext *gc = (GraphContext *)arg;
	uint len;

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

	if(gc->node_schemas) {
		len = arr_len(gc->node_schemas);
		for(uint32_t i = 0; i < len; i ++) {
			Schema_Free(gc->node_schemas[i]);
		}
		arr_free(gc->node_schemas);
	}

	//--------------------------------------------------------------------------
	// free relation schemas
	//--------------------------------------------------------------------------

	if(gc->relation_schemas) {
		len = arr_len(gc->relation_schemas);
		for(uint32_t i = 0; i < len; i ++) {
			Schema_Free(gc->relation_schemas[i]);
		}
		arr_free(gc->relation_schemas);
	}

	if(should_lock) {
		RedisModule_ThreadSafeContextUnlock(ctx);
		RedisModule_FreeThreadSafeContext(ctx);
	}

	//--------------------------------------------------------------------------
	// free queries log
	//--------------------------------------------------------------------------

	QueriesLog_Free(gc->queries_log);

	//--------------------------------------------------------------------------
	// free attribute mappings
	//--------------------------------------------------------------------------

	if(gc->attributes) raxFree(gc->attributes);

	if(gc->string_mapping) {
		len = arr_len(gc->string_mapping);
		for(uint32_t i = 0; i < len; i ++) {
			rm_free(gc->string_mapping[i]);
		}
		arr_free(gc->string_mapping);
	}

	int res = pthread_rwlock_destroy (&gc->_schema_rwlock) ;
	ASSERT (res == 0) ;

	if(gc->slowlog) SlowLog_Free(gc->slowlog);

	//--------------------------------------------------------------------------
	// clear cache
	//--------------------------------------------------------------------------

	if(gc->cache) Cache_Free(gc->cache);

	//--------------------------------------------------------------------------
	// free pending write queue
	//--------------------------------------------------------------------------

	if(gc->pending_write_queue != NULL) {
		ASSERT(CircularBuffer_Empty(gc->pending_write_queue));
		CircularBuffer_Free(gc->pending_write_queue, NULL);
	}

	GraphEncodeContext_Free(gc->encoding_context);
	GraphDecodeContext_Free(gc->decoding_context);
	rm_free(gc->graph_name);
	rm_free(gc);
}

