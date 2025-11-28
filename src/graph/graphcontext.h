/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "graph.h"
#include "../redismodule.h"
#include "../index/index.h"
#include "../schema/schema.h"
#include "../util/cache/cache.h"
#include "../slow_log/slow_log.h"
#include "../queries_log/queries_log.h"
#include "../serializers/encode_context.h"
#include "../serializers/decode_context.h"

#include <stdatomic.h>

// GraphContext holds refrences to various elements of a graph object
// it is the value sitting behind a Redis graph key
//
// the graph context is versioned, the version value itself is meaningless
// it is used as a "signature" for the graph schema: (labels, relationship-types
// and attribute set) client libraries which cache the mapping between graph
// schema elements and their internal IDs (see COMPACT reply formatter)
// can use the graph version to understand if the schema was modified
// and take action accordingly

typedef struct {
	Graph *g;                              // container for all matrices and entity properties
	int ref_count;                         // number of active references
	rax *attributes;                       // from strings to attribute IDs
	pthread_rwlock_t _attribute_rwlock;    // read-write lock to protect access to the attribute maps
	char *graph_name;                      // string associated with graph
	char **string_mapping;                 // from attribute IDs to strings
	Schema **node_schemas;                 // array of schemas for each node label
	Schema **relation_schemas;             // array of schemas for each relation type
	unsigned short index_count;            // number of indicies
	SlowLog *slowlog;                      // slowlog associated with graph
	QueriesLog queries_log;                // log last x executed queries
	GraphEncodeContext *encoding_context;  // encode context of the graph
	GraphDecodeContext *decoding_context;  // decode context of the graph
	Cache *cache;                          // global cache of execution plans
	XXH32_hash_t version;                  // graph version
	RedisModuleString *telemetry_stream;   // telemetry stream name
	
	atomic_bool write_in_progress;         // write query in progess
	CircularBuffer pending_write_queue;    // pending write queries queue
} GraphContext;

//------------------------------------------------------------------------------
// GraphContext API
//------------------------------------------------------------------------------

// Creates and initializes a graph context struct.
GraphContext *GraphContext_New
(
	const char *graph_name
);

// increase graph context ref count by 1
void GraphContext_IncreaseRefCount
(
	GraphContext *gc
);

// decrease graph context ref count by 1
void GraphContext_DecreaseRefCount
(
	GraphContext *gc
);

// retrive the graph context according to the graph name
// readOnly is the access mode to the graph key
GraphContext *GraphContext_Retrieve
(
	RedisModuleCtx *ctx,
	RedisModuleString *graphID,
	bool readOnly,
	bool shouldCreate
);

// decrease graph context reference count
// graph context will be free once reference count reaches 0
void GraphContext_Release
(
	GraphContext *gc // graph context to release
);

// mark graph key as "dirty" for Redis to pick up on
void GraphContext_MarkWriter
(
	RedisModuleCtx *ctx,
	GraphContext *gc
);

void GraphContext_LockForCommit
(
	RedisModuleCtx *ctx,
	GraphContext *gc
);

void GraphContext_UnlockCommit
(
	RedisModuleCtx *ctx,
	GraphContext *gc
);

// attempt to acquire exclusive write access to the given graph
// returns true if the calling thread successfully acquired write ownership
// returns false if another write is already in progress
bool GraphContext_TryEnterWrite
(
	GraphContext *gc  // graph context
);

// release exclusive write access to the graph
// this should be called by a thread that previously acquired write ownership
// via GraphContext_TryEnterWrite, it clears the write-in-progress flag
void GraphContext_ExitWrite
(
	GraphContext *gc  // graph context
);

// enqueue a write query for deferred execution on the specified graph
// returns true if the query was successfully enqueued
// false if the enqueue operation failed (e.g., due to allocation failure)
bool GraphContext_EnqueueWriteQuery
(
	GraphContext *gc,  // graph context
	void *query_ctx    // query context
);

// dequeue the next pending write query for the specified graph
// returns a query context pointer if a query was dequeued,
// or NULL if the pending write queue is empty
void *GraphContext_DequeueWriteQuery
(
	GraphContext *gc  // graph context
);

// checks if the graph's pending write queue is empty
bool GraphContext_WriteQueueEmpty
(
	const GraphContext *gc  // graph context
);

// get graph name out of graph context
const char *GraphContext_GetName
(
	const GraphContext *gc
);

// get graph context's telemetry stream name
const RedisModuleString *GraphContext_GetTelemetryStreamName
(
	const GraphContext *gc
);

// rename a graph context
void GraphContext_Rename
(
	RedisModuleCtx *ctx,  // redis module context
	GraphContext *gc,     // graph context to rename
	const char *name      // new name
);

// Get graph context version
XXH32_hash_t GraphContext_GetVersion
(
	const GraphContext *gc
);

// get graph from graph context
Graph *GraphContext_GetGraph
(
	const GraphContext *gc
);

//------------------------------------------------------------------------------
// Schema API
//------------------------------------------------------------------------------

// retrieve number of schemas created for given type
unsigned short GraphContext_SchemaCount
(
	const GraphContext *gc,
	SchemaType t
);

// enable all constraints
void GraphContext_EnableConstrains
(
	const GraphContext *gc
);

// disable all constraints
void GraphContext_DisableConstrains
(
	GraphContext *gc
);

// retrieve the specific schema for the provided ID
Schema *GraphContext_GetSchemaByID
(
	const GraphContext *gc,
	int id,
	SchemaType t
);

// retrieve the specific schema for the provided node label
// or relation type string
Schema *GraphContext_GetSchema
(
	const GraphContext *gc,
	const char *label,
	SchemaType t
);

// add a new schema and matrix for the given label
Schema *GraphContext_AddSchema
(
	GraphContext *gc,
	const char *label,
	SchemaType t
);

// removes a schema with a specific id
void GraphContext_RemoveSchema
(
	GraphContext *gc,
	int schema_id,
	SchemaType t
);

// returns the label string for a given Node object
const char *GraphContext_GetNodeLabel
(
	const GraphContext *gc,
	Node *n
);

// returns the relation type string for a given edge object
const char *GraphContext_GetEdgeRelationType
(
	const GraphContext *gc,
	Edge *e
);

// returns number of unique attribute keys
uint GraphContext_AttributeCount
(
	GraphContext *gc
);

// returns an attribute ID given a string, creating one if not found
AttributeID GraphContext_FindOrAddAttribute
(
	GraphContext *gc,
	const char *attribute,
	bool* created
);

// returns an attribute string given an ID
const char *GraphContext_GetAttributeString
(
	GraphContext *gc,
	AttributeID id
);

// returns an attribute ID given a string
// or ATTRIBUTE_ID_NONE if attribute doesn't exist
AttributeID GraphContext_GetAttributeID
(
	GraphContext *gc,
	const char *str
);

// removes an attribute from the graph
void GraphContext_RemoveAttribute
(
	GraphContext *gc,
	AttributeID id
);

//------------------------------------------------------------------------------
// Index API
//------------------------------------------------------------------------------

// returns true if the passed graph context has indices, false otherwise.
bool GraphContext_HasIndices
(
	GraphContext *gc
);

// returns the number of node indices within the passed graph context.
uint64_t GraphContext_NodeIndexCount
(
	const GraphContext *gc
);

// returns the number of edge indices within the passed graph context.
uint64_t GraphContext_EdgeIndexCount
(
	const GraphContext *gc
);

// attempt to retrieve an index on the given label and attribute IDs
Index GraphContext_GetIndexByID
(
	const GraphContext *gc,      // graph context
	int lbl_id,                  // label / rel-type ID
	const AttributeID *attrs,    // attributes
	uint n,                      // attributes count
	IndexFieldType t,            // all index attributes must be of this type
	GraphEntityType entity_type  // schema type NODE / EDGE
);

// attempt to retrieve an index on the given label and attribute
Index GraphContext_GetIndex
(
	const GraphContext *gc,
	const char *label,
	AttributeID *attrs,
	uint n,
	IndexFieldType type,
	SchemaType schema_type
);

// remove and free an index
int GraphContext_DeleteIndex
(
	GraphContext *gc,
	SchemaType schema_type,
	const char *label,
	const char *field,
	IndexFieldType t
);

// remove a single node from all indices that refer to it
void GraphContext_DeleteNodeFromIndices
(
	GraphContext *gc,  // graph context
	Node *n,           // node to remove from index
	LabelID *labels,   // [optional] node labels to remove from index
	uint label_count   // [optional] number of labels
);

// remove a single edge from all indices that refer to it
void GraphContext_DeleteEdgeFromIndices
(
	GraphContext *gc,  // graph context
	Edge *e            // edge to remove from index
);

// add node to any relevant index
void GraphContext_AddNodeToIndices
(
	GraphContext *gc,  // graph context
	Node *n            // node to add to index
);

// add edge to any relevant index
void GraphContext_AddEdgeToIndices
(
	GraphContext *gc,  // graph context
	Edge *e            // edge to add to index
);

// add GraphContext to global array
void GraphContext_RegisterWithModule
(
	GraphContext *gc
);

// retrive GraphContext from the global array
// graph isn't registered, NULL is returned
// graph's references count isn't increased!
// this is OK as long as only a single thread has access to the graph
GraphContext *GraphContext_UnsafeGetGraphContext
(
	const char *graph_name  // graph name
);

//------------------------------------------------------------------------------
// Slowlog API
//------------------------------------------------------------------------------

SlowLog *GraphContext_GetSlowLog
(
	const GraphContext *gc
);

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
);

//------------------------------------------------------------------------------
// Cache API
//------------------------------------------------------------------------------

// return cache associated with graph context and current thread id
Cache *GraphContext_GetCache
(
	const GraphContext *gc
);

