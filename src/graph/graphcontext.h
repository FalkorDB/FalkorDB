/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
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

// GraphContext holds refrences to various elements of a graph object
// It is the value sitting behind a Redis graph key
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

// retrieve the label string for a given Node object
const char *GraphContext_GetNodeLabel
(
	const GraphContext *gc,
	Node *n
);

// retrieve the relation type string for a given edge object
const char *GraphContext_GetEdgeRelationType
(
	const GraphContext *gc,
	Edge *e
);

// retrieve number of unique attribute keys
uint GraphContext_AttributeCount
(
	GraphContext *gc
);

// retrieve an attribute ID given a string, creating one if not found
Attribute_ID GraphContext_FindOrAddAttribute
(
	GraphContext *gc,
	const char *attribute,
	bool* created
);

// retrieve an attribute string given an ID
const char *GraphContext_GetAttributeString
(
	GraphContext *gc,
	Attribute_ID id
);

// retrieve an attribute ID given a string
// or ATTRIBUTE_NOTFOUND if attribute doesn't exist
Attribute_ID GraphContext_GetAttributeID
(
	GraphContext *gc,
	const char *str
);

// removes an attribute from the graph
void GraphContext_RemoveAttribute
(
	GraphContext *gc,
	Attribute_ID id
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
	const GraphContext *gc,        // graph context
	int lbl_id,                    // label / rel-type ID
	const Attribute_ID *attrs,     // attributes
	uint n,                        // attributes count
	IndexType idx_type,            // index type
	GraphEntityType entity_type    // schema type NODE / EDGE
);

// attempt to retrieve an index on the given label and attribute
Index GraphContext_GetIndex
(
	const GraphContext *gc,
	const char *label,
	Attribute_ID *attrs,
	uint n,
	IndexType type,
	SchemaType schema_type
);

bool GraphContext_AddExactMatchIndex
(
	Index *idx,                 // [input/output] index created
	GraphContext *gc,           // graph context
	SchemaType schema_type,     // type of entities to index nodes/edges
	const char *label,          // label of indexed entities
	const char **fields_str,    // fields to index
	uint fields_count,          // number of fields to index
	bool should_reply           // should reply to client
);

// create a full text index for the given label and attribute
bool GraphContext_AddFullTextIndex
(
	Index *idx,              // [input/output] index created
	GraphContext *gc,        // graph context
	const char *label,       // label of indexed entities
	const char **fields,     // fields to index
	uint fields_count,       // number of fields to index
	double *weights,         // fields weights
	bool *nostems,           //
	const char **phonetics,  //
	char **stopwords,
	const char *language
);

// remove and free an index
int GraphContext_DeleteIndex
(
	GraphContext *gc,
	SchemaType schema_type,
	const char *label,
	const char *field,
	IndexType type
);

// remove a single node from all indices that refer to it
void GraphContext_DeleteNodeFromIndices
(
	GraphContext *gc,
	Node *n
);

// remove a single edge from all indices that refer to it
void GraphContext_DeleteEdgeFromIndices
(
	GraphContext *gc,
	Edge *e
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
	const GraphContext *gc,       // graph context
	uint64_t received,            // query received timestamp
	double wait_duration,         // waiting time
	double execution_duration,    // executing time
	double report_duration,       // reporting time
	bool parameterized,           // uses parameters
	bool utilized_cache,          // utilized cache
	bool write,    		          // write query
	bool timeout,    		      // timeout query
	const char *query             // query string
);

//------------------------------------------------------------------------------
// Cache API
//------------------------------------------------------------------------------

// return cache associated with graph context and current thread id
Cache *GraphContext_GetCache
(
	const GraphContext *gc
);

