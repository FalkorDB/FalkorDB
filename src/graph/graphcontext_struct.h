/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// GraphContext holds refrences to various elements of a graph object
// it is the value sitting behind a Redis graph key
//
// the graph context is versioned, the version value itself is meaningless
// it is used as a "signature" for the graph schema: (labels, relationship-types
// and attribute set) client libraries which cache the mapping between graph
// schema elements and their internal IDs (see COMPACT reply formatter)
// can use the graph version to understand if the schema was modified
// and take action accordingly

typedef struct GraphContext {
	Graph *g;                              // container for all matrices and entity properties
	int ref_count;                         // number of active references
	rax *attributes;                       // from strings to attribute IDs
	pthread_rwlock_t _schema_rwlock;       // read-write lock to protect access to the graph's schema
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

