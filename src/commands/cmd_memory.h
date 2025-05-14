/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../redismodule.h"

typedef struct {
	int schema_id; // schema ID
	size_t sz;     // size
} SchemaSize;

typedef struct {
	size_t lbl_matrices_sz;     // [output] label matrices memory usage
	size_t rel_matrices_sz;     // [output] relation matrices memory usage
	size_t node_storage_sz;     // [output] node storage memory usage
	size_t edge_storage_sz;     // [output] edge storage memory usage
	size_t indices_sz;          // [output] indices memory usage
	size_t total_graph_sz_mb;      // [output] total size in MB
	SchemaSize *node_by_label_sz;  // [output] node storage memory usage by label
	SchemaSize *edge_by_type_sz;   // [output] edge storage memory usage by type
} MemoryUsageResult;

// GRAPH.MEMORY USAGE <key> command reports the number of bytes that a graph
// require to be stored in RAM
// usage:
// GRAPH.MEMORY USAGE g
// GRAPH.MEMORY USAGE g [SAMPLES count]
int Graph_Memory
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // arguments
	int argc                   // number of arguments
);

