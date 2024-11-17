/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "stdlib.h"
#include "stdbool.h"
#include "stdint.h"
#include "rax.h"

// A struct that maintains the state of a graph decoding from RDB.
typedef struct {
	uint64_t keys_processed;      // count the number of procssed graph keys
	uint64_t graph_keys_count;    // the number of keys representing the graph
	rax *meta_keys;               // the meta keys encountered so far in the decode process
	uint64_t *multi_edge;         // is relation contains multi edge values.
	uint64_t node_count;          // number of nodes to decode
	uint64_t edge_count;          // number of edges to decode
	uint64_t deleted_node_count;  // number of deleted nodes to decode
	uint64_t deleted_edge_count;  // number of deleted edges to decode
} GraphDecodeContext;

// creates a new graph decoding context
GraphDecodeContext *GraphDecodeContext_New();

// reset a graph decoding context
void GraphDecodeContext_Reset
(
	GraphDecodeContext *ctx
);

// sets the number of keys required for decoding the graph
void GraphDecodeContext_SetKeyCount
(
	GraphDecodeContext *ctx,
	uint64_t key_count
);

// returns the number of keys required for decoding the graph
uint64_t GraphDecodeContext_GetKeyCount
(
	const GraphDecodeContext *ctx
);

// add a meta key name, required for encoding the graph
void GraphDecodeContext_AddMetaKey
(
	GraphDecodeContext *ctx,
	const char *key
);

// returns a dynamic array with copies of the meta key names
unsigned char **GraphDecodeContext_GetMetaKeys
(
	const GraphDecodeContext *ctx
);

// removes the stored meta key names from the context
void GraphDecodeContext_ClearMetaKeys
(
	GraphDecodeContext *ctx
);

// returns if the number of processed keys is equal to the total number of graph keys
bool GraphDecodeContext_Finished
(
	const GraphDecodeContext *ctx
);

// increment the number of processed keys by one
void GraphDecodeContext_IncreaseProcessedKeyCount
(
	GraphDecodeContext *ctx
);

// returns the number of processed keys
bool GraphDecodeContext_GetProcessedKeyCount
(
	const GraphDecodeContext *ctx
);

// free graph decoding context
void GraphDecodeContext_Free
(
	GraphDecodeContext *ctx
);

