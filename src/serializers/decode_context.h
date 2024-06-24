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
	uint64_t keys_processed;    // Count the number of procssed graph keys.
	uint64_t graph_keys_count;  // The number of keys representing the graph.
	rax *meta_keys;             // The meta keys encountered so far in the decode process.
	uint64_t *multi_edge;       // Is relation contains multi edge values.
} GraphDecodeContext;

// Creates a new graph decoding context.
GraphDecodeContext *GraphDecodeContext_New();

// Reset a graph decoding context.
void GraphDecodeContext_Reset(GraphDecodeContext *ctx);

// Sets the number of keys required for decoding the graph.
void GraphDecodeContext_SetKeyCount(GraphDecodeContext *ctx, uint64_t key_count);

// Returns the number of keys required for decoding the graph.
uint64_t GraphDecodeContext_GetKeyCount(const GraphDecodeContext *ctx);

// Add a meta key name, required for encoding the graph.
void GraphDecodeContext_AddMetaKey(GraphDecodeContext *ctx, const char *key);

// Returns a dynamic array with copies of the meta key names.
unsigned char **GraphDecodeContext_GetMetaKeys(const GraphDecodeContext *ctx);

// Removes the stored meta key names from the context.
void GraphDecodeContext_ClearMetaKeys(GraphDecodeContext *ctx);

// Returns if the number of processed keys is equal to the total number of graph keys.
bool GraphDecodeContext_Finished(const GraphDecodeContext *ctx);

// Increment the number of processed keys by one.
void GraphDecodeContext_IncreaseProcessedKeyCount(GraphDecodeContext *ctx);

// Returns the number of processed keys.
bool GraphDecodeContext_GetProcessedKeyCount(const GraphDecodeContext *ctx);

// Free graph decoding context.
void GraphDecodeContext_Free(GraphDecodeContext *ctx);
