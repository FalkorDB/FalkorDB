/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_context.h"
#include "../RG.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../util/rax_extensions.h"

// creates a new graph decoding context
GraphDecodeContext *GraphDecodeContext_New() {
	GraphDecodeContext *ctx = rm_calloc(1, sizeof(GraphDecodeContext));

	ctx->meta_keys        = raxNew();
	ctx->multi_edge       = NULL;
	ctx->keys_processed   = 0;
	ctx->graph_keys_count = 1;

	return ctx;
}

// reset a graph decoding context
void GraphDecodeContext_Reset
(
	GraphDecodeContext *ctx
) {
	ASSERT(ctx);

	ctx->node_count         = 0;
	ctx->edge_count         = 0;
	ctx->keys_processed     = 0;
	ctx->graph_keys_count   = 1;
	ctx->deleted_node_count = 0;
	ctx->deleted_edge_count = 0;

	if(ctx->multi_edge) {
		array_free(ctx->multi_edge);
		ctx->multi_edge = NULL;
	}
}

// sets the number of keys required for decoding the graph
void GraphDecodeContext_SetKeyCount
(
	GraphDecodeContext *ctx,
	uint64_t key_count
) {
	ASSERT(ctx);
	ctx->graph_keys_count = key_count;
}

// returns the number of keys required for decoding the graph
uint64_t GraphDecodeContext_GetKeyCount
(
	const GraphDecodeContext *ctx
) {
	ASSERT(ctx);
	return ctx->graph_keys_count;
}

// add a meta key name, required for encoding the graph
void GraphDecodeContext_AddMetaKey
(
	GraphDecodeContext *ctx,
	const char *key
) {
	ASSERT(ctx);
	raxInsert(ctx->meta_keys, (unsigned char *)key, strlen(key), NULL, NULL);
}

// returns a dynamic array with copies of the meta key names
unsigned char **GraphDecodeContext_GetMetaKeys
(
	const GraphDecodeContext *ctx
) {
	ASSERT(ctx);
	return raxKeys(ctx->meta_keys);
}

// removes the stored meta key names from the context
void GraphDecodeContext_ClearMetaKeys
(
	GraphDecodeContext *ctx
) {
	ASSERT(ctx);
	raxFree(ctx->meta_keys);
	ctx->meta_keys = raxNew();
}

// returns if the the number of processed keys is equal to the total number of graph keys.
bool GraphDecodeContext_Finished
(
	const GraphDecodeContext *ctx
) {
	ASSERT(ctx);
	return ctx->keys_processed == ctx->graph_keys_count;
}

// increment the number of processed keys by one
void GraphDecodeContext_IncreaseProcessedKeyCount
(
	GraphDecodeContext *ctx
) {
	ASSERT(ctx);
	ctx->keys_processed++;
}

// returns the number of processed keys
bool GraphDecodeContext_GetProcessedKeyCount
(
	const GraphDecodeContext *ctx
) {
	ASSERT(ctx);
	return ctx->keys_processed;
}

// free graph decoding context
void GraphDecodeContext_Free
(
	GraphDecodeContext *ctx
) {
	if(ctx) {
		raxFree(ctx->meta_keys);

		if(ctx->multi_edge) {
			array_free(ctx->multi_edge);
			ctx->multi_edge = NULL;
		}

		rm_free(ctx);
	}
}

