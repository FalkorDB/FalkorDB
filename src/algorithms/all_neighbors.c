/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "../util/arr.h"
#include "all_neighbors.h"
#include "../util/rmalloc.h"

static void _AllNeighborsCtx_CollectNeighbors
(
	AllNeighborsCtx *ctx,
	EntityID id
) {
	ctx->current_level++;
	if(ctx->current_level == array_len(ctx->levels)) {
		Delta_MatrixTupleIter iter = {0};
		Delta_MatrixTupleIter_AttachRange(&iter, ctx->M, id, id);
		array_append(ctx->levels, iter);
	} else {
		Delta_MatrixTupleIter_iterate_row(&ctx->levels[ctx->current_level], id);
	}
}

void AllNeighborsCtx_Reset
(
	AllNeighborsCtx *ctx,  // all neighbors context to reset
	EntityID src,          // source node from which to traverse
	Delta_Matrix M,        // matrix describing connections
	uint minLen,           // minimum traversal depth
	uint maxLen            // maximum traversal depth
) {
	ASSERT(M            != NULL);
	ASSERT(src          != INVALID_ENTITY_ID);
	ASSERT(ctx          != NULL);
	ASSERT(ctx->levels  != NULL);
	ASSERT(ctx->visited != NULL);

	ctx->M             = M;
	ctx->src           = src;
	ctx->minLen        = minLen;
	ctx->maxLen        = maxLen;
	ctx->first_pull    = true;
	ctx->current_level = 0;

	array_clear(ctx->levels);
	array_clear(ctx->visited);

	// reset visited nodes
	hashmap_clear(ctx->visited_nodes, true);

	// dummy iterator at level 0
	array_append(ctx->levels, (Delta_MatrixTupleIter) {0});
}

AllNeighborsCtx *AllNeighborsCtx_New
(
	EntityID src,    // source node from which to traverse
	Delta_Matrix M,  // matrix describing connections
	uint minLen,     // minimum traversal depth
	uint maxLen      // maximum traversal depth
) {
	ASSERT(M   != NULL);
	ASSERT(src != INVALID_ENTITY_ID);

	AllNeighborsCtx *ctx = rm_calloc(1, sizeof(AllNeighborsCtx));

	ctx->M             = M;
	ctx->src           = src;
	ctx->minLen        = minLen;
	ctx->maxLen        = maxLen;
	ctx->levels        = array_new(Delta_MatrixTupleIter, 1);
	ctx->visited       = array_new(EntityID, 1);
	ctx->first_pull    = true;
	ctx->current_level = 0;
	ctx->visited_nodes = hashmap_new_with_redis_allocator(0, 0, 0, 0, NULL,
			NULL, NULL, NULL);

	// Dummy iterator at level 0
	array_append(ctx->levels, (Delta_MatrixTupleIter) {0});

	return ctx;
}

EntityID AllNeighborsCtx_NextNeighbor
(
	AllNeighborsCtx *ctx
) {
	if(unlikely(ctx == NULL)) return INVALID_ENTITY_ID;

	if(unlikely(ctx->first_pull)) {
		ASSERT(ctx->current_level == 0);
		ctx->first_pull = false;

		// update visited path, replace frontier with current node
		array_append(ctx->visited, ctx->src);
		hashmap_set_with_hash(ctx->visited_nodes, NULL, ctx->src);

		// current_level >= ctx->minLen
		// see if we should expand further?
		if(ctx->current_level < ctx->maxLen) {
			// we can expand further
			_AllNeighborsCtx_CollectNeighbors(ctx, ctx->src);
		}

		if(ctx->minLen == 0) {
			return ctx->src;
		}
	}

	while(ctx->current_level > 0) {
		ASSERT(ctx->current_level < array_len(ctx->levels));
		Delta_MatrixTupleIter *it = &ctx->levels[ctx->current_level];

		GrB_Index dest_id;
		GrB_Info info = Delta_MatrixTupleIter_next_BOOL(it, NULL, &dest_id,
				NULL);

		if(info == GxB_EXHAUSTED) {
			// backtrack
			ctx->current_level--;
			dest_id = array_pop(ctx->visited);
			const void *res = hashmap_delete_with_hash(ctx->visited_nodes, NULL,
					dest_id);
			ASSERT(res != NULL);
			continue;
		}

		// update visited path, replace frontier with current node
		bool visited =
			hashmap_set_with_hash(ctx->visited_nodes, NULL, dest_id) != NULL;

		if(ctx->current_level < ctx->minLen && !visited) {
			array_append(ctx->visited, dest_id);
			// continue traversing
			_AllNeighborsCtx_CollectNeighbors(ctx, dest_id);
			continue;
		}

		// current_level >= ctx->minLen
		// see if we should expand further?
		if(ctx->current_level < ctx->maxLen && !visited) {
			array_append(ctx->visited, dest_id);
			// we can expand further
			_AllNeighborsCtx_CollectNeighbors(ctx, dest_id);
		}

		return dest_id;
	}

	// couldn't find a neighbor
	return INVALID_ENTITY_ID;
}

void AllNeighborsCtx_Free
(
	AllNeighborsCtx *ctx
) {
	if(!ctx) return;

	uint count = array_len(ctx->levels);
	for (uint i = 0; i < count; i++) {
		Delta_MatrixTupleIter_detach(ctx->levels + i);
	}
	array_free(ctx->levels);
	array_free(ctx->visited);

	hashmap_free(ctx->visited_nodes);

	rm_free(ctx);
}

