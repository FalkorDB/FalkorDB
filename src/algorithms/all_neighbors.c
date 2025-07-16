/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "all_neighbors.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"

static void _AllNeighborsCtx_CollectNeighbors
(
	AllNeighborsCtx *ctx,
	EntityID id
) {
	ctx->current_level++;
	if(ctx->current_level == array_len(ctx->levels)) {
		RG_MatrixTupleIter iter = {0};
		RG_MatrixTupleIter_AttachRange(&iter, ctx->M, id, id);
		array_append(ctx->levels, iter);
	} else {
		RG_MatrixTupleIter_iterate_row(&ctx->levels[ctx->current_level], id);
	}
}

void AllNeighborsCtx_Reset
(
	AllNeighborsCtx *ctx,  // all neighbors context to reset
	EntityID src,          // source node from which to traverse
	RG_Matrix M,           // matrix describing connections
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
	HashTableRelease(ctx->visited_nodes);
	ctx->visited_nodes = HashTableCreate(&def_dt);

	// dummy iterator at level 0
	array_append(ctx->levels, (RG_MatrixTupleIter) {0});
}

AllNeighborsCtx *AllNeighborsCtx_New
(
	EntityID src,  // source node from which to traverse
	RG_Matrix M,   // matrix describing connections
	uint minLen,   // minimum traversal depth
	uint maxLen    // maximum traversal depth
) {
	ASSERT(M   != NULL);
	ASSERT(src != INVALID_ENTITY_ID);

	AllNeighborsCtx *ctx = rm_calloc(1, sizeof(AllNeighborsCtx));

	ctx->M              = M;
	ctx->src            = src;
	ctx->minLen         = minLen;
	ctx->maxLen         = maxLen;
	ctx->levels         = array_new(RG_MatrixTupleIter, 1);
	ctx->visited        = array_new(EntityID, 1);
	ctx->first_pull     = true;
	ctx->current_level  = 0;
	ctx->visited_nodes  = HashTableCreate(&def_dt);

	// Dummy iterator at level 0
	array_append(ctx->levels, (RG_MatrixTupleIter) {0});

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
		HashTableAdd(ctx->visited_nodes, (void*)(ctx->src), NULL);

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
		RG_MatrixTupleIter *it = &ctx->levels[ctx->current_level];

		GrB_Index dest_id;
		GrB_Info info = RG_MatrixTupleIter_next_UINT64(it, NULL, &dest_id, NULL);

		if(info == GxB_EXHAUSTED) {
			// backtrack
			ctx->current_level--;
			dest_id = array_pop(ctx->visited);
			int res = HashTableDelete(ctx->visited_nodes, (void*)(dest_id));
			ASSERT(res == DICT_OK);
			continue;
		}

		// update visited path, replace frontier with current node
		bool visited =
			HashTableAdd(ctx->visited_nodes, (void*)(dest_id), NULL) != DICT_OK;

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
		RG_MatrixTupleIter_detach(ctx->levels + i);
	}
	array_free(ctx->levels);
	array_free(ctx->visited);

	HashTableRelease(ctx->visited_nodes);

	rm_free(ctx);
}

