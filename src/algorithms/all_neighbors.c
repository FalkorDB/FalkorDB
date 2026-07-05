/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../util/arr.h"
#include "../util/dict.h"
#include "all_neighbors.h"
#include "../util/rmalloc.h"
#include "graph/tensor/tensor.h"

static void _AllNeighborsCtx_CollectNeighbors
(
	AllNeighborsCtx *ctx,
	EntityID id
) {
	ctx->current_level++ ;

	if (ctx->current_level == ctx->n_levels) {
		Delta_MatrixTupleIter iter ;
		Delta_MatrixTupleIter_AttachRange (&iter, ctx->M, id, id) ;

		ctx->levels = rm_realloc (ctx->levels,
				sizeof (Delta_MatrixTupleIter) * (ctx->n_levels+1)) ;

		ctx->levels [ctx->n_levels] = iter ;
		ctx->n_levels++ ;
	} else {
		Delta_MatrixTupleIter_iterate_row (&ctx->levels [ctx->current_level],
				id) ;
	}
}

// linear scan used when path depth is below VISITED_HASHMAP_THRESHOLD.
static inline bool _AllNeighborsCtx_EdgeVisited
(
	AllNeighborsCtx *ctx,
	EntityID edge_id
) {
	if (ctx->use_hashmap) {
		return (HashTableFind (ctx->visited_edges, (void *) edge_id)) != NULL ;
	}

	const EntityID *restrict visited = ctx->visited ;
	int len = arr_len(ctx->visited) ;

	for (int i = 0; i < len; i++) {
		if (visited[i] == edge_id) return true ;
	}

	return false ;
}

void AllNeighborsCtx_Reset
(
	AllNeighborsCtx *ctx,  // all neighbors context to reset
	EntityID src,          // source node from which to traverse
	Delta_Matrix M,        // matrix describing connections
	uint minLen,           // minimum traversal depth
	uint maxLen            // maximum traversal depth
) {
	ASSERT (M            != NULL) ;
	ASSERT (src          != INVALID_ENTITY_ID) ;
	ASSERT (ctx          != NULL) ;
	ASSERT (ctx->levels  != NULL) ;
	ASSERT (ctx->visited != NULL) ;

	ctx->M             = M ;
	ctx->src           = src ;
	ctx->minLen        = minLen ;
	ctx->maxLen        = maxLen ;
	ctx->first_pull    = true ;
	ctx->current_level = 0 ;

	arr_clear (ctx->visited) ;

	// reset lookup strategy for the new source
	ctx->use_hashmap = (maxLen >= VISITED_HASHMAP_THRESHOLD) ;

	if (ctx->use_hashmap) {
		if (ctx->visited_edges != NULL) {
			HashTableEmpty (ctx->visited_edges, NULL) ;
		} else {
			ctx->visited_edges = HashTableCreate (&def_dt) ;
		}
	} else {
		// array mode: discard any hashmap left from a previous dynamic promotion
		if (ctx->visited_edges != NULL) {
			HashTableRelease (ctx->visited_edges) ;
			ctx->visited_edges = NULL ;
		}
	}
}

AllNeighborsCtx *AllNeighborsCtx_New
(
	EntityID src,    // source node from which to traverse
	Delta_Matrix M,  // matrix describing connections
	uint minLen,     // minimum traversal depth
	uint maxLen      // maximum traversal depth
) {
	ASSERT (M   != NULL) ;
	ASSERT (src != INVALID_ENTITY_ID) ;

	AllNeighborsCtx *ctx = rm_calloc (1, sizeof (AllNeighborsCtx)) ;

	ctx->M             = M ;
	ctx->src           = src ;
	ctx->minLen        = minLen ;
	ctx->maxLen        = maxLen ;
	ctx->levels        = rm_malloc (sizeof(Delta_MatrixTupleIter)) ;
	ctx->n_levels      = 1 ;
	ctx->visited       = arr_new (EntityID, MIN (128, maxLen)) ;
	ctx->first_pull    = true ;
	ctx->current_level = 0 ;
	ctx->use_hashmap   = (maxLen >= VISITED_HASHMAP_THRESHOLD) ;
	ctx->visited_edges = ctx->use_hashmap ? HashTableCreate (&def_dt) : NULL ;

	// dummy iterator at level 0
	ctx->levels [0] = (Delta_MatrixTupleIter) {0} ;

	return ctx ;
}

EntityID AllNeighborsCtx_NextNeighbor
(
	AllNeighborsCtx *ctx
) {
	if (unlikely(ctx == NULL)) {
		return INVALID_ENTITY_ID ;
	}

	if(unlikely (ctx->first_pull)) {
		ASSERT (ctx->current_level == 0) ;
		ctx->first_pull = false ;

		// current_level >= ctx->minLen
		// see if we should expand further?
		if (ctx->current_level < ctx->maxLen) {
			// we can expand further
			_AllNeighborsCtx_CollectNeighbors (ctx, ctx->src);
		}

		if (ctx->minLen == 0) {
			return ctx->src ;
		}
	}

	while (ctx->current_level > 0) {
		ASSERT (ctx->current_level < ctx->n_levels) ;
		Delta_MatrixTupleIter *it = &ctx->levels [ctx->current_level] ;

		// unique stable position into the matrix's internal arrays
		// works on both BOOL and UINT64 matrices (no Ax read)
		GrB_Index edge_id = Delta_Matrix_Iterator_getp (it) ;

		GrB_Index dest_id ;
		GrB_Info info =
			Delta_MatrixTupleIter_next_BOOL (it, NULL, &dest_id, NULL);

		if (info == GxB_EXHAUSTED) {
			// backtrack: only pop if we pushed an edge to reach this level
			--ctx->current_level ;
			if (arr_len (ctx->visited) > 0) {
				EntityID popped = arr_pop (ctx->visited);
				if (ctx->use_hashmap) {
					HashTableDelete (ctx->visited_edges, (void *) popped) ;
				}
			}
			continue ;
		}

		if (_AllNeighborsCtx_EdgeVisited (ctx, edge_id)) {
			// if we have visted this edge already, do not continue down this 
			// path, and do not emmit
			continue ;
		}

		if (ctx->current_level < ctx->minLen) {
			arr_append (ctx->visited, edge_id) ;
			if (ctx->use_hashmap) {
				HashTableAdd (
				ctx->visited_edges, (void *) edge_id, (void *) edge_id);
			}

			// continue traversing
			_AllNeighborsCtx_CollectNeighbors (ctx, dest_id) ;
			continue ;
		}

		// current_level >= ctx->minLen
		// see if we should expand further?
		if (ctx->current_level < ctx->maxLen) {
			arr_append (ctx->visited, edge_id);
			if (ctx->use_hashmap) {
				HashTableAdd (
					ctx->visited_edges, (void *)edge_id, (void *) edge_id) ;
			}

			// we can expand further
			_AllNeighborsCtx_CollectNeighbors (ctx, dest_id) ;
		}

		return dest_id ;
	}

	// couldn't find a neighbor
	return INVALID_ENTITY_ID ;
}

void AllNeighborsCtx_Free
(
	AllNeighborsCtx *ctx
) {
	if (!ctx) {
		return ;
	}

	uint count = ctx->n_levels ;
	for (uint i = 0; i < count; i++) {
		Delta_MatrixTupleIter_detach (ctx->levels + i) ;
	}

	rm_free (ctx->levels) ;
	arr_free (ctx->visited) ;
	if (ctx->visited_edges != NULL) {
		HashTableRelease (ctx->visited_edges) ;
	}

	rm_free (ctx) ;
}

