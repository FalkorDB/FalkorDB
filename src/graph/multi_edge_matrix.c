/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "util/arr.h"
#include "multi_edge_matrix.h"

#include <query_ctx.h>
#include <undo_log/undo_log.h>

#include "delta_matrix/delta_matrix_iter.h"

bool _DepletedIter
(
	MultiEdgeIterator *it,
	NodeID *src,
	NodeID *dest,
	EdgeID *edge_id
) {
	ASSERT(it != NULL);

	return false;
}

bool _SourceTransposeIter
(
	MultiEdgeIterator *it,
	NodeID *src,
	NodeID *dest,
	EdgeID *edge_id
) {
	ASSERT(it != NULL);

	if(Delta_MatrixTupleIter_is_attached(&it->e_it, it->M->E)) {
		GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&it->e_it, NULL, edge_id, NULL);
		if(info == GrB_SUCCESS) {
			if(src) *src = it->src;
			if(dest) *dest = it->dest;
			return true;
		}
		Delta_MatrixTupleIter_detach(&it->e_it);
	}

	GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&it->r_it, &it->dest, &it->src, NULL);
	if(info == GrB_SUCCESS) {
		if(src) *src = it->src;
		if(dest) *dest = it->dest;
		Delta_Matrix_extractElement_UINT64(edge_id, it->M->R, it->src, it->dest);
		if(!SINGLE_EDGE(*edge_id)) {
			Delta_MatrixTupleIter_AttachRange(&it->e_it, it->M->E, CLEAR_MSB(*edge_id), CLEAR_MSB(*edge_id));
			GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&it->e_it, NULL, edge_id, NULL);
			ASSERT(info == GrB_SUCCESS);
		}
		return true;
	}
	it->iter_func = _DepletedIter;
	return false;
}

bool _SourceIter
(
	MultiEdgeIterator *it,
	NodeID *src,
	NodeID *dest,
	EdgeID *edge_id
) {
	ASSERT(it != NULL);

	if(Delta_MatrixTupleIter_is_attached(&it->e_it, it->M->E)) {
		GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&it->e_it, NULL, edge_id, NULL);
		if(info == GrB_SUCCESS) {
			if(src) *src = it->src;
			if(dest) *dest = it->dest;
			return true;
		}
		Delta_MatrixTupleIter_detach(&it->e_it);
	}

	GrB_Info info = Delta_MatrixTupleIter_next_UINT64(&it->r_it, &it->src, &it->dest, edge_id);
	if(info == GrB_SUCCESS) {
		if(src) *src = it->src;
		if(dest) *dest = it->dest;
		if(!SINGLE_EDGE(*edge_id)) {
			Delta_MatrixTupleIter_AttachRange(&it->e_it, it->M->E, CLEAR_MSB(*edge_id), CLEAR_MSB(*edge_id));
			info = Delta_MatrixTupleIter_next_BOOL(&it->e_it, NULL, edge_id, NULL);
			ASSERT(info == GrB_SUCCESS);
		}
		return true;
	}
	it->iter_func = _DepletedIter;
	return false;
}

bool _SourceDestSingleEdgeIter
(
	MultiEdgeIterator *it,
	NodeID *src,
	NodeID *dest,
	EdgeID *edge_id
) {
	ASSERT(it != NULL);

	*edge_id = it->edge_id;
	it->iter_func = _DepletedIter;
	return true;
}

bool _SourceDestMultiEdgeIter
(
	MultiEdgeIterator *it,
	NodeID *src,
	NodeID *dest,
	EdgeID *edge_id
) {
	ASSERT(it != NULL);

	GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&it->e_it, NULL, edge_id, NULL);
	return info == GrB_SUCCESS;
}

void MultiEdgeIterator_AttachSourceRange
(
	MultiEdgeIterator *it,
	MultiEdgeMatrix *M,
	NodeID min_src_id,
	NodeID max_src_id,
	bool transposed
) {
	ASSERT(it != NULL);
	ASSERT(M != NULL);

	it->M = M;
	if(transposed) {
		Delta_MatrixTupleIter_AttachRange(&it->r_it, Delta_Matrix_getTranspose(M->R), min_src_id, max_src_id);
		it->iter_func = _SourceTransposeIter;
	} else {
		Delta_MatrixTupleIter_AttachRange(&it->r_it, M->R, min_src_id, max_src_id);
		it->iter_func = _SourceIter;
	}
}

void MultiEdgeIterator_AttachSourceDest
(
	MultiEdgeIterator *it,
	MultiEdgeMatrix *M,
	NodeID src_id,
	NodeID dest_id
) {
	ASSERT(it != NULL);
	ASSERT(M != NULL);

	it->M = M;
	GrB_Info res = Delta_Matrix_extractElement_UINT64(&it->edge_id, it->M->R, src_id, dest_id);
	if(res == GrB_SUCCESS) {
		if(SINGLE_EDGE(it->edge_id)) {
			it->iter_func = _SourceDestSingleEdgeIter;
		} else {
			Delta_MatrixTupleIter_AttachRange(&it->e_it, it->M->E, CLEAR_MSB(it->edge_id), CLEAR_MSB(it->edge_id));
			it->iter_func = _SourceDestMultiEdgeIter;
		}
	} else {
		it->iter_func = _DepletedIter;
	}
}

bool MultiEdgeIterator_next
(
	MultiEdgeIterator *it,
	NodeID *src,
	NodeID *dest,
	EdgeID *edge_id
) {
	ASSERT(it != NULL);

	return it->iter_func(it, src, dest, edge_id);
}

bool MultiEdgeIterator_is_attached
(
	MultiEdgeIterator *it,
	MultiEdgeMatrix *M
) {
	ASSERT(it != NULL);
	ASSERT(M != NULL);

	return it->M == M;
}

void MultiEdgeMatrix_init
(
	MultiEdgeMatrix *M,
	GrB_Index nrows,
	GrB_Index ncols,
	GrB_Index me_nrows,
	GrB_Index me_ncols
) {
	ASSERT(M != NULL);

	Delta_Matrix_new(&M->R, GrB_UINT64, nrows, ncols, true);
	Delta_Matrix_new(&M->E, GrB_BOOL, me_nrows, me_ncols, false);
	M->row_id = 0;
	M->freelist = array_new(uint64_t, 0);
}

void MultiEdgeMatrix_FormConnection
(
	MultiEdgeMatrix *M,
	NodeID src,
	NodeID dest,
	EdgeID edge_id
) {
	ASSERT(M != NULL);

	GrB_Index current_edge;
	GrB_Info info = Delta_Matrix_extractElement_UINT64(&current_edge, M->R, src, dest);
	if(info == GrB_NO_VALUE) {
		info = Delta_Matrix_setElement_UINT64(M->R, edge_id, src, dest);
		ASSERT(info == GrB_SUCCESS);
	} else if(SINGLE_EDGE(current_edge)) {
		GrB_Index meid = array_len(M->freelist) > 0 
			? array_pop(M->freelist) 
			: M->row_id++;
		info = Delta_Matrix_setElement_UINT64(M->R, SET_MSB(meid), src, dest);
		ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_setElement_BOOL(M->E, meid, current_edge);
		ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_setElement_BOOL(M->E, meid, edge_id);
		ASSERT(info == GrB_SUCCESS);
	} else {
		info = Delta_Matrix_setElement_BOOL(M->E, CLEAR_MSB(current_edge), edge_id);
		ASSERT(info == GrB_SUCCESS);
	}
}

void MultiEdgeMatrix_FormConnections(
	const struct MultiEdgeCreationCtx* ctx,
	const size_t edge_count
)
{
	MultiEdgeMatrix* M = ctx->M;
	ASSERT(M != NULL);

	// If no value exists yet, and we intent to only add a single edge, add it normally
	const bool has_value = (ctx->current_value != -1);
	if (!has_value && edge_count == 1)
	{
		Edge* edge = ctx->edges_to_add[0];
		printf("%lu %lu %lu\n", edge->src_id, edge->dest_id, edge->id);

		const GrB_Info info = Delta_Matrix_setElement_UINT64(M->R, edge->id, ctx->src, ctx->dest);
		ASSERT(info == GrB_SUCCESS);
		return;
	}

	// If we currently are at a single_edge we need to set it to the selected meid
	GrB_Index meid;
	if (!has_value || !ctx->is_me)
	{
		meid = array_len(M->freelist) > 0
									   ? array_pop(M->freelist)
									   : M->row_id++;
		GrB_Info info = Delta_Matrix_setElement_UINT64(M->R, SET_MSB(meid), ctx->src, ctx->dest);
		ASSERT(info == GrB_SUCCESS);

		if (has_value) // If we were a single edge, we need to readd that edge it after switching to multi-edge
		{
			info = Delta_Matrix_setElement_BOOL(M->E, meid, ctx->current_value);
			ASSERT(info == GrB_SUCCESS);
		}
	} else // Has value, but is not single edge
	{
		meid = ctx->current_value;
	}

	for (size_t i = 0; i < edge_count; i++)
	{
		const GrB_Info info = Delta_Matrix_setElement_BOOL(M->E, meid, ctx->edges_to_add[i]->id);
		ASSERT(info == GrB_SUCCESS);
	}
}

void  MultiEdgeMatrix_free
(
	MultiEdgeMatrix *M
) {
	ASSERT(M != NULL);

	Delta_Matrix_free(&M->R);
	Delta_Matrix_free(&M->E);
	array_free(M->freelist);
}