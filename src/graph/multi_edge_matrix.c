/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "util/arr.h"
#include "util/dict.h"
#include "multi_edge_matrix.h"
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

void MultiEdgeMatrix_FormConnections
(
	MultiEdgeMatrix *M,
	Edge **edges
) {
	ASSERT(M != NULL);

	GrB_Info   info;
	dictEntry *entry;
	dict  *multi           = HashTableCreate(&def_dt);
	uint   edge_count      = array_len(edges);
	Edge  *single_to_multi = array_new(Edge, 0);
	EdgeID meid            = INVALID_ENTITY_ID;
	NodeID prev_src        = INVALID_ENTITY_ID;
	NodeID prev_dest       = INVALID_ENTITY_ID;
	EdgeID prev_edge_id    = INVALID_ENTITY_ID;

	// detect multi edges and create multi edge id
	for(uint i = 0; i < edge_count; i++) {
		Edge  *e       = edges[i];
		NodeID src     = e->src_id;
		NodeID dest    = e->dest_id;
		EdgeID edge_id = e->id;

		if(src == prev_src && dest == prev_dest) {
			if(meid == INVALID_ENTITY_ID) {
				meid = array_len(M->freelist) > 0 
						? array_pop(M->freelist) 
						: M->row_id++;
				entry = HashTableAddOrFind(multi, (void *)prev_edge_id);
				HashTableSetVal(multi, entry, (void *)(SET_MSB(meid)));
			}
			entry = HashTableAddOrFind(multi, (void *)edge_id);
			HashTableSetVal(multi, entry, (void *)(SET_MSB(meid)));
			continue;
		}

		meid = INVALID_ENTITY_ID;
		GrB_Index current_edge;
		GrB_Info info = Delta_Matrix_extractElement_UINT64(&current_edge, M->R, src, dest);
		if(info == GrB_SUCCESS) {
			if(SINGLE_EDGE(current_edge)) {
				meid = array_len(M->freelist) > 0 
					? array_pop(M->freelist) 
					: M->row_id++;
				entry = HashTableAddRaw(multi, (void *)edge_id, NULL);
				HashTableSetVal(multi, entry, (void *)(SET_MSB(meid)));
				Edge e = { .src_id = src, .dest_id = dest, .id = current_edge };
				array_append(single_to_multi, e);
				entry = HashTableAddRaw(multi, (void *)current_edge, NULL);
				HashTableSetVal(multi, entry, (void *)(SET_MSB(meid)));
			} else {
				entry = HashTableAddRaw(multi, (void *)edge_id, NULL);
				HashTableSetVal(multi, entry, (void *)current_edge);
				meid = current_edge;
			}
		} 

		prev_src = src;
		prev_dest = dest;
		prev_edge_id = edge_id;
	}

	// create edges
	for(uint i = 0; i < edge_count; i++) {
		Edge *e = edges[i];
		NodeID src = e->src_id;
		NodeID dest = e->dest_id;
		EdgeID edge_id = e->id;

		entry = HashTableFind(multi, (void *)edge_id);
		if(entry == NULL) {
			info = Delta_Matrix_setElement_UINT64(M->R, edge_id, src, dest);
			ASSERT(info == GrB_SUCCESS);
		} else {
			meid = (GrB_Index)HashTableGetVal(entry);
			info = Delta_Matrix_setElement_UINT64(M->R, meid, src, dest);
			ASSERT(info == GrB_SUCCESS);
			info = Delta_Matrix_setElement_BOOL(M->E, CLEAR_MSB(meid), edge_id);
			ASSERT(info == GrB_SUCCESS);
		}
	}

	// add single edge to multi edge mapping
	uint count = array_len(single_to_multi);
	for(uint i = 0; i < count; i++) {
		GrB_Index meid;
		Edge      e       = single_to_multi[i];
		GrB_Index edge_id = e.id;
		GrB_Index src     = e.src_id;
		GrB_Index dest    = e.dest_id;

		entry = HashTableFind(multi, (void *)edge_id);
		meid  = (GrB_Index)HashTableGetVal(entry);
		info  = Delta_Matrix_setElement_BOOL(M->E, CLEAR_MSB(meid), edge_id);
		ASSERT(info == GrB_SUCCESS);
	}

	// cleanup
	HashTableRelease(multi);
	array_free(single_to_multi);
}

void MultiEdgeMatrix_free
(
	MultiEdgeMatrix *M
) {
	ASSERT(M != NULL);

	Delta_Matrix_free(&M->R);
	Delta_Matrix_free(&M->E);
	array_free(M->freelist);
}