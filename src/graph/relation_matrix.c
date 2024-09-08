/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "util/arr.h"
#include "util/dict.h"
#include "util/roaring.h"
#include "relation_matrix.h"
#include "delta_matrix/delta_matrix_iter.h"

// don't iterate
bool _DepletedIter
(
	RelationIterator *it,
	NodeID *src,
	NodeID *dest,
	EdgeID *edge_id
) {
	ASSERT(it != NULL);

	return false;
}

// iterate over R matrix scanning a range of rows entry by entry
// entry can a edge id or multi edge id
// when an entry is multi-edge
// iterate over E matrix at row = multi-edge id to get the edge id
// transposed version
bool _SourceTransposeIter
(
	RelationIterator *it,
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

// iterate over R matrix scanning a range of rows entry by entry
// entry can a edge id or multi edge id
// when an entry is multi-edge
// iterate over E matrix at row = multi-edge id to get the edge id
bool _SourceIter
(
	RelationIterator *it,
	NodeID *src,
	NodeID *dest,
	EdgeID *edge_id
) {
	ASSERT(it != NULL);

	// resuming scan on edges matrix
	if(Delta_MatrixTupleIter_is_attached(&it->e_it, it->M->E)) {
		GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&it->e_it, NULL, edge_id, NULL);
		if(info == GrB_SUCCESS) {
			if(src) *src = it->src;
			if(dest) *dest = it->dest;
			return true;
		}
		Delta_MatrixTupleIter_detach(&it->e_it);
	}

	// trying to advance to the next set of src and dest nodes
	GrB_Info info = Delta_MatrixTupleIter_next_UINT64(&it->r_it, &it->src, &it->dest, edge_id);
	if(info == GrB_SUCCESS) {
		if(src) *src = it->src;
		if(dest) *dest = it->dest;
		if(!SINGLE_EDGE(*edge_id)) {
			// src and dest nodes have multiple edge connecting them
			// attach edges iterator and get the first edge
			Delta_MatrixTupleIter_AttachRange(&it->e_it, it->M->E, CLEAR_MSB(*edge_id), CLEAR_MSB(*edge_id));
			info = Delta_MatrixTupleIter_next_BOOL(&it->e_it, NULL, edge_id, NULL);
			ASSERT(info == GrB_SUCCESS);
		}
		return true;
	}
	it->iter_func = _DepletedIter;
	return false;
}

// single edge iterator
bool _SourceDestSingleEdgeIter
(
	RelationIterator *it,
	NodeID *src,
	NodeID *dest,
	EdgeID *edge_id
) {
	ASSERT(it != NULL);

	*edge_id = it->edge_id;
	if(src) *src = it->src;
	if(dest) *dest = it->dest;
	it->iter_func = _DepletedIter;
	return true;
}

// iterate over E matrix at row = multi-edge id to get the edge id
bool _SourceDestMultiEdgeIter
(
	RelationIterator *it,
	NodeID *src,
	NodeID *dest,
	EdgeID *edge_id
) {
	ASSERT(it != NULL);

	if(src) *src = it->src;
	if(dest) *dest = it->dest;
	GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&it->e_it, NULL, edge_id, NULL);
	return info == GrB_SUCCESS;
}

void RelationIterator_AttachSourceRange
(
	RelationIterator *it,
	RelationMatrix M,
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

void RelationIterator_AttachSourceDest
(
	RelationIterator *it,
	RelationMatrix M,
	NodeID src_id,
	NodeID dest_id
) {
	ASSERT(it != NULL);
	ASSERT(M != NULL);

	it->M = M;
	it->src = src_id;
	it->dest = dest_id;
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

bool RelationIterator_next
(
	RelationIterator *it,
	NodeID *src,
	NodeID *dest,
	EdgeID *edge_id
) {
	ASSERT(it != NULL);

	return it->iter_func(it, src, dest, edge_id);
}

bool RelationIterator_is_attached
(
	const RelationIterator *it,
	const RelationMatrix M
) {
	ASSERT(it != NULL);
	ASSERT(M != NULL);

	return it->M == M;
}

RelationMatrix RelationMatrix_new
(
	GrB_Index nrows,
	GrB_Index ncols
) {
	RelationMatrix M = rm_malloc(sizeof(struct RelationMatrix));

	Delta_Matrix_new(&M->R, GrB_UINT64, nrows, ncols, true);
	// delay matrix dimentions will be set on first sync
	Delta_Matrix_new(&M->E, GrB_BOOL, 0, 0, false);
	M->row_id = 0;
	M->freelist = array_new(uint64_t, 0);

	return M;
}

void RelationMatrix_FormConnection
(
	RelationMatrix M,
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

#define NEW_MULTI_EDGE_ID(M) array_len(M->freelist) > 0 ? array_pop(M->freelist) : M->row_id++;

void RelationMatrix_FormConnections
(
	RelationMatrix M,
	const Edge **edges // assume is sorted by src and dest
) {
	ASSERT(M != NULL);
	ASSERT(edges != NULL);

	GrB_Info            info;
	Edge               *multi        = array_new(Edge, 0); // newly added multi-edges
	uint                edge_count   = array_len(edges);
	EdgeID              meid         = INVALID_ENTITY_ID;  // multi-edge id
	NodeID              prev_src     = INVALID_ENTITY_ID;
	NodeID              prev_dest    = INVALID_ENTITY_ID;
	EdgeID              prev_edge_id = INVALID_ENTITY_ID;
	roaring64_bitmap_t *single       = roaring64_bitmap_create(); // marked single edges indices

	roaring64_bitmap_add_range(single, 0, edge_count);

	for(uint i = 0; i < edge_count; i++) {
		const Edge  *e       = edges[i];
		NodeID       src     = e->src_id;
		NodeID       dest    = e->dest_id;
		EdgeID       edge_id = e->id;

		if(src == prev_src && dest == prev_dest) {
			if(meid == INVALID_ENTITY_ID) {
				meid = NEW_MULTI_EDGE_ID(M);
				info = Delta_Matrix_setElement_BOOL(M->E, meid, prev_edge_id);
				ASSERT(info == GrB_SUCCESS);
				// when the R[src, dest] is not exists yet delay setting R[src, dest] = meid
				array_append(multi, ((Edge){.id = meid, .src_id = src, .dest_id = dest}));
				roaring64_bitmap_remove(single, i - 1);
			}
			info = Delta_Matrix_setElement_BOOL(M->E, meid, edge_id);
			ASSERT(info == GrB_SUCCESS);
			roaring64_bitmap_remove(single, i);
			continue;
		}

		meid = INVALID_ENTITY_ID;
		GrB_Index current_edge;
		GrB_Info info = Delta_Matrix_extractElement_UINT64(&current_edge, M->R, src, dest);
		if(info == GrB_SUCCESS) {
			if(SINGLE_EDGE(current_edge)) {
				meid = NEW_MULTI_EDGE_ID(M);
				info = Delta_Matrix_setElement_BOOL(M->E, meid, current_edge);
				ASSERT(info == GrB_SUCCESS);
				info = Delta_Matrix_setElement_BOOL(M->E, meid, edge_id);
				ASSERT(info == GrB_SUCCESS);
				info = Delta_Matrix_setElement_UINT64(M->R, SET_MSB(meid), src, dest);
				ASSERT(info == GrB_SUCCESS);
			} else {
				meid = CLEAR_MSB(current_edge);
				info = Delta_Matrix_setElement_BOOL(M->E, meid, edge_id);
				ASSERT(info == GrB_SUCCESS);
			}
			roaring64_bitmap_remove(single, i);
		} 

		prev_src = src;
		prev_dest = dest;
		prev_edge_id = edge_id;
	}

	uint count = array_len(multi);
	for(uint i = 0; i < count; i++) {
		Edge     *e       = multi + i;
		GrB_Index meid    = e->id;
		GrB_Index src     = e->src_id;
		GrB_Index dest    = e->dest_id;

		info = Delta_Matrix_setElement_UINT64(M->R, SET_MSB(meid), src, dest);
		ASSERT(info == GrB_SUCCESS);
	}

	roaring64_iterator_t *it = roaring64_iterator_create(single);
	while(roaring64_iterator_has_value(it)) {
		GrB_Index i = roaring64_iterator_value(it);
		const Edge *e = edges[i];
		info = Delta_Matrix_setElement_UINT64(M->R, e->id, e->src_id, e->dest_id);
		ASSERT(info == GrB_SUCCESS);
		roaring64_iterator_advance(it);
	}

	// cleanup
	array_free(multi);
	roaring64_iterator_free(it);
	roaring64_bitmap_free(single);
}

// checks to see if matrix has pending operations
bool RelationMatrix_pending
(
	RelationMatrix M   // relation matrix
) {
	ASSERT(M != NULL);
	
	bool pending;

	Delta_Matrix_pending(M->R, &pending);
	if(pending) {
		return true;
	}
	Delta_Matrix_pending(M->E, &pending);
	return pending;
}

void RelationMatrix_free
(
	RelationMatrix *M
) {
	ASSERT(M != NULL && *M != NULL);

	RelationMatrix m = *M;

	Delta_Matrix_free(&m->R);
	Delta_Matrix_free(&m->E);
	array_free(m->freelist);

	rm_free(m);
	*M = NULL;
}