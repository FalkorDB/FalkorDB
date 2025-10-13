/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "tensor.h"
#include "util/arr.h"
#include "globals.h"
#include "../delta_matrix/delta_matrix.h"
#include "../delta_matrix/delta_utils.h"
#include "../delta_matrix/delta_matrix_iter.h"

// init new tensor
Tensor Tensor_new
(
	GrB_Index nrows,  // # rows
	GrB_Index ncols   // # columns
) {
	Tensor T;
	Delta_Matrix_new(&T, GrB_UINT64, nrows, ncols, true);

	return T;
}

// set entry at T[row, col] = x
void Tensor_SetElement
(
	Tensor T,       // tensor
	GrB_Index row,  // row
	GrB_Index col,  // col
	uint64_t x      // value
) {
	ASSERT(T != NULL);
	GrB_BinaryOp accum = Global_GrB_Ops_Get()->push_id;
	Delta_Matrix_Assign_Element_UINT64(T, accum, x, row, col);
}

// different set methods available for tensor
enum SetMethod {
    NEW_SCALAR,      // new element creates a new scalar entry
    NEW_VECTOR,      // new element and existing entry are added to a new vector
    EXISTING_VECTOR  // new element is added to existing vector
};

// set multiple entries
void Tensor_SetElements
(
	Tensor T,                        // tensor
	const GrB_Index *restrict rows,  // array of row indices
	const GrB_Index *restrict cols,  // array of column indices
	const uint64_t *restrict vals,   // values
	uint64_t n                       // number of elements
) {
	ASSERT(n    > 0);
	ASSERT(T    != NULL);
	ASSERT(rows != NULL);
	ASSERT(cols != NULL);
	ASSERT(vals != NULL);

	for(uint64_t i = 0; i < n; ++i){
		Tensor_SetElement(T, rows[i], cols[i], vals[i]);
	}
}

// set multiple entries
void Tensor_SetEdges
(
	Tensor T,               // tensor
	const Edge **elements,  // assume edges are sorted by src and dest
	uint64_t n              // number of elements
) {
	ASSERT(T        != NULL);
	ASSERT(elements != NULL);

	// assert assumption
	#ifdef RG_DEBUG
	for(uint64_t i = 0; i < n-1; i++) {
		const Edge *e    = elements[i];
		const Edge *next = elements[i+1];

		// make sure current edge has either a lower source node id
		// or has the same source node id but a lower destination node id
		ASSERT((Edge_GetSrcNodeID(e)  < Edge_GetSrcNodeID(next))   ||
			   (Edge_GetSrcNodeID(e)  == Edge_GetSrcNodeID(next)   &&
				Edge_GetDestNodeID(e) <= Edge_GetDestNodeID(next)));
	}
	#endif
	for(uint64_t i = 0; i < n; ++i) {
		const Edge *e = elements[i];
		NodeID     src    = Edge_GetSrcNodeID(e);
		NodeID     dest   = Edge_GetDestNodeID(e);
		EdgeID     id     = e->id;
		Tensor_SetElement(T, src, dest, id);
	}
}

// qsort element compare function
// compare elements by their value
static int _value_cmp
(
	const void *a,
	const void *b
) {
	Edge *ea = (Edge *)a;
	Edge *eb = (Edge *)b;
	uint64_t a_id = ENTITY_GET_ID(ea);  // A's value
	uint64_t b_id = ENTITY_GET_ID(eb);  // B's value
	return a_id - b_id;
}

// remove multiple entries
// assuming T's entries are all scalar
void Tensor_RemoveElements_Flat
(
	Tensor T,              // tensor
	const Edge *elements,  // elements to remove
	uint64_t n             // number of elements
) {
	ASSERT(T        != NULL);
	ASSERT(elements != NULL);

	for(uint64_t i = 0; i < n; i++) {
		const Edge *e   = elements + i;           // tuple (row, col, x)
		GrB_Index   row = Edge_GetSrcNodeID(e);   // element row index
		GrB_Index   col = Edge_GetDestNodeID(e);  // element column index

		GrB_Info info = Delta_Matrix_removeElement_UINT64(T, row, col);
		ASSERT(info == GrB_SUCCESS);
	}
}

// remove multiple entries
void Tensor_RemoveElements
(
	Tensor T,                   // tensor
	const Edge *elements,       // elements to remove
	uint64_t n,                 // number of elements
	uint64_t **cleared_entries  // [optional] cleared entries, referes elements
) {
	ASSERT(T        != NULL);
	ASSERT(elements != NULL);

	// removing multiple elements from a tensor is done in two steps
	// 1. for each element determine the modification type
	//    modification types:
	//    1.1. removing an entry from a vector while maintaining the vector
	//    1.2. removing an entry from a vector and transitioning into a scalar
	//    1.3. removing an entry from a vector and transitioning to empty entry
	//    1.4. removing a scalar and transitioning to empty entry
	//    
	// 2. act according to the computed changes
	// assuming edges are sorted by src ID

	GrB_Info info;

	// array of indexes
	// delayed[i] points to elements which need to be deleted
	// these elements are introducing pendding changes to the tensor
	uint64_t *delayed = array_new(uint64_t, 0); 

	// i's is advanced within the loop's body
	for(uint64_t i = 0; i < n;) {
		const Edge *e   = elements + i;           // tuple (row, col, x)
		uint64_t    x   = ENTITY_GET_ID(e);       // element value
		GrB_Index   row = Edge_GetSrcNodeID(e);   // element row index
		GrB_Index   col = Edge_GetDestNodeID(e);  // element column index

		// consecutive elements sharing the same row, col indexes
		uint64_t j = i;
		while(j < n) {
			const Edge *next     = elements + j;
			GrB_Index   next_row = Edge_GetSrcNodeID(next);   // next row index
			GrB_Index   next_col = Edge_GetDestNodeID(next);  // next column index

			if(row != next_row || col != next_col) {
				break;
			}

			// consecutive elements, advance
			j++;
		}

		// check tensor at T[row,col]
		uint64_t _x;
		info = Delta_Matrix_extractElement_UINT64(&_x, T, row, col);
		ASSERT(info != GrB_NO_VALUE);

		uint64_t d = j - i;  // number of consecutive elements

		// expecting entry to exists
		if(SCALAR_ENTRY(_x)) {
			// removing a single entry
			ASSERT(d == 1);
			// postpone clear entry
			array_append(delayed, i);
		} else {
			// entry is a vector
			// determine if vector needs to be removed
			GrB_Vector V = AS_VECTOR(_x);

			GrB_Index nvals;
			info = GrB_Vector_nvals(&nvals, V);
			ASSERT(info == GrB_SUCCESS);

			if(nvals == d) {
				// entire vector needs to be removed
				// postpone entry removal
				GrB_free(&V);
				array_append(delayed, i);
			} else if(d+1 == nvals) {
				// transition from vector to scalar
				// determine which vector element becomes a scalar

				struct GB_Iterator_opaque _it;
				GxB_Iterator it = &_it;

				info = GxB_Vector_Iterator_attach(it, V, NULL);
				ASSERT(info == GrB_SUCCESS);

				// seek to the first entry
				info = GxB_Vector_Iterator_seek(it, 0);

				uint64_t  idx = 0;    // vector element

				// Use bitwise xor to cancel all equal values, leaving only the 
				// odd one out.
				
				for(uint64_t  k = i; k < j; k++){
					e = elements + k;
					idx ^= ENTITY_GET_ID(e); 
				}

				while(info != GxB_EXHAUSTED) {
					// get element index within the vector
					idx ^= GxB_Vector_Iterator_getIndex(it);
					// move to the next entry in V
					info = GxB_Vector_Iterator_next(it);
				}

				ASSERT(GxB_Vector_isStoredElement(V, idx) == GrB_SUCCESS);

				// free vector and set scalar
				GrB_free(&V);
				info = Delta_Matrix_setElement_UINT64(T, idx, row, col);
				ASSERT(info == GrB_SUCCESS);
			} else {
				// remove entries from vector
				for(uint64_t k = i; k < j; k++) {
					e = elements + k;
					GrB_Index col = ENTITY_GET_ID(e);  // element value
					info = GrB_Vector_removeElement(V, col);
					ASSERT(info == GrB_SUCCESS);
				}

				// flush vector
				info = GrB_wait(V, GrB_MATERIALIZE);
				ASSERT(info == GrB_SUCCESS);
			}
		}

		i = j;

	}

	// handel delayed deletions
	n = array_len(delayed);
	for(uint64_t i = 0; i < n; i++) {
		const Edge *e   = elements + delayed[i];  // tuple (row, col, x)
		GrB_Index   row = Edge_GetSrcNodeID(e);   // element row index
		GrB_Index   col = Edge_GetDestNodeID(e);  // element column index

		info = Delta_Matrix_removeElement_UINT64(T, row, col);
		ASSERT(info == GrB_SUCCESS);
	}

	if(cleared_entries != NULL) {
		*cleared_entries = delayed;
	} else {
		array_free(delayed);
	}
}

// computes row degree of T[row:]
uint64_t Tensor_RowDegree
(
	const Tensor T,  // tensor
	GrB_Index row    // row
) {
	ASSERT(T != NULL);

	uint64_t degree = 0;

	GrB_Info              info;
	GrB_Index             nvals;
	Delta_MatrixTupleIter it;

	// iterate over T[row:]
	info = Delta_MatrixTupleIter_attach(&it, T);
	ASSERT(info == GrB_SUCCESS);

	uint64_t x;
	info = Delta_MatrixTupleIter_iterate_row(&it, row);
	ASSERT(info == GrB_SUCCESS);

	// scan T[row:]
	while(Delta_MatrixTupleIter_next_UINT64(&it, NULL, NULL, &x) == GrB_SUCCESS) {
		if(SCALAR_ENTRY(x)) {
			// scalar entry, increase degree by 1
			degree++;
		} else {
			// Vector entry, increase degree by number of entries in vector
			GrB_Vector V = AS_VECTOR(x);
			info = GrB_Vector_nvals(&nvals, V);
			ASSERT(info == GrB_SUCCESS);
			degree += nvals;
		}
	}

	return degree;
}

// computes col degree of T[:col]
uint64_t Tensor_ColDegree
(
	const Tensor T,  // tensor
	GrB_Index col    // col
) {
	ASSERT(T != NULL);

	uint64_t degree = 0;

	uint64_t              x;
	GrB_Info              info;
	GrB_Index             row;
	GrB_Index             nvals;
	Delta_MatrixTupleIter it;

	// scan transpose matrix
	Delta_Matrix TT = Delta_Matrix_getTranspose(T);

	// iterate over T[col:]
	info = Delta_MatrixTupleIter_attach(&it, TT);
	ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_iterate_row(&it, col);
	ASSERT(info == GrB_SUCCESS);

	// scan TT[col:]
	while(Delta_MatrixTupleIter_next_BOOL(&it, NULL, &row, NULL) == GrB_SUCCESS) {
		// inspect T[row, col]
		info = Delta_Matrix_extractElement_UINT64(&x, T, row, col);
		ASSERT(info == GrB_SUCCESS);

		if(SCALAR_ENTRY(x)) {
			// scalar entry, increase degree by 1
			degree++;
		} else {
			// vector entry, increase degree by number of entries in vector
			GrB_Vector V = AS_VECTOR(x);
			info = GrB_Vector_nvals(&nvals, V);
			ASSERT(info == GrB_SUCCESS);
			degree += nvals;
		}
	}

	return degree;
}

// free tensor
void Tensor_free
(
	Tensor *T  // tensor
) {
	ASSERT(T != NULL && *T != NULL);
	GrB_Info info;
	Tensor t = *T;

	// apply _free_vectors on every entry of the tensor
	GrB_OK (Delta_Matrix_apply(t, Global_GrB_Ops_Get()->free_tensors, t));

	// free tensor internals
	Delta_Matrix_free(T);
}

