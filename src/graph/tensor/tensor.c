/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "tensor.h"
#include "util/arr.h"
#include "../delta_matrix/delta_matrix.h"
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

	// _x = A[row, col]
	GrB_Index _x;
	GrB_Vector V;
	GrB_Info info = Delta_Matrix_extractElement_UINT64(&_x, T, row, col);

	//--------------------------------------------------------------------------
	// new entry
	//--------------------------------------------------------------------------

	if(info == GrB_NO_VALUE) {
		info = Delta_Matrix_setElement_UINT64(T, x, row, col);
		ASSERT(info == GrB_SUCCESS);
		return;
	}

	//--------------------------------------------------------------------------
	// single entry -> vector
	//--------------------------------------------------------------------------

	if(SCALAR_ENTRY(_x)) {
		info = GrB_Vector_new(&V, GrB_BOOL, GrB_INDEX_MAX);
		ASSERT(info == GrB_SUCCESS);

		// T[row, col] = V
		uint64_t vec_entry = SET_MSB((uint64_t)(uintptr_t)V);
		info = Delta_Matrix_setElement_UINT64(T, vec_entry, row, col);
		ASSERT(info == GrB_SUCCESS);

		// populate vector with both original entry and newly added value
		info = GrB_Vector_setElement_BOOL(V, true, _x);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Vector_setElement_BOOL(V, true, x);
		ASSERT(info == GrB_SUCCESS);

		// flush vector
		info = GrB_wait(V, GrB_MATERIALIZE);
		ASSERT(info == GrB_SUCCESS);

		return;
	}

	//--------------------------------------------------------------------------
	// add x to existing vector
	//--------------------------------------------------------------------------

	V = AS_VECTOR(_x);
	info = GrB_Vector_setElement_BOOL(V, true, x);
	ASSERT(info == GrB_SUCCESS);

	// flush vector
	info = GrB_wait(V, GrB_MATERIALIZE);
	ASSERT(info == GrB_SUCCESS);
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

	// a new entry can cause one of the following transitions:
	// 1. the new entry creates a new scalar at T[i,j]
	// 2. the new entry convers an existing scalar at T[i,j] to a vector
	// 3. the new entry is added to an already existing vector

	// array of indexes pairs
	// delayed[i, i+1] points to a range of consecutive elements to be inserted
	// these elements are creating a new entries either scalar or vector
	uint64_t *delayed = array_new(uint64_t, 0);

	GrB_Vector V;
	GrB_Info info;

	// i's is advanced within the loop's body
	for(uint64_t i = 0; i < n;) {
		enum SetMethod method;  // insert method

		uint64_t  x   = vals[i];  // element value
		GrB_Index row = rows[i];  // element row index
		GrB_Index col = cols[i];  // element column index

		//----------------------------------------------------------------------
		// determine insert method
		//----------------------------------------------------------------------

		// check tensor at T[row,col]
		uint64_t _x;
		info = Delta_Matrix_extractElement_UINT64(&_x, T, row, col);

		if(info == GrB_NO_VALUE) {
			// new entry
			method = NEW_SCALAR;
		} else if(SCALAR_ENTRY(_x)) {
			// switch from scalar entry to vector
			method = NEW_VECTOR;
		} else {
			// add to existing vector
			method = EXISTING_VECTOR;
		}

		// consecutive elements sharing the same row, col indexes
		uint j = i;
		while(j < n) {
			GrB_Index next_row = rows[j];  // next row index
			GrB_Index next_col = cols[j];  // next column index

			if(row != next_row || col != next_col) {
				break;
			}

			// consecutive elements, advance
			j++;
		}

		// act according to insert method
		switch(method) {
			case NEW_SCALAR:
				// save elements IDs
				// we can't insert at this time as we'll be introducing pendding
				// changes in a READ / WRITE scenario
				array_append(delayed, i);
				array_append(delayed, j);
				break;
			case NEW_VECTOR:
				info = GrB_Vector_new(&V, GrB_BOOL, GrB_INDEX_MAX);
				ASSERT(info == GrB_SUCCESS);

				// update scalar entry to a vector
				// it is OK to write to T->A as we're updating an existing entry
				uint64_t vec_entry = SET_MSB((uint64_t)(uintptr_t) V);
				info = Delta_Matrix_setElement_UINT64(T, vec_entry, row, col);
				ASSERT(info == GrB_SUCCESS);

				// add existing entry to vector
				info = GrB_Vector_setElement_BOOL(V, true, _x);
				ASSERT(info == GrB_SUCCESS);

				// add new entries
				for(; i < j; i++) {
					x = vals[i];  // element value
					// add entry to vector
					info = GrB_Vector_setElement_BOOL(V, true, x);
					ASSERT(info == GrB_SUCCESS);
				}

				// flush vector
				info = GrB_wait(V, GrB_MATERIALIZE);
				ASSERT(info == GrB_SUCCESS);

				break;
			case EXISTING_VECTOR:
				V = AS_VECTOR(_x);

				// add new entries
				for(; i < j; i++) {
					x = vals[i];  // element value
					// add entry to vector
					info = GrB_Vector_setElement_BOOL(V, true, x);
					ASSERT(info == GrB_SUCCESS);
				}

				// flush vector
				info = GrB_wait(V, GrB_MATERIALIZE);
				ASSERT(info == GrB_SUCCESS);

				break;
			default:
				ASSERT(false);
		}

		// advance i, skipping all consecutive elements
		i = j;
	}

	// process delayed inserts
	n = array_len(delayed);
	for(uint64_t i = 0; i < n; i+=2) {
		uint64_t a = delayed[i];
		uint64_t z = delayed[i+1];

		uint64_t  x   = vals[a];  // element value
		GrB_Index row = rows[a];  // element row index
		GrB_Index col = cols[a];  // element column index

		// determine insert method:
		// single element -> scalar
		// multiple entries -> vector
		enum SetMethod method = (z - a == 1) ? NEW_SCALAR : NEW_VECTOR;

		switch(method) {
			case NEW_SCALAR:
				info = Delta_Matrix_setElement_UINT64(T, x, row, col);
				ASSERT(info == GrB_SUCCESS);
				break;
			case NEW_VECTOR:
				// set vector entry
				info = GrB_Vector_new(&V, GrB_BOOL, GrB_INDEX_MAX);
				ASSERT(info == GrB_SUCCESS);

				uint64_t vec_entry = SET_MSB((uint64_t)(uintptr_t) V);
				info = Delta_Matrix_setElement_UINT64(T, vec_entry, row, col);
				ASSERT(info == GrB_SUCCESS);

				// add elements to vector
				for(uint j = a; j < z; j++) {
					x = vals[j];
					info = GrB_Vector_setElement_BOOL(V, true, x);
					ASSERT(info == GrB_SUCCESS);
				}

				// flush vector
				info = GrB_wait(V, GrB_MATERIALIZE);
				ASSERT(info == GrB_SUCCESS);

				break;
			default:
				break;
		}
	}

	array_free(delayed);
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

	// a new entry can cause one of the following transitions:
	// 1. the new entry creates a new scalar at T[i,j]
	// 2. the new entry convers an existing scalar at T[i,j] to a vector
	// 3. the new entry is added to an already existing vector

	// array of indexes pairs
	// delayed[i, i+1] points to a range of consecutive elements to be inserted
	// these elements are creating a new entries either scalar or vector
	uint64_t *delayed = array_new(uint64_t, 0); 

	GrB_Vector V;
	GrB_Info info;

	// i's is advanced within the loop's body
	for(uint i = 0; i < n;) {
		enum SetMethod method;                    // insert method
		const Edge *e   = elements[i];            // tuple (row, col, x)
		uint64_t    x   = ENTITY_GET_ID(e);       // element value
		GrB_Index   row = Edge_GetSrcNodeID(e);   // element row index
		GrB_Index   col = Edge_GetDestNodeID(e);  // element column index

		//----------------------------------------------------------------------
		// determine insert method
		//----------------------------------------------------------------------

		// check tensor at T[row,col]
		uint64_t _x;
		info = Delta_Matrix_extractElement_UINT64(&_x, T, row, col);

		if(info == GrB_NO_VALUE) {
			// new entry
			method = NEW_SCALAR;
		} else if(SCALAR_ENTRY(_x)) {
			// switch from scalar entry to vector
			method = NEW_VECTOR;
		} else {
			// add to existing vector
			method = EXISTING_VECTOR;
		}

		// consecutive elements sharing the same row, col indexes
		uint j = i;
		while(j < n) {
			const Edge *next = elements[j];
			GrB_Index next_row = Edge_GetSrcNodeID(next);   // next row index
			GrB_Index next_col = Edge_GetDestNodeID(next);  // next column index

			if(row != next_row || col != next_col) {
				break;
			}

			// consecutive elements, advance
			j++;
		}

		// act according to insert method
		switch(method) {
			case NEW_SCALAR:
				// save elements IDs
				// we can't insert at this time as we'll be introducing pendding
				// changes in a READ / WRITE scenario
				array_append(delayed, i);
				array_append(delayed, j);
				break;
			case NEW_VECTOR:
				info = GrB_Vector_new(&V, GrB_BOOL, GrB_INDEX_MAX);
				ASSERT(info == GrB_SUCCESS);

				// update scalar entry to a vector
				// it is OK to write to T->A as we're updating an existing entry
				uint64_t vec_entry = SET_MSB((uint64_t)(uintptr_t) V);
				info = Delta_Matrix_setElement_UINT64(T, vec_entry, row, col);
				ASSERT(info == GrB_SUCCESS);

				// add existing entry to vector
				info = GrB_Vector_setElement_BOOL(V, true, _x);
				ASSERT(info == GrB_SUCCESS);

				// add new entries
				for(; i < j; i++) { 
					e = elements[i];          // tuple (row, col, x)
					x = ENTITY_GET_ID(e);  // element value
					// add entry to vector
					info = GrB_Vector_setElement_BOOL(V, true, x);
					ASSERT(info == GrB_SUCCESS);
				}

				// flush vector
				info = GrB_wait(V, GrB_MATERIALIZE);
				ASSERT(info == GrB_SUCCESS);

				break;
			case EXISTING_VECTOR:
				V = AS_VECTOR(_x);

				// add new entries
				for(; i < j; i++) { 
					e = elements[i];          // tuple (row, col, x)
					x = ENTITY_GET_ID(e);  // element value
					// add entry to vector
					info = GrB_Vector_setElement_BOOL(V, true, x);
					ASSERT(info == GrB_SUCCESS);
				}

				// flush vector
				info = GrB_wait(V, GrB_MATERIALIZE);
				ASSERT(info == GrB_SUCCESS);

				break;
			default:
				ASSERT(false);
		}

		// advance i, skipping all consecutive elements
		i = j;
	}

	// process delayed inserts
	n = array_len(delayed);
	for(uint64_t i = 0; i < n; i+=2) {
		uint64_t a = delayed[i];
		uint64_t z = delayed[i+1];

		const Edge *e   = elements[a];
		uint64_t    x   = ENTITY_GET_ID(e);       // element value
		GrB_Index   row = Edge_GetSrcNodeID(e);   // element row index
		GrB_Index   col = Edge_GetDestNodeID(e);  // element column index

		// determine insert method:
		// single element -> scalar
		// multiple entries -> vector
		enum SetMethod method = (z - a == 1) ? NEW_SCALAR : NEW_VECTOR;

		switch(method) {
			case NEW_SCALAR:
				info = Delta_Matrix_setElement_UINT64(T, x, row, col);
				ASSERT(info == GrB_SUCCESS);
				break;
			case NEW_VECTOR:
				// set vector entry
				info = GrB_Vector_new(&V, GrB_BOOL, GrB_INDEX_MAX);
				ASSERT(info == GrB_SUCCESS);

				uint64_t vec_entry = SET_MSB((uint64_t)(uintptr_t) V);
				info = Delta_Matrix_setElement_UINT64(T, vec_entry, row, col);
				ASSERT(info == GrB_SUCCESS);

				// add elements to vector
				for(uint j = a; j < z; j++) {
					e = elements[j];
					x = ENTITY_GET_ID(e);
					info = GrB_Vector_setElement_BOOL(V, true, x);
					ASSERT(info == GrB_SUCCESS);
				}

				// flush vector
				info = GrB_wait(V, GrB_MATERIALIZE);
				ASSERT(info == GrB_SUCCESS);

				break;
			default:
				break;
		}
	}

	array_free(delayed);
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

		GrB_Info info = Delta_Matrix_removeElement(T, row, col);
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
		uint j = i;
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

				// sort consecutive elements i..j by element value
				qsort((void*)(elements + i), d, sizeof(Edge), _value_cmp);

				// iterate over vector entries
				// determine which of the vector's entries transitions into
				// a scalar

				struct GB_Iterator_opaque _it;
				GxB_Iterator it = &_it;

				info = GxB_Vector_Iterator_attach(it, V, NULL);
				ASSERT(info == GrB_SUCCESS);

				// seek to the first entry
				info = GxB_Vector_Iterator_seek(it, 0);

				GrB_Index idx;    // vector element
				uint64_t  v;      // element's value
				uint64_t  k = i;  // first consecutive element
				while(info != GxB_EXHAUSTED) {
					// get current element
					e = elements + k;
					v = ENTITY_GET_ID(e);

					// get element index within the vector
					idx = GxB_Vector_Iterator_getIndex(it);

					// break if vector's entry isn't in the list of
					// deleted elements
					if(idx != v) break;

					// move to the next entry in V
					info = GxB_Vector_Iterator_next(it);

					// advance to next element
					k = MIN(k + 1, i + d - 1);
				}

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

		info = Delta_Matrix_removeElement(T, row, col);
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

// free vector entries of a tensor
static void _free_vectors
(
	void *z,       // [ignored] new value
	const void *x  // current entry
) {
	// see if entry is a vector
	uint64_t _x = *(uint64_t*)(x);
	if(!SCALAR_ENTRY(_x)) {
		// free vector
		GrB_Vector V = AS_VECTOR(_x);
		GrB_free(&V);
	}
}

// free tensor
void Tensor_free
(
	Tensor *T  // tensor
) {
	ASSERT(T != NULL && *T != NULL);

	Tensor t = *T;

	// flush all pendding changes in T
	// TODO: we might be able to avoid this if we had access to
	// the tensor underline matrices: DP, DM & M
	GrB_Info info = Delta_Matrix_wait(t, true);
	ASSERT(info == GrB_SUCCESS);

	// get delta matrix M matrix
	GrB_Matrix M = Delta_Matrix_M(t);

	// initialize unaryop only once
	static GrB_UnaryOp unaryop = NULL;
	if(unaryop == NULL) {
		info = GrB_UnaryOp_new(&unaryop, _free_vectors, GrB_UINT64, GrB_UINT64);
		ASSERT(info == GrB_SUCCESS);
	}

	// apply _free_vectors on every entry of the tensor
	info = GrB_Matrix_apply(M, NULL, NULL, unaryop, M, NULL);
	ASSERT(info == GrB_SUCCESS);

	// free tensor internals
	Delta_Matrix_free(T);
}

