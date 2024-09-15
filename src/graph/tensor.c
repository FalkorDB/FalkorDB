/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "tensor.h"
#include "util/arr.h"
#include "delta_matrix/delta_matrix.h"
#include "delta_matrix/delta_matrix_iter.h"

// get a new vector ID from tensor T
#define NEW_VECTOR_ID(T)         \
	array_len(T->freelist) > 0   \
	? array_pop(T->freelist) :   \ 
	T->row_id++;

// A tensor is a 3D matrix where A[i,j] is a vector
// we represent a tensor using two matrices
//   1. A a uint64 matrix, A[i,j] = Vector
//   2. B a boolean matrix B[l,k] = True if vector l[k] is active
struct Tensor {
	Delta_Matrix A;      // uint64 matrix, points to vectors
	Delta_Matrix V;      // boolean matrix, B[l:] = vector l 
	uint64_t row_id;     // number of active vectors
	uint64_t *freelist;  // deleted vectors IDs
};

// init new tensor
Tensor Tensor_new
(
	GrB_Index nrows,  // # rows
	GrB_Index ncols   // # columns
) {
	Tensor T = rm_malloc(sizeof(struct Tensor));

	Delta_Matrix_new(&T->A, GrB_UINT64, nrows, ncols, true);

	// E's dimentions will be set on first sync
	Delta_Matrix_new(&M->E, GrB_BOOL, 0, 0, false);

	// init T's free-list
	T->row_id = 0;
	T->freelist = array_new(uint64_t, 0);

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
	GrB_Index vec_id;
	GrB_Info info = Delta_Matrix_extractElement_UINT64(&_x, T->A, row, col);

	//--------------------------------------------------------------------------
	// new entry
	//--------------------------------------------------------------------------

	if(info == GrB_NO_VALUE) {
		info = Delta_Matrix_setElement_UINT64(T->A, x, row, col);
		ASSERT(info == GrB_SUCCESS);
		return;
	}

	//--------------------------------------------------------------------------
	// single entry -> vector
	//--------------------------------------------------------------------------

	if(SINGLE_ENTRY(_x)) {
		vec_id = NEW_VECTOR_ID(T);
		info = Delta_Matrix_setElement_UINT64(T->A, SET_MSB(vec_id), row, col);
		ASSERT(info == GrB_SUCCESS);

		// populate vector with both original entry and newly added value
		info = Delta_Matrix_setElement_BOOL(T->V, vec_id, _x);
		ASSERT(info == GrB_SUCCESS);

		info = Delta_Matrix_setElement_BOOL(T->V, vec_id, x);
		ASSERT(info == GrB_SUCCESS);
		return;
	}

	//--------------------------------------------------------------------------
	// add x to existing vector
	//--------------------------------------------------------------------------

	vec_id = CLEAR_MSB(_x);
	info = Delta_Matrix_setElement_BOOL(T->V, vec_id, x);
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
	Tensor T,              // tensor
	const Edge **elements  // assume edges are sorted by src and dest
) {
	ASSERT(T        != NULL);
	ASSERT(elements != NULL);

	// a new entry can cause one of the following transitions:
	// 1. the new entry creates a new scalar at T[i,j]
	// 2. the new entry convers an existing scalar at T[i,j] to a vector
	// 3. the new entry is added to an already existing vector

	// array of indexes pairs
	// delayed[i, i+1] points to a range of consecutive elements to be inserted
	// these elements are creating a new entries either scalar or vector
	uint64_t delayed = array_new(uint64_t, 0); 

	uint n = array_len(elements);

	// i's is advanced within the loop's body
	for(uint i = 0; i < n;) {
		enum SetMethod method;                  // insert method
		const edge *e = elements[i];               // tuple (row, col, x)
		uint64_t  x   = ENTITY_GET_ID(e);       // element value
		GrB_Index row = Edge_GetSrcNodeID(e);   // element row index
		GrB_Index col = Edge_GetDestNodeID(e);  // element column index

		//----------------------------------------------------------------------
		// determine insert method
		//----------------------------------------------------------------------

		// check tensor at T[row,col]
		uint64_t _x;
		info = Delta_Matrix_extractElement_UINT64(&_x, T->A, row, col);

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
			Edge *next = elements[j];
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
			case NEW_VECTOR:
				uint64_t vec_id = NEW_VECTOR_ID(T);
				// update scalar entry to a vector
				// it is OK to write to T->A as we're updating an existing entry
				info = Delta_Matrix_setElement_UINT64(T->A, SET_MSB(vec_id),
						row, col);
				ASSERT(info == GrB_SUCCESS);

				// add existing entry to vector
				info = Delta_Matrix_setElement_BOOL(T->V, vec_id, _x);
				ASSERT(info == GrB_SUCCESS);

				// add new entries
				for(; i < j; i++) { 
					e = elements[i];          // tuple (row, col, x)
					x = ENTITY_GET_ID(e);  // element value
					// add entry to vector
					info = Delta_Matrix_setElement_BOOL(T->V, vec_id, x);
					ASSERT(info == GrB_SUCCESS);
				}

				break;
			case EXISTING_VECTOR:
				_x = CLEAR_MSB(_x);

				// add new entries
				for(; i < j; i++) { 
					e = elements[i];          // tuple (row, col, x)
					x = ENTITY_GET_ID(e);  // element value
					// add entry to vector
					info = Delta_Matrix_setElement_BOOL(T->V, _x, x);
					ASSERT(info == GrB_SUCCESS);
				}

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

		Edge     *e   = elements[a];
		uint64_t  x   = ENTITY_GET_ID(e);       // element value
		GrB_Index row = Edge_GetSrcNodeID(e);   // element row index
		GrB_Index col = Edge_GetDestNodeID(e);  // element column index

		// determine insert method:
		// single element -> scalar
		// multiple entries -> vector
		enum SetMethod method = (a-z == 0) ? NEW_SCALAR : NEW_VECTOR;

		switch(method) {
			case NEW_SCALAR:
				info = Delta_Matrix_setElement_UINT64(T->A, x, row, col);
				ASSERT(info == GrB_SUCCESS);
				break;
			case NEW_VECTOR:
				// set vector entry
				uint64_t vec_id = NEW_VECTOR_ID(T);
				info = Delta_Matrix_setElement_UINT64(T->A, SET_MSB(vec_id),
						row, col);
				ASSERT(info == GrB_SUCCESS);

				// add elements to vector
				for(uint j = a; j < z; j++) {
					e = elements[j];
					x = ENTITY_GET_ID(e);
					info = Delta_Matrix_setElement_BOOL(T->V, vec_id, x);
					ASSERT(info == GrB_SUCCESS);
				}
				break;
			default:
				break;
		}
	}

	array_free(delayed);
}

// checks to see if tensor has pending changes
bool Tensor_pending
(
	const Tensor T   // tensor
) {
	ASSERT(T != NULL);
	
	bool pending;

	Delta_Matrix_pending(T->A, &pending);
	if(pending) return true;

	Delta_Matrix_pending(T->V, &pending);
	return pending;
}

// free tensor
void Tensor_free
(
	Tensor *T  // tensor
) {
	ASSERT(T != NULL && *T != NULL);

	Tensor t = *T;

	// free tensor internals
	Delta_Matrix_free(&t->A);
	Delta_Matrix_free(&t->V);
	array_free(t->freelist);

	// free tensor and nullify
	rm_free(t);
	*T = NULL;
}

