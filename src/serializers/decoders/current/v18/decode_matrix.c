/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "decode_v18.h"
#include "../../../serializer_io.h"
#include "../../../../graph/graphcontext.h"
#include "../../../../graph/tensor/tensor.h"
#include "../../../../graph/graph_statistics.h"
#include "../../../../graph/delta_matrix/delta_matrix.h"

// decode tensors
static bool _DecodeTensors
(
	SerializerIO io,      // stream
	Delta_Matrix D,       // matrix to populate with tensors
	uint64_t *n_tensors,  // [output] number of tensors loaded
	uint64_t *n_elem      // [output] number of edges loaded
) {
	// format:
	//  total number of tensors
	//
	//  M - number of tensors
	//  tensors:
	//   tensor i index
	//   tensor j index
	//   tensor
	//
	//  DP - number of tensors
	//  tensors:
	//   tensor i index
	//   tensor j index
	//   tensor

	ASSERT (D         != NULL) ;
	ASSERT (io        != NULL) ;
	ASSERT (n_elem    != NULL) ;
	ASSERT (n_tensors != NULL) ;

	*n_elem    = 0 ;
	*n_tensors = 0 ;

	// read number of tensors
	uint64_t n ;
	TRY_READ (io, n) ;

	// no tensors, simply return
	if (n == 0) {
		return true ;
	}

	GrB_Matrix M  = Delta_Matrix_M  (D) ;
	GrB_Matrix DP = Delta_Matrix_DP (D) ;

	GrB_Matrix matrices[2] = {M, DP} ;

	for (int l = 0; l < 2; l++) {
		GrB_Matrix A = matrices[l] ;

		uint64_t _n_tensors ;
		TRY_READ (io, _n_tensors) ;

		// decode and set tensors
		for (uint64_t k = 0; k < _n_tensors; k++) {
			// read tensor i,j indicies
			GrB_Index i ;
			GrB_Index j ;

			TRY_READ (io, i) ;
			TRY_READ (io, j) ;

			// read tensor blob
			void *blob ;
			GrB_Index blob_size ;
			if (!SerializerIO_TryReadBuffer (io, &blob, (size_t*)&blob_size)) {
				return false ;
			}

			GrB_Vector u ;
			GrB_Info info =
				GxB_Vector_deserialize (&u, NULL, blob, blob_size, NULL) ;
			ASSERT (info == GrB_SUCCESS) ;

			// update number of elements loaded
			GrB_Index nvals ;
			info = GrB_Vector_nvals (&nvals, u) ;
			ASSERT (info == GrB_SUCCESS) ;
			ASSERT (nvals > 0) ;
			*n_elem += nvals ;

			// set tensor
			uint64_t v = (uint64_t)(uintptr_t) SET_MSB (u) ;
			info = GrB_Matrix_setElement_UINT64 (A, v, i, j) ;
			ASSERT (info == GrB_SUCCESS) ;

			rm_free (blob) ;
		}

		// set number of loaded tensors
		*n_tensors += _n_tensors ;
	}

	return true ;
}

static bool _decode_and_load_vector
(
	SerializerIO io,  // stream
	GrB_Vector *v     // vector
) {
	// format:
	//   array
	//   type name
	//   number of entries
	//   number of bytes
	//   handeling

	ASSERT (v  != NULL) ;
	ASSERT (io != NULL) ;

	void *arr ;              // vector's data
	size_t n ;               // number of bytes read
	uint64_t n_entries ;     // number of entries
	uint64_t n_bytes ;       // data size in bytes
	int64_t handling ;       // memory owner GraphBLAS / App
	char   *t_name ;         // type name
	size_t t_name_len = 0 ;  // type name length
	GrB_Info info ;

	// load vector from stream
	if (!SerializerIO_TryReadBuffer (io, &arr, &n)) {
		return false ;
	}

	// type name
	if (!SerializerIO_TryReadBuffer (io, (void**)&t_name, &t_name_len)) {
		return false ;
	}

	TRY_READ (io, n_entries) ;
	TRY_READ (io, n_bytes)   ;
	TRY_READ (io, handling)  ;

	// get GrB_Type
	GrB_Type t ;  // data type
	info = GxB_Type_from_name (&t, t_name) ;
	ASSERT (info == GrB_SUCCESS) ;
	rm_free (t_name) ;

	// load vector
	info = GrB_Vector_new (v, t, 0) ;
	ASSERT (info == GrB_SUCCESS) ;

	info = GxB_Vector_load (*v, &arr, t, n_entries, n_bytes, handling, NULL) ;
	ASSERT (info == GrB_SUCCESS) ;

	return true ;
}

// decode a GraphBLAS matrix
static bool _Decode_GrB_Matrix
(
	SerializerIO io,  // stream
	GrB_Matrix *A     // matrix
) {
	// format:
	//  GraphBLAS container
	//  unloaded matrix components

	// decode container
	size_t n ;
	GxB_Container container ;

	if (!SerializerIO_TryReadBuffer (io, (void**)&container,  &n)) {
		return false ;
	}

	ASSERT (n == sizeof (struct GxB_Container_struct)) ;

	// nullify container's vectors
    container->p = NULL ;
    container->h = NULL ;
    container->b = NULL ;
    container->i = NULL ;
    container->x = NULL ;
    container->Y = NULL ;

	//--------------------------------------------------------------------------
	// load individual vectors
	//--------------------------------------------------------------------------

	if (!_decode_and_load_vector (io, &container->x)) {
		return false ;
	}

	if (!_decode_and_load_vector (io, &container->h)) {
		return false ;
	}

	if (!_decode_and_load_vector (io, &container->p)) {
		return false ;
	}

	if (!_decode_and_load_vector (io, &container->i)) {
		return false ;
	}

	if (!_decode_and_load_vector (io, &container->b)) {
		return false ;
	}

	// load A from the container
	GrB_Info info ;

	info = GrB_Matrix_new (A, GrB_BOOL, 0, 0) ;  // matrix type doesn't matter
	ASSERT (info == GrB_SUCCESS) ;

	info = GxB_load_Matrix_from_Container (*A, container, NULL) ;
	ASSERT (info == GrB_SUCCESS) ;

	// A is now back to its original state. The container and its p,h,b,i,x
	// GrB_Vectors exist but its vectors all have length 0.

	info = GxB_Container_free (&container) ; // does several O(1)-sized free’s
	ASSERT (info == GrB_SUCCESS) ;

	return true ;
}

// decode matrix
static bool _Decode_Delta_Matrix
(
	SerializerIO io,  // stream
	Delta_Matrix D    // delta matrix to populate
) {
	// format:
	//  M
	//  DP
	//  DM

	ASSERT (D  != NULL) ;
	ASSERT (io != NULL) ;

	GrB_Matrix M  ;
	GrB_Matrix DP ;
	GrB_Matrix DM ;

	if (!_Decode_GrB_Matrix (io, &M)) {
		return false ;
	}

	if (!_Decode_GrB_Matrix (io, &DP)) {
		return false ;
	}

	if (!_Decode_GrB_Matrix (io, &DM)) {
		return false ;
	}

	ASSERT (M  != NULL) ;
	ASSERT (DP != NULL) ;
	ASSERT (DM != NULL) ;

	GrB_Info info = Delta_Matrix_setMatrices (D, M, DP, DM) ;
	ASSERT (info == GrB_SUCCESS) ;

	return true ;
}

// decode label matrices from stream
bool LoadLabelMatrices_v18
(
	SerializerIO io,  // stream
	GraphContext *gc  // graph context
) {
	// format:
	//  number of label matrices
	//   label id
	//   matrix

	ASSERT (gc != NULL) ;
	ASSERT (io != NULL) ;

	GrB_Info info ;
	Graph *g = gc->g ;

	// read number of label matricies
	uint64_t n ;
	TRY_READ (io, n) ;
	
	// decode each label matrix
	for (int i = 0; i < n; i++) {
		// read label ID
		LabelID l ;
		uint64_t v ;

		TRY_READ(io, v);
		l = v;

		Delta_Matrix lbl = Graph_GetLabelMatrix (g, l) ;
		if (!_Decode_Delta_Matrix(io, lbl)) {
			return false ;
		}
	}

	return true ;
}

// decode relationship matrices from stream
bool LoadRelationMatrices_v18
(
	SerializerIO io,  // stream
	GraphContext *gc  // graph context
) {
	// format:
	//   relation id   X N
	//   matrix        X N
	//   tensors count X N
	//   tensors

	ASSERT (gc != NULL) ;
	ASSERT (io != NULL) ;

	GrB_Info info ;
	Graph *g = gc->g ;

	// number of relation matricies
	int n = Graph_RelationTypeCount (g) ;

	// decode relationship matrices
	for (int i = 0; i < n; i++) {
		// read relation ID
		uint64_t v ;
		RelationID r ;

		TRY_READ (io, v) ;
		r = v ;
		ASSERT (r == i) ;

		// plant M matrix
		Delta_Matrix DR = Graph_GetRelationMatrix (g, r, false) ;

		GrB_Index nvals ;
		Delta_Matrix_nvals (&nvals, DR) ;
		ASSERT (nvals == 0) ;

		if (!_Decode_Delta_Matrix(io, DR)) {
			return false ;
		}

		// decode tensors
		uint64_t n_elem    = 0 ;  // number of tensor edges
		uint64_t n_tensors = 0 ;  // number of tensors in matrix
		if (!_DecodeTensors (io, DR, &n_tensors, &n_elem)) {
			return false ;
		}

		// update graph edge statistics
		// number of edges of type 'r' equals to:
		// |R| - n_tensors + n_elem
		info = Delta_Matrix_nvals (&nvals, DR) ;
		ASSERT (info == GrB_SUCCESS) ;

		GraphStatistics_IncEdgeCount (&g->stats, r, nvals - n_tensors + n_elem) ;
	}

	return true ;
}

// decode adjacency matrix
bool LoadAdjMatrix_v18
(
	SerializerIO io,  // stream
	GraphContext *gc  // graph context
) {
	// format:
	//   adjacency matrix

	ASSERT (gc != NULL) ;
	ASSERT (io != NULL) ;

	Delta_Matrix adj = Graph_GetAdjacencyMatrix (gc->g, false) ;
	return _Decode_Delta_Matrix (io, adj) ;
}

// decode labels matrix
bool LoadLblsMatrix_v18
(
	SerializerIO io,  // stream
	GraphContext *gc  // graph context
) {
	// format:
	//   lbls matrix

	ASSERT (gc != NULL) ;
	ASSERT (io != NULL) ;

	Delta_Matrix lbl = Graph_GetNodeLabelMatrix (gc->g) ;
	return _Decode_Delta_Matrix (io, lbl) ;
}

