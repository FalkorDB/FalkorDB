/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "../../../serializer_io.h"
#include "../../../../graph/graphcontext.h"
#include "../../../../graph/tensor/tensor.h"
#include "../../../../graph//graph_statistics.h"
#include "../../../../graph/delta_matrix/delta_matrix.h"

// decode tensors
static void _DecodeTensors
(
	SerializerIO rdb,     // RDB
	Delta_Matrix D,       // matrix to populate with tensors
	uint64_t *n_tensors,  // [output] number of tensors loaded
	uint64_t *n_elem      // [output] number of edges loaded
) {
	// format:
	//  number of tensors
	//  tensors:
	//   tensor i index
	//   tensor j index
	//   tensor

	ASSERT(D         != NULL);
	ASSERT(n_elem    != NULL);
	ASSERT(n_tensors != NULL);

	*n_elem    = 0;
	*n_tensors = 0;

	// read number of tensors
	uint64_t n = SerializerIO_ReadUnsigned(rdb);

	// no tensors, simply return
	if(n == 0) {
		return;
	}

	GrB_Matrix M  = Delta_Matrix_M  (D) ;
	GrB_Matrix DP = Delta_Matrix_DP (D) ;

	GrB_Matrix matrices[2] = {M, DP} ;

	for (int l = 0; l < 2; l++) {
		GrB_Matrix A = matrices[l] ;

		uint64_t _n_tensors = SerializerIO_ReadUnsigned (rdb) ;

		// decode and set tensors
		for (uint64_t k = 0; k < _n_tensors; k++) {
			// read tensor i,j indicies
			GrB_Index i = SerializerIO_ReadUnsigned (rdb) ;
			GrB_Index j = SerializerIO_ReadUnsigned (rdb) ;

			// read tensor blob
			GrB_Index blob_size;
			void *blob = SerializerIO_ReadBuffer (rdb, (size_t*)&blob_size) ;
			ASSERT (blob != NULL) ;

			GrB_Vector u;
			GrB_Info info =
				GxB_Vector_deserialize (&u, NULL, blob, blob_size, NULL) ;
			ASSERT (info == GrB_SUCCESS) ;

			// update number of elements loaded
			GrB_Index nvals;
			info = GrB_Vector_nvals (&nvals, u) ;
			ASSERT (info == GrB_SUCCESS) ;
			*n_elem += nvals;

			// set tensor
			uint64_t v = (uint64_t)(uintptr_t) SET_MSB (u) ;
			info = GrB_Matrix_setElement_UINT64 (A, v, i, j) ;
			ASSERT (info == GrB_SUCCESS) ;

			rm_free(blob);
		}

		// set number of loaded tensors
		*n_tensors += _n_tensors;
	}
}

static void _decode_and_load_vector
(
	SerializerIO rdb,
	GrB_Vector *v
) {
	// format:
	//   array
	//   type name
	//   number of entries
	//   number of bytes
	//   handeling

	ASSERT (v   != NULL) ;
	ASSERT (rdb != NULL) ;

	void *arr;              // vector's data
	size_t n;               // number of bytes read
	uint64_t n_entries;     // number of entries
	uint64_t n_bytes;       // data size in bytes
	int handling;           // memory owner GraphBLAS / App
	char   *t_name;         // type name
	size_t t_name_len = 0;  // type name length
	GrB_Info info;

	// load vector from stream
	arr       = SerializerIO_ReadBuffer   (rdb, &n) ;
	t_name    = SerializerIO_ReadBuffer   (rdb, &t_name_len) ;
	n_entries = SerializerIO_ReadUnsigned (rdb) ;
	n_bytes   = SerializerIO_ReadUnsigned (rdb) ;
	handling  = SerializerIO_ReadSigned   (rdb) ;

	// get GrB_Type
	GrB_Type t;  // data type
	info = GxB_Type_from_name (&t, t_name) ;
	ASSERT (info == GrB_SUCCESS) ;

	// load vector
	info = GrB_Vector_new (v, t, 0) ;
	ASSERT (info == GrB_SUCCESS) ;

	info = GxB_Vector_load (*v, &arr, t, n_entries, n_bytes, handling, NULL) ;
	ASSERT (info == GrB_SUCCESS) ;
}

// decode a GraphBLAS matrix
static GrB_Matrix _Decode_GrB_Matrix
(
	SerializerIO rdb  // stream
) {
	// format:
	//  GraphBLAS container
	//  unloaded matrix components

	// decode container
	size_t n;
	GxB_Container container;

	container = SerializerIO_ReadBuffer (rdb, &n) ;
	ASSERT (n == sizeof(struct GxB_Container_struct)) ;

	// nullify container's vectors
    container->p = NULL ;
    container->h = NULL ;
    container->b = NULL ;
    container->i = NULL ;
    container->x = NULL ;
    container->Y = NULL ;

	_decode_and_load_vector (rdb, &container->x) ;

	switch (container->format) {
		case GxB_HYPERSPARSE :
			_decode_and_load_vector (rdb, &container->h) ;

		case GxB_SPARSE :
			_decode_and_load_vector (rdb, &container->p) ;
			_decode_and_load_vector (rdb, &container->i) ;

			break ;

		case GxB_BITMAP :
			_decode_and_load_vector (rdb, &container->b) ;

			break ;
	}

	// load A from the container
	GrB_Matrix A;
	GrB_Info info;

	info = GrB_Matrix_new (&A, GrB_BOOL, 0, 0) ;  // matrix type doesn't matter
	ASSERT (info == GrB_SUCCESS) ;

	info = GxB_load_Matrix_from_Container (A, container, NULL) ;
	ASSERT (info == GrB_SUCCESS) ;

	// A is now back to its original state. The container and its p,h,b,i,x
	// GrB_Vectors exist but its vectors all have length 0.

	info = GxB_Container_free (&container) ; // does several O(1)-sized free’s
	ASSERT (info == GrB_SUCCESS) ;

	return A;
}

// decode matrix
static void _Decode_Delta_Matrix
(
	SerializerIO rdb,  // RDB
	Delta_Matrix D     // delta matrix to populate
) {
	ASSERT (D   != NULL) ;
	ASSERT (rdb != NULL) ;

	GrB_Matrix M  = _Decode_GrB_Matrix (rdb) ;
	GrB_Matrix DP = _Decode_GrB_Matrix (rdb) ;
	GrB_Matrix DM = _Decode_GrB_Matrix (rdb) ;

	ASSERT (M  != NULL) ;
	ASSERT (DP != NULL) ;
	ASSERT (DM != NULL) ;

	GrB_Info info = Delta_Matrix_setMatrices (D, M, DP, DM) ;
	ASSERT (info == GrB_SUCCESS) ;
}

// decode label matrices from rdb
void RdbLoadLabelMatrices_v18
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
) {
	// format:
	//  number of label matrices
	//   label id
	//   matrix

	ASSERT(gc  != NULL);
	ASSERT(rdb != NULL);

	GrB_Info info;
	Graph *g = gc->g;

	// read number of label matricies
	int n = SerializerIO_ReadUnsigned(rdb);
	
	// decode each label matrix
	for(int i = 0; i < n; i++) {
		// read label ID
		LabelID l = SerializerIO_ReadUnsigned(rdb);
		Delta_Matrix lbl = Graph_GetLabelMatrix(g, l);
		_Decode_Delta_Matrix(rdb, lbl);
	}
}

// decode relationship matrices from rdb
void RdbLoadRelationMatrices_v18
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
) {
	// format:
	//   relation id X N
	//   matrix      X N

	ASSERT(gc  != NULL);
	ASSERT(rdb != NULL);

	GrB_Info info;
	Graph *g = gc->g;

	// number of relation matricies
	int n = Graph_RelationTypeCount (g) ;

	// decode relationship matrices
	for (int i = 0; i < n; i++) {
		// read relation ID
		RelationID r = SerializerIO_ReadUnsigned (rdb) ;

		// plant M matrix
		Delta_Matrix DR = Graph_GetRelationMatrix (g, r, false) ;
		_Decode_Delta_Matrix(rdb, DR);

		// decode tensors
		uint64_t n_elem    = 0;  // number of tensor edges
		uint64_t n_tensors = 0;  // number of tensors in matrix
		_DecodeTensors (rdb, DR, &n_tensors, &n_elem) ;

		// update graph edge statistics
		// number of edges of type 'r' equals to:
		// |R| - n_tensors + n_elem
		GrB_Index nvals;
		info = Delta_Matrix_nvals (&nvals, DR) ;
		ASSERT (info == GrB_SUCCESS) ;

		GraphStatistics_IncEdgeCount (&g->stats, r, nvals - n_tensors + n_elem) ;
	}
}

// decode adjacency matrix
void RdbLoadAdjMatrix_v18
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
) {
	// format:
	//   adjacency matrix

	ASSERT(gc  != NULL);
	ASSERT(rdb != NULL);

	Delta_Matrix adj = Graph_GetAdjacencyMatrix(gc->g, false);
	_Decode_Delta_Matrix(rdb, adj);
}

// decode labels matrix
void RdbLoadLblsMatrix_v18
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
) {
	// format:
	//   lbls matrix

	ASSERT(gc  != NULL);
	ASSERT(rdb != NULL);

	Delta_Matrix lbl = Graph_GetNodeLabelMatrix(gc->g);
	_Decode_Delta_Matrix(rdb, lbl);
}

