/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "../../../serializer_io.h"
#include "../../../../graph/graphcontext.h"
#include "../../../../graph/tensor/tensor.h"
#include "../../../../graph/graph_statistics.h"
#include "../../../../graph/delta_matrix/delta_matrix.h"

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
	rm_free (t_name) ;

	// load vector
	info = GrB_Vector_new (v, t, 0) ;
	ASSERT (info == GrB_SUCCESS) ;

	info = GxB_Vector_load (*v, &arr, t, n_entries, n_bytes, handling, NULL) ;
	ASSERT (info == GrB_SUCCESS) ;
}

// decode a multiedge entry
static GrB_Vector _Decode_multiedge
(
	SerializerIO rdb  // stream
) {
	// format:
	//  unloaded i vector

	// load into v
	GrB_Vector v = NULL;
	GrB_Vector i = NULL;
	GrB_Scalar one = NULL;
	GrB_OK (GrB_Scalar_new(&one, GrB_BOOL));
	GrB_OK (GrB_Scalar_setElement_BOOL(one, true));
	GrB_OK (GrB_Vector_new (&v, GrB_BOOL, GrB_INDEX_MAX));

	// make multiedge
	_decode_and_load_vector(rdb, &i);
	GrB_OK (GxB_Vector_assign_Scalar_Vector(v, NULL, NULL, one, i, NULL));

	// v is now back to its original state
	GrB_OK (GrB_free(&i));
	GrB_OK (GrB_free(&one));
	return v;
}

// decode a GraphBLAS matrix
static GrB_Matrix _Decode_GrB_Matrix
(
	SerializerIO rdb,  // stream
	bool tensors       // matrix has tensor entries?
) {
	// format:
	//  GraphBLAS container
	//  unloaded matrix components
	//  [if tensors: (k, multiedge)* terminated by UINT64_MAX]

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
	_decode_and_load_vector (rdb, &container->h) ;
	_decode_and_load_vector (rdb, &container->p) ;
	_decode_and_load_vector (rdb, &container->i) ;
	_decode_and_load_vector (rdb, &container->b) ;

	// patch tensor entries in x before loading the matrix
	if (tensors) {
		uint64_t k = SerializerIO_ReadUnsigned (rdb) ;
		while (k != UINT64_MAX) {
			GrB_Vector u = _Decode_multiedge (rdb) ;
			uint64_t v = SET_MSB ((uint64_t)(uintptr_t) u) ;
			GrB_OK (GrB_Vector_setElement_UINT64 (container->x, v, k)) ;
			k = SerializerIO_ReadUnsigned (rdb) ;
		}
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
	Delta_Matrix D,    // delta matrix to populate
	bool tensors       // matrix has tensor entries?
) {
	// format:
	//  M
	//  DP
	//  DM

	ASSERT (D   != NULL) ;
	ASSERT (rdb != NULL) ;

	GrB_Matrix M  = _Decode_GrB_Matrix (rdb, tensors) ;
	GrB_Matrix DP = _Decode_GrB_Matrix (rdb, tensors) ;
	GrB_Matrix DM = _Decode_GrB_Matrix (rdb, false) ;

	GrB_Info info = Delta_Matrix_setMatrices (D, &M, &DP, &DM) ;
	ASSERT (info == GrB_SUCCESS) ;
}

// decode label matrices from rdb
void RdbLoadLabelMatrices_v19
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
		_Decode_Delta_Matrix(rdb, lbl, false);
	}
}

// decode relationship matrices from rdb
void RdbLoadRelationMatrices_v19
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
) {
	// format:
	//   relation id      X N
	//   encode_tensors   X N
	//   matrix           X N

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
		ASSERT (r == i) ;

		// read tensors flag
		bool encode_tensors = SerializerIO_ReadUnsigned (rdb) ;

		// plant M matrix
		Delta_Matrix DR = Graph_GetRelationMatrix (g, r, false) ;

		GrB_Index nvals;
		Delta_Matrix_nvals (&nvals, DR) ;
		ASSERT (nvals == 0) ;

		_Decode_Delta_Matrix(rdb, DR, encode_tensors);

		// update graph edge statistics
		// number of edges of type 'r' equals to:
		// |R| - n_tensors + n_elem
		// where n_tensors is the count of tensor entries and
		// n_elem is the total number of edges across all tensors
		info = Delta_Matrix_nvals (&nvals, DR) ;
		ASSERT (info == GrB_SUCCESS) ;

		uint64_t n_tensors = 0;
		uint64_t n_elem    = 0;

		if (encode_tensors) {
			GrB_Matrix M  = Delta_Matrix_M  (DR) ;
			GrB_Matrix DP = Delta_Matrix_DP (DR) ;
			GrB_Matrix matrices[2] = {M, DP} ;

			for (int l = 0; l < 2; l++) {
				GxB_Iterator it;
				GrB_OK (GxB_Iterator_new (&it)) ;
				GrB_OK (GxB_Matrix_Iterator_attach (it, matrices[l], NULL)) ;
				info = GxB_Matrix_Iterator_seek (it, 0) ;
				while (info != GxB_EXHAUSTED) {
					uint64_t aij = GxB_Iterator_get_UINT64 (it) ;
					if (!SCALAR_ENTRY (aij)) {
						n_tensors++ ;
						GrB_Index nv ;
						GrB_OK (GrB_Vector_nvals (&nv, AS_VECTOR (aij))) ;
						n_elem += nv ;
					}
					info = GxB_Matrix_Iterator_next (it) ;
				}
				GrB_free (&it) ;
			}
		}

		GraphStatistics_IncEdgeCount (&g->stats, r, nvals - n_tensors + n_elem) ;
	}
}

// decode adjacency matrix
void RdbLoadAdjMatrix_v19
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
) {
	// format:
	//   adjacency matrix

	ASSERT(gc  != NULL);
	ASSERT(rdb != NULL);

	Delta_Matrix adj = Graph_GetAdjacencyMatrix(gc->g, false);
	_Decode_Delta_Matrix(rdb, adj, false);
}

// decode labels matrix
void RdbLoadLblsMatrix_v19
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
) {
	// format:
	//   lbls matrix

	ASSERT(gc  != NULL);
	ASSERT(rdb != NULL);

	Delta_Matrix lbl = Graph_GetNodeLabelMatrix(gc->g);
	_Decode_Delta_Matrix(rdb, lbl, false);
}
