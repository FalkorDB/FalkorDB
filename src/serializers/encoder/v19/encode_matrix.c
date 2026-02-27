/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "../../../globals.h"
#include "../../serializer_io.h"
#include "../../../graph/graphcontext.h"
#include "../../../graph/tensor/tensor.h"
#include "../../../graph/delta_matrix/delta_matrix.h"

// unloads vector to C array and encodes the vector to stream
static void _unload_and_encode_vector
(
	SerializerIO rdb,  // stream
	GrB_Vector v,      // vector to unload and encode
	bool reload        // should the vector be reloaded?
) {
	// format:
	//   array
	//   type name
	//   number of entries
	//   number of bytes
	//   handeling

	ASSERT(v   != NULL);
	ASSERT(rdb != NULL);

	GrB_Info info;
	void *arr;           // vector's data
	GrB_Type t;          // data type
	uint64_t n_entries;  // number of entries
	uint64_t n_bytes;    // data size in bytes
	int handling;        // memory owner GraphBLAS / Application

	// unload vector to C array
	GrB_OK (GxB_Vector_unload (v, &arr, &t, &n_entries, &n_bytes, &handling,
			NULL)) ;

	// get type name
	size_t t_name_len = 0 ;
	char   t_name [GxB_MAX_NAME_LEN] = {0} ;

	GrB_OK (GrB_get (t, t_name, GrB_NAME)) ;

	t_name_len = strlen ((char *)t_name) + 1 ;  // account for NULL byte

	//--------------------------------------------------------------------------
	// encode vector
	//--------------------------------------------------------------------------

	SerializerIO_WriteBuffer   (rdb, arr, n_bytes) ;
	SerializerIO_WriteBuffer   (rdb, t_name, t_name_len) ;
	SerializerIO_WriteUnsigned (rdb, n_entries) ;
	SerializerIO_WriteUnsigned (rdb, n_bytes) ;
	SerializerIO_WriteSigned   (rdb, handling) ;

	if (reload) {
		// reload vector
		info = GxB_Vector_load (v, &arr, t, n_entries, n_bytes, handling, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;
	} else {
		// free array
		// NOTE: this destroys the matrix!
		// rm_free(arr);
	}
}

static void _Encode_multiedge
(
	SerializerIO rdb,  // stream
	GrB_Vector v,      // multi edge to encode
	bool reload        // reload vector
) {
	// format:
	//  unloaded i vector

	ASSERT (v   != NULL) ;
	ASSERT (rdb != NULL) ;

    GxB_Container container;
	GrB_OK (GxB_Container_new (&container)) ;

	// unload matrix into a container
	GrB_OK (GxB_unload_Vector_into_Container(v, container, NULL)) ;

	//--------------------------------------------------------------------------
	// encode vector
	//--------------------------------------------------------------------------

	_unload_and_encode_vector(rdb, container->i, reload);

	if (reload) {
		// reload vector
		GrB_OK (GxB_load_Vector_from_Container (v, container, NULL)) ;
	}

	// clean up
	GxB_Container_free(&container);
}

static void _encode_multiedge_array
(
	SerializerIO rdb,  // stream
	GrB_Vector v,      // array
	bool reload
) {
	// format:
	//   i (index of multi edge)
	//   multiedge
	//   terminated by -1
	uint64_t arr_n;
	GrB_OK (GrB_Vector_size(&arr_n, v));
	for (uint64_t i = 0; i < arr_n; i++) {
		uint64_t x;
		GrB_OK (GrB_Vector_extractElement_UINT64(&x, v, i));
		if(!SCALAR_ENTRY(x)) {
			GrB_Vector u = AS_VECTOR(x);
			SerializerIO_WriteUnsigned (rdb, i) ;
			_Encode_multiedge(rdb, u, reload);
		}
	}
	SerializerIO_WriteUnsigned (rdb, UINT64_MAX) ;  // sentinel
}

// encode a GraphBLAS matrix
static void _Encode_GrB_Matrix
(
	SerializerIO rdb,  // stream
	GrB_Matrix A,      // GraphBLAS matrix to encode
	bool reload,       // reload matrix
	bool tensors       // are there tensors in this matrix?
) {
	// format:
	//  GraphBLAS container
	//  unloaded matrix components

	ASSERT (A   != NULL) ;
	ASSERT (rdb != NULL) ;

	GrB_Info info;

    GxB_Container container;
	info = GxB_Container_new (&container) ;
	ASSERT (info == GrB_SUCCESS);

	// unload matrix into a container
	info = GxB_unload_Matrix_into_Container (A, container, NULL) ;
	ASSERT(info == GrB_SUCCESS);

	// encode entire container
	SerializerIO_WriteBuffer (rdb, container,
			sizeof (struct GxB_Container_struct)) ;

	//--------------------------------------------------------------------------
	// encode vectors
	//--------------------------------------------------------------------------

	_unload_and_encode_vector(rdb, container->x, tensors || reload);
	_unload_and_encode_vector(rdb, container->h, reload);
	_unload_and_encode_vector(rdb, container->p, reload);
	_unload_and_encode_vector(rdb, container->i, reload);
	_unload_and_encode_vector(rdb, container->b, reload);

	if (tensors) {
		_encode_multiedge_array(rdb, container->x, reload);
	}

	if (reload) {
		// reload matrix
		info = GxB_load_Matrix_from_Container (A, container, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;
	}

	// clean up
	GxB_Container_free(&container);
}

// encode delta matrix
static void _Encode_Delta_Matrix
(
	SerializerIO rdb,  // RDB
	Delta_Matrix D,    // delta matrix to encode
	bool reload,       // reload matrix
	bool tensors       // are there tensor values to encode?
) {
	// format:
	//  M
	//  DP
	//  DM

	ASSERT (D   != NULL) ;
	ASSERT (rdb != NULL) ;

	GrB_Info info;

	GrB_Matrix M  = Delta_Matrix_M  (D) ;
	GrB_Matrix DP = Delta_Matrix_DP (D) ;
	GrB_Matrix DM = Delta_Matrix_DM (D) ;

	_Encode_GrB_Matrix(rdb, M,  reload, tensors);
	_Encode_GrB_Matrix(rdb, DP, reload, tensors);
	_Encode_GrB_Matrix(rdb, DM, reload, false);
}

// encode label matrices to rdb
void RdbSaveLabelMatrices_v19
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//  number of label matrices
	//   label id
	//   matrix

	ASSERT (g   != NULL) ;
	ASSERT (rdb != NULL) ;

	int n = Graph_LabelTypeCount (g) ;
	bool reload = !Globals_Get_ProcessIsChild () ;

	// write number of matrices
	SerializerIO_WriteUnsigned (rdb, n) ;
	
	// encode label matrices
	for (LabelID i = 0 ; i < n ; i++) {
		// write label ID
		SerializerIO_WriteUnsigned (rdb, i) ;

		// dump matrix to rdb
		Delta_Matrix L = Graph_GetLabelMatrix (g, i) ;
		_Encode_Delta_Matrix (rdb, L, reload, false) ;
	}
}

// encode relationship matrices to rdb
void RdbSaveRelationMatrices_v19
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   relation id   X N
	//   matrix        X N
	//   tensors count X N
	//   tensors

	ASSERT (g   != NULL) ;
	ASSERT (rdb != NULL) ;

	int n = Graph_RelationTypeCount (g) ;
	bool reload = !Globals_Get_ProcessIsChild () ;

	for (RelationID i = 0 ; i < n ; i++) {
		// write relation ID
		SerializerIO_WriteUnsigned (rdb, i) ;

		// dump matrix to rdb
		Delta_Matrix R   = Graph_GetRelationMatrix (g, i, false) ;

		bool encode_tensors = Graph_RelationshipContainsMultiEdge (g, i) ;
		SerializerIO_WriteUnsigned(rdb, encode_tensors);

		_Encode_Delta_Matrix (rdb, R, reload, encode_tensors) ;
	}
}

// encode graph's adjacency matrix
void RdbSaveAdjMatrix_v19
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   lbls matrix

	ASSERT (g   != NULL) ;
	ASSERT (rdb != NULL) ;

	bool reload = !Globals_Get_ProcessIsChild () ;

	Delta_Matrix ADJ = Graph_GetAdjacencyMatrix (g, false) ;
	_Encode_Delta_Matrix (rdb, ADJ, reload, false) ;
}

// encode graph's node labels matrix
void RdbSaveLblsMatrix_v19
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   lbls matrix

	ASSERT (g   != NULL) ;
	ASSERT (rdb != NULL) ;

	bool reload = !Globals_Get_ProcessIsChild () ;

	Delta_Matrix lbls = Graph_GetNodeLabelMatrix (g) ;
	_Encode_Delta_Matrix (rdb, lbls, reload, false) ;
}

