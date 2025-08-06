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

// extract D's tensors
static void _ExtractTensors
(
	Delta_Matrix D,    // matrix from which to extract tensors
	GrB_Matrix *TM,    // M's tensors
	GrB_Matrix *TDP    // DP's tensors
) {
	ASSERT(D   != NULL);
	ASSERT(TM  != NULL);
	ASSERT(TDP != NULL);

	GrB_Info info;
	GrB_Matrix M  = DELTA_MATRIX_M  (D) ;
	GrB_Matrix DP = DELTA_MATRIX_DELTA_PLUS (D) ;
	GrB_Matrix DM = DELTA_MATRIX_DELTA_MINUS (D) ;

	// create a temporary matrix which will contain A's tensors
	GrB_Index nrows;
	GrB_Index ncols;

	info = GrB_Matrix_nrows (&nrows, M) ;
	ASSERT (info == GrB_SUCCESS) ;

	info = GrB_Matrix_ncols (&ncols, M) ;
	ASSERT (info == GrB_SUCCESS) ;

	// tensors only matrix
	info = GrB_Matrix_new (TM, GrB_UINT64, nrows, ncols) ;
	ASSERT (info == GrB_SUCCESS) ;

	info = GrB_Matrix_new (TDP, GrB_UINT64, nrows, ncols) ;
	ASSERT (info == GrB_SUCCESS) ;

	// extract A's tensors
	// keep entries A[i,j] with MSB on
	// copy tensor entries from A to T
	info = GrB_Matrix_select_UINT64(*TM, DM, NULL, GrB_VALUEGT_UINT64, M,
			MSB_MASK, GrB_DESC_SC) ;
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_select_UINT64(*TDP, DM, NULL, GrB_VALUEGT_UINT64, DP,
			MSB_MASK, GrB_DESC_SC) ;
	ASSERT(info == GrB_SUCCESS);

	// expecting at least a single tensor was extracted
	GrB_Index tm_nvals;
	GrB_Index tdp_nvals;

	info = GrB_Matrix_nvals (&tm_nvals, *TM) ;
	ASSERT (info == GrB_SUCCESS) ;

	info = GrB_Matrix_nvals (&tdp_nvals, *TDP) ;
	ASSERT (info == GrB_SUCCESS) ;

	ASSERT ((tm_nvals + tdp_nvals) > 0) ;
}

// encode tensors
static void _EncodeTensors
(
	SerializerIO rdb,  // RDB
	GrB_Matrix TM,     // M's tensors
	GrB_Matrix TDP     // DP's tensors
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

	// either both matrices are specified or both are NULL
	ASSERT ( (TM == NULL && TDP == NULL) || (TM != NULL && TDP != NULL)) ;

	if (TM == NULL && TDP == NULL) {
		// no tensors
		SerializerIO_WriteUnsigned (rdb, 0) ;
		return;
	}

	GrB_Info info;

	//--------------------------------------------------------------------------
	// encode total number of tensors
	//--------------------------------------------------------------------------

	GrB_Index tm_nvals;
	info = GrB_Matrix_nvals (&tm_nvals, TM) ;
	ASSERT (info == GrB_SUCCESS) ;

	GrB_Index tdp_nvals;
	info = GrB_Matrix_nvals (&tdp_nvals, TDP) ;
	ASSERT (info == GrB_SUCCESS) ;

	GrB_Index nvals = tm_nvals + tdp_nvals ;
	ASSERT (nvals > 0) ;

	// encode number of tensors in matrix R
	SerializerIO_WriteUnsigned (rdb, nvals) ;

	GrB_Matrix matrices[2] = { TM, TDP } ;

	for (int l = 0; l < 2; l++) {
		GrB_Matrix T = matrices[l];

		// how many tensors are there?
		GrB_Index nvals;
		info = GrB_Matrix_nvals (&nvals, T) ;
		ASSERT (info == GrB_SUCCESS) ;

		// encode number of tensors
		SerializerIO_WriteUnsigned (rdb, nvals) ;

		if (nvals == 0) {
			continue ;
		}

		// encode each tensor
		GxB_Iterator it;
		info = GxB_Iterator_new (&it) ;
		ASSERT (info == GrB_SUCCESS) ;

		info = GxB_Matrix_Iterator_attach (it, T, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;

		info = GxB_Matrix_Iterator_seek (it, 0) ;
		while (info != GxB_EXHAUSTED) {
			// iterate over entries
			GrB_Index i ;
			GrB_Index j ;
			GxB_Matrix_Iterator_getIndex (it, &i, &j) ;

			// get the entry T(i,j)
			uint64_t aij = GxB_Iterator_get_UINT64 (it) ;
			ASSERT (aij & MSB_MASK) ;

			GrB_Vector u = AS_VECTOR (aij) ;  // treat entry as a vector
			ASSERT (info == GrB_SUCCESS) ;

			//--------------------------------------------------------------
			// serialize the tensor
			//--------------------------------------------------------------

			void *blob;           // the blob
			GrB_Index blob_size;  // size of the blob

			info = GxB_Vector_serialize (&blob, &blob_size, u, NULL) ;
			ASSERT (info == GrB_SUCCESS) ;

			// write tensor i,j position
			SerializerIO_WriteUnsigned (rdb, i) ;
			SerializerIO_WriteUnsigned (rdb, j) ;

			// write blob to rdb
			SerializerIO_WriteBuffer (rdb, blob, blob_size) ;

			rm_free (blob) ;

			// move to the next entry
			info = GxB_Matrix_Iterator_next (it) ;
		}

		// clean up
		GrB_free (&it) ;
	}
}

// unloads vector to C array and encodes the vector to stream
static void _unload_and_encode_vector
(
	SerializerIO rdb,  // stream
	GrB_Vector v,      // vector to unload and encode
	bool reload        // reload vector
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
	info = GxB_Vector_unload (v, &arr, &t, &n_entries, &n_bytes, &handling,
			NULL) ;
	ASSERT (info == GrB_SUCCESS) ;

	// get type name
	size_t t_name_len = 0 ;
	char   t_name [GxB_MAX_NAME_LEN] = {0} ;

	info = GrB_get (t, t_name, GrB_NAME) ;
	ASSERT (info == GrB_SUCCESS) ;

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

// encode a GraphBLAS matrix
static void _Encode_GrB_Matrix
(
	SerializerIO rdb,  // stream
	GrB_Matrix A,      // GraphBLAS matrix to encode
	bool reload        // reload matrix
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

	_unload_and_encode_vector(rdb, container->x, reload);
	_unload_and_encode_vector(rdb, container->h, reload);
	_unload_and_encode_vector(rdb, container->p, reload);
	_unload_and_encode_vector(rdb, container->i, reload);
	_unload_and_encode_vector(rdb, container->b, reload);

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
	bool reload        // reload matrix
) {
	// format:
	//  M
	//  DP
	//  DM

	ASSERT (D   != NULL) ;
	ASSERT (rdb != NULL) ;

	GrB_Info info;

	GrB_Matrix M  = DELTA_MATRIX_M  (D) ;
	GrB_Matrix DP = DELTA_MATRIX_DELTA_PLUS (D) ;
	GrB_Matrix DM = DELTA_MATRIX_DELTA_MINUS (D) ;

	_Encode_GrB_Matrix(rdb, M,  reload);
	_Encode_GrB_Matrix(rdb, DP, reload);
	_Encode_GrB_Matrix(rdb, DM, reload);
}

// encode label matrices to rdb
void RdbSaveLabelMatrices_v18
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//  number of label matrices
	//   label id
	//   matrix

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	int n = Graph_LabelTypeCount(g);
	bool reload = !Globals_Get_ProcessIsChild();

	// write number of matrices
	SerializerIO_WriteUnsigned(rdb, n);
	
	// encode label matrices
	Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);
	for(LabelID i = 0; i < n; i++) {
		// write label ID
		SerializerIO_WriteUnsigned(rdb, i);

		// dump matrix to rdb
		Delta_Matrix L = Graph_GetLabelMatrix(g, i);
		_Encode_Delta_Matrix(rdb, L, reload);
	}
}

// encode relationship matrices to rdb
void RdbSaveRelationMatrices_v18
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   relation id   X N
	//   matrix        X N
	//   tensors count X N
	//   tensors

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	int n = Graph_RelationTypeCount(g);
	bool reload = !Globals_Get_ProcessIsChild();

	Graph_SetMatrixPolicy (g, SYNC_POLICY_NOP) ;
	for (RelationID i = 0; i < n; i++) {
		// write relation ID
		SerializerIO_WriteUnsigned (rdb, i) ;

		// dump matrix to rdb
		Delta_Matrix R   = Graph_GetRelationMatrix (g, i, false) ;
		GrB_Matrix   TM  = NULL ;  // R's M's tensors
		GrB_Matrix   TDP = NULL ;  // R's DP's tensors

		bool encode_tensors = Graph_RelationshipContainsMultiEdge (g, i) ;
		if (encode_tensors) {
			_ExtractTensors (R, &TM, &TDP) ;
		}

		_Encode_Delta_Matrix (rdb, R, reload) ;

		_EncodeTensors (rdb, TM, TDP) ;

		if (encode_tensors) {
			GrB_free (&TM) ;
			GrB_free (&TDP) ;
		}
	}
}

// encode graph's adjacency matrix
void RdbSaveAdjMatrix_v18
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   lbls matrix

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	bool reload = !Globals_Get_ProcessIsChild();

	Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);
	Delta_Matrix ADJ = Graph_GetAdjacencyMatrix(g, false);
	_Encode_Delta_Matrix(rdb, ADJ, reload);
}

// encode graph's node labels matrix
void RdbSaveLblsMatrix_v18
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   lbls matrix

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	bool reload = !Globals_Get_ProcessIsChild();

	Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);
	Delta_Matrix lbls = Graph_GetNodeLabelMatrix(g);
	_Encode_Delta_Matrix(rdb, lbls, reload);
}

