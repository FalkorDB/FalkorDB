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

#include <mach/mach.h>
#include <mach/task_info.h>

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
	GrB_Matrix M  = Delta_Matrix_M  (D) ;
	GrB_Matrix DP = Delta_Matrix_DP (D) ;

	// create a scalar with MSB on
	GrB_Scalar s;
	info = GrB_Scalar_new(&s, GrB_UINT64);
	ASSERT(info == GrB_SUCCESS);

	// tensor entries have their MSB set
	info = GrB_Scalar_setElement_UINT64(s, (uint64_t)1 << 63);
	ASSERT(info == GrB_SUCCESS);

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
	info = GrB_select (*TM, NULL, NULL, GrB_VALUEGE_UINT64, M, s, NULL) ;
	ASSERT(info == GrB_SUCCESS);

	info = GrB_select (*TDP, NULL, NULL, GrB_VALUEGE_UINT64, DP, s, NULL) ;
	ASSERT(info == GrB_SUCCESS);

	GrB_free (&s) ;
}

// encode tensors
static void _EncodeTensors
(
	SerializerIO rdb,  // RDB
	GrB_Matrix TM,
	GrB_Matrix TDP
) {
	// format:
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
	SerializerIO_WriteUnsigned (rdb, nvals) ;

	// return if no tensors
	if (nvals == 0) {
		return;
	}

	GrB_Matrix matrices[2] = { TM, TDP } ;

	for (int i = 0; i < 2; i++) {
		GrB_Matrix T = matrices[i];

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

		info = GxB_rowIterator_attach (it, T, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;

		info = GxB_rowIterator_seekRow (it, 0) ;
		while (info != GxB_EXHAUSTED) {
			// iterate over entries in T(i,:)
			GrB_Index i = GxB_rowIterator_getRowIndex (it) ;
			while (info == GrB_SUCCESS) {
				// get the entry T(i,j)
				GrB_Index  j   = GxB_rowIterator_getColIndex (it) ;
				uint64_t   aij = GxB_Iterator_get_UINT64 (it) ;
				GrB_Vector u   = AS_VECTOR (aij) ;  // treat entry as a vector

				//------------------------------------------------------------------
				// serialize the tensor
				//------------------------------------------------------------------

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

				// move to the next entry in T(i,:)
				info = GxB_rowIterator_nextCol (it) ;
			}

			// move to the next row, T(i+1,:)
			info = GxB_rowIterator_nextRow (it) ;
		}

		// clean up
		GrB_free (&it) ;
	}
}

static void print_mem_cons(void) {
	struct task_vm_info _info;
	mach_msg_type_number_t count = TASK_VM_INFO_COUNT;

	if (task_info(mach_task_self(), TASK_VM_INFO,
				  (task_info_t)&_info, &count) == KERN_SUCCESS) {
		printf("Internal: %llu bytes\n", _info.internal);
		//printf("Compressed: %llu bytes\n", _info.compressed);
		//printf("Purgeable: %llu bytes\n", _info.purgeable_volatile_pmap);
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
	int handling;       // memory owner GraphBLAS / Application

	// unload vector to C array
	info = GxB_Vector_unload (v, &arr, &t, &n_entries, &n_bytes, &handling, NULL) ;
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
		rm_free(arr);
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

	ASSERT (A != NULL) ;

	GrB_Info info;
    GxB_set (GxB_BURBLE, true) ;

	print_mem_cons () ;

    GxB_Container container;
	info = GxB_Container_new (&container) ;
	ASSERT (info == GrB_SUCCESS);

	// unload matrix into a container
	info = GxB_unload_Matrix_into_Container (A, container, NULL);
	ASSERT(info == GrB_SUCCESS);

	// encode entire container
	SerializerIO_WriteBuffer (rdb, container,
			sizeof (struct GxB_Container_struct)) ;

	print_mem_cons();

	//--------------------------------------------------------------------------
	// encode vectors
	//--------------------------------------------------------------------------

	_unload_and_encode_vector(rdb, container->x, reload);

	// extract the sparsity pattern from the container
	switch (container->format)
	{
		case GxB_HYPERSPARSE :
			_unload_and_encode_vector(rdb, container->h, reload);

		case GxB_SPARSE :
			_unload_and_encode_vector(rdb, container->p, reload);
			_unload_and_encode_vector(rdb, container->i, reload);

				break ;

		case GxB_BITMAP :
			_unload_and_encode_vector(rdb, container->b, reload);

				break ;
	}

	print_mem_cons();

	if (reload) {
		// reload matrix
		info = GxB_load_Matrix_from_Container (A, container, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;
	}

	// clean up
	GxB_Container_free(&container);

    GxB_set (GxB_BURBLE, false) ;
}

// encode delta matrix
static void _Encode_Delta_Matrix
(
	SerializerIO rdb,  // RDB
	Delta_Matrix D,    // delta matrix to encode
	bool reload        // reload matrix
) {
	// format:
	//  blob size
	//  blob

	ASSERT(D != NULL);

	GrB_Info info;
    GxB_set (GxB_BURBLE, true) ;

	GrB_Matrix M  = Delta_Matrix_M(D);
	GrB_Matrix DP = Delta_Matrix_DP(D);
	GrB_Matrix DM = Delta_Matrix_DM(D);

	print_mem_cons();
	_Encode_GrB_Matrix(rdb, M, reload);

	print_mem_cons();
	_Encode_GrB_Matrix(rdb, DP, reload);

	print_mem_cons();
	_Encode_GrB_Matrix(rdb, DM, reload);

	print_mem_cons();
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

	Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);
	for(RelationID i = 0; i < n; i++) {
		// write relation ID
		SerializerIO_WriteUnsigned(rdb, i);

		// dump matrix to rdb
		Delta_Matrix R = Graph_GetRelationMatrix (g, i, false) ;
		GrB_Matrix TM  = NULL ;
		GrB_Matrix TDP = NULL ;

		bool encode_tensors = Graph_RelationshipContainsMultiEdge (g, i) ;
		if (encode_tensors) {
			_ExtractTensors (R, &TM, &TDP) ;
		}

		_Encode_Delta_Matrix (rdb, R, reload) ;
		_EncodeTensors (rdb, TM, TDP) ;

		// clean up
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

