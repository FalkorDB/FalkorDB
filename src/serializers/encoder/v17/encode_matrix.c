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

// encode tensors
static void _EncodeTensors
(
	SerializerIO rdb,  // RDB
	const Graph *g,    // graph
	RelationID r,      // matrix relation id
	GrB_Matrix A       // matrix from which to extract tensors
) {
	// format:
	//  number of tensors
	//  tensors:
	//   tensor i index
	//   tensor j index
	//   tensor

	ASSERT(g != NULL);
	ASSERT(A != NULL);

	GrB_Info info;

	// in case the matrix doesn't contains any tensors simply return
	if(!Graph_RelationshipContainsMultiEdge(g, r)) {
		// no tensors
		SerializerIO_WriteUnsigned(rdb, 0);
		return;
	}

	// extract A's tensors
	GrB_Scalar s;
	info = GrB_Scalar_new(&s, GrB_UINT64);
	ASSERT(info == GrB_SUCCESS);

	// tensor entries have their MSB set
	info = GrB_Scalar_setElement_UINT64(s, (uint64_t)1 << 63);
	ASSERT(info == GrB_SUCCESS);

	// create a temporary matrix which will contain A's tensors
	GrB_Index nrows;
	GrB_Index ncols;
	info = GrB_Matrix_nrows(&nrows, A);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols, A);
	ASSERT(info == GrB_SUCCESS);

	// keep entries A[i,j] with MSB on
	GrB_Matrix T;
	info = GrB_Matrix_new(&T, GrB_UINT64, nrows, ncols);
	ASSERT(info == GrB_SUCCESS);

	// copy tensor entries from A to T
	info = GrB_select(T, NULL, NULL, GrB_VALUEGE_UINT64, A, s, NULL);
	ASSERT(info == GrB_SUCCESS);

	// how many tensors are there?
	GrB_Index nvals;
	info = GrB_Matrix_nvals(&nvals, T);
	ASSERT(info == GrB_SUCCESS);

	// encode number of tensors
	SerializerIO_WriteUnsigned(rdb, nvals);

	// encode each tensor
	GxB_Iterator it;
	info = GxB_Iterator_new(&it);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_rowIterator_attach(it, T, NULL);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_rowIterator_seekRow(it, 0);
	while(info != GxB_EXHAUSTED) {
		// iterate over entries in T(i,:)
		GrB_Index i = GxB_rowIterator_getRowIndex(it);
		while(info == GrB_SUCCESS) {
			// get the entry T(i,j)
			GrB_Index  j   = GxB_rowIterator_getColIndex(it);
			uint64_t   aij = GxB_Iterator_get_UINT64(it);
			GrB_Vector u   = AS_VECTOR(aij);  // treat entry as a vector

			//------------------------------------------------------------------
			// serialize the tensor
			//------------------------------------------------------------------

			void *blob;           // the blob
			GrB_Index blob_size;  // size of the blob

			info = GxB_Vector_serialize(&blob, &blob_size, u, NULL);
			ASSERT(info == GrB_SUCCESS);

			// write tensor i,j position
			SerializerIO_WriteUnsigned(rdb, i);
			SerializerIO_WriteUnsigned(rdb, j);

			// write blob to rdb
			SerializerIO_WriteBuffer(rdb, (const char*)blob, blob_size);

			rm_free(blob);

			// optimize memory consumption
			// if dump is being taken on a fork process, as a result of calling:
			// BGSAVE or BGREWRITEAOF
			// to avoid increase in memory consumption due to copy-on-write
			// we can free processed tensor at the child process end
			if(Globals_Get_ProcessIsChild()) {
				info = GrB_free(&u);
				ASSERT(info == GrB_SUCCESS);
			}

			// move to the next entry in A(i,:)
			info = GxB_rowIterator_nextCol(it);
		}

		// move to the next row, A(i+1,:)
		info = GxB_rowIterator_nextRow(it);
	}

	// clean up
	GrB_free(&s);
	GrB_free(&T);
	GrB_free(&it);
}

// encode matrix
static void _EncodeMatrix
(
	SerializerIO rdb,  // RDB
	Delta_Matrix A     // matrix to encode
) {
	// format:
	//  blob size
	//  blob

	ASSERT(A != NULL);

	GrB_Info info;

	// flush delta matrix
	info = Delta_Matrix_wait(A, false);
	ASSERT(info == GrB_SUCCESS);

	GrB_Matrix M  = Delta_Matrix_M(A);
	GrB_Matrix DP = Delta_Matrix_DP(A);
	GrB_Matrix DM = Delta_Matrix_DM(A);

	//--------------------------------------------------------------------------
	// serialize M to a blob
	//--------------------------------------------------------------------------

	void *blob;           // the blob
	GrB_Index blob_size;  // size of the blob
	// TODO: experiment with the different compression methods
	info = GxB_Matrix_serialize(&blob, &blob_size, M, NULL);
	ASSERT(info == GrB_SUCCESS);

	// write blob to rdb
	SerializerIO_WriteBuffer(rdb, (const char*)blob, blob_size);

	rm_free(blob);

	//--------------------------------------------------------------------------
	// serialize Delta-Plus to a blob
	//--------------------------------------------------------------------------

	info = GxB_Matrix_serialize(&blob, &blob_size, DP, NULL);
	ASSERT(info == GrB_SUCCESS);

	// write blob to rdb
	SerializerIO_WriteBuffer(rdb, (const char*)blob, blob_size);

	rm_free(blob);

	//--------------------------------------------------------------------------
	// serialize Delta-Minus to a blob
	//--------------------------------------------------------------------------

	info = GxB_Matrix_serialize(&blob, &blob_size, DM, NULL);
	ASSERT(info == GrB_SUCCESS);

	// write blob to rdb
	SerializerIO_WriteBuffer(rdb, (const char*)blob, blob_size);

	rm_free(blob);
}

// encode label matrices to rdb
void RdbSaveLabelMatrices_v17
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

	// write number of matrices
	SerializerIO_WriteUnsigned(rdb, n);
	
	// encode label matrices
	for(LabelID i = 0; i < n; i++) {
		// write label ID
		SerializerIO_WriteUnsigned(rdb, i);

		// dump matrix to rdb
		Delta_Matrix L = Graph_GetLabelMatrix(g, i);
		_EncodeMatrix(rdb, L);
	}
}

// encode relationship matrices to rdb
void RdbSaveRelationMatrices_v17
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

	for(RelationID i = 0; i < n; i++) {
		// write relation ID
		SerializerIO_WriteUnsigned(rdb, i);

		// dump matrix to rdb
		Delta_Matrix R = Graph_GetRelationMatrix(g, i, false);
		_EncodeMatrix(rdb, R);
		_EncodeTensors(rdb, g, i, Delta_Matrix_M(R));
	}
}

// encode graph's adjacency matrix
void RdbSaveAdjMatrix_v17
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   lbls matrix

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	Delta_Matrix ADJ = Graph_GetAdjacencyMatrix(g, false);
	_EncodeMatrix(rdb, ADJ);
}

// encode graph's node labels matrix
void RdbSaveLblsMatrix_v17
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   lbls matrix

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	Delta_Matrix lbls = Graph_GetNodeLabelMatrix(g);
	_EncodeMatrix(rdb, lbls);
}

