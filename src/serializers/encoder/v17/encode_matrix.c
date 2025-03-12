/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "../../serializer_io.h"
#include "../../../graph/graphcontext.h"
#include "../../../graph/delta_matrix/delta_matrix.h"

// encode matrix
static void _EncodeMatrix
(
	SerializerIO rdb,     // RDB
	const Delta_Matrix A  // matrix to encode
) {
	// format:
	//  blob size
	//  blob

	ASSERT(A != NULL);

	GrB_Info info;

	// flush delta matrix
	info = Delta_Matrix_wait(A, true);
	ASSERT(info == GrB_SUCCESS);

	GrB_Matrix M = Delta_Matrix_M(A);

	//--------------------------------------------------------------------------
	// serialize a GrB_Matrix to a blob
	//--------------------------------------------------------------------------

	void *blob;           // the blob
	GrB_Index blob_size;  // size of the blob
	// TODO: experiment with the different compression methods
	info = GxB_Matrix_serialize(&blob, &blob_size, M, NULL);
	ASSERT(info == GrB_SUCCESS);

	// write blob size
	SerializerIO_WriteUnsigned(rdb, blob_size);
	
	// write blob to rdb
	SerializerIO_WriteBuffer(rdb, (const char*)blob, blob_size);
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
		Delta_Matrix l = Graph_GetLabelMatrix(g, i);
		_EncodeMatrix(rdb, l);
	}
}

// encode relationship matrices to rdb
void RdbSaveRelationMatrices_v17
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   relation id X N
	//   matrix      X N

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	int n = Graph_RelationTypeCount(g);

	for(RelationID i = 0; i < n; i++) {
		// write relation ID
		SerializerIO_WriteUnsigned(rdb, i);

		// dump matrix to rdb
		Delta_Matrix r = Graph_GetRelationMatrix(g, i, false);
		_EncodeMatrix(rdb, r);
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

// encode graph's labels matrix
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

