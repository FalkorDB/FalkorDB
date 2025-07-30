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

// decode tensors
static void _DecodeTensors
(
	SerializerIO rdb,     // RDB
	GrB_Matrix A,         // matrix to populate with tensors
	uint64_t *n_tensors,  // [output] number of tensors loaded
	uint64_t *n_elem      // [output] number of edges loaded
) {
	// format:
	//  number of tensors
	//  tensors:
	//   tensor i index
	//   tensor j index
	//   tensor

	ASSERT(A         != NULL);
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

	// decode and set tensors
	for(uint64_t i = 0; i < n; i++) {
		// read tensor i,j indicies
		GrB_Index row = SerializerIO_ReadUnsigned(rdb);
		GrB_Index col = SerializerIO_ReadUnsigned(rdb);

		// read tensor blob
		GrB_Index blob_size;
		void *blob = SerializerIO_ReadBuffer(rdb, (size_t*)&blob_size);
		ASSERT(blob != NULL);

		GrB_Vector u;
		GrB_Info info = GxB_Vector_deserialize(&u, NULL, blob, blob_size, NULL);
		ASSERT(info == GrB_SUCCESS);

		// update number of elements loaded
		GrB_Index nvals;
		info = GrB_Vector_nvals(&nvals, u);
		ASSERT(info == GrB_SUCCESS);
		*n_elem += nvals;

		// set tensor
		uint64_t v = (uint64_t)(uintptr_t)SET_MSB(u);
		info = GrB_Matrix_setElement_UINT64(A, v, row, col);
		ASSERT(info == GrB_SUCCESS);

		rm_free(blob);
	}

	// set number of loaded tensors
	*n_tensors = n;
}

// decode matrix
static GrB_Matrix _DecodeMatrix
(
	SerializerIO rdb  // RDB
) {
	// format:
	//  blob size
	//  blob

	//--------------------------------------------------------------------------
	// read blob
	//--------------------------------------------------------------------------

	// read matrix blob
	GrB_Index blob_size;
	void *blob = SerializerIO_ReadBuffer(rdb, (size_t*)&blob_size);
	ASSERT(blob != NULL);

	//--------------------------------------------------------------------------
	// deserialize a GrB_Matrix from blob
	//--------------------------------------------------------------------------

	GrB_Matrix A;
	GrB_Info info = GxB_Matrix_deserialize(&A, NULL, blob, blob_size, NULL);
	ASSERT(info == GrB_SUCCESS);

	rm_free(blob);

	return A;
}

// decode label matrices from rdb
void RdbLoadLabelMatrices_v17
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

		// decode matrix
		GrB_Matrix L = _DecodeMatrix(rdb);

		//----------------------------------------------------------------------
		// update graph's label matrix with L
		//----------------------------------------------------------------------

		Delta_Matrix lbl = Graph_GetLabelMatrix(g, l);
		ASSERT(lbl != NULL);

		// replace lbl's current M matrix with L
		info = Delta_Matrix_setM(lbl, L);
		ASSERT(info == GrB_SUCCESS);
	}
}

// decode relationship matrices from rdb
void RdbLoadRelationMatrices_v17
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
	int n = Graph_RelationTypeCount(g);

	// decode relationship matrices
	for(int i = 0; i < n; i++) {
		// read relation ID
		RelationID r = SerializerIO_ReadUnsigned(rdb);

		// decode matrix
		GrB_Matrix R = _DecodeMatrix(rdb);

		// decode tensors
		uint64_t n_elem    = 0;  // number of tensor edges
		uint64_t n_tensors = 0;  // number of tensors in matrix
		_DecodeTensors(rdb, R, &n_tensors, &n_elem);

		// plant M matrix
		Delta_Matrix DR = Graph_GetRelationMatrix(g, r, false);
		ASSERT(DR != NULL);

		info = Delta_Matrix_setM(DR, R);
		ASSERT(info == GrB_SUCCESS);

		// update graph edge statistics
		// number of edges of type 'r' equals to:
		// |R| - n_tensors + n_elem
		GrB_Index nvals;
		info = GrB_Matrix_nvals(&nvals, R);
		ASSERT(info == GrB_SUCCESS);

		GraphStatistics_IncEdgeCount(&g->stats, r, nvals - n_tensors + n_elem);
	}
}

// decode adjacency matrix
void RdbLoadAdjMatrix_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
) {
	// format:
	//   adjacency matrix

	ASSERT(gc  != NULL);
	ASSERT(rdb != NULL);

	GrB_Matrix A = _DecodeMatrix(rdb);

	Delta_Matrix adj = Graph_GetAdjacencyMatrix(gc->g, false);
	ASSERT(adj != NULL);

	// replace adj's current M matrix with A
	GrB_Info info = Delta_Matrix_setM(adj, A);
	ASSERT(info == GrB_SUCCESS);
}

// decode labels matrix
void RdbLoadLblsMatrix_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
) {
	// format:
	//   lbls matrix

	ASSERT(gc  != NULL);
	ASSERT(rdb != NULL);

	GrB_Matrix A = _DecodeMatrix(rdb);

	Delta_Matrix lbl = Graph_GetNodeLabelMatrix(gc->g);
	ASSERT(lbl != NULL);

	// replace lbl's current M matrix with A
	GrB_Info info = Delta_Matrix_setM(lbl, A);
	ASSERT(info == GrB_SUCCESS);
}

