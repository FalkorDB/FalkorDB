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
	uint64_t n = SerializerIO_TryReadUnsigned(rdb);

	// no tensors, simply return
	if(n == 0) {
		return;
	}

	// decode and set tensors
	for(uint64_t i = 0; i < n; i++) {
		// read tensor i,j indicies
		GrB_Index i = SerializerIO_TryReadUnsigned(rdb);
		GrB_Index j = SerializerIO_TryReadUnsigned(rdb);

		// read tensor blob (a short read jumps to the landing pad)
		GrB_Index blob_size;
		void *blob = SerializerIO_TryReadBuffer(rdb, (size_t*)&blob_size);
		ASSERT(blob != NULL);

		GrB_Vector u;
		GrB_Info info = GxB_Vector_deserialize(&u, NULL, blob, blob_size, NULL);
		rm_free(blob);
		if (info != GrB_SUCCESS) {
			SerializerIO_SetError(rdb,
					EMSG_DECODE_GRB, "deserializing tensor",
					(int) info, (unsigned long long) i, (unsigned long long) j);
			return;
		}

		// update number of elements loaded
		GrB_Index nvals;
		info = GrB_Vector_nvals(&nvals, u);
		ASSERT(info == GrB_SUCCESS);
		*n_elem += nvals;

		// set tensor
		uint64_t v = SET_MSB((uint64_t)(uintptr_t) u);
		info = GrB_Matrix_setElement_UINT64(A, v, i, j);
		if (info != GrB_SUCCESS) {
			GrB_Vector_free(&u);
			SerializerIO_SetError(rdb,
					EMSG_DECODE_GRB, "setting tensor",
					(int) info, (unsigned long long) i, (unsigned long long) j);
			return;
		}
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

	// read matrix blob (a short read jumps to the landing pad)
	GrB_Index blob_size;
	void *blob = SerializerIO_TryReadBuffer(rdb, (size_t*)&blob_size);
	ASSERT(blob != NULL);

	//--------------------------------------------------------------------------
	// deserialize a GrB_Matrix from blob
	//--------------------------------------------------------------------------

	GrB_Matrix A;
	GrB_Info info = GxB_Matrix_deserialize(&A, NULL, blob, blob_size, NULL);
	rm_free(blob);
	if (info != GrB_SUCCESS) {
		SerializerIO_SetError(rdb, EMSG_DECODE_GRB, "deserializing a matrix",
				(int) info);
		return NULL;
	}

	return A;
}

// decode label matrices from rdb
void RdbLoadLabelMatrices_v17
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

	GrB_Info info;

	// read number of label matricies: one per label
	uint64_t n = SerializerIO_TryReadUnsigned(rdb);
	uint64_t label_count = Graph_LabelTypeCount(g);
	if(n != label_count) {
		SerializerIO_SetError(rdb, EMSG_DECODE_LABEL_MATRIX_COUNT,
				(unsigned long long) n, (unsigned long long) label_count);
		return;
	}

	// decode each label matrix
	for(uint64_t i = 0; i < n; i++) {
		// read label ID, validated before it indexes the label array
		uint64_t l = SerializerIO_TryReadUnsigned(rdb);
		if(l >= label_count) {
			SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "label matrix id",
					(unsigned long long) l);
			return;
		}

		// decode matrix
		GrB_Matrix L = _DecodeMatrix(rdb);
		if(L == NULL) return;  // SetError already latched (unarmed safety)

		//----------------------------------------------------------------------
		// update graph's label matrix with L
		//----------------------------------------------------------------------

		Delta_Matrix lbl = Graph_GetLabelMatrix(g, l);
		ASSERT(lbl != NULL);

		// replace lbl's current M matrix with L
		info = Delta_Matrix_setM(lbl, &L);
		ASSERT(info == GrB_SUCCESS);
	}
}

// decode relationship matrices from rdb
void RdbLoadRelationMatrices_v17
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   relation id X N
	//   matrix      X N

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	GrB_Info info;

	// number of relation matricies
	int n = Graph_RelationTypeCount(g);

	// decode relationship matrices
	for(int i = 0; i < n; i++) {
		// read relation ID, validated before it indexes the relation array
		uint64_t r = SerializerIO_TryReadUnsigned(rdb);
		if(r >= (uint64_t) n) {
			SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "relation matrix id",
					(unsigned long long) r);
			return;
		}

		// decode matrix
		GrB_Matrix R = _DecodeMatrix(rdb);
		if(R == NULL) return;  // SetError already latched (unarmed safety)

		// decode tensors
		uint64_t n_elem    = 0;  // number of tensor edges
		uint64_t n_tensors = 0;  // number of tensors in matrix
		_DecodeTensors(rdb, R, &n_tensors, &n_elem);

		// plant M matrix
		Delta_Matrix DR = Graph_GetRelationMatrix(g, r, false);
		ASSERT(DR != NULL);

		// update graph edge statistics
		// number of edges of type 'r' equals to:
		// |R| - n_tensors + n_elem
		GrB_Index nvals;
		info = GrB_Matrix_nvals(&nvals, R);
		ASSERT(info == GrB_SUCCESS);

		info = Delta_Matrix_setM(DR, &R);
		ASSERT(info == GrB_SUCCESS);

		GraphStatistics_IncEdgeCount(&g->stats, r, nvals - n_tensors + n_elem);
	}
}

// decode adjacency matrix
void RdbLoadAdjMatrix_v17
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph context
) {
	// format:
	//   adjacency matrix

	ASSERT (g   != NULL) ;
	ASSERT (rdb != NULL) ;

	GrB_Matrix A = _DecodeMatrix (rdb) ;
	if(A == NULL) return;  // SetError already latched (unarmed safety)

	Delta_Matrix adj = Graph_GetAdjacencyMatrix (g, false) ;
	ASSERT (adj != NULL) ;

	// replace adj's current M matrix with A
	GrB_Info info = Delta_Matrix_setM (adj, &A) ;
	ASSERT (info == GrB_SUCCESS) ;
}

// decode labels matrix
void RdbLoadLblsMatrix_v17
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   lbls matrix

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	GrB_Matrix A = _DecodeMatrix(rdb);
	if(A == NULL) return;  // SetError already latched (unarmed safety)

	Delta_Matrix lbl = Graph_GetNodeLabelMatrix (g) ;
	ASSERT (lbl != NULL) ;

	// replace lbl's current M matrix with A
	GrB_Info info = Delta_Matrix_setM (lbl, &A) ;
	ASSERT (info == GrB_SUCCESS) ;
}
