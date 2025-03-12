/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "../../../serializer_io.h"
#include "../../../../graph/graphcontext.h"
#include "../../../../graph/delta_matrix/delta_matrix.h"

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

	// read blob size
	GrB_Index blob_size = SerializerIO_ReadUnsigned(rdb);
	ASSERT(blob_size > 0);

	// read matrix blob
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

	// get graph's adjacency matrix and its transpose
	Delta_Matrix adj    = Graph_GetAdjacencyMatrix(g, false);
	GrB_Matrix   adj_m  = Delta_Matrix_M(adj);
	GrB_Matrix   adj_tm = Delta_Matrix_M(Delta_Matrix_getTranspose(adj));

	// decode relationship matrices
	for(int i = 0; i < n; i++) {
		// read relation ID
		RelationID r = SerializerIO_ReadUnsigned(rdb);

		// decode matrix
		GrB_Matrix R = _DecodeMatrix(rdb);

		Delta_Matrix DR = Graph_GetRelationMatrix(g, r, false);
		ASSERT(DR != NULL);

		// check if r contains tensors
		GraphDecodeContext *decode_ctx = gc->decoding_context;
		if(decode_ctx->multi_edge[r] == false) {
			// relationship matrix doesn't contains tensors
			// we can simply replace its internal M matrix with R
			info = Delta_Matrix_setM(DR, R);
			ASSERT(info == GrB_SUCCESS);
		} else {
			// clear tensor entries, these will be created via edge load
			GrB_Index nvals;
			info = GrB_Matrix_nvals(&nvals, R);
			ASSERT(info == GrB_SUCCESS);

			GrB_Scalar s;
			info = GrB_Scalar_new(&s, GrB_UINT64);
			ASSERT(info == GrB_SUCCESS);

			// 64bit number with its only MSB set
			info = GrB_Scalar_setElement_UINT64(s, (uint64_t)1 << 63);
			ASSERT(info == GrB_SUCCESS);

			// keep entries R[i,j] with MSB off
			info = GrB_select(R, NULL, NULL, GrB_VALUELT_UINT64, R, s, NULL);
			ASSERT(info == GrB_SUCCESS);

			// free scalar
			info = GrB_free(&s);
			ASSERT(info == GrB_SUCCESS);

			info = GrB_wait(R, GrB_MATERIALIZE);
			ASSERT(info == GrB_SUCCESS);

			// validate number of entries decreased
			GrB_Index new_nvals;
			info = GrB_Matrix_nvals(&nvals, R);
			ASSERT(info == GrB_SUCCESS);
			ASSERT(new_nvals < nvals);

			// update relation matrix
			// DRM = R + DRM
			GrB_Matrix DRM = Delta_Matrix_M(DR);
			info = GrB_Matrix_apply(DRM, NULL, NULL, GrB_IDENTITY_UINT64, R, NULL);
			ASSERT(info == GrB_SUCCESS);
		}
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

