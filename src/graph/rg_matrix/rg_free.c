/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "rg_matrix.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"
#include "../entities/graph_entity.h"

// free multi-edge arrays GraphBLAS unary operation
static GrB_UnaryOp free_multi_edge_op = NULL;

// unary GraphBLAS operation which frees all multi-edge array entries
// within a matrix
static void free_multiedge_array(void *out, const void *in) {
	EdgeID *_out = (EdgeID *)out;
	const EdgeID *id = (const EdgeID *)in;

	if(!(SINGLE_EDGE(*id))) {
		// entry is a pointer to dynamic array, free it
		EdgeID *ids = (EdgeID *)(CLEAR_MSB(*id));
		array_free(ids);
	}

	// set out to 0
	*_out = 0;
}

// free RG_Matrix's internal matrices:
// M, delta-plus, delta-minus and transpose
void RG_Matrix_free
(
	RG_Matrix *C
) {
	ASSERT(C != NULL);
	RG_Matrix M = *C;

	GrB_Info info;
	UNUSED(info);

	if(RG_MATRIX_MAINTAIN_TRANSPOSE(M)) RG_Matrix_free(&M->transposed);

	GrB_Matrix m  = RG_MATRIX_M(M);
	GrB_Matrix dp = RG_MATRIX_DELTA_PLUS(M);

	// free edges
	if(RG_MATRIX_MULTI_EDGE(M)) {
		if(free_multi_edge_op == NULL) {
			// create unary operation
			info = GrB_UnaryOp_new(&free_multi_edge_op, free_multiedge_array, 
				GrB_UINT64, GrB_UINT64);
			ASSERT(info == GrB_SUCCESS);
		}

		// frees multi-edge arrays
		info = GrB_Matrix_apply(m, NULL, NULL, free_multi_edge_op, m, NULL);
		ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_apply(dp, NULL, NULL, free_multi_edge_op, dp, NULL);
		ASSERT(info == GrB_SUCCESS);
	}

	info = GrB_Matrix_free(&M->matrix);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_free(&M->delta_plus);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_free(&M->delta_minus);
	ASSERT(info == GrB_SUCCESS);

	pthread_mutex_destroy(&M->mutex);

	rm_free(M);
	
	*C = NULL;
}

