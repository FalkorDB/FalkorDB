/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"
#include "../entities/graph_entity.h"



// free RG_Matrix's internal matrices:
// M, delta-plus, delta-minus and transpose
void Delta_Matrix_free
(
	Delta_Matrix *C
) {
	ASSERT(C != NULL);
	Delta_Matrix M = *C;

	GrB_Info info;
	UNUSED(info);

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(M)) Delta_Matrix_free(&M->transposed);

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

