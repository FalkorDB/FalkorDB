/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
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
	if(M == NULL) return;

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(M)){
		Delta_Matrix T = M->transposed;
		M->transposed = NULL;
		GrB_OK (GrB_Matrix_free(&T->matrix));
		GrB_OK (GrB_Matrix_free(&T->delta_plus));
		GrB_OK (GrB_Matrix_free(&T->delta_minus));
	} 

	GrB_OK (GrB_Matrix_free(&M->matrix));
	GrB_OK (GrB_Matrix_free(&M->delta_plus));
	GrB_OK (GrB_Matrix_free(&M->delta_minus));

	pthread_mutex_destroy(&M->mutex);

	rm_free(M);
	
	*C = NULL;
}

