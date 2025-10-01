#include "RG.h"
#include "delta_matrix.h"

// replace C's internal M matricies with given
// the operation can only succeed if C's interal matrices:
// M, DP, DM are all empty
// C->M will point to *M and *M will be set to NULL
GrB_Info Delta_Matrix_setMatrices
(
	Delta_Matrix C,  // delta matrix
	GrB_Matrix M,    // new M
	GrB_Matrix DP,   // new delta-plus
	GrB_Matrix DM    // new delta-minus
) {
	ASSERT(C  != NULL);
	ASSERT(M  != NULL);
	ASSERT(DP != NULL);
	ASSERT(DM != NULL);

	GrB_Index nvals = 0;

	// Verify that C is empty
	ASSERT(Delta_Matrix_Synced(C));
	Delta_Matrix_nvals(&nvals, C);
	ASSERT(nvals == 0);

	GrB_OK(GrB_free(&DELTA_MATRIX_M(C)));
	GrB_OK(GrB_free(&DELTA_MATRIX_DELTA_PLUS(C)));
	GrB_OK(GrB_free(&DELTA_MATRIX_DELTA_MINUS(C)));

	DELTA_MATRIX_M(C)           = M;
	DELTA_MATRIX_DELTA_PLUS(C)  = DP;
	DELTA_MATRIX_DELTA_MINUS(C) = DM;

	return GrB_SUCCESS;
}

GrB_Info Delta_Matrix_setM
(
	Delta_Matrix C,  // delta matrix
	GrB_Matrix M     // new M
) {
	GrB_Index nvals = 0;
	GrB_Index tot   = 0;
	GrB_Info info   = GrB_Matrix_nvals(&nvals, DELTA_MATRIX_M(C));
	ASSERT(info == GrB_SUCCESS);
	tot |= nvals;
	info = GrB_Matrix_nvals(&nvals, DELTA_MATRIX_DELTA_PLUS(C));
	ASSERT(info == GrB_SUCCESS);
	tot |= nvals;
	info = GrB_Matrix_nvals(&nvals, DELTA_MATRIX_DELTA_MINUS(C));
	ASSERT(info == GrB_SUCCESS);
	tot |= nvals;
	if (tot != 0)
		return GrB_ALREADY_SET;
	info = GrB_free(&DELTA_MATRIX_M(C));
	ASSERT(info == GrB_SUCCESS);

	DELTA_MATRIX_M(C) = M;
	return GrB_SUCCESS;
}

const GrB_Matrix Delta_Matrix_M
(
	const Delta_Matrix C  // delta matrix
) {
	return DELTA_MATRIX_M(C);
}

const GrB_Matrix Delta_Matrix_DP
(
	const Delta_Matrix C  // delta matrix
) {
	return DELTA_MATRIX_DELTA_PLUS(C);
}

const GrB_Matrix Delta_Matrix_DM
(
	const Delta_Matrix C  // delta matrix
) {
	return DELTA_MATRIX_DELTA_MINUS(C);
}