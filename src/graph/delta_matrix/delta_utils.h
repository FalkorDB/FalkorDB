/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "delta_matrix.h"

void Delta_Matrix_checkBounds
(
	const Delta_Matrix C,
	GrB_Index i,
	GrB_Index j
);

void Delta_Matrix_checkCompatible
(
	const Delta_Matrix M,
	const Delta_Matrix N
);

// validate 'C' isn't in an invalid state
void Delta_Matrix_validateState
(
	const Delta_Matrix C,
	GrB_Index i,
	GrB_Index j
);

// Check every assumption for the Delta Matrix
//         ∅ = m  ∩ dp
//         ∅ = dp ∩ dm
// {zombies} = m  ∩ dm
// Transpose
//    Check it is actually M^T
// Types / Dimensions
//    m BOOL / UINT64
//    dp BOOL / UINT64
//    dm BOOL
void Delta_Matrix_validate
(
	const Delta_Matrix C,
	bool check_transpose
);

#define Delta_Matrix_print(C, p)                                          \
{                                                                         \
	GxB_Matrix_fprint(DELTA_MATRIX_M(C), #C, p, stdout);                  \
	GxB_Matrix_fprint(DELTA_MATRIX_DELTA_PLUS(C), #C "-DP", p, stdout);   \
	GxB_Matrix_fprint(DELTA_MATRIX_DELTA_MINUS(C), #C "-DM", p, stdout);  \
}
