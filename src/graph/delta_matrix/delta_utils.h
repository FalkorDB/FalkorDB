/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
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

// check if the dimensions of C, A and B are compatible for addition
void Delta_Matrix_addCompatible
(
	const Delta_Matrix C,
	const Delta_Matrix A,
	const Delta_Matrix B
) ;

// check if the dimensions of C, A and B are compatible for multiplication
void Delta_Matrix_mulCompatible
(
	const Delta_Matrix C,
	const Delta_Matrix A,
	const Delta_Matrix B
) ;

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
//         m \superset dm
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
) ;
