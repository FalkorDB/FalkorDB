/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "RG.h"
#include "delta_matrix.h"

#if RG_DEBUG
#define GRB_MATRIX_TYPE_ASSERT(M, TYPE)  \
do {                                     \
	GrB_Type _ty = NULL;                 \
	GrB_OK(GxB_Matrix_type(&_ty, M));    \
	ASSERT(_ty == TYPE);                 \
} while(0);
#else
#define GRB_MATRIX_TYPE_ASSERT(M, TYPE)
#endif

typedef enum
{
	VAL_BASIC,
	VAL_T_SHORT,
	VAL_T_FULL
} DM_validation_level;

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
	DM_validation_level lvl
) ;

