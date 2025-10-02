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

// validate 'C' isn't in an invalid state
void Delta_Matrix_validateState
(
	const Delta_Matrix C,
	GrB_Index i,
	GrB_Index j
);

