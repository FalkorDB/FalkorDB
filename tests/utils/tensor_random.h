/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "src/graph/tensor/tensor.h"

// Make a random delta matrix
void Delta_Random_Matrix
(
	Delta_Matrix *A,
	GrB_Type type,
	GrB_Index n,
	double density,
	double add_density,
	double del_density,
	uint64_t seed
) ;

// Make a random tensor
void Random_Tensor
(
	Tensor *A,
	GrB_Index n,
	double density,
	double add_density,
	double del_density,
	uint64_t seed
) ;
