/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "src/graph/tensor/tensor.h"

// Make a random delta matrix
void Delta_Random_Matrix
(
	Delta_Matrix *A,     // delta matrix to be initialized and output
	GrB_Type type,       // type of the matrix
	GrB_Index n,         // dimension of the matrix (nxn)
	double density,      // estimated density of entries in M
	double add_density,  // estimated density of entries in DP
	double del_density,  // estimated density of entries in DM
	uint64_t seed        // seed to be used for generating the matrix
) ;

// Make a random tensor
void Random_Tensor
(
	Tensor *A,           // Tensor to allocate and add rendom entries to
	GrB_Index n,         // dimension of Tensor (nxn)
	double density,      // estimated density of edges in M
	double add_density,  // estimated density of edges in DP
	double del_density,  // estimated density of edges in DM
	uint64_t seed        // random seed
) ;
