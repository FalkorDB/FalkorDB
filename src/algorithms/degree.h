/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "GraphBLAS.h"

// compute in/out degree for all nodes
//
// arguments:
// 'degree' output degree vector degree[i] contains the degree of node i
// 'A' adjacency matrix
//
// returns:
// GrB_SUCCESS on success otherwise a GraphBLAS error
GrB_Info Degree
(
	GrB_Vector *degree,  // [output] degree vector
	GrB_Matrix A         // graph matrix
);

