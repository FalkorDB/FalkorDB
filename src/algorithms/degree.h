/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "GraphBLAS.h"
#include "../graph/graph.h"

enum {
	DEG_DEFAULT = 0,
	DEG_INDEGREE = 1 << 0,
	DEG_OUTDEGREE = 1 << 1,
	DEG_TENSOR = 1 << 2,
} ;

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

GrB_Info TensorDegree  
(
	GrB_Vector degree,  // [input / output] degree vector with values where 
						// the degree should be added
	GrB_Vector dest,  	// [input] possible destination / source nodes
	Tensor T,         	// matrix with tensor entries
	int ops
);

