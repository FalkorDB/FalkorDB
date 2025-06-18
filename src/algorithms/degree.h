/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "GraphBLAS.h"
#include "../graph/graph.h"

typedef enum {
	DEG_INDEGREE = 1 << 0,
	DEG_OUTDEGREE = 1 << 1,
	DEG_TENSOR = 1 << 2,
} Degree_Options;

// Compute the in or out degree of each node in degree
//
// arguments:
// 'degree' input / output degree vector. Degree of node i is added to degree[i]
// 'dest' input vector. Boolean vector with entries to be counted for degree.
// 'T' Tensor being used to find the degree.
// 'ops' input. DEG_[OUT/IN]DEGREE: compute [out/in]degree. 
// 				DEG_TENSOR: compute tensor degree
// returns:
// GrB_SUCCESS on success otherwise a GraphBLAS error
GrB_Info TensorDegree  
(
	GrB_Vector degree,  // [input / output] degree vector with values where 
						//         the degree should be added
	GrB_Vector dest,    // [input] possible destination / source nodes
	Tensor T,           // matrix with tensor entries
	Degree_Options ops
) ;

typedef struct{
    const Graph *g;         // Graph
    AttributeID attribute;  // Weight Attribute to consider
} FDB_degree_ctx ;

// Compute the in or out degree of each node in degree
//
// arguments:
// 'degree' input / output degree vector. Degree of node i is added to degree[i]
//    degree is a double value, the sum of the weights of adjacent relationships
// 'dest' input vector. Boolean vector with entries to be counted for degree.
// 'T' Tensor being used to find the degree.
// 'ops' input. DEG_[OUT/IN]DEGREE: compute [out/in]degree. 
// 				DEG_TENSOR: compute tensor degree
// returns:
// GrB_SUCCESS on success otherwise a GraphBLAS error
GrB_Info TensorDegree_weighted
(
	GrB_Vector degree,  // [input / output] degree vector with values where 
						//         the degree should be added
	GrB_Vector dest,    // [input] possible destination / source nodes
	Tensor T,           // matrix with tensor entries
	Degree_Options ops,
	FDB_degree_ctx ctx
) ;

