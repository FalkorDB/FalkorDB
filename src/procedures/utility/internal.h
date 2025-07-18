/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "GraphBLAS.h"
#include "../../graph/graph.h"


// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = L * (R0 U R1 U ... Rn) * L
//
// rows = L's main diagonal
// in case no labels are specified rows is a dense 1 vector: [1,1,...1]
GrB_Info Build_Matrix
(
	GrB_Matrix *A,           // [output] matrix
	GrB_Vector *rows,        // [output] filtered rows
	const Graph *g,          // graph
	const LabelID *lbls,     // [optional] labels to consider
	unsigned short n_lbls,   // number of labels
	const RelationID *rels,  // [optional] relationships to consider
	unsigned short n_rels,   // number of relationships
	bool symmetric,          // build a symmetric matrix
	bool compact             // remove unused row & columns
);

// reduction strategy to be used by build weighted matrix
typedef enum {
	BWM_MIN,  // choose the minimum Edge 
	BWM_MAX   // choose the maximum Edge
} BWM_reduce_strategy;

// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = L * (R0 + R1 + ... Rn) * L 
//
// if a weight attribute is specified, this function will pick which Edge to 
// return given a BWM_reduce strategy
// for example, BWM_MIN returns the edge with minimum weight
// 
// A_w  = [Attribute values of A]
// rows = L's main diagonal
// in case no labels are specified rows is a dense 1 vector: [1,1,...1]
GrB_Info Build_Weighted_Matrix
(
	GrB_Matrix *A,                 // [output] matrix (EdgeIDs)
	GrB_Matrix *A_w,               // [output] matrix (weights)
	GrB_Vector *rows,              // [output] filtered rows
	const Graph *g,                // graph
	const LabelID *lbls,           // [optional] labels to consider
	unsigned short n_lbls,         // number of labels
	const RelationID *rels,        // [optional] relationships to consider
	unsigned short n_rels,         // number of relationships
	const AttributeID weight,      // weight attribute to consider
	BWM_reduce_strategy strategy,  // use either maximum or minimum weight
	bool symmetric,                // build a symmetric matrix
	bool compact                   // remove unused row & columns
);

