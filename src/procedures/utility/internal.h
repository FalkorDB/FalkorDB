/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "GraphBLAS.h"
#include "../../graph/graph.h"
#include "value.h"


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
// A = (R0 + R1 + ... Rn) (compressed to only include the rows/cols from L)
//
// if a weight attribute is specified, this function will pick which edge to 
// return given a BWM_reduce_strategy
// for example, BWM_MIN returns the edge with minimum weight
// 
// A_w  = [attribute values of A]
// rows = nodes with specified labels
// in case no labels are specified rows is a dense 1 vector: [1, 1, ...1]
GrB_Info get_sub_weight_matrix
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
	bool symmetric                 // build a symmetric matrix
) ;

// reduction strategy to project graph
typedef enum {
	PROJECT_TO_ANY,  // choose any Edge
	PROJECT_TO_MIN,  // choose the minimum Edge
	PROJECT_TO_MAX   // choose the maximum Edge
} project_strategy;

typedef struct {
	const Graph      *g;           // graph
	const LabelID    *lbls;        // labels to consider
								   // will consider all labels if NULL
	unsigned short    n_lbls;      // number of labels
	const RelationID *rels;        // relationships to consider
								   // will consider all relationships if NULL
	unsigned short    n_rels;      // number of relationships
	AttributeID       edge_weight; // Attribute to use for edge weights. Will
								   // use default_ew, or boolean true if NULL
	SIValue           default_ew;  // Default edge weight. SI_NullVal will
	                               // error if a candidate edge does not have
	                               // the given weight attribute
	AttributeID       node_weight; // Attribute to use for node weights will
								   // use default_nw, or boolean true if NULL
	SIValue           default_nw;  // Default node weight, SI_NullVal will
	                               // error if a candidate node does not have
	                               // the given weight attribute
	project_strategy  strategy;    // strategy for deduping edges
	GRAPH_EDGE_DIR    direction;   // projection direction:
								   // OUTGOING->default
								   // INCOMING->transpose
								   // BOTH->symmetric
	bool              compact;     // if true, return only the rows which were
								   // selected (ie nvals of rows equals nrows of
								   // A)
} PGTM_config ;
// In the default config, all edges and nodes are considered. No weights are
// added, and A is returned as a boolean true matrix.

#define DEFAULT_PGTM_CONFIG (PGTM_config) {                                    \
	.g = NULL, .lbls = NULL, .n_lbls = 0, .rels = NULL, .n_rels = 0,           \
	.edge_weight = ATTRIBUTE_ID_NONE, .default_ew = SI_NullVal(),               \
	.node_weight = ATTRIBUTE_ID_NONE, .default_nw = SI_NullVal(),               \
	.strategy = PROJECT_TO_ANY, .direction = GRAPH_EDGE_DIR_OUTGOING,           \
	.compact = false                                                             \
}

// Make a matrix out of a graph, given an input configuration object
GrB_Info project_graph_to_matrix
(
	GrB_Matrix *A,     // [optional output] matrix weights
	GrB_Vector *rows,  // [optional output] filtered rows
	PGTM_config conf   // input configuration
) ;
