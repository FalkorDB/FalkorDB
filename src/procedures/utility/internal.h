/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "GraphBLAS.h"
#include "../../graph/graph.h"
#include "value.h"


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

//------------------------------------------------------------------------------
// Matrix_EdgeID - see matrix_edge_id.c for design notes.
//------------------------------------------------------------------------------

// strategy for disambiguating which edge produced a given matrix entry
typedef enum {
	MEID_EQUAL,  // disambiguate by matching the weight value exactly
	MEID_ANY     // grab any edge connecting src -> dest
} MEID_strategy;

// given a matrix of edge weights, recover a matrix of the EdgeIDs that
// produced those weights
GrB_Info Matrix_EdgeID
(
	GrB_Matrix *A_eid,             // [output] matrix of EdgeIDs
	const GrB_Matrix A,            // [input]  matrix of edge weights
	const Graph *g,                // graph
	const RelationID *rels,        // [optional] relationship types
	unsigned short n_rels,         // number of relationship types
	const GrB_Vector rows,         // [optional] compacted-index -> NodeID,
	                               // same vector project_graph_to_matrix
	                               // returns when conf.compact == true
	GRAPH_EDGE_DIR direction,      // direction A was projected with (must
	                               // match the PGTM_config.direction used to
	                               // build A)
	const AttributeID weight,      // weight attribute (MEID_EQUAL only)
	double default_value,          // default weight for attribute-less edges
	MEID_strategy strategy         // MEID_EQUAL / MEID_ANY
) ;
