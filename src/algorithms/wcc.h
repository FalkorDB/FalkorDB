/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "LAGraph.h"
#include "GraphBLAS.h"
#include "../graph/graph.h"
#include "../graph/entities/node.h"
#include "../graph/entities/edge.h"

GrB_Info WCC
(
	GrB_Vector *components, // [output] components
	GrB_Vector *N,          // [output] list computed nodes
	const Graph *g,         // graph
	LabelID *lbls,          // [optional] labels to consider
	unsigned short n_lbls,  // number of labels
	RelationID *rels,       // [optional] relationships to consider
	unsigned short n_rels   // number of relationships
);

