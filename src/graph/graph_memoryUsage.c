/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "graph_memoryUsage.h"

// get graph's memory usage
void Graph_memoryUsage
(
	const Graph *g,            // graph
	MemoryUsageResult *result  // [output] memory usage
) {
	ASSERT(g      != NULL);
	ASSERT(result != NULL);

	size_t n = 0;  // matrix memory consumption

	Tensor       T;
	Delta_Matrix D;
	GrB_Info     info;

	//--------------------------------------------------------------------------
	// graph's adjacency matrix
	//--------------------------------------------------------------------------

	D = Graph_GetAdjacencyMatrix(g, false);

	info = Delta_Matrix_memoryUsage(&n, D);
	ASSERT(info == GrB_SUCCESS);

	result->rel_matrices_sz += n;

	D = Graph_GetAdjacencyMatrix(g, true);

	info = Delta_Matrix_memoryUsage(&n, D);
	ASSERT(info == GrB_SUCCESS);

	result->rel_matrices_sz += n;

	//--------------------------------------------------------------------------
	// graph's label matrices
	//--------------------------------------------------------------------------

	int n_lbl = Graph_LabelTypeCount(g);

	for(LabelID lbl = 0; lbl < n_lbl; lbl++) {
		D = Graph_GetLabelMatrix(g, lbl);

		info = Delta_Matrix_memoryUsage(&n, D);
		ASSERT(info == GrB_SUCCESS);

		result->lbl_matrices_sz += n;
	}

	// account for graph's node labels matrix
	D = Graph_GetNodeLabelMatrix(g);

	info = Delta_Matrix_memoryUsage(&n, D);
	ASSERT(info == GrB_SUCCESS);

	result->lbl_matrices_sz += n;

	//--------------------------------------------------------------------------
	// graph's relation matrices
	//--------------------------------------------------------------------------

	int n_rel = Graph_RelationTypeCount(g);
	for(RelationID rel = 0; rel < n_rel; rel++) {
		T = Graph_GetRelationMatrix(g, rel, false);

		info = Delta_Matrix_memoryUsage(&n, T);
		ASSERT(info == GrB_SUCCESS);

		result->rel_matrices_sz += n;
	}

	//--------------------------------------------------------------------------
	// graph's datablocks
	//--------------------------------------------------------------------------

	result->node_storage_sz += DataBlock_memoryUsage(g->nodes);
	result->edge_storage_sz += DataBlock_memoryUsage(g->edges);
}

