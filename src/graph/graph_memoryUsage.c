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
	ASSERT (g      != NULL) ;
	ASSERT (result != NULL) ;

	size_t n = 0 ;  // matrix memory consumption

	Tensor       T ;
	Delta_Matrix D ;
	GrB_Info     info ;

	//--------------------------------------------------------------------------
	// graph's adjacency matrix
	//--------------------------------------------------------------------------

	D = Graph_GetAdjacencyMatrix (g, false) ;

	GrB_OK (Delta_Matrix_memoryUsage (&n, D)) ;

	result->rel_matrices_sz += n ;

	//--------------------------------------------------------------------------
	// graph's label matrices
	//--------------------------------------------------------------------------

	int n_lbl = Graph_LabelTypeCount (g) ;

	for (LabelID lbl = 0 ; lbl < n_lbl ; lbl++) {
		D = Graph_GetLabelMatrix (g, lbl) ;

		GrB_OK (Delta_Matrix_memoryUsage (&n, D)) ;

		result->lbl_matrices_sz += n ;
	}

	// account for graph's node labels matrix
	D = Graph_GetNodeLabelMatrix (g) ;

	GrB_OK (Delta_Matrix_memoryUsage (&n, D)) ;

	result->lbl_matrices_sz += n ;

	//--------------------------------------------------------------------------
	// graph's relation matrices
	//--------------------------------------------------------------------------

	int n_rel = Graph_RelationTypeCount (g) ;
	for (RelationID rel = 0; rel < n_rel; rel++) {
		T = Graph_GetRelationMatrix (g, rel, false) ;

		if (Graph_RelationshipContainsMultiEdge(g, rel)) {
			GrB_OK (Tensor_memoryUsage (&n, T)) ;
		} else {
			GrB_OK (Delta_Matrix_memoryUsage (&n, T)) ;
		}

		result->rel_matrices_sz += n ;
	}

	//--------------------------------------------------------------------------
	// graph's zero matrix
	//--------------------------------------------------------------------------
	D = Graph_GetZeroMatrix (g) ;

	GrB_OK (Delta_Matrix_memoryUsage (&n, D)) ;

	result->rel_matrices_sz += n ;

	//--------------------------------------------------------------------------
	// graph's datablocks
	//--------------------------------------------------------------------------

	result->node_block_storage_sz = DataBlock_memoryUsage (g->nodes) ;
	result->edge_block_storage_sz = DataBlock_memoryUsage (g->edges) ;
}

