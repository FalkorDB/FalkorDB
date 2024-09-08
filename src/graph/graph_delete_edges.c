/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graph.h"
#include "../util/arr.h"
#include "delta_matrix/delta_matrix_iter.h"

// removes edges from Graph and updates graph relevant matrices
//
// edge deletion is performed in two steps
// 1. simple edge deletion
//    each edge is removed from the relation matrix if it is the
//    only edge check if need to remove from the adjacency matrix
//    otherwise delete it from the multi-edge matrix
// 2. update multi-edge state
//    each edge that was connected as multi edge check if we can
//    transform it to single edge or delete it completly
void Graph_DeleteEdges
(
	Graph *g,
	Edge *edges,
	uint64_t n
) {
	ASSERT(g != NULL);
	ASSERT(n > 0);
	ASSERT(edges != NULL);

	GrB_Info       info;
	Delta_Matrix   M;
	Delta_Matrix   E;

	MATRIX_POLICY policy = Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);

	// delete edges without considering multi edge state changes
	for (uint i = 0; i < n; i++) {
		Edge       *e         =  edges + i;
		int         r         =  Edge_GetRelationID(e);
		NodeID      src_id    =  Edge_GetSrcNodeID(e);
		NodeID      dest_id   =  Edge_GetDestNodeID(e);
		EdgeID      edge_id   =  ENTITY_GET_ID(e);

		ASSERT(!DataBlock_ItemIsDeleted((void *)e->attributes));

		// an edge of type r has just been deleted, update statistics
		GraphStatistics_DecEdgeCount(&g->stats, r, 1);

		M = Graph_GetRelationMatrix(g, r, false);

		GrB_Index me_id;
		info = Delta_Matrix_extractElement_UINT64(&me_id, M, src_id, dest_id);
		ASSERT(info == GrB_SUCCESS);

		if(SINGLE_EDGE(me_id)) {
			info = Delta_Matrix_removeElement(M, src_id, dest_id);
			ASSERT(info == GrB_SUCCESS);
			ASSERT(me_id == edge_id);

			// see if source is connected to destination with additional edges
			bool connected = false;
			int relationCount = Graph_RelationTypeCount(g);
			for(int j = 0; j < relationCount; j++) {
				if(j == r) continue;
				Delta_Matrix r = Graph_GetRelationMatrix(g, j, false);
				info = Delta_Matrix_extractElement_BOOL(NULL, r, src_id, dest_id);
				if(info == GrB_SUCCESS) {
					connected = true;
					break;
				}
			}

			// there are no additional edges connecting source to destination
			// remove edge from THE adjacency matrix
			if(!connected) {
				Delta_Matrix adj = Graph_GetAdjacencyMatrix(g, false);
				info = Delta_Matrix_removeElement(adj, src_id, dest_id);
				ASSERT(info == GrB_SUCCESS);
			}
		} else {
			E  = Graph_GetMultiEdgeRelationMatrix(g, r);
			me_id = CLEAR_MSB(me_id);
			info = Delta_Matrix_removeElement(E, me_id, edge_id);
			ASSERT(info == GrB_SUCCESS);
		}

		// free and remove edges from datablock.
		DataBlock_DeleteItem(g->edges, edge_id);
	}

	// check if multi edge can be transformed to single edge or deleted completely
	for (uint i = 0; i < n; i++) {
		Edge       *e         =  edges + i;
		int         r         =  Edge_GetRelationID(e);
		NodeID      src_id    =  Edge_GetSrcNodeID(e);
		NodeID      dest_id   =  Edge_GetDestNodeID(e);
		EdgeID      edge_id   =  ENTITY_GET_ID(e);

		M = Graph_GetRelationMatrix(g, r, false);

		GrB_Index id;
		info = Delta_Matrix_extractElement_UINT64(&id, M, src_id, dest_id);
		if(info != GrB_SUCCESS || SINGLE_EDGE(id)) continue;

		E  = Graph_GetMultiEdgeRelationMatrix(g, r);
		id = CLEAR_MSB(id);
		Delta_MatrixTupleIter it = {0};
		Delta_MatrixTupleIter_AttachRange(&it, E, id, id);
		GrB_Index last_edge_id;
		uint count = 0;
		while (Delta_MatrixTupleIter_next_BOOL(&it, NULL, &last_edge_id, NULL) == GrB_SUCCESS) {
			count++;
			if(count == 2) break;
		}

		if(count == 0) {
			info = Delta_Matrix_removeElement(M, src_id, dest_id);
			ASSERT(info == GrB_SUCCESS);

			// see if source is connected to destination with additional edges
			bool connected = false;
			int relationCount = Graph_RelationTypeCount(g);
			for(int j = 0; j < relationCount; j++) {
				if(j == r) continue;
				Delta_Matrix r = Graph_GetRelationMatrix(g, j, false);
				info = Delta_Matrix_extractElement_BOOL(NULL, r, src_id, dest_id);
				if(info == GrB_SUCCESS) {
					connected = true;
					break;
				}
			}

			// there are no additional edges connecting source to destination
			// remove edge from THE adjacency matrix
			if(!connected) {
				Delta_Matrix adj = Graph_GetAdjacencyMatrix(g, false);
				info = Delta_Matrix_removeElement(adj, src_id, dest_id);
				ASSERT(info == GrB_SUCCESS);
			}
		} else if(count == 1) {
			Delta_Matrix_removeElement(E, id, last_edge_id);
			Delta_Matrix_setElement_UINT64(M, last_edge_id, src_id, dest_id);
			array_append(g->relations[r]->freelist, id);
		}
	}

	Graph_SetMatrixPolicy(g, policy);
}
