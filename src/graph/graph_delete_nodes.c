/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graph.h"
#include "../util/arr.h"
#include "delta_matrix/delta_matrix_iter.h"

// deletes nodes from the graph
//
// nodes deletion is performed in two steps
// 1. update label matrices
//    each deleted node is removed from all applicable label matrices
//    suppose node N with internal ID 9 is labeld with labels 0 and 4
//    in which case entry [9,9] is cleared from both label matrix 0 and 4
//    to determine which labels are associated with a given node we consult
//    with the labels-matrix we don't want to alter this matrix at this
//    phase as we're constantly querying it.
//    lastly each inspected entry e.g. [9,0] and [9,4]
//    is collected within a temporary matrix used in the second phase
//
// 2. update labels-matrix
//    using the temporary matrix we clear all relevant entries from the
//    labels-matrix
//
// this seperation to two phases avoids multiple flushes of labels-matrix

void Graph_DeleteNodes
(
	Graph *g,       // graph to delete nodes from
	Node *nodes,    // nodes to delete
	uint64_t count  // number of nodes
) {
	// assumption, nodes are detached
	// there are no incoming nor outgoing edges leading to / from nodes
	ASSERT(g != NULL);
	ASSERT(count > 0);
	ASSERT(nodes != NULL);

	// set matrix sync policy to NOP
	MATRIX_POLICY policy = Graph_GetMatrixPolicy(g);
	Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);

#if RG_DEBUG
	Edge *es = array_new(Edge, 0);
	for(uint i = 0; i < count; i++) {
		Node *n = nodes + i;
		// validate assumption
		Graph_GetNodeEdges(g, n, GRAPH_EDGE_DIR_BOTH, GRAPH_NO_RELATION, &es);
		ASSERT(array_len(es) == 0);
	}
	array_free(es);
#endif

	//--------------------------------------------------------------------------
	// update label matrices
	//--------------------------------------------------------------------------

	GrB_Index j;               // iterated entry col idx
	GrB_Info info;             // GraphBLAS return code
	GrB_Index nrows;           // lbls row count
	GrB_Index ncols;           // lbls col count
	// GrB_Matrix elems;          // elements to delete
	Delta_MatrixTupleIter it;  // matrix iterator

	// get labels matrix
	Delta_Matrix lbls = Graph_GetNodeLabelMatrix(g);

	// create lbls mask
	info = Delta_Matrix_nrows(&nrows, lbls);
	ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_ncols(&ncols, lbls);
	ASSERT(info == GrB_SUCCESS);
	// info = GrB_Matrix_new(&elems, GrB_BOOL, nrows, ncols);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// phase one
	//--------------------------------------------------------------------------

	// iterate over each lbls row coresponding to a deleted node
	// and clear relevant label matrices
	for(uint i = 0; i < count; i++) {
		Node *n = nodes + i;
		EntityID id = ENTITY_GET_ID(n);

		info = Delta_MatrixTupleIter_AttachRange(&it, lbls, id, id);
		ASSERT(info == GrB_SUCCESS);

		// for each deleted node label
		while(Delta_MatrixTupleIter_next_BOOL(&it, NULL, &j, NULL) == GrB_SUCCESS) {
			// populate lbls mask
			// info = GrB_Matrix_setElement_BOOL(elems, true, id, j);
			info = Delta_Matrix_removeElement_BOOL(lbls, id, j);
			ASSERT(info == GrB_SUCCESS);

			// clear label matrix j at position [id,id]
			Delta_Matrix L = Graph_GetLabelMatrix(g, j);
			info = Delta_Matrix_removeElement_BOOL(L, id, id);
			ASSERT(info == GrB_SUCCESS);

			// a label was removed from node, update statistics
			GraphStatistics_DecNodeCount(&g->stats, j, 1);
		}

		// remove node from datablock
		DataBlock_DeleteItem(g->nodes, id);
	}

	//--------------------------------------------------------------------------
	// phase two
	//--------------------------------------------------------------------------

	// Delta_Matrix_removeElements(lbls, elems);

	// restore matrix sync policy
	Graph_SetMatrixPolicy(g, policy);

	// GrB_free(&elems);
}

