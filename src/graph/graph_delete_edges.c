/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graph.h"
#include "../util/arr.h"
#include "tensor/tensor.h"
#include "delta_matrix/delta_matrix_iter.h"

// qsort compare function
// compare edges by relationship-type, src ID and dest ID
static int _edge_cmp
(
	const void *a,
	const void *b
) {
	Edge *ea = (Edge*)a;
	Edge *eb = (Edge*)b;

	// get edges relationship-type, src and dest node IDs
	NodeID as     = Edge_GetSrcNodeID(ea);   // A's src node ID
	NodeID bs     = Edge_GetSrcNodeID(eb);   // B's src node ID
	NodeID at     = Edge_GetDestNodeID(ea);  // A's dest node ID
	NodeID bt     = Edge_GetDestNodeID(eb);  // B's dest node ID
	RelationID ar = Edge_GetRelationID(ea);  // A's relationship-type
	RelationID br = Edge_GetRelationID(eb);  // B's relationship-type

	// different relationship-type
	if(ar != br) return ar - br;

	// same relationship-type, different source node ID
	if(as != bs) return as - bs;

	// same relationship-type and src node ID
	// compare base on destination node ID
	return at - bt;
}

static void _clear_adj
(
	Graph *g,
	Delta_Matrix ADJ,
	const Edge *e
) {
	RelationID r    = Edge_GetRelationID(e);
	NodeID     src  = Edge_GetSrcNodeID(e);
	NodeID     dest = Edge_GetDestNodeID(e);

	// see if source is connected to destination with additional edges
	// TODO: this is expensive, consider switching to numeric ADJ matrix
	// where ADJ[i, j] = k the number of edges of any type connecting
	// node i to node j, the entry can be dropped once ADJ[i, j] = 0
	GrB_Info info;
	bool connected = false;
	int relationCount = Graph_RelationTypeCount(g);
	for(int ri = 0; ri < relationCount; ri++) {
		if(ri == r) continue;

		Delta_Matrix A = Graph_GetRelationMatrix(g, ri);
		info = Delta_Matrix_extractElement_BOOL(NULL, A, src, dest);
		if(info == GrB_SUCCESS) {
			connected = true;
			break;
		}
	}

	// there are no additional edges connecting source to destination
	// remove edge from THE adjacency matrix
	if(!connected) {
		info = Delta_Matrix_removeElement(ADJ, src, dest);
		ASSERT(info == GrB_SUCCESS);
	}
}

// clears connections from the graph by updating relevent matrices
void Graph_ClearConnections
(
	Graph *g,     // graph to update
	Edge *edges,  // edges to clear
	uint64_t n    // number of edges
) {
	ASSERT(n > 0);
	ASSERT(g     != NULL);
	ASSERT(edges != NULL);

	GrB_Info info;

	// update matrix sync policy to NOP
	MATRIX_POLICY policy = Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);

	// sort edges by:
	// 1. relationship-type
	// 2. src node ID
	// 3. dest node ID
	qsort(edges, n, sizeof(Edge), _edge_cmp);

	// handle each relationship-type seperetly
	for(uint64_t i = 0; i < n;) {
		Edge      *e = edges + i;
		RelationID r = Edge_GetRelationID(e);

		// gather edges by relationship-type
		uint64_t j = i;
		while(Edge_GetRelationID(e) == r) {
			// advance
			if(++j == n) break;

			e = edges + j;
		}

		// in case relationship-type 'r' doesn't contains any vector entries
		// we can improve deletion performance by performin "flat deletion"
		// otherwise the deletion process needs to take into account vectors
		// which is a bit more expenssive
		bool flat_deletion = !Graph_RelationshipContainsMultiEdge(g, r);

		// edge(s) of type r has just been deleted, update statistics
		uint64_t d = j - i;  // number of edges sharing the same relationship
		GraphStatistics_DecEdgeCount(&g->stats, r, d);

		// delete edges[i..j]
		Delta_Matrix R   = Graph_GetRelationMatrix(g, r);
		Delta_Matrix ADJ = Graph_GetAdjacencyMatrix(g);

		if(flat_deletion) {
			// tensor R doesn't contains any vector
			// perform a simple "flat" deletion
			Tensor_RemoveElements_Flat(R, edges + i, d);
			// for each removed edge E see if ADJ[E.src, E.dest] needs clearing
			for (uint64_t k = 0; k < d; k++) {
				e = edges + (i + k);
				_clear_adj(g, ADJ, e);
			}
		} else {
			// tensor R contains vectors
			// perform deletion which handels vector entries
			uint64_t *cleared_entries = NULL;
			Tensor_RemoveElements(R, edges + i, d, &cleared_entries);

			// for each cleared entry E see if ADJ[E.src, E.dest] needs clearing
			uint64_t m = array_len(cleared_entries);
			for (uint k = 0; k < m; k++) {
				e = edges + (i + cleared_entries[k]);
				_clear_adj(g, ADJ, e);
			}

			// free reported cleared entries
			array_free(cleared_entries);
		}

		i = j;
	}

	Graph_SetMatrixPolicy(g, policy);
}

