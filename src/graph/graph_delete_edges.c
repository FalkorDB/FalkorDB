/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graph.h"
#include "GraphBLAS.h"
#include "../util/arr.h"
#include "tensor/tensor.h"
#include "delta_matrix/delta_matrix_iter.h"
#include "graph/delta_matrix/delta_matrix.h"

#include <stdint.h>

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

// clears connections from the graph by updating relevent matrices
void Graph_ClearConnections
(
	Graph *g,     // graph to update
	Edge *edges,  // edges to clear
	uint64_t n    // number of edges
) {
	ASSERT (n > 0) ;
	ASSERT (g     != NULL) ;
	ASSERT (edges != NULL) ;

	GrB_Info info ;

	// update matrix sync policy to NOP
	MATRIX_POLICY policy = Graph_SetMatrixPolicy (g, SYNC_POLICY_NOP) ;

	Delta_Matrix ADJ = Graph_GetAdjacencyMatrix (g, false) ;
	uint64_t *deleted_entries = array_new (uint64_t, n / 4) ;

	// sort edges by:
	// 1. relationship-type
	// 2. src node ID
	// 3. dest node ID
	qsort (edges, n, sizeof (Edge), _edge_cmp) ;

	// handle each relationship-type seperetly
	for (uint64_t i = 0; i < n;) {
		Edge      *e = edges + i ;
		RelationID r = Edge_GetRelationID (e) ;

		// gather edges by relationship-type
		uint64_t j = i ;
		while (Edge_GetRelationID (e) == r) {
			// advance
			if (++j == n) {
				break ;
			}

			e = edges + j ;
		}

		// in case relationship-type 'r' doesn't contains any vector entries
		// we can improve deletion performance by performin "flat deletion"
		// otherwise the deletion process needs to take into account vectors
		// which is a bit more expenssive
		bool flat_deletion = !Graph_RelationshipContainsMultiEdge (g, r) ;

		// edge(s) of type r has just been deleted, update statistics
		uint64_t d = j - i ;  // number of edges sharing the same relationship
		GraphStatistics_DecEdgeCount (&g->stats, r, d) ;

		// delete edges[i..j]
		Delta_Matrix R = Graph_GetRelationMatrix  (g, r, false) ;

		if (flat_deletion) {
			// tensor R doesn't contains any vector
			// perform a simple "flat" deletion
			Tensor_RemoveElements_Flat (R, edges + i, d) ;
		} else {
			// tensor R contains vectors
			// perform deletion which handels vector entries
			Tensor_RemoveElements (R, edges + i, d, NULL) ;
		}

		// for each cleared entry E see if ADJ[E.src, E.dest] needs clearing
		// TODO: could take advantage of this being sorted to reduce
		// elements by more than one at a time (for tensors).

		// NOTE: nothing in this for loop actually causes pending changes
		for (uint k = 0; k < d; k++) {
			e = edges + (i + k);
			uint16_t v = 0;

			NodeID src  = Edge_GetSrcNodeID  (e) ;
			NodeID dest = Edge_GetDestNodeID (e) ;
			GrB_OK (Delta_Matrix_extractElement_UINT16 (&v, ADJ, src, dest)) ;
			ASSERT (v > 0) ;

			if (--v == 0) {
				array_append (deleted_entries, (uint64_t) src) ;
				array_append (deleted_entries, (uint64_t) dest) ;
			}

			GrB_OK (Delta_Matrix_setElement_UINT16 (ADJ, v, src, dest)) ;
		}

		i = j ;  // skip processed batch
	}

	// if any slots have been left with zero edges, remove them from ADJ
	// NOTE: done like this to reduce read after writes
	uint64_t m = array_len (deleted_entries) ;
	for (uint64_t k = 0; k < m; k++) {
		NodeID src  = deleted_entries[k] ;
		NodeID dest = deleted_entries[++k] ;
		GrB_OK (Delta_Matrix_removeElement (ADJ, src, dest)) ;
	}

	array_free (deleted_entries) ;

	// restore sync policy
	Graph_SetMatrixPolicy (g, policy) ;
}

