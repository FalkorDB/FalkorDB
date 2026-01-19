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
	Edge *ea = (Edge*)a ;
	Edge *eb = (Edge*)b ;

	// get edges relationship-type, src and dest node IDs
	NodeID as     = Edge_GetSrcNodeID  (ea) ;  // A's src node ID
	NodeID bs     = Edge_GetSrcNodeID  (eb) ;  // B's src node ID
	NodeID at     = Edge_GetDestNodeID (ea) ;  // A's dest node ID
	NodeID bt     = Edge_GetDestNodeID (eb) ;  // B's dest node ID
	RelationID ar = Edge_GetRelationID (ea) ;  // A's relationship-type
	RelationID br = Edge_GetRelationID (eb) ;  // B's relationship-type

	// different relationship-type
	if (ar != br) {
		return ar - br ;
	}

	// same relationship-type, different source node ID
	if (as != bs) {
		return as - bs ;
	}

	// same relationship-type and src node ID
	// compare base on destination node ID
	return at - bt ;
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

		Delta_Matrix A = Graph_GetRelationMatrix(g, ri, false);
		info = Delta_Matrix_isStoredElement(A, src, dest);
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

// delete edges implicitly
// as a result of a query e.g. MATCH (n) DELETE n
// all edges associated with n (incoming / outgoing) are deleted implicitly
// unlike MATCH ()-[e]->() DELETE e
//
// implict deletion allows up to perform faster update to the underline matrices
// as the entire entries are removed
static void Graph_ImplicitClearConnections
(
	Graph *g,     // graph to update
	Edge *edges,  // edges to clear
	uint64_t n    // number of edges
) {
	ASSERT (n > 0) ;
	ASSERT (g     != NULL) ;
	ASSERT (edges != NULL) ;

	GrB_Info info ;

	//--------------------------------------------------------------------------
	// create mask matrix
	//--------------------------------------------------------------------------
	
	GrB_Matrix M;
	GrB_Matrix MT = NULL ;

	// update matrix sync policy to NOP
	MATRIX_POLICY policy = Graph_SetMatrixPolicy (g, SYNC_POLICY_NOP) ;

	int relation_count = Graph_RelationTypeCount (g) ;

	// track number of edges deleted from each relation
	uint64_t *encountered_relations =
		rm_calloc (relation_count, sizeof (uint64_t)) ;

    GrB_Index *_I = rm_malloc (sizeof (GrB_Index) * n) ; 
    GrB_Index *_J = rm_malloc (sizeof (GrB_Index) * n) ;  

	for (uint64_t i = 0; i < n; i++) {
		Edge *e = edges + i ;

		encountered_relations[Edge_GetRelationID (e)] += 1 ;

		_I[i] = e->src_id;
		_J[i] = e->dest_id;
	}

	GrB_Scalar s ;
	GrB_OK (GrB_Scalar_new (&s, GrB_BOOL)) ;
	GrB_OK (GrB_Scalar_setElement (s, true)) ;

	GrB_Index m = Graph_RequiredMatrixDim (g) ;
	GrB_OK (GrB_Matrix_new (&M, GrB_BOOL, m, m)) ;
	GrB_OK (GxB_Matrix_build_Scalar (M, _I, _J, s, n)) ;

	GrB_OK (GrB_Matrix_new (&MT, GrB_BOOL, m, m)) ;
	GrB_OK (GrB_transpose (MT, NULL, NULL, M, NULL)) ;

	GxB_print (M, GxB_COMPLETE_VERBOSE) ;
	GxB_print (MT, GxB_COMPLETE_VERBOSE) ;

	for (int r = 0; r < relation_count; r++) {
		uint64_t n_deletions = encountered_relations[r] ;

		// no edges associated with current relationship-type, skip
		if (n_deletions == 0) {
			continue ;
		}

		// get relationship matrix
		Delta_Matrix R = Graph_GetRelationMatrix  (g, r, false) ;

		// in case relationship-type 'r' doesn't contains any vector entries
		// we can improve deletion performance by performin "flat deletion"
		// otherwise the deletion process needs to take into account vectors
		// which is a bit more expenssive
		bool flat_deletion = !Graph_RelationshipContainsMultiEdge (g, r) ;

		if (flat_deletion) {
			// tensor R doesn't contains any vector
			// perform a simple "flat" deletion
			Delta_Matrix_fprint (R, GxB_COMPLETE_VERBOSE, stdout) ;
			GrB_OK (Delta_Matrix_removeElements (R, M, MT)) ;
		} else {
			Tensor_ClearElements (R, M, MT) ;
		}

		// edge(s) of type r has just been deleted, update statistics
		GraphStatistics_DecEdgeCount (&g->stats, r, n_deletions) ;
	}

	// fast ADJ update
	Delta_Matrix ADJ = Graph_GetAdjacencyMatrix (g, false) ;

	GrB_Matrix adj_m  = Delta_Matrix_M  (ADJ) ;
	GrB_Matrix adj_dp =	Delta_Matrix_DP (ADJ) ;
	GrB_Matrix adj_dm =	Delta_Matrix_DM (ADJ) ;

	GxB_fprint (adj_m,  GxB_SHORT, stdout) ;
	GxB_fprint (adj_dp, GxB_SHORT, stdout) ;
	GxB_fprint (adj_dm, GxB_SHORT, stdout) ;

	GrB_OK (Delta_Matrix_removeElements (ADJ, M, MT)) ;

	// clean up
	GrB_OK (GrB_free (&s)) ;
	rm_free (_I) ;
	rm_free (_J) ;

	GrB_free (&M) ;
	GrB_free (&MT) ;
	rm_free (encountered_relations) ;

	Graph_SetMatrixPolicy (g, policy) ;
}

// clears connections from the graph by updating relevent matrices
void Graph_ClearConnections
(
	Graph *g,      // graph to update
	Edge *edges,   // edges to clear
	uint64_t n,    // number of edges
	bool implicit  // edge deleted due to node deletion
) {
	ASSERT (n > 0) ;
	ASSERT (g     != NULL) ;
	ASSERT (edges != NULL) ;

	if (implicit) {
		return Graph_ImplicitClearConnections (g, edges, n) ;
	}

	GrB_Info info ;

	//--------------------------------------------------------------------------
	// create mask matrix
	//--------------------------------------------------------------------------
	
	// update matrix sync policy to NOP
	MATRIX_POLICY policy = Graph_SetMatrixPolicy (g, SYNC_POLICY_NOP) ;

	// sort edges by:
	// 1. relationship-type
	// 2. src node ID
	// 3. dest node ID
	qsort (edges, n, sizeof(Edge), _edge_cmp) ;
	
	Delta_Matrix ADJ = Graph_GetAdjacencyMatrix (g, false) ;

	// handle each relationship-type seperetly
	for (uint64_t i = 0; i < n;) {
		Edge      *e = edges + i ;
		RelationID r = Edge_GetRelationID (e) ;

		// gather edges by relationship-type
		uint64_t j = i;
		while (e->relationID == r) {
			// advance
			if (++j == n) {
				break ;
			}

			e = edges + j;
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

		if (flat_deletion)
		{
			// tensor R doesn't contains any vector
			// perform a simple "flat" deletion
			Tensor_RemoveElements_Flat (R, edges + i, d) ;

			// update ADJ matrix
			for (uint k = 0; k < d; k++) {
				e = edges + i + k ;
				_clear_adj (g, ADJ, e) ;
			}
		}
		else
		{
			// tensor R contains vectors
			// perform deletion which handels vector entries
			uint64_t *cleared_entries = NULL ;
			Tensor_RemoveElements (R, edges + i, d, &cleared_entries) ;

			// for each cleared entry E see if ADJ[E.src, E.dest] needs clearing
			uint64_t m = array_len (cleared_entries) ;
			for (uint k = 0; k < m; k++) {
				e = edges + (i + cleared_entries[k]) ;
				_clear_adj (g, ADJ, e) ;
			}

			// free reported cleared entries
			array_free (cleared_entries) ;
		}

		i = j ;
	}

	Graph_SetMatrixPolicy (g, policy) ;
}

