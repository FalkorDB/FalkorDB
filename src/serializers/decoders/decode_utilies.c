/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "current/v18/decode_v18.h"
#include "graph/delta_matrix/delta_matrix.h"

// unary op, determines the number of edges represented by an entry:
// 1 if the entry is scalar
// |V| if the entry is a GrB_Vector
static void _vector_size
(
	uint16_t *z,
	uint64_t *x
) {
	if (SCALAR_ENTRY (*x)) {
		*z = 1 ;
	} else {
		uint64_t v = 0;
		GrB_OK (GrB_Vector_nvals(&v, AS_VECTOR (*x))) ;
		ASSERT (v <= UINT16_MAX);
		*z = v;
	}
}

// if the rdb we are loading is old, then we must recalculate the number of
// edges connecting ech pair of nodes
// precondition: relation matricies have been calculated and fully synced
void RdbNormalizeAdjMatrix
(
	const Graph *g  // graph
) {
	ASSERT (g != NULL) ;

	Delta_Matrix adj = Graph_GetAdjacencyMatrix (g, false) ;

	// get ADJ matrix type
	GrB_Type ty = NULL ;
	GrB_OK (GxB_Matrix_type (&ty, Delta_Matrix_M (adj))) ;

	// ADJ is numeric, we can return
	if (ty == GrB_UINT16) {
		return;
	}

	// ADJ is boolean, transition to numeric
	// ADJ[i,j] = number of edges (i)-[]->(j)
	ASSERT (ty == GrB_BOOL) ;

	GrB_Index nrows ;
	GrB_Index ncols ;

	GrB_OK (Delta_Matrix_nrows (&nrows, adj)) ;
	GrB_OK (Delta_Matrix_ncols (&ncols, adj)) ;

	Delta_Matrix_clear (adj) ;

	GrB_Matrix a_m = NULL ;
	GrB_UnaryOp op = NULL ;

	// TODO: once the GB kernel is fast, since you already know the stucture,
	// it may be faster to compute a_m <(struct) a_m> += R
	GrB_UnaryOp_new (
		&op, (GxB_unary_function) _vector_size, GrB_UINT16, GrB_UINT64) ;

	GrB_OK (GrB_Matrix_new (&a_m, GrB_UINT16,  nrows, ncols)) ;

	int n = Graph_RelationTypeCount (g) ;

	// count number of edges in each relation matrix
	// ADJ[i,j] += |R[i,j]|
	for (RelationID r = 0; r < n; r++) {
		Delta_Matrix R = Graph_GetRelationMatrix (g, r, false) ;
		ASSERT (Delta_Matrix_Synced(R));

		GrB_OK (GrB_Matrix_apply (
			a_m, NULL, GrB_PLUS_UINT16, op, Delta_Matrix_M (R), NULL)) ;
	}

	GrB_Matrix a_dp = NULL ;
	GrB_Matrix a_dm = NULL ;
	GrB_OK (GrB_Matrix_new (&a_dp, GrB_UINT16, nrows, ncols)) ;
	GrB_OK (GrB_Matrix_new (&a_dm, GrB_BOOL,   nrows, ncols)) ;

#if RG_DEBUG
	// check edge count is correct
	// NOTE: if this fails, there might be too many edges between a single node
	// aka more than UINT16_MAX, causing an overflow
	uint64_t edge_count = 0;
	GrB_OK(GrB_Matrix_reduce_UINT64(&edge_count, NULL, GrB_PLUS_MONOID_UINT64,
		a_m, NULL));
	ASSERT (edge_count == Graph_EdgeCount(g));
#endif

	Delta_Matrix_setMatrices (adj, &a_m, &a_dp, &a_dm) ;

	// clean up
	GrB_free (&op) ;
}
