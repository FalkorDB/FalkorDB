/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "./detect_cycle.h"
#include "rax.h"
#include "../util/arr.h"
#include "../graph/entities/qg_node.h"
#include "../../deps/GraphBLAS/Include/GraphBLAS.h"

bool IsAcyclicGraph(const QueryGraph *qg) {
	ASSERT(qg);

	bool acyclic = true;

	// Give an ID for each node, abuse of `labelID`.
	uint node_count = QueryGraph_NodeCount(qg);
	uint edge_count = QueryGraph_EdgeCount(qg);

	GrB_Info res;
	GrB_Matrix m;   // Matrix representation of QueryGraph.
	GrB_Matrix c;   // Intermidate matrix, c = m^i.
	GrB_Matrix t;   // Temporary matrix, t = c .* i. elementwise boolean multiplication.

	// Build matrix representation of query graph.
	m = QueryGraph_MatrixRepresentation(qg);
	res = GrB_Matrix_dup(&c, m);
	UNUSED(res);
	ASSERT(res == GrB_SUCCESS);
	res = GrB_Matrix_new(&t, GrB_BOOL, node_count, node_count);
	ASSERT(res == GrB_SUCCESS);

	/* Perform traversals, stop when:
	* 1. Node i manged to reach itself, x[i,i] is set (cycle detected).
	* 2. After edge_count multiplications, no cycles. */
	for(uint i = 0; i < edge_count; i++) {
		// c = c * m.
		res = GrB_mxm(c, GrB_NULL, GrB_NULL, GxB_ANY_PAIR_BOOL, c, m, GrB_NULL);
		ASSERT(res == GrB_SUCCESS);

		/*  Extract main diagonal of `c` into `t`.
		 * Check if C[i,i] is set.
		 * t = c<identity> */
		res = GxB_Matrix_select(t, GrB_NULL, GrB_NULL, GxB_DIAG, c, GrB_NULL, GrB_NULL);
		ASSERT(res == GrB_SUCCESS);

		/* How many entries are there in `t`?
		 * if there are any entires in `t` this means node `k` was able to reach itself cycle! */
		GrB_Index nvals = 0;
		res = GrB_Matrix_nvals(&nvals, t);
		ASSERT(res == GrB_SUCCESS);
		if(nvals != 0) {
			acyclic = false;
			break;
		}
	}

	// Cleanup.
	GrB_free(&m);
	GrB_free(&c);
	GrB_free(&t);

	return acyclic;
}

