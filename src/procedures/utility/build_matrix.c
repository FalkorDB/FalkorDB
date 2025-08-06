/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "./internal.h"

// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = L * (R0 U R1 U ... Rn) * L
//
// rows = L's main diagonal
// in case no labels are specified rows is a dense 1 vector: [1,1,...1]
GrB_Info Build_Matrix
(
	GrB_Matrix *A,           // [output] matrix
	GrB_Vector *rows,        // [output] filtered rows
	const Graph *g,          // graph
	const LabelID *lbls,     // [optional] labels to consider
	unsigned short n_lbls,   // number of labels
	const RelationID *rels,  // [optional] relationships to consider
	unsigned short n_rels,   // number of relationships
	bool symmetric,          // build a symmetric matrix
	bool compact             // remove unused row & columns
) {
	ASSERT(g != NULL);
	ASSERT(A != NULL);

	ASSERT((lbls != NULL && n_lbls > 0) || (lbls == NULL && n_lbls == 0));
	ASSERT((rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0));

	GrB_Info info;
	Delta_Matrix D;   // graph delta matrix
	GrB_Index nrows;  // number of rows in matrix
	GrB_Index ncols;  // number of columns in matrix
	GrB_Matrix _A;    // output matrix
	GrB_Vector _N;    // output filtered rows

	// if no relationships are specified use the adjacency matrix
	// otherwise use specified relation matrices
	if(n_rels == 0) {
		D = Graph_GetAdjacencyMatrix(g, false);
	} else {
		RelationID id = rels[0];
		D = Graph_GetRelationMatrix(g, id, false);
	}
	ASSERT(D != NULL);

	// export relation matrix to A
	// TODO: extend Delta_Matrix_export to include a exported matrix type
	// cast if needed
	info = Delta_Matrix_export(&_A, D);
	ASSERT(info == GrB_SUCCESS);

	// in case there are multiple relation types, include them in A
	for(unsigned short i = 1; i < n_rels; i++) {
		D = Graph_GetRelationMatrix(g, rels[i], false);

		GrB_Matrix M;
		info = Delta_Matrix_export(&M, D);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Matrix_eWiseAdd_Monoid(_A, NULL, NULL, GxB_ANY_BOOL_MONOID,
				_A, M, NULL);
		ASSERT(info == GrB_SUCCESS);

		GrB_Matrix_free(&M);
	}

	info = GrB_Matrix_nrows(&nrows, _A);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols, _A);
	ASSERT(info == GrB_SUCCESS);

	// expecting a square matrix
	ASSERT(nrows == ncols);

	// create vector N denoting all nodes passing the labels filter
	if(rows != NULL) {
		info = GrB_Vector_new(&_N, GrB_BOOL, nrows);
		ASSERT(info == GrB_SUCCESS);
	}

	// enforce labels
	if(n_lbls > 0) {
		Delta_Matrix DL = Graph_GetLabelMatrix(g, lbls[0]);

		GrB_Matrix L;
		info = Delta_Matrix_export(&L, DL);
		ASSERT(info == GrB_SUCCESS);

		// L = L U M
		for(unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, lbls[i]);

			GrB_Matrix M;
			info = Delta_Matrix_export(&M, DL);
			ASSERT(info == GrB_SUCCESS);

			info = GrB_Matrix_eWiseAdd_Monoid(L, NULL, NULL,
					GxB_ANY_BOOL_MONOID, L, M, NULL);
			ASSERT(info == GrB_SUCCESS);

			GrB_Matrix_free(&M);
		}

		// A = L * A * L
		info = GrB_mxm(_A, NULL, NULL, GxB_ANY_PAIR_BOOL, L, _A, NULL);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_mxm(_A, NULL, NULL, GxB_ANY_PAIR_BOOL, _A, L, NULL);
		ASSERT(info == GrB_SUCCESS);

		// set N to L's main diagonal denoting all participating nodes 
		if(rows != NULL) {
			info = GxB_Vector_diag(_N, L, 0, NULL);
			ASSERT(info == GrB_SUCCESS);
		}

		// free L matrix
		info = GrB_Matrix_free(&L);
		ASSERT(info == GrB_SUCCESS);
	} else if(rows != NULL) {
		// N = [1,....1]
		GrB_Scalar scalar;
		info = GrB_Scalar_new(&scalar, GrB_BOOL);
		ASSERT(info == GrB_SUCCESS);

		info = GxB_Scalar_setElement_BOOL(scalar, true);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Vector_assign_Scalar(_N, NULL, NULL, scalar, GrB_ALL, nrows,
				NULL);
		ASSERT(info == GrB_SUCCESS);
    
		info = GrB_free(&scalar);
		ASSERT(info == GrB_SUCCESS);
	}

	if(symmetric) {
		// make A symmetric A = A + At
		info = GrB_Matrix_eWiseAdd_Semiring(_A, NULL, NULL, GxB_ANY_PAIR_BOOL,
				_A, _A, GrB_DESC_T1);
		ASSERT(info == GrB_SUCCESS);
	}

	if(compact) {
		// determine the number of nodes in the graph
		// this includes deleted nodes
		size_t n = Graph_UncompactedNodeCount(g);

		// get rid of extra unused rows and columns
		info = GrB_Matrix_resize(_A, n, n);
		ASSERT(info == GrB_SUCCESS);

		if(rows != NULL) {
			info = GrB_Vector_resize(_N, n);
			ASSERT(info == GrB_SUCCESS);
		}
	}

	// set outputs
	*A = _A;
	if(rows) *rows = _N;

	return info;
}

