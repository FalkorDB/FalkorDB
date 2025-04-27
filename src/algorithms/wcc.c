/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "wcc.h"

// build adjacency matrix which will be used to compute WCC
static GrB_Matrix _Build_Matrix
(
	const Graph *g,         // graph
	GrB_Vector *N,          // list nodes participating in WCC
	LabelID *lbls,          // [optional] labels to consider
	unsigned short n_lbls,  // number of labels
	RelationID *rels,       // [optional] relationships to consider
	unsigned short n_rels   // number of relationships
) {
	GrB_Info info;
	Delta_Matrix D;       // graph delta matrix
	GrB_Index nrows;      // number of rows in matrix
	GrB_Index ncols;      // number of columns in matrix
	GrB_Matrix A = NULL;  // returned WCC matrix

	// if no relationships are specified use the adjacency matrix
	// otherwise use specified relation matrices
	if(n_rels == 0) {
		D = Graph_GetAdjacencyMatrix(g, false);
	} else {
		RelationID id = rels[0];
		D = Graph_GetRelationMatrix(g, id, false);
	}

	// export relation matrix to A
	info = Delta_Matrix_export(&A, D);
	ASSERT(info == GrB_SUCCESS);

	// in case there are multiple relation types, include them in A
	for(unsigned short i = 1; i < n_rels; i++) {
		D = Graph_GetRelationMatrix(g, rels[i], false);

		GrB_Matrix M;
		info = Delta_Matrix_export(&M, D);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Matrix_eWiseAdd_Semiring(A, NULL, NULL, GxB_ANY_PAIR_BOOL,
				A, M, NULL);
		ASSERT(info == GrB_SUCCESS);

		GrB_Matrix_free(&M);
	}

	info = GrB_Matrix_nrows(&nrows, A);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols, A);
	ASSERT(info == GrB_SUCCESS);

	// expecting a squsre matrix
	ASSERT(nrows == ncols);

	// create vector N denoting all nodes participating in WCC
	info = GrB_Vector_new(N, GrB_BOOL, nrows);
	ASSERT(info == GrB_SUCCESS);

	// enforece labels
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

			info = GrB_Matrix_eWiseAdd_Semiring(L, NULL, NULL,
					GxB_ANY_PAIR_BOOL, L, M, NULL);
			ASSERT(info == GrB_SUCCESS);

			GrB_Matrix_free(&M);
		}

		// A = L * M * L
		info = GrB_mxm(A, NULL, NULL, GxB_ANY_PAIR_BOOL, L, A, NULL);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_mxm(A, NULL, NULL, GxB_ANY_PAIR_BOOL, A, L, NULL);
		ASSERT(info == GrB_SUCCESS);

		// set N to L's main diagonal denoting all nodes participating in WCC
		info = GxB_Vector_diag(*N, L, 0, NULL);
		ASSERT(info == GrB_SUCCESS);
	} else {
		// N = [1,....1]
		GrB_Scalar scalar;
		info = GrB_Scalar_new(&scalar, GrB_BOOL);
		ASSERT(info == GrB_SUCCESS);

		info = GxB_Scalar_setElement_BOOL(scalar, true);
		ASSERT(info == GrB_SUCCESS);

		//info = GxB_Vector_build_Scalar(*N, GrB_ALL, scalar, nrows);
		//ASSERT(info == GrB_SUCCESS);

		info = GrB_Vector_assign_Scalar(*N, NULL, NULL, scalar, GrB_ALL, nrows, NULL);
		ASSERT(info == GrB_SUCCESS);
    
		info = GrB_free(&scalar);
		ASSERT(info == GrB_SUCCESS);
	}

	// make A symmetric A = A + At
	info = GrB_Matrix_eWiseAdd_Semiring(A, NULL, NULL, GxB_ANY_PAIR_BOOL, A, A, GrB_DESC_T1);

	return A;
}

GrB_Info WCC
(
	GrB_Vector *components, // [output] components
	GrB_Vector *N,          // [output] list computed nodes
	const Graph *g,         // graph
	LabelID *lbls,          // [optional] labels to consider
	unsigned short n_lbls,  // number of labels
	RelationID *rels,       // [optional] relationships to consider
	unsigned short n_rels   // number of relationships
) {
	ASSERT(g          != NULL);
	ASSERT(N          != NULL);
	ASSERT(components != NULL);

	ASSERT( (lbls != NULL && n_lbls > 0) || (lbls == NULL && n_lbls == 0) );
	ASSERT( (rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0) );

	// nullify outputs
	*N          = NULL;
    *components = NULL;

	// build matrix on which we'll compute WCC
	GrB_Matrix A = _Build_Matrix(g, N, lbls, n_lbls, rels, n_rels);
	ASSERT(A  != NULL);
	ASSERT(*N != NULL);

	GrB_Info info;
    LAGraph_Graph G = NULL;

	char msg [LAGRAPH_MSG_LEN];
    LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg);

	info = LAGr_ConnectedComponents(components, G, msg);
	if(info != GrB_SUCCESS) {

	}

	info = GrB_wait(*components, GrB_MATERIALIZE);
	ASSERT(info == GrB_SUCCESS);

    // free the graph, the connected components, and finish LAGraph
    info = LAGraph_Delete(&G, msg);
	ASSERT(info == GrB_SUCCESS);

	return info;
}

