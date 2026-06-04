/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graph.h"
#include "../util/arr.h"
#include "../util/rwlock.h"
#include "../util/rmalloc.h"
#include "delta_matrix/delta_matrix_iter.h"
#include "../util/datablock/oo_datablock.h"

bool debug_matrix_print_diff
(
	const GrB_Matrix A,
	const GrB_Matrix B
) {
	GrB_Index      nrows   = 0 ;
	GrB_Index      ncols   = 0 ;
	GrB_Index      brows   = 0 ;
	GrB_Index      bcols   = 0 ;
	GrB_Index      c_nvals = 0 ;
	GxB_Container cont = NULL;
	
	GrB_OK (GrB_Matrix_nrows(&nrows, A));
	GrB_OK (GrB_Matrix_ncols(&ncols, A));
	GrB_OK (GrB_Matrix_nrows(&brows, B));
	GrB_OK (GrB_Matrix_ncols(&bcols, B));

	if(nrows != brows || ncols != bcols) {
		return false;
	}

	GrB_Matrix C = NULL;
	GrB_OK (GrB_Matrix_new(&C, GrB_BOOL, nrows, ncols));
	GrB_OK (GrB_Matrix_assign (
		C, B, NULL, A, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_RSC)) ;
	GrB_OK (GrB_Matrix_nvals(&c_nvals, C)) ;
	if (c_nvals > 0) {
		GrB_OK (GxB_Matrix_fprint (C, "A - B", GxB_COMPLETE, stdout)) ;
	}

	GrB_OK (GrB_Matrix_assign (
		C, A, NULL, B, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_RSC)) ;
	GrB_OK (GrB_Matrix_nvals(&c_nvals, C)) ;
	if (c_nvals > 0) {
		GrB_OK (GxB_Matrix_fprint (C, "B - A", GxB_COMPLETE, stdout)) ;
	}

	GrB_free(&C);
	return true;
}

bool _matrix_leq
(
	const GrB_BinaryOp leq,
	const GrB_Matrix A,
	const GrB_Matrix B,
	bool transpose
) ;

void Graph_CheckConsistency
(
	Graph *g
) {
	ASSERT(g != NULL);

	GrB_Matrix temp = NULL;
	GrB_Vector diag = NULL;
	GrB_Index nrows = Graph_RequiredMatrixDim(g);
	GrB_OK(GrB_Matrix_new(&temp, GrB_BOOL, nrows, nrows));

	// For the labels matrix, check that the indivudual label matricies agree
	// with THE labels matrix
	Delta_Matrix lbls = Graph_GetNodeLabelMatrix(g);
	Delta_Matrix_wait(lbls, true);
	GrB_Matrix M = Delta_Matrix_M(lbls);

	uint32_t n = arr_len(g->labels);
	for (uint32_t i = 0; i < n; i++) {
		Delta_Matrix L = Graph_GetLabelMatrix(g, i);
		Delta_Matrix_wait(L, true);
		GrB_OK(GrB_Vector_new(&diag, GrB_BOOL, nrows));
		GrB_OK(GxB_Vector_diag(diag, Delta_Matrix_M(L), 0, NULL));
		GrB_OK(GrB_Col_assign(temp, NULL, NULL, diag, GrB_ALL, 0, i, NULL));
		GrB_free (&diag) ;
	}

	printf("labels: \n");
	debug_matrix_print_diff (M, temp);

	// for the adj matrix, check that THE adj matrix agrees with the relation
	// ship matricies
	Delta_Matrix ADJ = Graph_GetAdjacencyMatrix(g, false);
	Delta_Matrix_wait(ADJ, true);
	M = Delta_Matrix_M(ADJ);
	GrB_Matrix_clear(temp);

	n = Graph_RelationTypeCount(g);
	for (uint32_t i = 0; i < n; i++) {
		Delta_Matrix R = Graph_GetRelationMatrix(g, i, false);
		Delta_Matrix_wait(R, true);
		GrB_OK (GrB_Matrix_assign_BOOL (temp, DELTA_MATRIX_M(R), NULL, true,
			GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S)) ;
	}

	printf("adjacency matrix: \n");
	debug_matrix_print_diff (M, temp);
	GrB_free (&temp);
}

void Graph_RepairLabels
(
	Graph *g
) {
	ASSERT(g != NULL);

	GrB_Matrix temp = NULL;
	GrB_Vector diag = NULL;
	GrB_Index nrows = Graph_RequiredMatrixDim(g);
	GrB_OK(GrB_Matrix_new(&temp, GrB_BOOL, nrows, nrows));

	Delta_Matrix lbls = Graph_GetNodeLabelMatrix(g);
	Delta_Matrix_clear(lbls);

	uint32_t n = arr_len(g->labels);
	for (uint32_t i = 0; i < n; i++) {
		Delta_Matrix L = Graph_GetLabelMatrix(g, i);
		Delta_Matrix_wait(L, true);
		GrB_OK(GrB_Vector_new(&diag, GrB_BOOL, nrows));
		GrB_OK(GxB_Vector_diag(diag, Delta_Matrix_M(L), 0, NULL));
		GrB_OK(GrB_Col_assign(temp, NULL, NULL, diag, GrB_ALL, 0, i, NULL));
		GrB_free (&diag) ;
	}

	Delta_Matrix_setM (lbls, &temp);
}
