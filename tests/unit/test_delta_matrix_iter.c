/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/configuration/config.h"
#include "src/graph/delta_matrix/delta_matrix.h"
#include "src/graph/delta_matrix/delta_matrix_iter.h"

void setup() {
	Alloc_Reset();	

	// initialize GraphBLAS
	GrB_init(GrB_NONBLOCKING);

	// all matrices in CSR format
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);

	// set delta matrix flush threshold
	Config_Option_set(Config_DELTA_MAX_PENDING_CHANGES, "10000", NULL);
}

void tearDown() {
	GrB_finalize();
}

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

// test DeltaMatrixTupleIter initialization
void test_DeltaMatrixTupleIter_attach() {
	Delta_Matrix          A                   =  NULL;
	GrB_Type              t                   =  GrB_BOOL;
	GrB_Info              info                =  GrB_SUCCESS;
	GrB_Index             nrows               =  100;
	GrB_Index             ncols               =  100;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(Delta_MatrixTupleIter_is_attached(&iter, A));

	Delta_MatrixTupleIter_detach(&iter);
	// TEST_ASSERT(iter.A == NULL);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
}

// test DeltaMatrixTupleIter iteration
void test_DeltaMatrixTupleIter_next() {
	Delta_Matrix          A                   =  NULL;
	GrB_Type              t                   =  GrB_BOOL;
	GrB_Info              info                =  GrB_SUCCESS;
	GrB_Index             i                   =  1;
	GrB_Index             j                   =  2;
	GrB_Index             row                 =  0;
	GrB_Index             col                 =  0;
	GrB_Index             nrows               =  100;
	GrB_Index             ncols               =  100;
	bool                  val                 =  0;
	bool                  sync                =  false;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i,j
	info = Delta_Matrix_setElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);

	//--------------------------------------------------------------------------
	// set pending changes
	//--------------------------------------------------------------------------

	// remove element at position i,j
	info = Delta_Matrix_removeElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i+1,j+1
	info = Delta_Matrix_setElement_BOOL(A, i+1, j+1);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(Delta_MatrixTupleIter_is_attached(&iter, A));

	info = Delta_MatrixTupleIter_next_BOOL(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	
	TEST_ASSERT(row == i+1);
	TEST_ASSERT(col == j+1);
	TEST_ASSERT(val == 1);

	info = Delta_MatrixTupleIter_next_BOOL(&iter, &row, &col, &val);

	TEST_ASSERT(info == GxB_EXHAUSTED);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	// TEST_ASSERT(iter.A == NULL);
}

// test DeltaMatrixTupleIter iteration for sparse matrix
void test_DeltaMatrixTupleIter_next_sparse() {
	Delta_Matrix          A                   =  NULL;
	GrB_Type              t                   =  GrB_BOOL;
	GrB_Info              info                =  GrB_SUCCESS;
	GrB_Index             row                 =  0;
	GrB_Index             col                 =  0;
	GrB_Index             nrows               =  100;
	GrB_Index             ncols               =  100;
	bool                  val                 =  0;
	bool                  sync                =  false;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	for (GrB_Index i = 25; i < 100; i++) {
		for (GrB_Index j = 25; j < 100; j++) {
			// set element at position i,j
			info = Delta_Matrix_setElement_BOOL(A, i, j);
			TEST_ASSERT(info == GrB_SUCCESS);
		}
	}

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);

	//--------------------------------------------------------------------------
	// check M is sparse
	//--------------------------------------------------------------------------

	GrB_Matrix M = Delta_Matrix_M(A);
	
	int sparsity;
	GxB_Matrix_Option_get(M, GxB_SPARSITY_STATUS, &sparsity);
	TEST_ASSERT(sparsity == GxB_SPARSE);

	//--------------------------------------------------------------------------
	// check iter start from correct row
	//--------------------------------------------------------------------------

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(Delta_MatrixTupleIter_is_attached(&iter, A));

	info = Delta_MatrixTupleIter_next_BOOL(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	
	TEST_ASSERT(row == 25);
	TEST_ASSERT(col == 25);
	TEST_ASSERT(val);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	// TEST_ASSERT(iter.A == NULL);
}

// test DeltaMatrixTupleIter iteration
void test_DeltaMatrixTupleIter_reuse() {
	Delta_Matrix          A                   =  NULL;
	Delta_Matrix          B                   =  NULL;
	GrB_Type              t                   =  GrB_BOOL;
	GrB_Info              info                =  GrB_SUCCESS;
	GrB_Index             i                   =  1;
	GrB_Index             j                   =  2;
	GrB_Index             row                 =  0;
	GrB_Index             col                 =  0;
	GrB_Index             nrows               =  100;
	GrB_Index             ncols               =  100;
	bool                  val                 =  0;
	bool                  sync                =  false;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_Matrix_new(&B, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i,j
	info = Delta_Matrix_setElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);

	info = Delta_MatrixTupleIter_attach(&iter, B);
	TEST_ASSERT(Delta_MatrixTupleIter_is_attached(&iter, B));

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(Delta_MatrixTupleIter_is_attached(&iter, A));

	info = Delta_MatrixTupleIter_next_BOOL(&iter, &row, &col, &val);

	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == i);
	TEST_ASSERT(col == j);
	TEST_ASSERT(val);

	info = Delta_MatrixTupleIter_next_BOOL(&iter, &row, &col, &val);

	TEST_ASSERT(info == GxB_EXHAUSTED);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_Matrix_free(&B);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	// TEST_ASSERT(iter.A == NULL);
}

// test DeltaMatrixTupleIter_iterate_row
void test_DeltaMatrixTupleIter_iterate_row() {
	Delta_Matrix          A                   =  NULL;
	GrB_Type              t                   =  GrB_BOOL;
	GrB_Info              info                =  GrB_SUCCESS;
	GrB_Index             i                   =  1;
	GrB_Index             j                   =  2;
	GrB_Index             row                 =  0;
	GrB_Index             col                 =  0;
	GrB_Index             nrows               =  100;
	GrB_Index             ncols               =  100;
	bool                  val                 =  0;
	bool                  sync                =  false;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i,j
	info = Delta_Matrix_setElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);

	//--------------------------------------------------------------------------
	// set pending changes
	//--------------------------------------------------------------------------

	// remove element at position i,j
	info = Delta_Matrix_removeElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// wait, DM can't have pendding changes
	sync = false;
	Delta_Matrix_wait(A, sync);

	// set element at position i+1,j+1
	info = Delta_Matrix_setElement_BOOL(A, i+1, j+1);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_attach(&iter, A);
	// TEST_ASSERT(iter.A == A);

	info = Delta_MatrixTupleIter_iterate_row(&iter, i);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_next_BOOL(&iter, &row, &col, &val);
	TEST_ASSERT(info == GxB_EXHAUSTED);

	info = Delta_MatrixTupleIter_reset(&iter);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_iterate_row(&iter, i+1);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_next_BOOL(&iter, &row, &col, &val);

	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == i+1);
	TEST_ASSERT(col == j+1);
	TEST_ASSERT(val);

	info = Delta_MatrixTupleIter_next_BOOL(&iter, &row, &col, &val);
	TEST_ASSERT(info == GxB_EXHAUSTED);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	// TEST_ASSERT(iter.A == NULL);
}

// test DeltaMatrixTupleiIter_iterate_range
void test_DeltaMatrixTupleIter_iterate_range() {
	Delta_Matrix          A                   =  NULL;
	GrB_Type              t                   =  GrB_BOOL;
	GrB_Info              info                =  GrB_SUCCESS;
	GrB_Index             i                   =  1;
	GrB_Index             j                   =  2;
	GrB_Index             row                 =  0;
	GrB_Index             col                 =  0;
	GrB_Index             nrows               =  100;
	GrB_Index             ncols               =  100;
	bool                  val                 =  0;
	bool                  sync                =  false;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i,j
	info = Delta_Matrix_setElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);

	//--------------------------------------------------------------------------
	// set pending changes
	//--------------------------------------------------------------------------

	// remove element at position i,j
	info = Delta_Matrix_removeElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i+1,j+1
	info = Delta_Matrix_setElement_BOOL(A, i+1, j+1);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(Delta_MatrixTupleIter_is_attached(&iter, A));

	info = Delta_MatrixTupleIter_iterate_range(&iter, i+1, i+1);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_next_BOOL(&iter, &row, &col, &val);

	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == i+1);
	TEST_ASSERT(col == j+1);
	TEST_ASSERT(val == 1);

	info = Delta_MatrixTupleIter_next_BOOL(&iter, &row, &col, &val);
	TEST_ASSERT(info == GxB_EXHAUSTED);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	// TEST_ASSERT(iter.A == NULL);
}

TEST_LIST = {
	{"DeltaMatrixTupleIter_attach", test_DeltaMatrixTupleIter_attach},
	{"DeltaMatrixTupleIter_next", test_DeltaMatrixTupleIter_next},
	{"DeltaMatrixTupleIter_next_sparse", test_DeltaMatrixTupleIter_next_sparse},
	{"DeltaMatrixTupleIter_reuse", test_DeltaMatrixTupleIter_reuse},
	{"DeltaMatrixTupleIter_iterate_row", test_DeltaMatrixTupleIter_iterate_row},
	{"DeltaMatrixTupleIter_iterate_range", test_DeltaMatrixTupleIter_iterate_range},
	{NULL, NULL}
};
