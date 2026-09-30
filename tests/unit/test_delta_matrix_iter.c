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

// test RGMatrixTupleIter initialization
void test_RGMatrixTupleIter_attach() {
	Delta_Matrix  A      =  NULL;
	GrB_Type      t      =  GrB_UINT64;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     nrows  =  100;
	GrB_Index     ncols  =  100;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(Delta_MatrixTupleIter_is_attached(&iter, A));

	Delta_MatrixTupleIter_detach(&iter);
	TEST_ASSERT(iter.A == NULL);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
}

// test RGMatrixTupleIter iteration
void test_RGMatrixTupleIter_next() {
	Delta_Matrix  A      =  NULL;
	GrB_Type      t      =  GrB_UINT64;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     i      =  1;
	GrB_Index     j      =  2;
	GrB_Index     row    =  0;
	GrB_Index     col    =  0;
	GrB_Index     nrows  =  100;
	GrB_Index     ncols  =  100;
	uint64_t      val    =  0;
	bool          sync   =  false;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i,j
	info = Delta_Matrix_setElement_UINT64(A, 0, i, j);
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
	info = Delta_Matrix_removeElement(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i+1,j+1
	info = Delta_Matrix_setElement_UINT64(A, 1, i+1, j+1);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(Delta_MatrixTupleIter_is_attached(&iter, A));

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	
	TEST_ASSERT(row == i+1);
	TEST_ASSERT(col == j+1);
	TEST_ASSERT(val == 1);

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);

	TEST_ASSERT(info == GxB_EXHAUSTED);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	TEST_ASSERT(iter.A == NULL);
}

// test RGMatrixTupleIter iteration for sparse matrix
void test_RGMatrixTupleIter_next_sparse() {
	Delta_Matrix          A      =  NULL;
	GrB_Type              t      =  GrB_UINT64;
	GrB_Info              info   =  GrB_SUCCESS;
	GrB_Index             row    =  0;
	GrB_Index             col    =  0;
	GrB_Index             nrows  =  100;
	GrB_Index             ncols  =  100;
	uint64_t              val    =  0;
	bool                  sync   =  false;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	for (GrB_Index i = 25; i < 100; i++) {
		for (GrB_Index j = 25; j < 100; j++) {
			// set element at position i,j
			info = Delta_Matrix_setElement_UINT64(A, 0, i, j);
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

	GrB_Matrix M = DELTA_MATRIX_M(A);
	
	int sparsity;
	GxB_Matrix_Option_get(M, GxB_SPARSITY_STATUS, &sparsity);
	TEST_ASSERT(sparsity == GxB_SPARSE);

	//--------------------------------------------------------------------------
	// check iter start from correct row
	//--------------------------------------------------------------------------

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(Delta_MatrixTupleIter_is_attached(&iter, A));

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	
	TEST_ASSERT(row == 25);
	TEST_ASSERT(col == 25);
	TEST_ASSERT(val == 0);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	TEST_ASSERT(iter.A == NULL);
}

// test RGMatrixTupleIter iteration
void test_RGMatrixTupleIter_reuse() {
	Delta_Matrix  A      =  NULL;
	Delta_Matrix  B      =  NULL;
	GrB_Type      t      =  GrB_UINT64;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     i      =  1;
	GrB_Index     j      =  2;
	GrB_Index     row    =  0;
	GrB_Index     col    =  0;
	GrB_Index     nrows  =  100;
	GrB_Index     ncols  =  100;
	uint64_t      val    =  0;
	bool          sync   =  false;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_Matrix_new(&B, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i,j
	info = Delta_Matrix_setElement_UINT64(A, 0, i, j);
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

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);

	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == i);
	TEST_ASSERT(col == j);
	TEST_ASSERT(val == 0);

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);

	TEST_ASSERT(info == GxB_EXHAUSTED);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_Matrix_free(&B);
	TEST_ASSERT(B == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	TEST_ASSERT(iter.A == NULL);
}

// test RGMatrixTupleIter_iterate_row
void test_RGMatrixTupleIter_iterate_row() {
	Delta_Matrix       A                   =  NULL;
	GrB_Type           t                   =  GrB_UINT64;
	GrB_Info           info                =  GrB_SUCCESS;
	GrB_Index          i                   =  1;
	GrB_Index          j                   =  2;
	GrB_Index          row                 =  0;
	GrB_Index          col                 =  0;
	GrB_Index          nrows               =  100;
	GrB_Index          ncols               =  100;
	uint64_t           val                 =  0;
	bool               sync                =  false;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i,j
	info = Delta_Matrix_setElement_UINT64(A, 1, i, j);
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
	info = Delta_Matrix_removeElement(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// wait, DM can't have pendding changes
	sync = false;
	Delta_Matrix_wait(A, sync);

	// set element at position i+1,j+1
	info = Delta_Matrix_setElement_UINT64(A, 2, i+1, j+1);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(iter.A == A);

	info = Delta_MatrixTupleIter_iterate_row(&iter, i);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);
	TEST_ASSERT(info == GxB_EXHAUSTED);

	info = Delta_MatrixTupleIter_reset(&iter);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_iterate_row(&iter, i+1);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);

	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == i+1);
	TEST_ASSERT(col == j+1);
	TEST_ASSERT(val == 2);

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);
	TEST_ASSERT(info == GxB_EXHAUSTED);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	TEST_ASSERT(iter.A == NULL);
}

// test RGMatrixTupleiIter_iterate_range
void test_RGMatrixTupleIter_iterate_range() {
	Delta_Matrix       A                   =  NULL;
	GrB_Type           t                   =  GrB_UINT64;
	GrB_Info           info                =  GrB_SUCCESS;
	GrB_Index          i                   =  1;
	GrB_Index          j                   =  2;
	GrB_Index          row                 =  0;
	GrB_Index          col                 =  0;
	GrB_Index          nrows               =  100;
	GrB_Index          ncols               =  100;
	uint64_t           val                 =  0;
	bool               sync                =  false;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i,j
	info = Delta_Matrix_setElement_UINT64(A, 0, i, j);
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
	info = Delta_Matrix_removeElement(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i+1,j+1
	info = Delta_Matrix_setElement_UINT64(A, 1, i+1, j+1);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(Delta_MatrixTupleIter_is_attached(&iter, A));

	info = Delta_MatrixTupleIter_iterate_range(&iter, i+1, i+1);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);

	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == i+1);
	TEST_ASSERT(col == j+1);
	TEST_ASSERT(val == 1);

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);
	TEST_ASSERT(info == GxB_EXHAUSTED);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	TEST_ASSERT(iter.A == NULL);
}

// the plain (non-sorted) iterator drains M entirely before ever yielding a
// delta-plus (DP) entry, so it can return a *higher* row (M) before a
// *lower* row that's still pending (DP). next_UINT64_sorted instead merges
// M and DP in true ascending (row, col) order.
void test_RGMatrixTupleIter_next_UINT64_sorted_merges_M_and_DP() {
	Delta_Matrix  A      =  NULL;
	GrB_Type      t      =  GrB_UINT64;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     row    =  0;
	GrB_Index     col    =  0;
	GrB_Index     nrows  =  100;
	GrB_Index     ncols  =  100;
	uint64_t      val    =  0;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// high-row entry, flushed into M
	info = Delta_Matrix_setElement_UINT64(A, 100, 50, 51);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_wait(A, true);  // force sync -> M

	// low-row entry, stays pending in delta-plus (well below the flush
	// threshold configured in setup())
	info = Delta_Matrix_setElement_UINT64(A, 200, 5, 6);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// plain iterator: M (row 50) before delta-plus (row 5) - out of order
	//--------------------------------------------------------------------------

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == 50 && col == 51 && val == 100);

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == 5 && col == 6 && val == 200);

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);
	TEST_ASSERT(info == GxB_EXHAUSTED);

	//--------------------------------------------------------------------------
	// sorted iterator: delta-plus (row 5) before M (row 50) - true order
	//--------------------------------------------------------------------------

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_next_UINT64_sorted(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == 5 && col == 6 && val == 200);

	info = Delta_MatrixTupleIter_next_UINT64_sorted(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == 50 && col == 51 && val == 100);

	info = Delta_MatrixTupleIter_next_UINT64_sorted(&iter, &row, &col, &val);
	TEST_ASSERT(info == GxB_EXHAUSTED);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	TEST_ASSERT(iter.A == NULL);
}

// next_UINT64_sorted must merge M and DP while still masking out M entries
// that have a pending deletion (DM) - exercises _MaskedIter_skip_masked
// through the sorted path specifically.
void test_RGMatrixTupleIter_next_UINT64_sorted_skips_deleted() {
	Delta_Matrix  A      =  NULL;
	GrB_Type      t      =  GrB_UINT64;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     row    =  0;
	GrB_Index     col    =  0;
	GrB_Index     nrows  =  100;
	GrB_Index     ncols  =  100;
	uint64_t      val    =  0;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// two entries, flushed into M
	info = Delta_Matrix_setElement_UINT64(A, 10, 20, 21);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_UINT64(A, 11, 60, 61);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_wait(A, true);  // force sync -> M

	// delete (20, 21) - pending in DM, not yet synced out of M
	info = Delta_Matrix_removeElement(A, 20, 21);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_wait(A, false);  // materialize DM without merging

	// a delta-plus entry between the two M rows
	info = Delta_Matrix_setElement_UINT64(A, 30, 40, 41);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	// row 20 is masked out - first entry is delta-plus row 40
	info = Delta_MatrixTupleIter_next_UINT64_sorted(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == 40 && col == 41 && val == 30);

	info = Delta_MatrixTupleIter_next_UINT64_sorted(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == 60 && col == 61 && val == 11);

	info = Delta_MatrixTupleIter_next_UINT64_sorted(&iter, &row, &col, &val);
	TEST_ASSERT(info == GxB_EXHAUSTED);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	TEST_ASSERT(iter.A == NULL);
}

// reproduces the batch-resume regression this iterator exists to fix:
// resuming a range scan at [last_row + 1, MAX) with the plain iterator can
// permanently skip a delta-plus row below the resume point, because the
// plain iterator drains M before DP within each attached range. The sorted
// iterator doesn't have this problem, since rows only ever increase.
void test_RGMatrixTupleIter_resume_across_batches() {
	Delta_Matrix  A      =  NULL;
	GrB_Type      t      =  GrB_UINT64;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     row    =  0;
	GrB_Index     col    =  0;
	GrB_Index     nrows  =  100;
	GrB_Index     ncols  =  100;
	uint64_t      val    =  0;
	int           count  =  0;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// "already flushed" high rows, simulating a batch of prior additions
	info = Delta_Matrix_setElement_UINT64(A, 100, 10, 10);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_UINT64(A, 101, 11, 11);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_UINT64(A, 102, 12, 12);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_wait(A, true);  // force sync -> M

	// "still pending" low row, simulating a recycled id whose addition
	// hasn't been flushed yet
	info = Delta_Matrix_setElement_UINT64(A, 200, 1, 1);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// plain iterator: batch 1 drains M's first row (10) before it would
	// ever reach delta-plus's row 1; resuming at [10 + 1, MAX) - exactly
	// what _Index_PopulateNodeIndex does between batches - permanently
	// excludes row 1
	//--------------------------------------------------------------------------

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_MatrixTupleIter_iterate_range(&iter, 0, UINT64_MAX);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == 10);  // M drained first - row 1 (DP) not reached yet

	info = Delta_MatrixTupleIter_iterate_range(&iter, row + 1, UINT64_MAX);
	TEST_ASSERT(info == GrB_SUCCESS);

	count = 0;
	while (Delta_MatrixTupleIter_next_UINT64(&iter, &row, &col, &val) == GrB_SUCCESS) {
		count++;
	}
	// row 1 is gone for good - only rows 11 and 12 remain visible
	TEST_ASSERT(count == 2);

	//--------------------------------------------------------------------------
	// sorted iterator: the lowest row (delta-plus row 1) is always visited
	// first, so the resume boundary derived from it never excludes it
	//--------------------------------------------------------------------------

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_MatrixTupleIter_iterate_range(&iter, 0, UINT64_MAX);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_next_UINT64_sorted(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == 1);  // delta-plus row visited first

	info = Delta_MatrixTupleIter_iterate_range(&iter, row + 1, UINT64_MAX);
	TEST_ASSERT(info == GrB_SUCCESS);

	count = 0;
	while (Delta_MatrixTupleIter_next_UINT64_sorted(&iter, &row, &col, &val) == GrB_SUCCESS) {
		count++;
	}
	// rows 10, 11, 12 all still visited - nothing was skipped
	TEST_ASSERT(count == 3);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	TEST_ASSERT(iter.A == NULL);
}

// BOOL counterpart (the shape used by node/label matrices): merges M and DP
// in ascending order and masks out a pending deletion.
void test_RGMatrixTupleIter_next_BOOL_sorted_merges_and_skips_deleted() {
	Delta_Matrix  A      =  NULL;
	GrB_Type      t      =  GrB_BOOL;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     row    =  0;
	GrB_Index     col    =  0;
	GrB_Index     nrows  =  100;
	GrB_Index     ncols  =  100;
	bool          val    =  false;
	Delta_MatrixTupleIter iter;
	memset(&iter, 0, sizeof(Delta_MatrixTupleIter));

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// two diagonal entries, flushed into M
	info = Delta_Matrix_setElement_BOOL(A, 10, 10);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_BOOL(A, 50, 50);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_wait(A, true);  // force sync -> M

	// delete (10, 10) - pending in DM
	info = Delta_Matrix_removeElement(A, 10, 10);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_wait(A, false);  // materialize DM without merging

	// a delta-plus entry below both M rows
	info = Delta_Matrix_setElement_BOOL(A, 5, 5);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_MatrixTupleIter_attach(&iter, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	// delta-plus row 5 first, then M row 50 - row 10 is masked out
	info = Delta_MatrixTupleIter_next_BOOL_sorted(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == 5 && col == 5 && val == true);

	info = Delta_MatrixTupleIter_next_BOOL_sorted(&iter, &row, &col, &val);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(row == 50 && col == 50 && val == true);

	info = Delta_MatrixTupleIter_next_BOOL_sorted(&iter, &row, &col, &val);
	TEST_ASSERT(info == GxB_EXHAUSTED);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_MatrixTupleIter_detach(&iter);
	TEST_ASSERT(iter.A == NULL);
}

TEST_LIST = {
	{"RGMatrixTupleIter_attach", test_RGMatrixTupleIter_attach},
	{"RGMatrixTupleIter_next", test_RGMatrixTupleIter_next},
	{"RGMatrixTupleIter_next_sparse", test_RGMatrixTupleIter_next_sparse},
	{"RGMatrixTupleIter_reuse", test_RGMatrixTupleIter_reuse},
	{"RGMatrixTupleIter_iterate_row", test_RGMatrixTupleIter_iterate_row},
	{"RGMatrixTupleIter_iterate_range", test_RGMatrixTupleIter_iterate_range},
	{"RGMatrixTupleIter_next_UINT64_sorted_merges_M_and_DP", test_RGMatrixTupleIter_next_UINT64_sorted_merges_M_and_DP},
	{"RGMatrixTupleIter_next_UINT64_sorted_skips_deleted", test_RGMatrixTupleIter_next_UINT64_sorted_skips_deleted},
	{"RGMatrixTupleIter_resume_across_batches", test_RGMatrixTupleIter_resume_across_batches},
	{"RGMatrixTupleIter_next_BOOL_sorted_merges_and_skips_deleted", test_RGMatrixTupleIter_next_BOOL_sorted_merges_and_skips_deleted},
	{NULL, NULL}
};
