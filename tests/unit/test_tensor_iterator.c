/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/configuration/config.h"
#include "src/graph/tensor/tensor.h"
#include "src/graph/delta_matrix/delta_matrix.h"

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

// TensorIterator_ScanEntry over a scalar cell (a single edge between a
// given pair of nodes) yields exactly one entry, tensor == false
void test_TensorIterator_ScanEntry_scalar() {
	Tensor T = Tensor_new(100, 100);

	Tensor_SetElement(T, 1, 2, 100);

	TensorIterator it;
	TensorIterator_ScanEntry(&it, T, 1, 2);

	GrB_Index row, col;
	uint64_t  x;
	bool      tensor;

	bool has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 1 && col == 2 && x == 100 && tensor == false);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(!has_next);

	Tensor_free(&T);
}

// TensorIterator_ScanEntry over a cell holding multiple edges (a parallel
// edge / "tensor" entry, promoted from scalar to vector by Tensor_SetElement)
// yields every edge in the vector, each with tensor == true
void test_TensorIterator_ScanEntry_vector() {
	Tensor T = Tensor_new(100, 100);

	Tensor_SetElement(T, 1, 2, 100);
	Tensor_SetElement(T, 1, 2, 200);  // second edge, promotes cell to a vector

	TensorIterator it;
	TensorIterator_ScanEntry(&it, T, 1, 2);

	GrB_Index row, col;
	uint64_t  x;
	bool      tensor;

	bool has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 1 && col == 2 && x == 100 && tensor == true);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 1 && col == 2 && x == 200 && tensor == true);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(!has_next);

	Tensor_free(&T);
}

// the plain (non-sorted) range scan drains M entirely before ever yielding a
// delta-plus (DP) entry - same underlying behavior as the delta matrix's
// plain iterator, since TensorIterator delegates to it directly for the
// non-vector part of each cell
void test_TensorIterator_ScanRange_plain_M_then_DP() {
	Tensor T = Tensor_new(100, 100);

	// high-row entry, flushed into M
	Tensor_SetElement(T, 50, 51, 500);
	Delta_Matrix_wait(T, true);  // force sync -> M

	// low-row entry, stays pending in delta-plus
	Tensor_SetElement(T, 5, 6, 56);

	TensorIterator it;
	TensorIterator_ScanRange(&it, T, 0, UINT64_MAX, false);

	GrB_Index row, col;
	uint64_t  x;
	bool      tensor;

	// M drained first: row 50 before the pending row 5
	bool has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 50 && col == 51 && x == 500 && tensor == false);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 5 && col == 6 && x == 56 && tensor == false);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(!has_next);

	Tensor_free(&T);
}

// TensorIterator_ScanRange_Sorted merges M and delta-plus in true ascending
// (row, col) order, so the pending low row is visited before the flushed
// high row - see TensorIterator_ScanRange_Sorted's declaration for why this
// matters when resuming a batched scan
void test_TensorIterator_ScanRange_Sorted_merges_M_and_DP() {
	Tensor T = Tensor_new(100, 100);

	// high-row entry, flushed into M
	Tensor_SetElement(T, 50, 51, 500);
	Delta_Matrix_wait(T, true);  // force sync -> M

	// low-row entry, stays pending in delta-plus
	Tensor_SetElement(T, 5, 6, 56);

	TensorIterator it;
	TensorIterator_ScanRange_Sorted(&it, T, 0, UINT64_MAX);

	GrB_Index row, col;
	uint64_t  x;
	bool      tensor;

	bool has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 5 && col == 6 && x == 56 && tensor == false);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 50 && col == 51 && x == 500 && tensor == false);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(!has_next);

	Tensor_free(&T);
}

// a pending (delta-plus) cell that holds multiple parallel edges must still
// be merged in order by the sorted iterator, and every edge within that
// vector must be yielded (tensor == true) before the iterator moves on to
// the next cell
void test_TensorIterator_ScanRange_Sorted_multiedge_vector() {
	Tensor T = Tensor_new(100, 100);

	// scalar entry, flushed into M
	Tensor_SetElement(T, 50, 51, 500);
	Delta_Matrix_wait(T, true);  // force sync -> M

	// parallel-edge (vector) entry at a lower row, stays pending in delta-plus
	Tensor_SetElement(T, 5, 6, 100);
	Tensor_SetElement(T, 5, 6, 200);

	TensorIterator it;
	TensorIterator_ScanRange_Sorted(&it, T, 0, UINT64_MAX);

	GrB_Index row, col;
	uint64_t  x;
	bool      tensor;

	// both parallel edges at (5, 6), in ascending edge-id order
	bool has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 5 && col == 6 && x == 100 && tensor == true);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 5 && col == 6 && x == 200 && tensor == true);

	// then the flushed scalar entry
	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 50 && col == 51 && x == 500 && tensor == false);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(!has_next);

	Tensor_free(&T);
}

// TensorIterator_Attach + TensorIterator_IterateRow reseek an already
// attached iterator onto a single row at a time, without re-attaching the
// underlying row iterators
void test_TensorIterator_Attach_and_IterateRow() {
	Tensor T = Tensor_new(100, 100);

	Tensor_SetElement(T, 1, 10, 1000);
	Tensor_SetElement(T, 1, 11, 1001);
	Tensor_SetElement(T, 2, 20, 2000);
	Delta_Matrix_wait(T, true);  // force sync -> M

	TensorIterator it;
	TensorIterator_Attach(&it, T, false);

	//--------------------------------------------------------------------------
	// row 1 - two entries
	//--------------------------------------------------------------------------

	TensorIterator_IterateRow(&it, 1);

	GrB_Index row, col;
	uint64_t  x;
	bool      tensor;

	bool has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 1 && col == 10 && x == 1000 && tensor == false);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 1 && col == 11 && x == 1001 && tensor == false);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(!has_next);

	//--------------------------------------------------------------------------
	// reseek to row 2 - one entry, no re-attach
	//--------------------------------------------------------------------------

	TensorIterator_IterateRow(&it, 2);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 2 && col == 20 && x == 2000 && tensor == false);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(!has_next);

	//--------------------------------------------------------------------------
	// reseek to an empty row
	//--------------------------------------------------------------------------

	TensorIterator_IterateRow(&it, 5);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(!has_next);

	Tensor_free(&T);
}

// scanning T's transpose visits entries in (col, row) order - i.e. grouped
// by column first, ascending row within each column - while still reporting
// the entry's original (row, col) coordinates and value
void test_TensorIterator_ScanRange_transpose() {
	Tensor T = Tensor_new(100, 100);

	Tensor_SetElement(T, 5, 1, 501);
	Tensor_SetElement(T, 2, 1, 201);
	Tensor_SetElement(T, 1, 3, 103);
	Delta_Matrix_wait(T, true);  // force sync, including the transpose

	TensorIterator it;
	TensorIterator_ScanRange(&it, T, 0, UINT64_MAX, true);

	GrB_Index row, col;
	uint64_t  x;
	bool      tensor;

	// column 1, ascending row: row 2 before row 5
	bool has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 2 && col == 1 && x == 201 && tensor == false);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 5 && col == 1 && x == 501 && tensor == false);

	// then column 3
	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(has_next);
	TEST_ASSERT(row == 1 && col == 3 && x == 103 && tensor == false);

	has_next = TensorIterator_next(&it, &row, &col, &x, &tensor);
	TEST_ASSERT(!has_next);

	Tensor_free(&T);
}

TEST_LIST = {
	{"TensorIterator_ScanEntry_scalar", test_TensorIterator_ScanEntry_scalar},
	{"TensorIterator_ScanEntry_vector", test_TensorIterator_ScanEntry_vector},
	{"TensorIterator_ScanRange_plain_M_then_DP", test_TensorIterator_ScanRange_plain_M_then_DP},
	{"TensorIterator_ScanRange_Sorted_merges_M_and_DP", test_TensorIterator_ScanRange_Sorted_merges_M_and_DP},
	{"TensorIterator_ScanRange_Sorted_multiedge_vector", test_TensorIterator_ScanRange_Sorted_multiedge_vector},
	{"TensorIterator_Attach_and_IterateRow", test_TensorIterator_Attach_and_IterateRow},
	{"TensorIterator_ScanRange_transpose", test_TensorIterator_ScanRange_transpose},
	{NULL, NULL}
};
