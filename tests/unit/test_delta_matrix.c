/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/configuration/config.h"
#include "src/graph/tensor/tensor.h"
#include "src/graph/delta_matrix/delta_utils.h"
#include <time.h>

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

#define MATRIX_EMPTY(M)               \
	({                                \
		GrB_Matrix_nvals(&nvals, M);  \
		TEST_CHECK(nvals == 0);       \
	}) 

#define MATRIX_NOT_EMPTY(M)           \
	({                                \
		GrB_Matrix_nvals(&nvals, M);  \
		TEST_CHECK(nvals != 0);       \
	}) 

#define M_EMPTY()   MATRIX_EMPTY(M)
#define DP_EMPTY()  MATRIX_EMPTY(DP)
#define DM_EMPTY()  MATRIX_EMPTY(DM)

#define M_NOT_EMPTY()   MATRIX_NOT_EMPTY(M)
#define DP_NOT_EMPTY()  MATRIX_NOT_EMPTY(DP)
#define DM_NOT_EMPTY()  MATRIX_NOT_EMPTY(DM)

void setup() {
	// use the malloc family for allocations
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

// nvals(A + B) == nvals(A) == nvals(B)
void CHECK_GrB_Matrices_EQ
(
	const GrB_Matrix A, 
	const GrB_Matrix B, 
	const GrB_BinaryOp eq
) {
	GrB_Type    t_A                 =  NULL;
	GrB_Type    t_B                 =  NULL;
	GrB_Matrix  C                   =  NULL;
	GrB_Info    info                =  GrB_SUCCESS;
	GrB_Index   nvals_A             =  0;
	GrB_Index   nvals_B             =  0;
	GrB_Index   nvals_C             =  0;
	GrB_Index   nrows_A             =  0;
	GrB_Index   ncols_A             =  0;
	GrB_Index   nrows_B             =  0;
	GrB_Index   ncols_B             =  0;

	//--------------------------------------------------------------------------
	// type(A) == type(B)
	//--------------------------------------------------------------------------

	info = GxB_Matrix_type(&t_A, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GxB_Matrix_type(&t_B, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(t_A == t_B);

	//--------------------------------------------------------------------------
	// dim(A) == dim(B)
	//--------------------------------------------------------------------------

	info = GrB_Matrix_nrows(&nrows_A, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols_A, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_nrows(&nrows_B, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols_B, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_CHECK(nrows_A == nrows_B);
	TEST_CHECK(ncols_A == ncols_B);

	//--------------------------------------------------------------------------
	// NNZ(A) == NNZ(B)
	//--------------------------------------------------------------------------

	GrB_Matrix_nvals(&nvals_A, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	GrB_Matrix_nvals(&nvals_B, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// structure(A) == structure(B)
	//--------------------------------------------------------------------------

	info = GrB_Matrix_new(&C, GrB_BOOL, nrows_A, ncols_A);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_eWiseMult_BinaryOp(C, NULL, NULL, eq, A, B, NULL);
	TEST_ASSERT(info == GrB_SUCCESS);

	GrB_Matrix_nvals(&nvals_C, C);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_CHECK(nvals_C == nvals_A);
	TEST_CHECK(nvals_C == nvals_B);

	bool ok = true;
	info = GrB_Matrix_reduce_BOOL(&ok, NULL, GrB_LAND_MONOID_BOOL, C, NULL);
	TEST_CHECK(ok);

	// clean up
	info = GrB_Matrix_free(&C);
	TEST_ASSERT(info == GrB_SUCCESS);
}

// nvals(A + B) == nvals(A) == nvals(B)
void ASSERT_GrB_Matrices_EQ(const GrB_Matrix A, const GrB_Matrix B) {
	GrB_Type    t_A                 =  NULL;
	GrB_Type    t_B                 =  NULL;
	GrB_Matrix  C                   =  NULL;
	GrB_Info    info                =  GrB_SUCCESS;
	GrB_Index   nvals_A             =  0;
	GrB_Index   nvals_B             =  0;
	GrB_Index   nvals_C             =  0;
	GrB_Index   nrows_A             =  0;
	GrB_Index   ncols_A             =  0;
	GrB_Index   nrows_B             =  0;
	GrB_Index   ncols_B             =  0;

	//--------------------------------------------------------------------------
	// type(A) == type(B)
	//--------------------------------------------------------------------------

	info = GxB_Matrix_type(&t_A, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GxB_Matrix_type(&t_B, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(t_A == t_B);

	//--------------------------------------------------------------------------
	// dim(A) == dim(B)
	//--------------------------------------------------------------------------

	info = GrB_Matrix_nrows(&nrows_A, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols_A, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_nrows(&nrows_B, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols_B, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_CHECK(nrows_A == nrows_B);
	TEST_CHECK(ncols_A == ncols_B);

	//--------------------------------------------------------------------------
	// NNZ(A) == NNZ(B)
	//--------------------------------------------------------------------------

	GrB_Matrix_nvals(&nvals_A, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	GrB_Matrix_nvals(&nvals_B, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// structure(A) == structure(B)
	//--------------------------------------------------------------------------

	info = GrB_Matrix_new(&C, GrB_BOOL, nrows_A, ncols_A);
	TEST_ASSERT(info == GrB_SUCCESS);

	GrB_BinaryOp op = t_A == GrB_BOOL? GrB_EQ_BOOL: GrB_EQ_UINT64;
	info = GrB_Matrix_eWiseMult_BinaryOp(C, NULL, NULL, op, A, B, NULL);
	TEST_ASSERT(info == GrB_SUCCESS);

	GrB_Matrix_nvals(&nvals_C, C);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_CHECK(nvals_C == nvals_A);
	TEST_CHECK(nvals_C == nvals_B);

	bool ok = true;
	info = GrB_Matrix_reduce_BOOL(&ok, NULL, GrB_LAND_MONOID_BOOL, C, NULL);
	TEST_CHECK(ok);

	// clean up
	info = GrB_Matrix_free(&C);
	TEST_ASSERT(info == GrB_SUCCESS);
}

// make a matrix for testing
// if transpose is false, will return:
// [ . . . . ] A row with entries in M
// [ - - - - ] A row with deleted entries (in M and DM)
// [ + + + + ] A row with added entries (in DP)
// [         ] A row with no entries
Delta_Matrix make_test_matrix
(
	bool transpose
) {
	GrB_Info info = GrB_SUCCESS;
	Delta_Matrix A = NULL;

	// create a new delta matrix
	info = Delta_Matrix_new(&A, GrB_BOOL, 4, 4, transpose);
	TEST_ASSERT(info == GrB_SUCCESS);

	for(int i = 0; i < 4; ++i){

		int j = 0;
		if (transpose){
			int tmp = i; i = j; j = tmp;
		}

		Delta_Matrix_setElement_BOOL(A, i, j);
		j = 1;

		if (transpose){
			int tmp = i; i = j; j = tmp;
		}

		Delta_Matrix_setElement_BOOL(A, i, j);
	}
	
	Delta_Matrix_wait(A, true);

	for(int i = 0; i < 4; ++i){

		int j = 1;
		if (transpose){
			int tmp = i; i = j; j = tmp;
		}

		Delta_Matrix_removeElement_BOOL(A, i, j);
		j = 2;

		if (transpose){
			int tmp = i; i = j; j = tmp;
		}

		Delta_Matrix_setElement_BOOL(A, i, j);
	}

	return A;
}

// test RGMatrix initialization
void test_RGMatrix_new() {
	Delta_Matrix  A     = NULL;
	GrB_Matrix    M     = NULL;
	GrB_Matrix    DP    = NULL;
	GrB_Matrix    DM    = NULL;
	GrB_Type      t     = GrB_UINT64;
	GrB_Info      info  = GrB_SUCCESS;
	GrB_Index     nvals = 0;
	GrB_Index     nrows = 100;
	GrB_Index     ncols = 100;

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M  = DELTA_MATRIX_M(A);
	DP = DELTA_MATRIX_DELTA_PLUS(A);
	DM = DELTA_MATRIX_DELTA_MINUS(A);

	// uint64 matrix always maintain transpose
	TEST_ASSERT(DELTA_MATRIX_MAINTAIN_TRANSPOSE(A));

	// a new empty matrix should be synced
	// no data in either DP or DM
	TEST_ASSERT(Delta_Matrix_Synced(A));

	// test M, DP and DM hyper switch
	int format;
	double hyper_switch;

	// M should be either hyper-sparse or sparse
	GxB_Matrix_Option_get(M, GxB_SPARSITY_CONTROL, &format);
	TEST_ASSERT(format == (GxB_SPARSE | GxB_HYPERSPARSE));

	// DP should always be hyper
	GxB_Matrix_Option_get(DP, GxB_HYPER_SWITCH, &hyper_switch);
	TEST_ASSERT(hyper_switch == GxB_ALWAYS_HYPER);
	GxB_Matrix_Option_get(DP, GxB_SPARSITY_CONTROL, &format);
	TEST_ASSERT(format == GxB_HYPERSPARSE);

	// DM should always be hyper
	GxB_Matrix_Option_get(DM, GxB_HYPER_SWITCH, &hyper_switch);
	TEST_ASSERT(hyper_switch == GxB_ALWAYS_HYPER);
	GxB_Matrix_Option_get(DM, GxB_SPARSITY_CONTROL, &format);
	TEST_ASSERT(format == GxB_HYPERSPARSE);

	// matrix should be empty
	M_EMPTY();
	DP_EMPTY(); 
	DM_EMPTY();
	Delta_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 0);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);

	t = GrB_BOOL;

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M   =  DELTA_MATRIX_M(A);
	DP  =  DELTA_MATRIX_DELTA_PLUS(A);
	DM  =  DELTA_MATRIX_DELTA_MINUS(A);

	TEST_ASSERT(!(DELTA_MATRIX_MAINTAIN_TRANSPOSE(A)));

	// a new empty matrix should be synced
	// no data in either DP or DM
	TEST_ASSERT(Delta_Matrix_Synced(A));

	// matrix should be empty
	M_EMPTY();
	DP_EMPTY(); 
	DM_EMPTY();

	Delta_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 0);

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
}

// setting an empty entry
// M[i,j] = 1
void test_RGMatrix_simple_set() {
	GrB_Type      t      =  GrB_UINT64;
	Delta_Matrix  A      =  NULL;
	GrB_Matrix    M      =  NULL;
	GrB_Matrix    DP     =  NULL;
	GrB_Matrix    DM     =  NULL;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     nvals  =  0;
	GrB_Index     nrows  =  100;
	GrB_Index     ncols  =  100;
	GrB_Index     i      =  0;
	GrB_Index     j      =  1;
	uint64_t      x      =  1;

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// set element at position i,j
	//--------------------------------------------------------------------------

	info = Delta_Matrix_setElement_UINT64(A, x, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// make sure element at position i,j exists
	info = Delta_Matrix_extractElement_UINT64(&x, A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_CHECK(x == 1);
	
	// matrix should contain a single element
	Delta_Matrix_nvals(&nvals, A);
	TEST_CHECK(nvals == 1);

	// matrix should be mark as dirty
	TEST_CHECK(Delta_Matrix_isDirty(A));

	// get internal matrices
	M   =  DELTA_MATRIX_M(A);
	DP  =  DELTA_MATRIX_DELTA_PLUS(A);
	DM  =  DELTA_MATRIX_DELTA_MINUS(A);

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// All matrices should be empty
	M_EMPTY();
	DM_EMPTY();
	DP_NOT_EMPTY();

	//--------------------------------------------------------------------------
	// set already existing entry
	//--------------------------------------------------------------------------

	// flush matrix
	Delta_Matrix_wait(A, false);

	// introduce existing entry
	info = Delta_Matrix_setElement_UINT64(A, x, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// All matrices should be empty
	M_EMPTY();
	DM_EMPTY();
	DP_NOT_EMPTY();

	// clean up
	Delta_Matrix_free(&A);
	TEST_CHECK(A == NULL);
}

// multiple delete scenarios
void test_RGMatrix_del() {
	GrB_Type       t                   =  GrB_UINT64;
	Delta_Matrix   A                   =  NULL;
	GrB_Matrix     M                   =  NULL;
	GrB_Matrix     DP                  =  NULL;
	GrB_Matrix     DM                  =  NULL;
	GrB_Info       info                =  GrB_SUCCESS;
	GrB_Index      nvals               =  0;
	GrB_Index      nrows               =  100;
	GrB_Index      ncols               =  100;
	GrB_Index      i                   =  0;
	GrB_Index      j                   =  1;
	uint64_t       x                   =  1;

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M                   =  DELTA_MATRIX_M(A);
	DP                  =  DELTA_MATRIX_DELTA_PLUS(A);
	DM                  =  DELTA_MATRIX_DELTA_MINUS(A);	

	//--------------------------------------------------------------------------
	// remove none flushed addition
	//--------------------------------------------------------------------------

	// set element at position i,j
	info = Delta_Matrix_setElement_UINT64(A, x, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// remove element at position i,j
	info = Delta_Matrix_removeElement_UINT64(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// matrix should be mark as dirty
	TEST_CHECK(Delta_Matrix_isDirty(A));

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// A should be empty
	Delta_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 0);

	// Matrix should be empty
	M_EMPTY();
	DM_EMPTY();
	DP_EMPTY();

	//--------------------------------------------------------------------------
	// remove flushed addition
	//--------------------------------------------------------------------------

	// set element at position i,j
	info = Delta_Matrix_setElement_UINT64(A, x, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// force sync
	// entry should migrated from 'delta-plus' to 'M'
	info = Delta_Matrix_wait(A, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// remove element at position i,j
	info = Delta_Matrix_removeElement_UINT64(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// A should be empty
	Delta_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 0);

	// Matrix should be empty
	M_NOT_EMPTY();
	DM_NOT_EMPTY();
	DP_EMPTY();

	//--------------------------------------------------------------------------
	// flush
	//--------------------------------------------------------------------------

	info = Delta_Matrix_wait(A, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// entry should be removed from both 'delta-minus' and 'M'
	// A should be empty
	Delta_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 0);

	// Matrix should be empty
	M_EMPTY();
	DM_EMPTY();
	DP_EMPTY();

	//--------------------------------------------------------------------------

	// commit an entry M[i,j] = 1
	// delete entry del DM[i,j] = true
	// re-introduce entry DM[i,j] = 0, M[i,j] = 2
	// delete entry DM[i,j] = true
	// commit
	// M[i,j] = 0, DP[i,j] = 0, DM[i,j] = 0

	//--------------------------------------------------------------------------
	// commit an entry M[i,j] = 1
	//--------------------------------------------------------------------------

	// set element at position i,j
	info = Delta_Matrix_setElement_UINT64(A, 1, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// force sync
	info = Delta_Matrix_wait(A, true);

	// M should contain a single element
	M_NOT_EMPTY();
	DP_EMPTY();
	DM_EMPTY();

	//--------------------------------------------------------------------------
	// delete entry del DM[i,j] = true
	//--------------------------------------------------------------------------

	// remove element at position i,j
	info = Delta_Matrix_removeElement_UINT64(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	M_NOT_EMPTY();
	DP_EMPTY();
	DM_NOT_EMPTY();

	//--------------------------------------------------------------------------
	// introduce an entry DP[i,j] = 2
	//--------------------------------------------------------------------------

	// set element at position i,j
	info = Delta_Matrix_setElement_UINT64(A, 2, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// M should contain a single element
	M_NOT_EMPTY();
	DP_EMPTY();
	DM_EMPTY();

	//--------------------------------------------------------------------------
	// commit
	//--------------------------------------------------------------------------

	// force sync
	info = Delta_Matrix_wait(A, true);

	//--------------------------------------------------------------------------
	// M[i,j] = 2, DP[i,j] = 0, DM[i,j] = 0
	//--------------------------------------------------------------------------

	info = Delta_Matrix_extractElement_UINT64(&x, A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_ASSERT(2 == x);

	M_NOT_EMPTY();
	DP_EMPTY();
	DM_EMPTY();

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
}

// multiple delete entry scenarios
void test_RGMatrix_del_entry() {
	GrB_Type    t               =  GrB_UINT64;
	Tensor      A               =  NULL;
	GrB_Matrix  M               =  NULL;
	GrB_Matrix  DP              =  NULL;
	GrB_Matrix  DM              =  NULL;
	GrB_Info    info            =  GrB_SUCCESS;
	GrB_Index   nvals           =  0;
	GrB_Index   nrows           =  100;
	GrB_Index   ncols           =  100;
	GrB_Index   i               =  0;
	GrB_Index   j               =  1;
	uint64_t    x               =  1;
	bool        entry_deleted   =  false;
	Edge        edges[1]        = {{.id = x, .src_id = i, .dest_id = j}};

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M  = DELTA_MATRIX_M(A);
	DP = DELTA_MATRIX_DELTA_PLUS(A);
	DM = DELTA_MATRIX_DELTA_MINUS(A);	

	//--------------------------------------------------------------------------
	// remove none existing entry ---- No longer supported ----
	//--------------------------------------------------------------------------
	// Tensor_RemoveElements(A, edges, 1, NULL);

	//--------------------------------------------------------------------------
	// remove none flushed addition
	//--------------------------------------------------------------------------

	// set element at position i,j
	Tensor_SetElement(A, i, j, x);

	// remove element at position i,j
	Tensor_RemoveElements(A, edges, 1, NULL);

	// matrix should be mark as dirty
	TEST_ASSERT(Delta_Matrix_isDirty(A));

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// A should be empty
	Delta_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 0);

	M_EMPTY();
	DM_EMPTY();
	DP_EMPTY();

	//--------------------------------------------------------------------------
	// remove flushed addition
	//--------------------------------------------------------------------------

	// set element at position i,j
	Tensor_SetElement(A, i, j, x);

	// force sync
	// entry should migrated from 'delta-plus' to 'M'
	info = Delta_Matrix_wait(A, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// remove element at position i,j
	Tensor_RemoveElements(A, edges, 1, NULL);

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// A should be empty
	Delta_Matrix_nvals(&nvals, A);
	TEST_CHECK(nvals == 0);

	M_NOT_EMPTY();
	DM_NOT_EMPTY();
	DP_EMPTY();

	//--------------------------------------------------------------------------
	// flush
	//--------------------------------------------------------------------------

	info = Delta_Matrix_wait(A, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// entry should be removed from both 'delta-minus' and 'M'
	// A should be empty
	Delta_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 0);

	M_EMPTY();
	DM_EMPTY();
	DP_EMPTY();

	//--------------------------------------------------------------------------

	// commit an entry M[i,j] = 1
	// delete entry del DM[i,j] = true
	// re-introduce entry DM[i,j] = 0, M[i,j] = 2
	// delete entry DM[i,j] = true
	// commit
	// M[i,j] = 0, DP[i,j] = 0, DM[i,j] = 0

	//--------------------------------------------------------------------------
	// commit an entry M[i,j] = 1
	//--------------------------------------------------------------------------

	// set element at position i,j
	Tensor_SetElement(A, i, j, 1);

	// force sync
	info = Delta_Matrix_wait(A, true);

	// M should contain a single element
	M_NOT_EMPTY();
	DP_EMPTY();
	DM_EMPTY();

	//--------------------------------------------------------------------------
	// delete entry del DM[i,j] = true
	//--------------------------------------------------------------------------

	// remove element at position i,j
	Tensor_RemoveElements(A, edges, 1, NULL);

	M_NOT_EMPTY();
	DP_EMPTY();
	DM_NOT_EMPTY();

	//--------------------------------------------------------------------------
	// introduce an entry DP[i,j] = 2
	//--------------------------------------------------------------------------

	// set element at position i,j
	Tensor_SetElement(A, i, j, 2);

	// M should contain a single element
	M_NOT_EMPTY();
	DP_EMPTY();
	DM_EMPTY();

	//--------------------------------------------------------------------------
	// commit
	//--------------------------------------------------------------------------

	// force sync
	info = Delta_Matrix_wait(A, true);

	//--------------------------------------------------------------------------
	// M[i,j] = 2, DP[i,j] = 0, DM[i,j] = 0
	//--------------------------------------------------------------------------

	M_NOT_EMPTY();
	DP_EMPTY();
	DM_EMPTY();

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	Tensor_free(&A);
	TEST_ASSERT(A == NULL);
}

void test_RGMatrix_set() {
	GrB_Type       t     = GrB_BOOL;
	Delta_Matrix   A     = NULL;
	GrB_Matrix     M     = NULL;
	GrB_Matrix     DP    = NULL;
	GrB_Matrix     DM    = NULL;
	GrB_Info       info  = GrB_SUCCESS;
	GrB_Index      nvals = 0;
	GrB_Index      nrows = 100;
	GrB_Index      ncols = 100;
	GrB_Index      i     = 0;
	GrB_Index      j     = 1;

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	M     = DELTA_MATRIX_M(A);
	DP    = DELTA_MATRIX_DELTA_PLUS(A);
	DM    = DELTA_MATRIX_DELTA_MINUS(A);	

	//--------------------------------------------------------------------------
	// Set element that marked for deletion
	//--------------------------------------------------------------------------

	// set element at position i,j
	info = Delta_Matrix_setElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// force sync
	// entry should migrated from 'delta-plus' to 'M'
	Delta_Matrix_wait(A, true);

	// set element at position i,j
	info = Delta_Matrix_removeElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);
	
	// set element at position i,j
	info = Delta_Matrix_setElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// A should be empty
	Delta_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 1);

	// M should contain a single element
	M_NOT_EMPTY();
	DM_EMPTY();
	DP_EMPTY();


	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
}

// flush simple addition
void test_RGMatrix_flus() {
	GrB_Type       t      =  GrB_BOOL;
	Delta_Matrix   A      =  NULL;
	GrB_Matrix     M      =  NULL;
	GrB_Matrix     DP     =  NULL;
	GrB_Matrix     DM     =  NULL;
	GrB_Info       info   =  GrB_SUCCESS;
	GrB_Index      nvals  =  0;
	GrB_Index      nrows  =  100;
	GrB_Index      ncols  =  100;
	GrB_Index      i      =  0;
	GrB_Index      j      =  1;
	bool           sync   =  false;

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i,j
	info = Delta_Matrix_setElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M  = DELTA_MATRIX_M(A);
	DP = DELTA_MATRIX_DELTA_PLUS(A);
	DM = DELTA_MATRIX_DELTA_MINUS(A);

	//--------------------------------------------------------------------------
	// flush matrix, no sync
	//--------------------------------------------------------------------------
	
	// wait, don't force sync
	sync = false;
	Delta_Matrix_wait(A, sync);

	M_EMPTY();
	DM_EMPTY();

	// DP should contain a single element
	DP_NOT_EMPTY();

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);

	Delta_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 1);

	M_NOT_EMPTY();
	DM_EMPTY();
	DP_EMPTY();

	// clean up
	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
}

//------------------------------------------------------------------------------
// transpose test
//------------------------------------------------------------------------------

// M[i,j] = x, M[i,j] = y
void test_GRMatrix_managed_transposed() {
	GrB_Type      t              =  GrB_UINT64;
	Delta_Matrix  T              =  NULL;  // A transposed
	GrB_Matrix    M              =  NULL;  // primary internal matrix
	GrB_Matrix    DP             =  NULL;  // delta plus internal matrix
	GrB_Matrix    DM             =  NULL;  // delta minus internal matrix
	GrB_Info      info           =  GrB_SUCCESS;
	GrB_Index     nvals          =  0;
	GrB_Index     nrows          =  100;
	GrB_Index     ncols          =  100;
	GrB_Index     i              =  0;
	GrB_Index     j              =  1;
	uint64_t      x              =  0;  // M[i,j] = x
	bool          b              =  false;
	bool          entry_deleted  =  false;
	Edge          edges[1]       = {{.id = x, .src_id = i, .dest_id = j}};

	//--------------------------------------------------------------------------
	// create RGMatrix
	//--------------------------------------------------------------------------

	Tensor A = Tensor_new(nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	// make sure transposed was created
	T = Delta_Matrix_getTranspose(A);
	TEST_CHECK(T != A);
	TEST_CHECK(T != NULL);

	// get internal matrices
	M  = DELTA_MATRIX_M(T);
	DP = DELTA_MATRIX_DELTA_PLUS(T);
	DM = DELTA_MATRIX_DELTA_MINUS(T);

	//--------------------------------------------------------------------------
	// set element at position i,j
	//--------------------------------------------------------------------------

	Tensor_SetElement(A, i, j, x);

	// make sure element at position j,i exists
	info = Delta_Matrix_extractElement_BOOL(&b, T, j, i);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_CHECK(true == b);
	
	// matrix should contain a single element
	Delta_Matrix_nvals(&nvals, T);
	TEST_CHECK(nvals == 1);

	// matrix should be mark as dirty
	TEST_CHECK(Delta_Matrix_isDirty(T));

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	M_EMPTY();
	DM_EMPTY();

	// TDP should contain a single element
	DP_NOT_EMPTY();

	//--------------------------------------------------------------------------
	// flush matrix
	//--------------------------------------------------------------------------

	Delta_Matrix_wait(A, true);

	// flushing 'A' should flush 'T' aswell

	// TM should contain a single element
	M_NOT_EMPTY();
	DM_EMPTY();
	DP_EMPTY();

	//--------------------------------------------------------------------------
	// delete element at position i,j
	//--------------------------------------------------------------------------
	
	Tensor_RemoveElements(A, edges, 1, NULL);

	// matrix should be mark as dirty
	TEST_CHECK(Delta_Matrix_isDirty(T));

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// TM should contain a single element
	M_NOT_EMPTY();

	// TDM should contain a single element
	DM_NOT_EMPTY();
	DP_EMPTY();

	//--------------------------------------------------------------------------
	// flush matrix
	//--------------------------------------------------------------------------

	// flushing 'A' should flush 'T' aswell

	Delta_Matrix_wait(A, true);

	M_EMPTY();
	DM_EMPTY();
	DP_EMPTY();

	//--------------------------------------------------------------------------
	// delete entry at position i,j
	//--------------------------------------------------------------------------

	Tensor_SetElement(A, i, j, x);
	Tensor_SetElement(A, i, j, x + 1);

	Tensor_RemoveElements(A, edges, 1, NULL);

	// make sure element at position j,i exists
	info = Delta_Matrix_extractElement_BOOL(&b, T, j, i);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_CHECK(true == b);

	edges[0].id = x + 1;
	Tensor_RemoveElements(A, edges, 1, NULL);
	info = Delta_Matrix_extractElement_BOOL(&b, T, j, i);
	TEST_ASSERT(info == GrB_NO_VALUE);

	//--------------------------------------------------------------------------
	// delete flushed entry at position i,j
	//--------------------------------------------------------------------------

	Tensor_SetElement(A, i, j, x);
	Tensor_SetElement(A, i, j, x + 1);

	Delta_Matrix_wait(A, true);

	edges[0].id = x;
	Tensor_RemoveElements(A, edges, 1, NULL);

	// make sure element at position j,i exists
	info = Delta_Matrix_extractElement_BOOL(&b, T, j, i);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_CHECK (true == b);

	edges[0].id = x + 1;
	Tensor_RemoveElements(A, edges, 1, NULL);

	info = Delta_Matrix_extractElement_BOOL(&b, T, j, i);
	TEST_ASSERT(info == GrB_NO_VALUE);

	//--------------------------------------------------------------------------
	// revive deleted entry at position i,j
	//--------------------------------------------------------------------------

	Tensor_SetElement(A, i, j, x);
	Delta_Matrix_wait(A, true);

	info = Delta_Matrix_removeElement_UINT64(A, i, j);

	Tensor_SetElement(A, i, j, x);
	TEST_ASSERT(info == GrB_SUCCESS);

	// make sure element at position j,i exists
	info = Delta_Matrix_extractElement_BOOL(&b, T, j, i);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_CHECK (true == b);

	// clean up
	Tensor_free(&A);
	TEST_ASSERT(A == NULL);
}

//------------------------------------------------------------------------------
// fuzzy test compare Delta_Matrix to GrB_Matrix
//------------------------------------------------------------------------------

void test_RGMatrix_fuzzy() {
	GrB_Type       t           =  GrB_BOOL;
	Delta_Matrix   A           =  NULL;
	Delta_Matrix   T           =  NULL;  // A transposed
	GrB_Matrix     M           =  NULL;  // primary internal matrix
	GrB_Matrix     MT          =  NULL;
	GrB_Matrix     N           =  NULL;
	GrB_Matrix     NT          =  NULL;
	GrB_Info       info        =  GrB_SUCCESS;
	GrB_Index      nrows       =  100;
	GrB_Index      ncols       =  100;
	GrB_Index      i           =  0;
	GrB_Index      j           =  1;
	GrB_Index*     II          =  NULL;
	GrB_Index*     J           =  NULL;
	uint32_t       operations  =  10000;

	//--------------------------------------------------------------------------
	// create RGMatrix
	//--------------------------------------------------------------------------

	srand(time(0));

	II = (GrB_Index*) malloc(sizeof(GrB_Index) * operations);
	J  = (GrB_Index*) malloc(sizeof(GrB_Index) * operations);

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// make sure transposed was created
	T = Delta_Matrix_getTranspose(A);
	TEST_ASSERT(T != A);
	TEST_ASSERT(T != NULL);

	// get internal matrices
	M   =  DELTA_MATRIX_M(A);
	MT  =  DELTA_MATRIX_M(T);

	info = GrB_Matrix_new(&N, t, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&NT, t, ncols, nrows);
	TEST_ASSERT(info == GrB_SUCCESS);

	uint additions = 0;
	for (size_t index = 0; index < operations; index++)
	{
		if (index < 10 || rand() % 100 > 20)
		{
			i = rand() % nrows;
			j = rand() % ncols;

			//------------------------------------------------------------------
			// set element at position i,j
			//------------------------------------------------------------------

			info = Delta_Matrix_setElement_BOOL(A, i, j);
			TEST_ASSERT(info == GrB_SUCCESS);

			info = GrB_Matrix_setElement_BOOL(N, true, i, j);
			TEST_ASSERT(info == GrB_SUCCESS);

			II[additions] = i;
			J[additions] = j;
			additions++;
		}
		else
		{
			uint32_t delete_pos = rand() % additions;
			i = II[delete_pos];
			j = J[delete_pos];

			//------------------------------------------------------------------
			// delete element at position i,j
			//------------------------------------------------------------------
			
			Delta_Matrix_removeElement_BOOL(A, i, j);

			GrB_Matrix_removeElement(N, i, j);
		}
	}

	//--------------------------------------------------------------------------
	// flush matrix
	//--------------------------------------------------------------------------

	Delta_Matrix_wait(A, true);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	info = GrB_transpose(NT, NULL, NULL, N, NULL);
	TEST_ASSERT(info == GrB_SUCCESS);

	ASSERT_GrB_Matrices_EQ(M, N);
	ASSERT_GrB_Matrices_EQ(MT, NT);

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);

	info = GrB_Matrix_free(&N);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_free(&NT);
	TEST_ASSERT(info == GrB_SUCCESS);

	free(II);
	free(J);
}

// test exporting Delta_Matrix to GrB_Matrix when there are no pending changes
// by exporting the matrix after flushing
void test_RGMatrix_export_no_changes() {
	GrB_Type       t      =  GrB_BOOL;
	Delta_Matrix   A      =  NULL; 
	GrB_Matrix     M      =  NULL;
	GrB_Matrix     N      =  NULL;  // exported matrix 
	GrB_Info       info   =  GrB_SUCCESS;
	GrB_Index      i      =  0;
	GrB_Index      j      =  1;
	GrB_Index      nrows  =  100;
	GrB_Index      ncols  =  100;
	bool           sync   =  false;

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M = DELTA_MATRIX_M(A);

	//--------------------------------------------------------------------------
	// export empty matrix
	//--------------------------------------------------------------------------

	info = Delta_Matrix_export(&N, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	ASSERT_GrB_Matrices_EQ(M, N);
	GrB_Matrix_free(&N);

	//--------------------------------------------------------------------------
	// export none empty matrix
	//--------------------------------------------------------------------------

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
	// validation
	//--------------------------------------------------------------------------

	info = Delta_Matrix_export(&N, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	ASSERT_GrB_Matrices_EQ(M, N);

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	Delta_Matrix_free(&A);
	GrB_Matrix_free(&N);
}

// test exporting Delta_Matrix to GrB_Matrix when there are pending changes
// by exporting the matrix after making changes
// then flush the matrix and compare the internal matrix to the exported matrix
void test_RGMatrix_export_pending_changes() {
	GrB_Type       t      =  GrB_BOOL;
	Delta_Matrix   A      =  NULL;
	GrB_Matrix     M      =  NULL;
	GrB_Matrix     N      =  NULL;  // exported matrix
	GrB_Info       info   =  GrB_SUCCESS;
	GrB_Index      nrows  =  100;
	GrB_Index      ncols  =  100;
	bool           sync   =  false;

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M = DELTA_MATRIX_M(A);

	// set elements
	info = Delta_Matrix_setElement_BOOL(A, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_BOOL(A, 1, 1);
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

	// remove element at position 0,0
	info = Delta_Matrix_removeElement_BOOL(A, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position 2,2
	info = Delta_Matrix_setElement_BOOL(A, 2, 2);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// export matrix
	//--------------------------------------------------------------------------

	info = Delta_Matrix_export(&N, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	ASSERT_GrB_Matrices_EQ(M, N);

	// clean up
	GrB_Matrix_free(&N);
	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
}

void test_Delta_Matrix_export_structure() {
	GrB_Type       t      =  GrB_UINT64;
	Delta_Matrix   A      =  NULL;
	GrB_Matrix     M      =  NULL;
	GrB_Matrix     N      =  NULL;  // exported matrix
	GrB_Matrix     C      =  NULL;  
	GrB_Info       info   =  GrB_SUCCESS;
	GrB_Index      nrows  =  100;
	GrB_Index      ncols  =  100;
	GrB_Index      nvals  =  0;
	GrB_Index      mvals  =  0;
	bool           sync   =  false;

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M = DELTA_MATRIX_M(A);

	// set elements
	info = Delta_Matrix_setElement_UINT64(A, 0, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_UINT64(A, 0, 1, 1);
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

	// remove element at position 0,0
	info = Delta_Matrix_removeElement_UINT64(A, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position 2,2
	info = Delta_Matrix_setElement_UINT64(A, 15, 2, 2);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// export matrix
	//--------------------------------------------------------------------------

	info = Delta_Matrix_export_structure(&N, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	GrB_Matrix_nvals(&nvals, N);
	TEST_ASSERT(info == GrB_SUCCESS);

	GrB_Matrix_nvals(&mvals, M);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_CHECK (nvals == mvals);

	//--------------------------------------------------------------------------
	// structure(A) == structure(B)
	//--------------------------------------------------------------------------

	info = GrB_Matrix_new(&C, GrB_BOOL, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_eWiseMult_BinaryOp(
		C, NULL, NULL, GrB_ONEB_BOOL, N, M, NULL);
	TEST_ASSERT(info == GrB_SUCCESS);

	GrB_Matrix_nvals(&nvals, C);
	TEST_ASSERT(info == GrB_SUCCESS);
	TEST_CHECK (nvals == mvals);

	// clean up
	GrB_Matrix_free(&N);
	Delta_Matrix_free(&A);
	TEST_CHECK (A == NULL);
}

void test_RGMatrix_copy() {
	GrB_Type       t      =  GrB_BOOL;
	Delta_Matrix   A      =  NULL;
	Delta_Matrix   B      =  NULL;
	GrB_Matrix     A_M    =  NULL;
	GrB_Matrix     B_M    =  NULL;
	GrB_Matrix     A_DP   =  NULL;
	GrB_Matrix     B_DP   =  NULL;
	GrB_Matrix     A_DM   =  NULL;
	GrB_Matrix     B_DM   =  NULL;
	GrB_Info       info   =  GrB_SUCCESS;
	GrB_Index      nrows  =  100;
	GrB_Index      ncols  =  100;
	bool           sync   =  false;

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_Matrix_new(&B, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set elements
	info = Delta_Matrix_setElement_BOOL(A, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_BOOL(A, 1, 1);
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

	// remove element at position 0,0
	info = Delta_Matrix_removeElement_BOOL(A, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position 2,2
	info = Delta_Matrix_setElement_BOOL(A, 2, 2);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// copy matrix
	//--------------------------------------------------------------------------

	info = Delta_Matrix_copy(B, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	A_M  = DELTA_MATRIX_M(A);
	B_M  = DELTA_MATRIX_M(B);
	A_DP = DELTA_MATRIX_DELTA_PLUS(A);
	B_DP = DELTA_MATRIX_DELTA_PLUS(B);
	A_DM = DELTA_MATRIX_DELTA_MINUS(A);
	B_DM = DELTA_MATRIX_DELTA_MINUS(B);
	
	ASSERT_GrB_Matrices_EQ(A_M, B_M);
	ASSERT_GrB_Matrices_EQ(A_DP, B_DP);
	ASSERT_GrB_Matrices_EQ(A_DM, B_DM);

	//--------------------------------------------------------------------------
	// free
	//--------------------------------------------------------------------------
	Delta_Matrix_free(&A);
	info = Delta_Matrix_new(&A, GrB_UINT64, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	Delta_Matrix_clear(A);


	// set elements
	info = Delta_Matrix_setElement_UINT64(A, 0, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_UINT64(A, 0, 1, 1);
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

	// remove element at position 0,0
	info = Delta_Matrix_removeElement_UINT64(A, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position 2,2
	info = Delta_Matrix_setElement_UINT64(A, 1, 2, 2);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// copy matrix
	//--------------------------------------------------------------------------

	info = Delta_Matrix_copy(B, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	A_M  = DELTA_MATRIX_M(A);
	B_M  = DELTA_MATRIX_M(B);
	A_DP = DELTA_MATRIX_DELTA_PLUS(A);
	B_DP = DELTA_MATRIX_DELTA_PLUS(B);
	A_DM = DELTA_MATRIX_DELTA_MINUS(A);
	B_DM = DELTA_MATRIX_DELTA_MINUS(B);
	
	CHECK_GrB_Matrices_EQ(A_M, B_M, GrB_ONEB_BOOL);
	CHECK_GrB_Matrices_EQ(A_DP, B_DP, GrB_ONEB_BOOL);
	CHECK_GrB_Matrices_EQ(A_DM, B_DM, GrB_ONEB_BOOL);

	// clean up
	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_Matrix_free(&B);
	TEST_ASSERT(B == NULL);
}

void test_Delta_Matrix_add() {
	GrB_Type      t      =  GrB_BOOL;
	Delta_Matrix  A      =  NULL;
	Delta_Matrix  B      =  NULL;
	Delta_Matrix  C      =  NULL;
	GrB_Matrix    A_GB   =  NULL;
	GrB_Matrix    B_GB   =  NULL;
	GrB_Matrix    C_GB   =  NULL;
	GrB_Matrix    C_M    =  NULL;
	GrB_Matrix    t_Ap   =  NULL;
	GrB_Matrix    t_Bp   =  NULL;
	GrB_Matrix    temp   =  NULL;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     nrows  =  4;
	GrB_Index     ncols  =  4;
	bool          sync   =  false;

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_Matrix_new(&B, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_Matrix_new(&C, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&t_Ap, t, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);
	
	info = GrB_Matrix_new(&t_Bp, t, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&A_GB, t, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&B_GB, t, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&C_GB, t, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	// rows of A / columns of B: [M, deleted, new, non-existent]
	// set elements
	for(int i = 0; i < 4; i++) {
		info = Delta_Matrix_setElement_BOOL(A, 0, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_setElement_BOOL(A, 1, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_setElement_BOOL(B, i, 0);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_setElement_BOOL(B, i, 1);
		TEST_ASSERT(info == GrB_SUCCESS);

		info = GrB_Matrix_setElement_BOOL(A_GB, true, 0, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_setElement_BOOL(A_GB, true, 1, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_setElement_BOOL(B_GB, true, i, 0);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_setElement_BOOL(B_GB, true, i, 1);
		TEST_ASSERT(info == GrB_SUCCESS);
	}

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);
	Delta_Matrix_wait(B, sync);

	//--------------------------------------------------------------------------
	// A + B
	//--------------------------------------------------------------------------

	info = Delta_eWiseAdd(C, GrB_LOR, A, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	Delta_Matrix_validate(C);
	Delta_Matrix_wait(C, true);

	info = GrB_Matrix_eWiseAdd_BinaryOp(
		C_GB, NULL, NULL, GrB_LOR, A_GB, B_GB, NULL);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	C_M  = DELTA_MATRIX_M(C);

	ASSERT_GrB_Matrices_EQ(C_M, C_GB);

	//--------------------------------------------------------------------------
	// set pending additions
	//--------------------------------------------------------------------------
	for(int i = 0; i < 4; i++) {
		info = Delta_Matrix_setElement_BOOL(A, 2, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_setElement_BOOL(B, i, 2);
		TEST_ASSERT(info == GrB_SUCCESS);

		info = GrB_Matrix_setElement_BOOL(A_GB, true, 2, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_setElement_BOOL(B_GB, true, i, 2);
		TEST_ASSERT(info == GrB_SUCCESS);
	}
	
	//--------------------------------------------------------------------------
	// A + B
	//--------------------------------------------------------------------------

	info = Delta_eWiseAdd(C, GrB_LOR, A, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	Delta_Matrix_validate(C);
	Delta_Matrix_wait(C, true);

	info = GrB_Matrix_eWiseAdd_BinaryOp(
		C_GB, NULL, NULL, GrB_LOR, A_GB, B_GB, NULL);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	C_M  = DELTA_MATRIX_M(C);

	ASSERT_GrB_Matrices_EQ(C_M, C_GB);
	
	//--------------------------------------------------------------------------
	// set pending removals
	//--------------------------------------------------------------------------
	for(int i = 0; i < 4; i++)
	{
		info = Delta_Matrix_removeElement_BOOL(A, 1, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_removeElement_BOOL(B, i, 1);
		TEST_ASSERT(info == GrB_SUCCESS);

		info = GrB_Matrix_removeElement(A_GB, 1, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_removeElement(B_GB, i, 1);
		TEST_ASSERT(info == GrB_SUCCESS);
	}

	//--------------------------------------------------------------------------
	// A + B
	//--------------------------------------------------------------------------

	info = Delta_eWiseAdd(C, GrB_LOR, A, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	Delta_Matrix_validate(C);
	Delta_Matrix_wait(C, sync);

	info = GrB_Matrix_eWiseAdd_BinaryOp(
		C_GB, NULL, NULL, GrB_LOR, A_GB, B_GB, NULL);
	TEST_ASSERT(info == GrB_SUCCESS);
	
	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	C_M  = DELTA_MATRIX_M(C);

	ASSERT_GrB_Matrices_EQ(C_M, C_GB);

	// clean up
	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_Matrix_free(&B);
	TEST_ASSERT(B == NULL);
	Delta_Matrix_free(&C);
	TEST_ASSERT(C == NULL);

	GrB_Matrix_free(&A_GB);
	TEST_ASSERT(A_GB == NULL);
	GrB_Matrix_free(&B_GB);
	TEST_ASSERT(B_GB == NULL);
	GrB_Matrix_free(&C_GB);
	TEST_ASSERT(C_GB == NULL);
}

void test_Delta_Matrix_union() {
	GrB_Type      t      =  GrB_UINT64;
	Delta_Matrix  A      =  NULL;
	Delta_Matrix  B      =  NULL;
	Delta_Matrix  C      =  NULL;
	GrB_Matrix    A_GB   =  NULL;
	GrB_Matrix    B_GB   =  NULL;
	GrB_Matrix    C_GB   =  NULL;
	GrB_Matrix    C_M    =  NULL;
	GrB_Scalar    empty  =  NULL;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     nrows  =  4;
	GrB_Index     ncols  =  4;
	bool          sync   =  false;

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_Matrix_new(&B, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_Matrix_new(&C, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&A_GB, t, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&B_GB, t, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&C_GB, t, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Scalar_new(&empty, GrB_UINT64);	
	info = GrB_Scalar_setElement_UINT64(empty, U64_ZOMBIE);	


	// set elements
	for(int i = 0; i < 4; i++)
	{
		info = Delta_Matrix_setElement_UINT64(A, (uint64_t) 1, 0, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_setElement_UINT64(A, (uint64_t) 1, 1, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_setElement_UINT64(B, (uint64_t) 1, i, 0);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_setElement_UINT64(B, (uint64_t) 1, i, 1);
		TEST_ASSERT(info == GrB_SUCCESS);

		info = GrB_Matrix_setElement_UINT64(A_GB, (uint64_t) 1, 0, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_setElement_UINT64(A_GB, (uint64_t) 1, 1, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_setElement_UINT64(B_GB, (uint64_t) 1, i, 0);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_setElement_UINT64(B_GB, (uint64_t) 1, i, 1);
		TEST_ASSERT(info == GrB_SUCCESS);
	}

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);
	Delta_Matrix_wait(B, sync);

	//--------------------------------------------------------------------------
	// set pending additions
	//--------------------------------------------------------------------------
	for(int i = 0; i < 4; i++) {
		info = Delta_Matrix_setElement_UINT64(A, (uint64_t) 2, 2, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_setElement_UINT64(B, (uint64_t) 2, i, 2);
		TEST_ASSERT(info == GrB_SUCCESS);

		info = GrB_Matrix_setElement_UINT64(A_GB, (uint64_t) 2, 2, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_setElement_UINT64(B_GB, (uint64_t) 2, i, 2);
		TEST_ASSERT(info == GrB_SUCCESS);
	}
	
	//--------------------------------------------------------------------------
	// A + B
	//--------------------------------------------------------------------------

	info = Delta_eWiseUnion(C, GrB_BOR_UINT64, A, empty, B, empty);
	TEST_ASSERT(info == GrB_SUCCESS);

	Delta_Matrix_wait(C, true);

	info = GxB_Matrix_eWiseUnion(
		C_GB, NULL, NULL, GrB_BOR_UINT64, A_GB, empty, B_GB, empty, NULL);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	C_M  = DELTA_MATRIX_M(C);

	ASSERT_GrB_Matrices_EQ(C_M, C_GB);
	
	//--------------------------------------------------------------------------
	// set pending removals
	//--------------------------------------------------------------------------
	for(int i = 0; i < 4; i++)
	{
		info = Delta_Matrix_removeElement_UINT64(A, 1, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = Delta_Matrix_removeElement_UINT64(B, i, 1);
		TEST_ASSERT(info == GrB_SUCCESS);

		info = GrB_Matrix_removeElement(A_GB, 1, i);
		TEST_ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_removeElement(B_GB, i, 1);
		TEST_ASSERT(info == GrB_SUCCESS);
	}

	//--------------------------------------------------------------------------
	// A + B
	//--------------------------------------------------------------------------

	info = Delta_eWiseUnion(C, GrB_BOR_UINT64, A, empty, B, empty);
	TEST_ASSERT(info == GrB_SUCCESS);

	Delta_Matrix_wait(C, true);

	info = GxB_Matrix_eWiseUnion(
		C_GB, NULL, NULL, GrB_BOR_UINT64, A_GB, empty, B_GB, empty, NULL);
	TEST_ASSERT(info == GrB_SUCCESS);
	
	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	ASSERT_GrB_Matrices_EQ(C_M, C_GB);
	
	// clean up
	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_Matrix_free(&B);
	TEST_ASSERT(B == NULL);
	Delta_Matrix_free(&C);
	TEST_ASSERT(C == NULL);

	GrB_Matrix_free(&A_GB);
	TEST_ASSERT(A_GB == NULL);
	GrB_Matrix_free(&B_GB);
	TEST_ASSERT(B_GB == NULL);
	GrB_Matrix_free(&C_GB);
	TEST_ASSERT(C_GB == NULL);
}

void test_Delta_Matrix_mxm() {
	GrB_Type      t      =  GrB_BOOL;
	GrB_Matrix    A      =  NULL;
	Delta_Matrix  B      =  NULL;
	GrB_Matrix    B_GB   =  NULL;
	GrB_Matrix    C      =  NULL;
	GrB_Matrix    D      =  NULL;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     nrows  =  4;
	GrB_Index     ncols  =  256;
	bool          sync   =  false;

	info = GrB_Matrix_new(&A, t, nrows, 4);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_Matrix_new(&B, t, 4, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&B_GB, t, 4, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&C, t, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&D, t, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	// The rows of B are every permutation of entry, deleted, new, empty.
	for(int i = 0; i < 256; i++)
	{
		if(((i / 128) % 2) == 0) {
			info = Delta_Matrix_setElement_BOOL(B, 0, i);
			TEST_ASSERT(info == GrB_SUCCESS);
			info = GrB_Matrix_setElement_BOOL(B_GB, true, 0, i);
			TEST_ASSERT(info == GrB_SUCCESS);
		}
		if(((i / 32) % 2) == 0) {
			info = Delta_Matrix_setElement_BOOL(B, 1, i);
			TEST_ASSERT(info == GrB_SUCCESS);
			info = GrB_Matrix_setElement_BOOL(B_GB, true, 1, i);
			TEST_ASSERT(info == GrB_SUCCESS);
		}
		if(((i / 8) % 2) == 0) {
			info = Delta_Matrix_setElement_BOOL(B, 2, i);
			TEST_ASSERT(info == GrB_SUCCESS);
			info = GrB_Matrix_setElement_BOOL(B_GB, true, 2, i);
			TEST_ASSERT(info == GrB_SUCCESS);
		}
		if(((i / 2) % 2) == 0) {
			info = Delta_Matrix_setElement_BOOL(B, 3, i);
			TEST_ASSERT(info == GrB_SUCCESS);
			info = GrB_Matrix_setElement_BOOL(B_GB, true, 3, i);
			TEST_ASSERT(info == GrB_SUCCESS);
		}
	}

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(B, sync);

	//--------------------------------------------------------------------------
	// set pending changes
	//--------------------------------------------------------------------------
	for(int i = 0; i < 256; i++)
	{
		switch ((i/64) % 4){
			case 1:
				info = Delta_Matrix_removeElement_BOOL(B, 0, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				info = GrB_Matrix_removeElement(B_GB, 0, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				break;
			case 2:
				info = Delta_Matrix_setElement_BOOL(B, 0, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				info = GrB_Matrix_setElement_BOOL(B_GB, true, 0, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				break;
			default:
				break;
		}

		switch ((i/16) % 4){
			case 1:
				info = Delta_Matrix_removeElement_BOOL(B, 1, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				info = GrB_Matrix_removeElement(B_GB, 1, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				break;
			case 2:
				info = Delta_Matrix_setElement_BOOL(B, 1, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				info = GrB_Matrix_setElement_BOOL(B_GB, true, 1, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				break;
			default:
				break;
		}

		switch ((i/4) % 4){
			case 1:
				info = Delta_Matrix_removeElement_BOOL(B, 2, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				info = GrB_Matrix_removeElement(B_GB, 2, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				break;
			case 2:
				info = Delta_Matrix_setElement_BOOL(B, 2, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				info = GrB_Matrix_setElement_BOOL(B_GB, true, 2, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				break;
			default:
				break;
		}

		switch ((i) % 4){
			case 1:
				info = Delta_Matrix_removeElement_BOOL(B, 3, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				info = GrB_Matrix_removeElement(B_GB, 3, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				break;
			case 2:
				info = Delta_Matrix_setElement_BOOL(B, 3, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				info = GrB_Matrix_setElement_BOOL(B_GB, true, 3, i);
				TEST_ASSERT(info == GrB_SUCCESS);
				break;
			default:
				break;
		}
	}
	//--------------------------------------------------------------------------
	// Set A 
	//--------------------------------------------------------------------------
	for(int i = 0; i < 4; ++i)
		for(int j = i; j < 4; ++j){
			GrB_Matrix_setElement_BOOL(A, true, i, j);
			TEST_ASSERT(info == GrB_SUCCESS);
		}

	//--------------------------------------------------------------------------
	// AB via identity
	//--------------------------------------------------------------------------

	info = Delta_mxm_identity(C, GrB_LOR_LAND_SEMIRING_BOOL, A, B);
	TEST_ASSERT(info == GrB_SUCCESS);
	
	info = GrB_mxm(D, NULL, NULL, GrB_LOR_LAND_SEMIRING_BOOL, A, B_GB, NULL);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	GrB_Matrix_select_BOOL(C, NULL, NULL, GrB_VALUEEQ_BOOL, C, true, NULL);
	bool d_ok = true;
	GrB_Matrix_reduce_BOOL(&d_ok, NULL, GrB_LAND_MONOID_BOOL, D, NULL);
	TEST_ASSERT(d_ok);
	
	ASSERT_GrB_Matrices_EQ(C, D);

	//--------------------------------------------------------------------------
	// AB Numerically
	//--------------------------------------------------------------------------
	GrB_Matrix_clear(C);
	info = Delta_mxm_count(C, GxB_PLUS_PAIR_UINT64, A, B);
	TEST_ASSERT(info == GrB_SUCCESS);
	ASSERT_GrB_Matrices_EQ(C, D);
	
	// clean up
	GrB_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_Matrix_free(&B);
	TEST_ASSERT(B == NULL);
	GrB_Matrix_free(&B_GB);
	TEST_ASSERT(B_GB == NULL);
	GrB_Matrix_free(&C);
	TEST_ASSERT(C == NULL);
	GrB_Matrix_free(&D);
	TEST_ASSERT(D == NULL);
}

void test_RGMatrix_mxm() {
	GrB_Type      t      =  GrB_BOOL;
	Delta_Matrix  A      =  NULL;
	Delta_Matrix  B      =  NULL;
	Delta_Matrix  C      =  NULL;
	Delta_Matrix  D      =  NULL;
	GrB_Matrix    C_M    =  NULL;
	GrB_Matrix    D_M    =  NULL;
	GrB_Info      info   =  GrB_SUCCESS;
	GrB_Index     nrows  =  100;
	GrB_Index     ncols  =  100;
	bool          sync   =  false;

	info = Delta_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_Matrix_new(&B, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_Matrix_new(&C, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = Delta_Matrix_new(&D, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set elements
	info = Delta_Matrix_setElement_BOOL(A, 0, 1);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_BOOL(A, 2, 3);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_BOOL(B, 1, 2);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_BOOL(B, 3, 4);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);
	Delta_Matrix_wait(B, sync);

	//--------------------------------------------------------------------------
	// set pending changes
	//--------------------------------------------------------------------------

	// remove element at position 0,0
	info = Delta_Matrix_removeElement_BOOL(B, 1, 2);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position 2,2
	info = Delta_Matrix_setElement_BOOL(B, 1, 3);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// mxm matrix
	//--------------------------------------------------------------------------

	info = Delta_mxm(C, GxB_ANY_PAIR_BOOL, A, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	Delta_Matrix_wait(B, sync);

	info = Delta_mxm(D, GxB_ANY_PAIR_BOOL, A, B);
	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	C_M  = DELTA_MATRIX_M(C);
	D_M  = DELTA_MATRIX_M(D);
	
	ASSERT_GrB_Matrices_EQ(C_M, D_M);


	// set elements
	Delta_Matrix_clear(A);
	Delta_Matrix_clear(B);
	Delta_Matrix_clear(C);
	Delta_Matrix_clear(D);
	info = Delta_Matrix_setElement_BOOL(A, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_BOOL(A, 0, 1);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_setElement_BOOL(A, 1, 1);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	Delta_Matrix_wait(A, sync);
	Delta_Matrix_copy(B, A);


	//--------------------------------------------------------------------------
	// set pending changes
	//--------------------------------------------------------------------------

	// remove element at position 0,0
	info = Delta_Matrix_removeElement_BOOL(B, 1, 1);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// mxm matrix
	//--------------------------------------------------------------------------

	info = Delta_mxm(C, GxB_ANY_PAIR_BOOL, A, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	Delta_Matrix_wait(B, sync);

	info = Delta_mxm(D, GxB_ANY_PAIR_BOOL, A, B);
	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	C_M  = DELTA_MATRIX_M(C);
	D_M  = DELTA_MATRIX_M(D);
	ASSERT_GrB_Matrices_EQ(C_M, D_M);

	// clean up
	Delta_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	Delta_Matrix_free(&B);
	TEST_ASSERT(B == NULL);
	Delta_Matrix_free(&C);
	TEST_ASSERT(C == NULL);
	Delta_Matrix_free(&D);
	TEST_ASSERT(C == NULL);
}

void test_RGMatrix_resize() {
	Delta_Matrix  A        =  NULL;
	Delta_Matrix  T        =  NULL;
	GrB_Info      info     =  GrB_SUCCESS;
	GrB_Type      t        =  GrB_UINT64;
	GrB_Index     nrows    =  10;
	GrB_Index     ncols    =  20;

	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
	T = Delta_Matrix_getTranspose(A);

	GrB_Index  A_nrows;
	GrB_Index  A_ncols;
	GrB_Index  T_nrows;
	GrB_Index  T_ncols;

	// verify A and T dimensions
	Delta_Matrix_nrows(&A_nrows, A);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_ncols(&A_ncols, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(A_nrows == nrows);
	TEST_ASSERT(A_ncols == ncols);

	Delta_Matrix_nrows(&T_nrows, T);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_ncols(&T_ncols, T);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(T_nrows == ncols);
	TEST_ASSERT(T_ncols == nrows);

	// resize matrix, increase size by 2
	nrows *= 2;
	ncols *= 2;

	info = Delta_Matrix_resize(A, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	// verify A and T dimensions
	Delta_Matrix_nrows(&A_nrows, A);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_ncols(&A_ncols, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(A_nrows == nrows);
	TEST_ASSERT(A_ncols == ncols);

	Delta_Matrix_nrows(&T_nrows, T);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_ncols(&T_ncols, T);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(T_nrows == ncols);
	TEST_ASSERT(T_ncols == nrows);

	// resize matrix decrease size by 2
	nrows /= 2;
	ncols /= 2;

	info = Delta_Matrix_resize(A, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	// verify A and T dimensions
	Delta_Matrix_nrows(&A_nrows, A);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_ncols(&A_ncols, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(A_nrows == nrows);
	TEST_ASSERT(A_ncols == ncols);

	Delta_Matrix_nrows(&T_nrows, T);
	TEST_ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_ncols(&T_ncols, T);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(T_nrows == ncols);
	TEST_ASSERT(T_ncols == nrows);

	Delta_Matrix_free(&A);
}

void test_RGMatrix_cache_transpose (){
	Delta_Matrix A     = make_test_matrix(false);
	GrB_Info     info  = GrB_SUCCESS;

	// verify the matrix
	Delta_Matrix_validate(A);

	// cache transpose
	info = Delta_cache_transpose(A);
	TEST_ASSERT(info == GrB_SUCCESS);

	// verify transpose is cached
	Delta_Matrix T = Delta_Matrix_getTranspose(A);
	TEST_CHECK (DELTA_MATRIX_MAINTAIN_TRANSPOSE(A));
	TEST_CHECK (T != NULL);
	Delta_Matrix_validate(T);

	// free A, T should be freed as well
	Delta_Matrix_free(&A);
	TEST_CHECK(A == NULL);
}

void test_RGMatrix_mxv (){
	Delta_Matrix A     = make_test_matrix(false);
	GrB_Matrix   GB_A  = NULL;
	GrB_Info     info  = GrB_SUCCESS;
	GrB_Index    nrows = 4;
	GrB_Vector   x     = NULL;
	GrB_Vector   y1    = NULL;
	GrB_Vector   y2    = NULL;
	GrB_Vector   y3    = NULL;

	Delta_Matrix_export(&GB_A, A);
	
	GrB_Vector_new(&x, GrB_BOOL, 4);
	GrB_Vector_new(&y1, GrB_UINT64, 4);
	GrB_Vector_new(&y2, GrB_UINT64, 4);
	GrB_Vector_new(&y3, GrB_UINT64, 4);


	for(int i = 0; i < 2; ++i){
		GrB_Descriptor desc = i == 0 ? NULL : GrB_DESC_T0;
		info = GrB_Vector_assign_BOOL(x, NULL, NULL, true, GrB_ALL, 4, desc);
		TEST_ASSERT(info == GrB_SUCCESS);

		info = Delta_mxv(y1, NULL, NULL, GxB_PLUS_FIRST_UINT64, A, x, desc);
		TEST_ASSERT(info == GrB_SUCCESS);

		info = Delta_mxv_count(y2, NULL, NULL, GxB_PLUS_PAIR_UINT64, A, x, desc);
		TEST_ASSERT(info == GrB_SUCCESS);

		info = GrB_mxv (y3, NULL, NULL, GxB_PLUS_PAIR_UINT64, GB_A, x, desc);
		TEST_ASSERT(info == GrB_SUCCESS);

		// validate results
		GrB_Vector_eWiseMult_BinaryOp(
			y1, NULL, NULL, GrB_MINUS_UINT64, y1, y2, desc);

		uint64_t sum = 0;
		GrB_Vector_reduce_UINT64(&sum, NULL, GrB_PLUS_MONOID_UINT64, y1, desc);
		TEST_CHECK (sum == 0);

		// validate results
		GrB_Vector_eWiseMult_BinaryOp(
			y2, NULL, NULL, GrB_MINUS_UINT64, y2, y3, desc);

		sum = 0;
		GrB_Vector_reduce_UINT64(&sum, NULL, GrB_PLUS_MONOID_UINT64, y2, desc);
		TEST_CHECK (sum == 0);
	}
	GrB_free(&GB_A);
	GrB_free(&x);
	GrB_free(&y1);
	GrB_free(&y2);
	GrB_free(&y3);
	Delta_Matrix_free(&A);
}

TEST_LIST = {
	{"RGMatrix_new", test_RGMatrix_new},
	{"RGMatrix_simple_set", test_RGMatrix_simple_set},
	{"RGMatrix_del", test_RGMatrix_del},
	{"RGMatrix_del_entry", test_RGMatrix_del_entry},
	{"RGMatrix_set", test_RGMatrix_set},
	{"RGMatrix_flus", test_RGMatrix_flus},
	{"GRMatrix_managed_transposed", test_GRMatrix_managed_transposed},
	{"RGMatrix_fuzzy", test_RGMatrix_fuzzy},
	{"RGMatrix_export_no_changes", test_RGMatrix_export_no_changes},
	{"RGMatrix_export_pending_changes", test_RGMatrix_export_pending_changes},
	{"Delta_Matrix_export_structure", test_Delta_Matrix_export_structure},
	{"Delta_Matrix_add", test_Delta_Matrix_add},
	{"Delta_Matrix_union", test_Delta_Matrix_union},
	{"Delta_Matrix_mxm", test_Delta_Matrix_mxm},
	{"RGMatrix_mxm", test_RGMatrix_mxm},
	{"RGMatrix_resize", test_RGMatrix_resize},
	{"RGMatrix_cache_transpose", test_RGMatrix_cache_transpose},
	{"RGMatrix_mxv", test_RGMatrix_mxv},
	{NULL, NULL}
};


//#ifndef RG_DEBUG
//// test RGMatrix_pending
//// if RG_DEBUG is defined, each call to setElement will flush all 3 matrices
//// causing this test to fail
//TEST_F(RGMatrixTest, RGMatrix_pending) {
//	Delta_Matrix  A        =  NULL;
//	GrB_Info   info     =  GrB_SUCCESS;
//	GrB_Type   t        =  GrB_UINT64;
//	GrB_Index  nrows    =  100;
//	GrB_Index  ncols    =  100;
//	bool       pending  =  false;
//
//	info = Delta_Matrix_new(&A, t, nrows, ncols, true);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// new Delta_Matrix shouldn't have any pending operations
//	info = Delta_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_FALSE(pending);
//
//	// set element, modifies delta-plus
//	info = Delta_Matrix_setElement_UINT64(A, 2, 2, 2);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// expecting pending changes
//	info = Delta_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_TRUE(pending);
//
//	// flush pending changes on both DP and DM
//	info = Delta_Matrix_wait(A, false);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// expecting no pending changes
//	info = Delta_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_FALSE(pending);
//
//	// remove entry, DP entry is now a zombie
//	info = Delta_Matrix_removeElement_UINT64(A, 2, 2);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// expecting pending changes
//	info = Delta_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_TRUE(pending);
//
//	// flush pending changes on both DP and DM
//	info = Delta_Matrix_wait(A, false);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// expecting no pending changes
//	info = Delta_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_FALSE(pending);
//
//	// set element, modifies delta-plus
//	info = Delta_Matrix_setElement_UINT64(A, 2, 2, 2);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// flush pending changes on M, DM and DP
//	info = Delta_Matrix_wait(A, true);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// expecting no pending changes
//	info = Delta_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_FALSE(pending);
//
//	// clean up
//	Delta_Matrix_free(&A);
//	ASSERT_TRUE(A == NULL);
//}
//#endif
