/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/configuration/config.h"
#include "src/graph/rg_matrix/rg_matrix.h"

#include <time.h>

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

#define MATRIX_EMPTY(M)               \
	({                                \
		GrB_Matrix_nvals(&nvals, M);  \
		TEST_ASSERT(nvals == 0);      \
	}) 

#define MATRIX_NOT_EMPTY(M)           \
	({                                \
		GrB_Matrix_nvals(&nvals, M);  \
		TEST_ASSERT(nvals != 0);      \
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
void ASSERT_GrB_Matrices_EQ(const GrB_Matrix A, const GrB_Matrix B)
{
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

	TEST_ASSERT(nrows_A == nrows_B);
	TEST_ASSERT(ncols_A == ncols_B);

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

	info = GrB_Matrix_new(&C, t_A, nrows_A, ncols_A);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_eWiseMult_BinaryOp(C, NULL, NULL, GrB_LAND, A, B, NULL);
	TEST_ASSERT(info == GrB_SUCCESS);

	GrB_Matrix_nvals(&nvals_C, C);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(nvals_C == nvals_A);

	// clean up
	info = GrB_Matrix_free(&C);
	TEST_ASSERT(info == GrB_SUCCESS);
}

// test RGMatrix initialization
void test_RGMatrix_new() {
	RG_Matrix  A     = NULL;
	GrB_Matrix M     = NULL;
	GrB_Matrix DP    = NULL;
	GrB_Matrix DM    = NULL;
	GrB_Type   t     = GrB_BOOL;
	GrB_Info   info  = GrB_SUCCESS;
	GrB_Index  nvals = 0;
	GrB_Index  nrows = 100;
	GrB_Index  ncols = 100;

	info = RG_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M   =  RG_Matrix_m(A);
	DP  =  RG_Matrix_dp(A);
	DM  =  RG_Matrix_dm(A);

	// bool matrix do not maintain transpose
	TEST_ASSERT(RG_Matrix_getTranspose(A) == NULL);

	// a new empty matrix should be synced
	// no data in either DP or DM
	TEST_ASSERT(!RG_Matrix_isDirty(A));

	// matrix should be empty
	M_EMPTY();
	DP_EMPTY(); 
	DM_EMPTY();
	RG_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 0);

	RG_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
}

void test_RGMatrix_set() {
	GrB_Type    t                   =  GrB_BOOL;
	RG_Matrix   A                   =  NULL;
	GrB_Matrix  M                   =  NULL;
	GrB_Matrix  DP                  =  NULL;
	GrB_Matrix  DM                  =  NULL;
	GrB_Info    info                =  GrB_SUCCESS;
	GrB_Index   nvals               =  0;
	GrB_Index   nrows               =  100;
	GrB_Index   ncols               =  100;
	GrB_Index   i                   =  0;
	GrB_Index   j                   =  1;

	info = RG_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M   =  RG_Matrix_m(A);
	DP  =  RG_Matrix_dp(A);
	DM  =  RG_Matrix_dm(A);

	//--------------------------------------------------------------------------
	// Set element that marked for deletion
	//--------------------------------------------------------------------------

	// set element at position i,j
	info = RG_Matrix_setElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// force sync
	// entry should migrated from 'delta-plus' to 'M'
	RG_Matrix_wait(A, true);

	// set element at position i,j
	info = RG_Matrix_removeElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);
	
	// set element at position i,j
	info = RG_Matrix_setElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// A should be empty
	RG_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 1);

	// M should contain a single element
	M_NOT_EMPTY();

	// DM should contain a single element
	DM_EMPTY();

	// DP should be empty
	DP_EMPTY();


	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	RG_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
}

// flush simple addition
void test_RGMatrix_flush() {
	GrB_Type    t                   =  GrB_BOOL;
	RG_Matrix   A                   =  NULL;
	GrB_Matrix  M                   =  NULL;
	GrB_Matrix  DP                  =  NULL;
	GrB_Matrix  DM                  =  NULL;
	GrB_Info    info                =  GrB_SUCCESS;
	GrB_Index   nvals               =  0;
	GrB_Index   nrows               =  100;
	GrB_Index   ncols               =  100;
	GrB_Index   i                   =  0;
	GrB_Index   j                   =  1;
	bool        sync                =  false;

	info = RG_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position i,j
	info = RG_Matrix_setElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M   =  RG_Matrix_m(A);
	DP  =  RG_Matrix_dp(A);
	DM  =  RG_Matrix_dm(A);

	//--------------------------------------------------------------------------
	// flush matrix, no sync
	//--------------------------------------------------------------------------
	
	// wait, don't force sync
	sync = false;
	RG_Matrix_wait(A, sync);

	// M should be empty
	M_EMPTY();

	// DM should be empty
	DM_EMPTY();

	// DP should contain a single element
	DP_NOT_EMPTY();

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	RG_Matrix_wait(A, sync);

	RG_Matrix_nvals(&nvals, A);
	TEST_ASSERT(nvals == 1);

	// M should be empty
	M_NOT_EMPTY();

	// DM should be empty
	DM_EMPTY();

	// DP should contain a single element
	DP_EMPTY();

	// clean up
	RG_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
}

//------------------------------------------------------------------------------
// fuzzy test compare RG_Matrix to GrB_Matrix
//------------------------------------------------------------------------------

void test_RGMatrix_fuzzy() {
	GrB_Type    t                   =  GrB_BOOL;
	RG_Matrix   A                   =  NULL;
	RG_Matrix   T                   =  NULL;  // A transposed
	GrB_Matrix  M                   =  NULL;  // primary internal matrix
	GrB_Matrix  MT                  =  NULL;
	GrB_Matrix  N                   =  NULL;
	GrB_Matrix  NT                  =  NULL;
	GrB_Info    info                =  GrB_SUCCESS;
	GrB_Index   nrows               =  100;
	GrB_Index   ncols               =  100;
	GrB_Index   i                   =  0;
	GrB_Index   j                   =  1;
	GrB_Index*  II                  =  NULL;
	GrB_Index*  J                   =  NULL;
	uint32_t    operations          =  10000;

	//--------------------------------------------------------------------------
	// create RGMatrix
	//--------------------------------------------------------------------------

	srand(time(0));

	II = (GrB_Index*) malloc(sizeof(GrB_Index) * operations);
	J  = (GrB_Index*) malloc(sizeof(GrB_Index) * operations);

	info = RG_Matrix_new(&A, t, nrows, ncols, true);
	TEST_ASSERT(info == GrB_SUCCESS);

	// make sure transposed was created
	T = RG_Matrix_getTranspose(A);
	TEST_ASSERT(T != A);
	TEST_ASSERT(T != NULL);

	// get internal matrices
	M   =  RG_Matrix_m(A);
	MT  =  RG_Matrix_m(T);

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

			info = RG_Matrix_setElement_BOOL(A, i, j);
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
			
			RG_Matrix_removeElement_BOOL(A, i, j);

			GrB_Matrix_removeElement(N, i, j);
		}
	}

	//--------------------------------------------------------------------------
	// flush matrix
	//--------------------------------------------------------------------------

	RG_Matrix_wait(A, true);

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

	RG_Matrix_free(&A);
	TEST_ASSERT(A == NULL);

	info = GrB_Matrix_free(&N);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_free(&NT);
	TEST_ASSERT(info == GrB_SUCCESS);

	free(II);
	free(J);
}

// test exporting RG_Matrix to GrB_Matrix when there are no pending changes
// by exporting the matrix after flushing
void test_RGMatrix_export_no_changes() {
	GrB_Type    t                   =  GrB_BOOL;
	RG_Matrix   A                   =  NULL; 
	GrB_Matrix  M                   =  NULL;
	GrB_Matrix  N                   =  NULL;  // exported matrix 
	GrB_Info    info                =  GrB_SUCCESS;
	GrB_Index   i                   =  0;
	GrB_Index   j                   =  1;
	GrB_Index   nrows               =  100;
	GrB_Index   ncols               =  100;
	bool        sync                =  false;

	info = RG_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M = RG_Matrix_m(A);

	//--------------------------------------------------------------------------
	// export empty matrix
	//--------------------------------------------------------------------------

	info = RG_Matrix_export(&N, A);
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
	info = RG_Matrix_setElement_BOOL(A, i, j);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	RG_Matrix_wait(A, sync);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	info = RG_Matrix_export(&N, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	ASSERT_GrB_Matrices_EQ(M, N);

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	RG_Matrix_free(&A);
	GrB_Matrix_free(&N);
}

// test exporting RG_Matrix to GrB_Matrix when there are pending changes
// by exporting the matrix after making changes
// then flush the matrix and compare the internal matrix to the exported matrix
void test_RGMatrix_export_pending_changes() {
	GrB_Type    t                   =  GrB_BOOL;
	RG_Matrix   A                   =  NULL;
	GrB_Matrix  M                   =  NULL;
	GrB_Matrix  N                   =  NULL;  // exported matrix
	GrB_Info    info                =  GrB_SUCCESS;
	GrB_Index   nrows               =  100;
	GrB_Index   ncols               =  100;
	bool        sync                =  false;

	info = RG_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// get internal matrices
	M = RG_Matrix_m(A);

	// set elements
	info = RG_Matrix_setElement_BOOL(A, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = RG_Matrix_setElement_BOOL(A, 1, 1);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	RG_Matrix_wait(A, sync);

	//--------------------------------------------------------------------------
	// set pending changes
	//--------------------------------------------------------------------------

	// remove element at position 0,0
	info = RG_Matrix_removeElement_BOOL(A, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position 2,2
	info = RG_Matrix_setElement_BOOL(A, 2, 2);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// export matrix
	//--------------------------------------------------------------------------

	info = RG_Matrix_export(&N, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	RG_Matrix_wait(A, sync);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	ASSERT_GrB_Matrices_EQ(M, N);

	// clean up
	GrB_Matrix_free(&N);
	RG_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
}

void test_RGMatrix_copy() {
	GrB_Type    t                   =  GrB_BOOL;
	RG_Matrix   A                   =  NULL;
	RG_Matrix   B                   =  NULL;
	GrB_Matrix  A_M                 =  NULL;
	GrB_Matrix  B_M                 =  NULL;
	GrB_Matrix  A_DP                =  NULL;
	GrB_Matrix  B_DP                =  NULL;
	GrB_Matrix  A_DM                =  NULL;
	GrB_Matrix  B_DM                =  NULL;
	GrB_Info    info                =  GrB_SUCCESS;
	GrB_Index   nrows               =  100;
	GrB_Index   ncols               =  100;
	bool        sync                =  false;

	info = RG_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = RG_Matrix_new(&B, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set elements
	info = RG_Matrix_setElement_BOOL(A, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = RG_Matrix_setElement_BOOL(A, 1, 1);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	RG_Matrix_wait(A, sync);

	//--------------------------------------------------------------------------
	// set pending changes
	//--------------------------------------------------------------------------

	// remove element at position 0,0
	info = RG_Matrix_removeElement_BOOL(A, 0, 0);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position 2,2
	info = RG_Matrix_setElement_BOOL(A, 2, 2);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// copy matrix
	//--------------------------------------------------------------------------

	info = RG_Matrix_copy(B, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	A_M  = RG_Matrix_m(A);
	B_M  = RG_Matrix_m(B);
	A_DP = RG_Matrix_dp(A);
	B_DP = RG_Matrix_dp(B);
	A_DM = RG_Matrix_dm(A);
	B_DM = RG_Matrix_dm(B);
	
	ASSERT_GrB_Matrices_EQ(A_M, B_M);
	ASSERT_GrB_Matrices_EQ(A_DP, B_DP);
	ASSERT_GrB_Matrices_EQ(A_DM, B_DM);

	// clean up
	RG_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	RG_Matrix_free(&B);
	TEST_ASSERT(B == NULL);
}

void test_RGMatrix_mxm() {
	GrB_Type    t                   =  GrB_BOOL;
	RG_Matrix   A                   =  NULL;
	RG_Matrix   B                   =  NULL;
	RG_Matrix   C                   =  NULL;
	RG_Matrix   D                   =  NULL;
	GrB_Matrix  C_M                 =  NULL;
	GrB_Matrix  D_M                 =  NULL;
	GrB_Info    info                =  GrB_SUCCESS;
	GrB_Index   nrows               =  100;
	GrB_Index   ncols               =  100;
	bool        sync                =  false;

	info = RG_Matrix_new(&A, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = RG_Matrix_new(&B, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = RG_Matrix_new(&C, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = RG_Matrix_new(&D, t, nrows, ncols, false);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set elements
	info = RG_Matrix_setElement_BOOL(A, 0, 1);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = RG_Matrix_setElement_BOOL(A, 2, 3);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = RG_Matrix_setElement_BOOL(B, 1, 2);
	TEST_ASSERT(info == GrB_SUCCESS);
	info = RG_Matrix_setElement_BOOL(B, 3, 4);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// flush matrix, sync
	//--------------------------------------------------------------------------
	
	// wait, force sync
	sync = true;
	RG_Matrix_wait(A, sync);
	RG_Matrix_wait(B, sync);

	//--------------------------------------------------------------------------
	// set pending changes
	//--------------------------------------------------------------------------

	// remove element at position 0,0
	info = RG_Matrix_removeElement_BOOL(B, 1, 2);
	TEST_ASSERT(info == GrB_SUCCESS);

	// set element at position 1,3
	info = RG_Matrix_setElement_BOOL(B, 1, 3);
	TEST_ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// mxm matrix
	//--------------------------------------------------------------------------

	info = RG_mxm(C, GxB_ANY_PAIR_BOOL, A, B);
	TEST_ASSERT(info == GrB_SUCCESS);

	RG_Matrix_wait(B, sync);

	info = RG_mxm(D, GxB_ANY_PAIR_BOOL, A, B);
	//--------------------------------------------------------------------------
	// validation
	//--------------------------------------------------------------------------

	C_M  = RG_Matrix_m(C);
	D_M  = RG_Matrix_m(D);
	
	ASSERT_GrB_Matrices_EQ(C_M, D_M);

	// clean up
	RG_Matrix_free(&A);
	TEST_ASSERT(A == NULL);
	RG_Matrix_free(&B);
	TEST_ASSERT(B == NULL);
	RG_Matrix_free(&C);
	TEST_ASSERT(C == NULL);
	RG_Matrix_free(&D);
	TEST_ASSERT(C == NULL);
}

void test_RGMatrix_resize() {
	RG_Matrix  A        =  NULL;
	RG_Matrix  T        =  NULL;
	GrB_Info   info     =  GrB_SUCCESS;
	GrB_Type   t        =  GrB_BOOL;
	GrB_Index  nrows    =  10;
	GrB_Index  ncols    =  20;

	info = RG_Matrix_new(&A, t, nrows, ncols, true);
	T = RG_Matrix_getTranspose(A);

	GrB_Index  A_nrows;
	GrB_Index  A_ncols;
	GrB_Index  T_nrows;
	GrB_Index  T_ncols;

	// verify A and T dimensions
	RG_Matrix_nrows(&A_nrows, A);
	TEST_ASSERT(info == GrB_SUCCESS);
	RG_Matrix_ncols(&A_ncols, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(A_nrows == nrows);
	TEST_ASSERT(A_ncols == ncols);

	RG_Matrix_nrows(&T_nrows, T);
	TEST_ASSERT(info == GrB_SUCCESS);
	RG_Matrix_ncols(&T_ncols, T);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(T_nrows == ncols);
	TEST_ASSERT(T_ncols == nrows);

	// resize matrix, increase size by 2
	nrows *= 2;
	ncols *= 2;

	info = RG_Matrix_resize(A, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	// verify A and T dimensions
	RG_Matrix_nrows(&A_nrows, A);
	TEST_ASSERT(info == GrB_SUCCESS);
	RG_Matrix_ncols(&A_ncols, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(A_nrows == nrows);
	TEST_ASSERT(A_ncols == ncols);

	RG_Matrix_nrows(&T_nrows, T);
	TEST_ASSERT(info == GrB_SUCCESS);
	RG_Matrix_ncols(&T_ncols, T);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(T_nrows == ncols);
	TEST_ASSERT(T_ncols == nrows);

	// resize matrix decrease size by 2
	nrows /= 2;
	ncols /= 2;

	info = RG_Matrix_resize(A, nrows, ncols);
	TEST_ASSERT(info == GrB_SUCCESS);

	// verify A and T dimensions
	RG_Matrix_nrows(&A_nrows, A);
	TEST_ASSERT(info == GrB_SUCCESS);
	RG_Matrix_ncols(&A_ncols, A);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(A_nrows == nrows);
	TEST_ASSERT(A_ncols == ncols);

	RG_Matrix_nrows(&T_nrows, T);
	TEST_ASSERT(info == GrB_SUCCESS);
	RG_Matrix_ncols(&T_ncols, T);
	TEST_ASSERT(info == GrB_SUCCESS);

	TEST_ASSERT(T_nrows == ncols);
	TEST_ASSERT(T_ncols == nrows);

	RG_Matrix_free(&A);
}

TEST_LIST = {
	{"RGMatrix_new", test_RGMatrix_new},
	{"RGMatrix_set", test_RGMatrix_set},
	{"RGMatrix_flush", test_RGMatrix_flush},
	{"RGMatrix_fuzzy", test_RGMatrix_fuzzy},
	{"RGMatrix_export_no_changes", test_RGMatrix_export_no_changes},
	{"RGMatrix_export_pending_changes", test_RGMatrix_export_pending_changes},
	{"RGMatrix_copy", test_RGMatrix_copy},
	{"RGMatrix_mxm", test_RGMatrix_mxm},
	{"RGMatrix_resize", test_RGMatrix_resize},
	{NULL, NULL}
};

//#ifndef RG_DEBUG
//// test RGMatrix_pending
//// if RG_DEBUG is defined, each call to setElement will flush all 3 matrices
//// causing this test to fail
//TEST_F(RGMatrixTest, RGMatrix_pending) {
//	RG_Matrix  A        =  NULL;
//	GrB_Info   info     =  GrB_SUCCESS;
//	GrB_Type   t        =  GrB_UINT64;
//	GrB_Index  nrows    =  100;
//	GrB_Index  ncols    =  100;
//	bool       pending  =  false;
//
//	info = RG_Matrix_new(&A, t, nrows, ncols);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// new RG_Matrix shouldn't have any pending operations
//	info = RG_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_FALSE(pending);
//
//	// set element, modifies delta-plus
//	info = RG_Matrix_setElement_UINT64(A, 2, 2, 2);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// expecting pending changes
//	info = RG_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_TRUE(pending);
//
//	// flush pending changes on both DP and DM
//	info = RG_Matrix_wait(A, false);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// expecting no pending changes
//	info = RG_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_FALSE(pending);
//
//	// remove entry, DP entry is now a zombie
//	info = RG_Matrix_removeElement_UINT64(A, 2, 2);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// expecting pending changes
//	info = RG_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_TRUE(pending);
//
//	// flush pending changes on both DP and DM
//	info = RG_Matrix_wait(A, false);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// expecting no pending changes
//	info = RG_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_FALSE(pending);
//
//	// set element, modifies delta-plus
//	info = RG_Matrix_setElement_UINT64(A, 2, 2, 2);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// flush pending changes on M, DM and DP
//	info = RG_Matrix_wait(A, true);
//	ASSERT_EQ(info, GrB_SUCCESS);
//
//	// expecting no pending changes
//	info = RG_Matrix_pending(A, &pending);
//	ASSERT_EQ(info, GrB_SUCCESS);
//	ASSERT_FALSE(pending);
//
//	// clean up
//	RG_Matrix_free(&A);
//	ASSERT_TRUE(A == NULL);
//}
//#endif
