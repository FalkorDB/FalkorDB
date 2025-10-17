/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

// check if i and j are within matrix boundries
// i < nrows
// j < ncols
void Delta_Matrix_checkBounds
(
	const Delta_Matrix C,
	GrB_Index i,
	GrB_Index j
) {
#ifdef RG_DEBUG
	ASSERT (C != NULL);
	GrB_Matrix m = DELTA_MATRIX_M(C);
	// check bounds
	GrB_Index nrows;
	GrB_Index ncols;
	GrB_Matrix_nrows(&nrows, m);
	GrB_Matrix_ncols(&ncols, m);
	ASSERT(i < nrows);
	ASSERT(j < ncols);
#endif
}

// check 2 matrices have same type nrows and ncols
void Delta_Matrix_checkCompatible
(
	const Delta_Matrix M,
	const Delta_Matrix N
) {
#ifdef RG_DEBUG
	ASSERT(M != NULL);
	ASSERT(N != NULL);
	GrB_Matrix m = DELTA_MATRIX_M(M);
	GrB_Matrix n = DELTA_MATRIX_M(N);

	GrB_Type  m_type;
	GrB_Type  n_type;
	GxB_Matrix_type(&m_type, m);
	GxB_Matrix_type(&n_type, n);
	ASSERT(m_type == n_type);

	GrB_Index m_nrows;
	GrB_Index m_ncols;
	GrB_Index n_nrows;
	GrB_Index n_ncols;
	GrB_Matrix_nrows(&m_nrows, m);
	GrB_Matrix_ncols(&m_ncols, m);
	GrB_Matrix_nrows(&n_nrows, n);
	GrB_Matrix_ncols(&n_ncols, n);
	ASSERT(m_nrows == n_nrows);
	ASSERT(m_ncols == n_ncols);
#endif
}

// check if the dimensions of C, A and B are compatible for addition
void Delta_Matrix_addCompatible
(
	const Delta_Matrix C,
	const Delta_Matrix A,
	const Delta_Matrix B
) {
#ifdef RG_DEBUG
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(B != NULL);

	GrB_Index c_rows;
	GrB_Index c_cols;
	GrB_Index a_rows;
	GrB_Index a_cols;
	GrB_Index b_rows;
	GrB_Index b_cols;

	Delta_Matrix_nrows(&c_rows, C);
	Delta_Matrix_ncols(&c_cols, C);
	Delta_Matrix_nrows(&a_rows, A);
	Delta_Matrix_ncols(&a_cols, A);
	Delta_Matrix_nrows(&b_rows, B);
	Delta_Matrix_ncols(&b_cols, B);

	ASSERT(c_rows == a_rows);
	ASSERT(c_cols == a_cols);
	ASSERT(c_rows == b_rows);
	ASSERT(c_cols == b_cols);
#endif
}

// check if the dimensions of C, A and B are compatible for multiplication
void Delta_Matrix_mulCompatible
(
	const Delta_Matrix C,
	const Delta_Matrix A,
	const Delta_Matrix B
) {
#ifdef RG_DEBUG
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(B != NULL);

	GrB_Index c_rows;
	GrB_Index c_cols;
	GrB_Index a_rows;
	GrB_Index a_cols;
	GrB_Index b_rows;
	GrB_Index b_cols;

	Delta_Matrix_nrows(&c_rows, C);
	Delta_Matrix_ncols(&c_cols, C);
	Delta_Matrix_nrows(&a_rows, A);
	Delta_Matrix_ncols(&a_cols, A);
	Delta_Matrix_nrows(&b_rows, B);
	Delta_Matrix_ncols(&b_cols, B);

	ASSERT(c_rows == a_rows);
	ASSERT(c_cols == b_cols);
	ASSERT(a_cols == b_rows);
#endif
}

bool _matrix_leq
(
	const GrB_BinaryOp leq,
	const GrB_Matrix A,
	const GrB_Matrix B,
	bool transpose
) {
	GrB_Index      a_nvals = 0;
	GrB_Index      b_nvals = 0;
	GrB_Index      c_nvals = 0;
	GrB_Index      nrows   = 0;
	GrB_Index      ncols   = 0;
	GrB_Index      brows   = 0;
	GrB_Index      bcols   = 0;
	GrB_Descriptor desc = transpose ? GrB_DESC_T1 : NULL;
	
	GrB_OK (GrB_Matrix_nvals(&a_nvals, A));
	GrB_OK (GrB_Matrix_nvals(&b_nvals, B));
	if (a_nvals > b_nvals) {
		return false;
	}

	GrB_OK (GrB_Matrix_nrows(&nrows, A));
	GrB_OK (GrB_Matrix_ncols(&ncols, A));
	GrB_OK (GrB_Matrix_nrows(&brows, B));
	GrB_OK (GrB_Matrix_ncols(&bcols, B));

	if (transpose) {
		GrB_Index temp = brows;
		brows = bcols;
		bcols = temp;
	}

	if(nrows != brows || ncols != bcols) {
		return false;
	}

	GrB_Matrix C = NULL;
	GrB_OK (GrB_Matrix_new(&C, GrB_BOOL, nrows, ncols));
	GrB_OK (GrB_eWiseMult(C, NULL, NULL, leq, A, B, desc));
	GrB_OK (GrB_Matrix_nvals(&c_nvals, C));

	bool result = true;
	GrB_OK(GrB_Matrix_reduce_BOOL(
		&result, NULL, GrB_LAND_MONOID_BOOL, C, NULL));
	GrB_free(&C);

	result = result && (c_nvals == a_nvals);
	return result;
}

// Check every assumption for the Delta Matrix
//         ∅ = m  ∩ dp
//         ∅ = dp ∩ dm
//         m \superset dm
// Transpose
//    Check it is actually M^T
// Types / Dimensions
//    m BOOL / UINT64
//    dp BOOL / UINT64
//    dm BOOL
void Delta_Matrix_validate
(
	const Delta_Matrix C,
	bool check_transpose
) {
#ifdef RG_DEBUG
	ASSERT (C != NULL);

	bool        m_dp_disjoint     =  false;
	bool        dp_dm_disjoint    =  false;
	bool        m_zombies_valid   =  true;
	bool        dm_iso            =  true;
	GrB_Info    info              =  GrB_SUCCESS;
	GrB_Matrix  m                 =  DELTA_MATRIX_M(C);
	GrB_Matrix  dp                =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm                =  DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix  temp              =  NULL;
	GrB_Index   nrows             = 0;
	GrB_Index   ncols             = 0;
	GrB_Index   nvals             = 0;
	GrB_Index   dp_nvals          = 0;
	GrB_Index   dm_nvals          = 0;
	GrB_Type    ty                = NULL;
	GrB_Type    ty_m              = NULL;
	GrB_Type    ty_dp             = NULL;

	GrB_OK (Delta_Matrix_nrows(&nrows, C));
	GrB_OK (Delta_Matrix_ncols(&ncols, C));
	
	//--------------------------------------------------------------------------
	// Check type is allowed
	//--------------------------------------------------------------------------

	// GrB_OK (Delta_Matrix_type(&ty, C));
	GrB_OK (GxB_Matrix_type(&ty_m, m));
	GrB_OK (GxB_Matrix_type(&ty_dp, dp));
	ty = ty_m;
	ASSERT(ty == ty_m);
	ASSERT(ty == ty_dp);
	ASSERT(ty == GrB_BOOL || ty == GrB_UINT64);

	//--------------------------------------------------------------------------
	// check sparcity control
	//--------------------------------------------------------------------------

	int32_t sparticy;
	GrB_OK(GrB_Matrix_get_INT32(m, &sparticy, GxB_SPARSITY_CONTROL));
	ASSERT(sparticy == GxB_SPARSE | GxB_HYPERSPARSE)

	GrB_OK(GrB_Matrix_get_INT32(dp, &sparticy, GxB_SPARSITY_CONTROL));
	ASSERT(sparticy == GxB_HYPERSPARSE)

	GrB_OK(GrB_Matrix_get_INT32(dm, &sparticy, GxB_SPARSITY_CONTROL));
	ASSERT(sparticy == GxB_HYPERSPARSE)

	//--------------------------------------------------------------------------
	// Check dm is iso
	//--------------------------------------------------------------------------

	#if 0 // less strict iso test:
	// if this passes, Graphblas may not recognize the matrix as iso
	// but it only has true values. 
	info = GrB_Matrix_reduce_BOOL(
		&dm_iso, GrB_LAND, GrB_LAND_MONOID_BOOL, dm, NULL);
	ASSERT(info == GrB_SUCCESS);
	#else
	GrB_OK (GxB_Matrix_iso (&dm_iso, dm));
	#endif

	GrB_OK (GrB_Matrix_nvals (&dm_nvals, dm));

	if(!dm_iso && dm_nvals > 0) {
		GxB_fprint(dm, GxB_SHORT, stdout);
	}

	ASSERT(dm_iso || dm_nvals == 0);

	//--------------------------------------------------------------------------
	// Check the transpose
	//--------------------------------------------------------------------------
	if(check_transpose && DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) { 
		// this may to too strict
		// the transpose should be structually the transpose
		// however doesn't need to have all pending changes be equal.
		GrB_Matrix tm        = DELTA_MATRIX_TM(C);
		GrB_Matrix tdp       = DELTA_MATRIX_TDELTA_PLUS(C);
		GrB_Matrix tdm       = DELTA_MATRIX_TDELTA_MINUS(C);

		// m = tm^t
		ASSERT(_matrix_leq(GrB_ONEB_BOOL, m, tm, true));
		ASSERT(_matrix_leq(GrB_ONEB_BOOL, tm, m, true));

		// dp = tdp^t
		ASSERT(_matrix_leq(GrB_ONEB_BOOL, dp, tdp, true));
		ASSERT(_matrix_leq(GrB_ONEB_BOOL, tdp, dp, true));

		// dm = tdm^t
		ASSERT(_matrix_leq(GrB_ONEB_BOOL, dm, tdm, true));
		ASSERT(_matrix_leq(GrB_ONEB_BOOL, tdm, dm, true));
	}
	
	//--------------------------------------------------------------------------
	// check assumptions
	//--------------------------------------------------------------------------
	GrB_OK (GrB_Matrix_new(&temp, GrB_BOOL, nrows, ncols));
	GrB_OK (GrB_eWiseMult(temp, NULL, NULL, GrB_ONEB_BOOL, m, dp, NULL));
	GrB_OK (GrB_Matrix_nvals(&nvals, temp));
	m_dp_disjoint = nvals == 0;

	// if(!m_dp_disjoint)
	// 	GxB_Matrix_fprint(temp, "m&dp",GxB_SHORT, stdout);
	
	GrB_OK (GrB_eWiseMult(temp, NULL, NULL, GrB_ONEB_BOOL, dp, dm, NULL));
	GrB_OK (GrB_Matrix_nvals(&nvals, temp));

	dp_dm_disjoint = nvals == 0;

	// if(!dp_dm_disjoint)
	// 	GxB_Matrix_fprint(temp, "dp&dm",GxB_SHORT, stdout);

	// m \superset dm
	ASSERT(_matrix_leq(GrB_ONEB_BOOL, dm, m, false));

	ASSERT(m_dp_disjoint);
	ASSERT(dp_dm_disjoint);

	// Free allocation.
	GrB_free(&temp);
#endif
}