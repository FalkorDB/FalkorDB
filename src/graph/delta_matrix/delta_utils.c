/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "delta_utils.h"

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
	GrB_Index      nrows   = 0 ;
	GrB_Index      ncols   = 0 ;
	GrB_Index      brows   = 0 ;
	GrB_Index      bcols   = 0 ;
	GrB_Index      a_nvals = 0 ;
	GrB_Index      b_nvals = 0 ;
	GrB_Index      c_nvals = 0 ;
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

static bool _GrB_transpose_eq
(
	const GrB_Matrix A,
	const GrB_Matrix B,
	DM_validation_level level
) {
	GrB_Vector reduced = NULL ;
	GrB_Vector x = NULL ;
	GrB_Matrix C = NULL ;
	GrB_Index nrows_a = 0, ncols_a = 0, nvals_a = 0 ;
	GrB_Index nrows_b = 0, ncols_b = 0, nvals_b = 0 ;
	GrB_Index nvals_c = 0 ;
	GrB_Semiring bxor_second = NULL ;
	GrB_Type type_a = NULL ;
	GrB_Type type_b = NULL ;
	GrB_BinaryOp eq = NULL ;
	int64_t degree_delta = 0 ;
	uint64_t hash_delta = 0 ;
	bool result = false;

	ASSERT (A != NULL);
	ASSERT (B != NULL);
	ASSERT (level >= DM_TVAL_BASIC);
	ASSERT (level <= DM_TVAL_FULL);

	GrB_OK (GrB_Matrix_nrows (&nrows_a, A));
	GrB_OK (GrB_Matrix_ncols (&ncols_a, A));
	GrB_OK (GrB_Matrix_nrows (&nrows_b, B));
	GrB_OK (GrB_Matrix_ncols (&ncols_b, B));
	GrB_OK (GrB_Matrix_nvals (&nvals_a, A));
	GrB_OK (GrB_Matrix_nvals (&nvals_b, B));

	if (nrows_a != ncols_b || ncols_a != nrows_b) {
		return false;
	}

	if (nvals_a != nvals_b) {
		return false;
	}

	// Level 0: do dimensions and nvals match?
	if (level == DM_TVAL_BASIC) {
		return true;
	}

	// Level 1: do degrees match?
	GrB_OK (GrB_Vector_new (&x, GrB_INT64, ncols_a)) ;
	GrB_OK (GrB_assign (x, NULL, NULL, (int64_t) 0, GrB_ALL, ncols_a, NULL)) ;
	GrB_OK (GrB_Vector_new (&reduced, GrB_INT64, nrows_a)) ;

	GrB_OK (GrB_mxv (reduced, NULL, NULL, GxB_PLUS_PAIR_INT64, A, x, NULL)) ;
	GrB_OK (GrB_apply (reduced, NULL, NULL, GrB_AINV_INT64, reduced, NULL)) ;
	GrB_OK (GrB_mxv (reduced, NULL, GrB_PLUS_INT64, GxB_PLUS_PAIR_INT64,
		B, x, GrB_DESC_T0)) ;
	GrB_OK (GrB_reduce (
		&degree_delta, NULL, GxB_BOR_UINT64_MONOID, reduced, NULL)) ;

	if (degree_delta != 0) {
		goto cleanup;
	}

	GrB_OK (GrB_Vector_clear (reduced)) ;
	GrB_OK (GrB_mxv (
		reduced, NULL, NULL, GxB_PLUS_PAIR_INT64, A, x, GrB_DESC_T0)) ;
	GrB_OK (GrB_apply (reduced, NULL, NULL, GrB_AINV_INT64, reduced, NULL)) ;
	GrB_OK (GrB_mxv (reduced, NULL, GrB_PLUS_INT64, GxB_PLUS_PAIR_INT64,
		B, x, NULL)) ;
	GrB_OK (GrB_reduce (
		&degree_delta, NULL, GxB_BOR_UINT64_MONOID, reduced, NULL)) ;

	if (degree_delta != 0) {
		goto cleanup;
	}

	if (level == DM_TVAL_FAST) {
		result = true;
		goto cleanup;
	}

	// Level 2: check for structural equality
	GrB_OK (GrB_Matrix_new (&C, GrB_BOOL, nrows_a, ncols_a)) ;
	GrB_OK (GrB_eWiseMult (C, NULL, NULL, GrB_ONEB_BOOL, A, B, GrB_DESC_T1)) ;
	GrB_OK (GrB_Matrix_nvals (&nvals_c, C)) ;
	result = nvals_a == nvals_c ;

cleanup:
	GrB_free (&reduced);
	GrB_free (&x);
	GrB_free (&C);
	GrB_free (&bxor_second);

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
bool Delta_Matrix_validate
(
	const Delta_Matrix C,
	DM_validation_level transpose_validation
) {
	if (C == NULL) {
		return false;
	}

	bool       valid          = true;
	bool       dm_iso         = true;
	GrB_Matrix m              = DELTA_MATRIX_M(C);
	GrB_Matrix dp             = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm             = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix temp           = NULL;
	GrB_Index  nrows          = 0;
	GrB_Index  ncols          = 0;
	GrB_Index  nvals          = 0;
	GrB_Index  dm_nvals       = 0;
	GrB_Type   ty             = NULL;
	GrB_Type   ty_m           = NULL;
	GrB_Type   ty_dp          = NULL;
	int32_t    sparticy       = 0;
	int32_t    hyper_hash     = 0;
	double     hyper_switch   = 0;

	if (m == NULL || dp == NULL || dm == NULL) {
		return false;
	}

	GrB_OK (Delta_Matrix_nrows (&nrows, C));
	GrB_OK (Delta_Matrix_ncols (&ncols, C));

	// Check type is allowed.
	GrB_OK (GxB_Matrix_type (&ty_m, m));
	GrB_OK (GxB_Matrix_type (&ty_dp, dp));
	ty = ty_m;

	if (ty != ty_m || ty != ty_dp) {
		valid = false;
		goto cleanup;
	}

	if (!(ty == GrB_BOOL || ty == GrB_UINT64)) {
		valid = false;
		goto cleanup;
	}

	// Check sparsity control.
	GrB_OK (GrB_Matrix_get_INT32 (m, &sparticy, GxB_SPARSITY_CONTROL));
	if (sparticy != (GxB_SPARSE | GxB_HYPERSPARSE)) {
		valid = false;
		goto cleanup;
	}

	GrB_OK (GrB_Matrix_get_INT32 (dp, &sparticy, GxB_SPARSITY_CONTROL));
	if (sparticy != GxB_HYPERSPARSE) {
		valid = false;
		goto cleanup;
	}

	GrB_OK (GrB_Matrix_get_INT32 (dm, &sparticy, GxB_SPARSITY_CONTROL));
	if (sparticy != GxB_HYPERSPARSE) {
		valid = false;
		goto cleanup;
	}

	GrB_OK (GrB_get (dp, &hyper_hash, GxB_HYPER_HASH));
	if (hyper_hash != false) {
		valid = false;
		goto cleanup;
	}

	GrB_OK (GrB_get (dm, &hyper_hash, GxB_HYPER_HASH));
	if (hyper_hash != false) {
		valid = false;
		goto cleanup;
	}

	// using historical method because modern one requires me to create a scalar
	GrB_OK (GxB_get (dp, GxB_HYPER_SWITCH, &hyper_switch));
	if (hyper_switch != GxB_ALWAYS_HYPER) {
		valid = false;
		goto cleanup;
	}

	GrB_OK (GxB_get (dm, GxB_HYPER_SWITCH, &hyper_switch));
	if (hyper_switch != GxB_ALWAYS_HYPER) {
		valid = false;
		goto cleanup;
	}

	// Check dm is iso (all true values) or empty.
	GrB_OK (GrB_Matrix_reduce_BOOL (
		&dm_iso, GrB_LAND, GrB_LAND_MONOID_BOOL, dm, NULL));
	GrB_OK (GrB_Matrix_nvals (&dm_nvals, dm));
	if (!dm_iso && dm_nvals > 0) {
		valid = false;
		goto cleanup;
	}

	// Check transpose cache.
	if (DELTA_MATRIX_MAINTAIN_TRANSPOSE (C)) {
		GrB_Matrix tm  = DELTA_MATRIX_TM           (C);
		GrB_Matrix tdp = DELTA_MATRIX_TDELTA_PLUS  (C);
		GrB_Matrix tdm = DELTA_MATRIX_TDELTA_MINUS (C);

		if (tm == NULL || tdp == NULL || tdm == NULL) {
			valid = false;
			goto cleanup;
		}

		if (!_GrB_transpose_eq (m,  tm,  transpose_validation) ||
			!_GrB_transpose_eq (dp, tdp, transpose_validation) ||
			!_GrB_transpose_eq (dm, tdm, transpose_validation)) {
			valid = false;
			goto cleanup;
		}
	}

	// check assumptions.
	GrB_OK (GrB_Matrix_new (&temp, GrB_BOOL, nrows, ncols));
	GrB_OK (GrB_eWiseMult (temp, NULL, NULL, GrB_ONEB_BOOL, m, dp, NULL));
	GrB_OK (GrB_Matrix_nvals (&nvals, temp));
	if (nvals != 0) {
		valid = false;
		goto cleanup;
	}

	GrB_OK (GrB_eWiseMult (temp, NULL, NULL, GrB_ONEB_BOOL, dp, dm, NULL));
	GrB_OK (GrB_Matrix_nvals (&nvals, temp));
	if (nvals != 0) {
		valid = false;
		goto cleanup;
	}

	if (!_matrix_leq (GrB_ONEB_BOOL, dm, m, false)) {
		valid = false;
		goto cleanup;
	}

cleanup:
	GrB_free (&temp);
	return valid;
}

