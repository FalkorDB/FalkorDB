/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
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

void Delta_Matrix_validateState
(
	const Delta_Matrix C,
	GrB_Index i,
	GrB_Index j
) {
#ifdef RG_DEBUG
	bool        x_m               =  false;
	bool        x_dp              =  false;
	bool        x_dm              =  false;
	bool        existing_entry    =  false;
	bool        pending_addition  =  false;
	bool        pending_deletion  =  false;
	GrB_Info    info_m            =  GrB_SUCCESS;
	GrB_Info    info_dp           =  GrB_SUCCESS;
	GrB_Info    info_dm           =  GrB_SUCCESS;
	GrB_Matrix  m                 =  DELTA_MATRIX_M(C);
	GrB_Matrix  dp                =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm                =  DELTA_MATRIX_DELTA_MINUS(C);

	// find out which entries exists
	info_m  = GrB_Matrix_extractElement(&x_m,  m,  i, j);
	info_dp = GrB_Matrix_extractElement(&x_dp, dp, i, j);
	info_dm = GrB_Matrix_extractElement(&x_dm, dm, i, j);

	UNUSED(existing_entry);
	UNUSED(pending_addition);
	UNUSED(pending_deletion);

	existing_entry    =  info_m  == GrB_SUCCESS;
	pending_addition  =  info_dp == GrB_SUCCESS;
	pending_deletion  =  info_dm == GrB_SUCCESS;

	//--------------------------------------------------------------------------
	// impossible states
	//--------------------------------------------------------------------------

	// matrix disjoint
	ASSERT(!(existing_entry   &&
			 pending_addition &&
			 pending_deletion));

	// deletion only
	ASSERT(!(!existing_entry   &&
			 !pending_addition &&
			 pending_deletion));

	// addition to already existing entry
	ASSERT(!(existing_entry   &&
			 pending_addition &&
			 !pending_deletion));

	// pending deletion and pending addition
	ASSERT(!(!existing_entry   &&
			  pending_addition &&
			  pending_deletion));
#endif
}

// Check every assumption for the Delta Matrix
//         ∅ = m  ∩ dp
//         ∅ = dp ∩ dm
// {zombies} = m  ∩ dm
// Transpose
//    Check it is actually M^T
// Types / Dimensions
//    m BOOL / UINT64
//    dp BOOL / UINT64
//    dm BOOL
void Delta_Matrix_validate
(
	const Delta_Matrix C
) {
#if RG_DEBUG
	bool        m_dp_disjoint     =  false;
	bool        dp_dm_disjoint    =  false;
	bool        m_zombies_valid   =  true;
	bool        dp_iso            =  true;
	bool        dm_iso            =  true;
	GrB_Info    info              =  GrB_SUCCESS;
	GrB_Matrix  m                 =  DELTA_MATRIX_M(C);
	GrB_Matrix  dp                =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm                =  DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix  m_bool            =  NULL;
	GrB_Matrix  dp_bool           =  NULL;
	GrB_Matrix  temp              =  NULL;
	GrB_Index   nrows             = 0;
	GrB_Index   ncols             = 0;
	GrB_Index   nvals             = 0;
	GrB_Index   dp_nvals          = 0;
	GrB_Index   dm_nvals          = 0;
	GrB_Type    ty                = NULL;
	GrB_Type    ty_m              = NULL;
	GrB_Type    ty_dp             = NULL;
	
	Delta_Matrix_type(&ty, C);
	GxB_Matrix_type(&ty_m, m);
	GxB_Matrix_type(&ty_dp, dp);
	Delta_Matrix_nrows(&nrows, C);
	Delta_Matrix_ncols(&ncols, C);
	
	ASSERT(ty == ty_m);
	ASSERT(ty == ty_dp);
	ASSERT(ty == GrB_BOOL || ty == GrB_UINT64);

	if(ty == GrB_UINT64)
	{
		info = GrB_Matrix_new(&m_bool, GrB_BOOL, nrows, ncols);
		ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_apply_BinaryOp1st_UINT64(
			m_bool, NULL, NULL, GrB_NE_UINT64, MSB_MASK, m, NULL
		);
		ASSERT(info == GrB_SUCCESS);
		m = m_bool;
		info = GrB_Matrix_new(&dp_bool, GrB_BOOL, nrows, ncols);
		ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_apply_BinaryOp1st_UINT64(
			dp_bool, NULL, NULL, GrB_NE_UINT64, MSB_MASK, dp, NULL
		);
		ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_reduce_BOOL(
               &dp_iso, GrB_LAND, GrB_LAND_MONOID_BOOL, dp_bool, NULL);
		ASSERT(info == GrB_SUCCESS);
		dp = dp_bool;
	} else{
		info = GxB_Matrix_iso (&dp_iso, dp);
		ASSERT(info == GrB_SUCCESS);
	}

	#if 0 // less strict iso test:
	// if this passes, Graphblas may not recognize the matrix as iso
	// but it only has true values. 
	info = GrB_Matrix_reduce_BOOL(
		&dm_iso, GrB_LAND, GrB_LAND_MONOID_BOOL, dm, NULL);
	ASSERT(info == GrB_SUCCESS);
	#else
	info = GxB_Matrix_iso (&dm_iso, dm);
	ASSERT(info == GrB_SUCCESS);
	#endif

	info = GrB_Matrix_nvals (&dm_nvals, dm);
	ASSERT(info == GrB_SUCCESS);
	info = GrB_Matrix_nvals (&dp_nvals, dp);
	ASSERT(info == GrB_SUCCESS);
	if(!dm_iso && dm_nvals > 0) {
		GxB_fprint(dm, GxB_SHORT, stdout);
	}
	if(!dp_iso && dp_nvals > 0) {
		GxB_fprint(dp, GxB_SHORT, stdout);
	}
	ASSERT(dm_iso || dm_nvals == 0);
	ASSERT(dp_iso || dp_nvals == 0);

	info = GrB_Matrix_new(&temp, GrB_BOOL, nrows, ncols);
	ASSERT(info == GrB_SUCCESS);
	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) { // this may to too strict
		// the transpose should be structually the transpose
		// however doesn't need to have all pending changes be equal.
		GrB_Matrix tm        = DELTA_MATRIX_TM(C);
		GrB_Matrix tdp       = DELTA_MATRIX_TDELTA_PLUS(C);
		GrB_Matrix tdm       = DELTA_MATRIX_TDELTA_MINUS(C);
		GrB_Index  t_eq_vals = 0;
		bool       all_t_eq  = true;

		// m = tm^t
		GrB_Matrix_nvals(&nvals, m);
		info = GrB_eWiseMult(temp, NULL, NULL, GrB_EQ_BOOL, m, tm, GrB_DESC_T1);
		ASSERT(info == GrB_SUCCESS);
		GrB_Matrix_nvals(&t_eq_vals, temp);
		ASSERT(t_eq_vals == nvals);

		info = GrB_Matrix_reduce_BOOL(
		&all_t_eq, GrB_LAND, GrB_LAND_MONOID_BOOL, temp, NULL);
		ASSERT(info == GrB_SUCCESS);
		ASSERT(all_t_eq);

		// dp = tdp^t
		GrB_Matrix_nvals(&nvals, dp);
		info = GrB_eWiseMult(temp, NULL, NULL, GrB_EQ_BOOL, dp, tdp, GrB_DESC_T1);
		ASSERT(info == GrB_SUCCESS);
		GrB_Matrix_nvals(&t_eq_vals, temp);
		ASSERT(t_eq_vals == nvals);

		info = GrB_Matrix_reduce_BOOL(
		&all_t_eq, GrB_LAND, GrB_LAND_MONOID_BOOL, temp, NULL);
		ASSERT(info == GrB_SUCCESS);
		ASSERT(all_t_eq);

		// dm = tdm^t
		GrB_Matrix_nvals(&nvals, dm);
		info = GrB_eWiseMult(temp, NULL, NULL, GrB_EQ_BOOL, dm, tdm, GrB_DESC_T1);
		ASSERT(info == GrB_SUCCESS);
		GrB_Matrix_nvals(&t_eq_vals, temp);
		ASSERT(t_eq_vals == nvals);

		info = GrB_Matrix_reduce_BOOL(
		&all_t_eq, GrB_LAND, GrB_LAND_MONOID_BOOL, temp, NULL);
		ASSERT(info == GrB_SUCCESS);
		ASSERT(all_t_eq);
	}
	
	info = GrB_eWiseMult(temp, NULL, NULL, GrB_ONEB_BOOL, m, dp, NULL);
	ASSERT(info == GrB_SUCCESS);
	GrB_Matrix_nvals(&nvals, temp);
	m_dp_disjoint = nvals == 0;
	if(!m_dp_disjoint)
		GxB_Matrix_fprint(temp, "m&dp",GxB_SHORT, stdout);
	info = GrB_eWiseMult(temp, NULL, NULL, GrB_ONEB_BOOL, dp, dm, NULL);
	ASSERT(info == GrB_SUCCESS);
	GrB_Matrix_nvals(&nvals, temp);
	dp_dm_disjoint = nvals == 0;
	if(!dp_dm_disjoint)
		GxB_Matrix_fprint(temp, "dp&dm",GxB_SHORT, stdout);
	info = GrB_eWiseAdd(temp, NULL, NULL, GrB_LXOR, dp, dm, NULL);
	ASSERT(info == GrB_SUCCESS);
	info = GrB_Matrix_reduce_BOOL(
		&m_zombies_valid, NULL, GrB_LAND_MONOID_BOOL, temp, NULL);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// check assumptions 
	//--------------------------------------------------------------------------

	ASSERT(m_dp_disjoint);
	ASSERT(dp_dm_disjoint);
	ASSERT(m_zombies_valid);

	// Free allocation.
	GrB_free(&m_bool);
	GrB_free(&dp_bool);
	GrB_free(&temp);
#endif
}

