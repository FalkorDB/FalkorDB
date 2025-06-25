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
#ifdef DELTA_DEBUG
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
#ifdef DELTA_DEBUG
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
#ifdef DELTA_DEBUG
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
// TODO Transpose
//    Check it is actually M^T
// Types / Dimensions
//    m BOOL / UINT64
//    dp BOOL / UINT64
//    dm BOOL
void Delta_Matrix_validate
(
	const Delta_Matrix C
) {
#ifdef DELTA_DEBUG
	// printf("Validating Matrix\n");
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
		info = GrB_Matrix_apply_BinaryOp1st_UINT64(
			m_bool, NULL, NULL, GrB_NE_UINT64, MSB_MASK, m, NULL
		);
		m = m_bool;
		info = GrB_Matrix_new(&dp_bool, GrB_BOOL, nrows, ncols);
		info = GrB_Matrix_apply_BinaryOp1st_UINT64(
			dp_bool, NULL, NULL, GrB_NE_UINT64, MSB_MASK, dp, NULL
		);
		dp = dp_bool;
	}
	info = GrB_Matrix_reduce_BOOL(
		&dm_iso, NULL, GrB_LAND_MONOID_BOOL, dm, NULL);
	ASSERT(info == GrB_SUCCESS);
	info = GrB_Matrix_reduce_BOOL(
		&dp_iso, NULL, GrB_LAND_MONOID_BOOL, dp, NULL);
	ASSERT(info == GrB_SUCCESS);
	// GxB_Matrix_iso (&dm_iso, dm);
	// GxB_Matrix_iso (&dp_iso, dp);
	ASSERT(dm_iso);
	ASSERT(dp_iso);
	info = GrB_Matrix_new(&temp, GrB_BOOL, nrows, ncols);
	ASSERT(info == GrB_SUCCESS);
	info = GrB_eWiseMult(temp, NULL, NULL, GrB_ONEB_BOOL, m, dp, NULL);
	ASSERT(info == GrB_SUCCESS);
	GrB_Matrix_nvals(&nvals, temp);
	m_dp_disjoint = nvals == 0;
	if(!m_dp_disjoint)
		GxB_fprint(temp, GxB_SHORT, stdout);

	info = GrB_eWiseMult(temp, NULL, NULL, GrB_ONEB_BOOL, dp, dm, NULL);
	ASSERT(info == GrB_SUCCESS);
	GrB_Matrix_nvals(&nvals, temp);
	dp_dm_disjoint = nvals == 0;
	
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
#endif
}

