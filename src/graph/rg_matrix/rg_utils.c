/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "rg_matrix.h"

// check if i and j are within matrix boundries
// i < nrows
// j < ncols
void RG_Matrix_checkBounds
(
	const RG_Matrix C,
	GrB_Index i,
	GrB_Index j
) {
#if RG_DEBUG
	GrB_Matrix m = RG_MATRIX_M(C);
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
void RG_Matrix_checkCompatible
(
	const RG_Matrix M,
	const RG_Matrix N
) {
#if RG_DEBUG
	GrB_Matrix m = RG_MATRIX_M(M);
	GrB_Matrix n = RG_MATRIX_M(N);

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

void RG_Matrix_validateState
(
	const RG_Matrix C,
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
	GrB_Matrix  m                 =  RG_MATRIX_M(C);
	GrB_Matrix  dp                =  RG_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm                =  RG_MATRIX_DELTA_MINUS(C);

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

