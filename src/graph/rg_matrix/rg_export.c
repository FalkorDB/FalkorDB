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
#include "../../util/rmalloc.h"

GrB_Info RG_Matrix_export
(
	GrB_Matrix *A,
	RG_Matrix C
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);

	GrB_Type    t;
	GrB_Index   nrows;
	GrB_Index   ncols;
	GrB_Index   dp_nvals;
	GrB_Index   dm_nvals;
	GrB_Matrix  a          =  NULL;
	GrB_Matrix  m          =  RG_MATRIX_M(C);
	GrB_Info    info       =  GrB_SUCCESS;
	GrB_Matrix  dp         =  RG_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm         =  RG_MATRIX_DELTA_MINUS(C);

	info = GxB_Matrix_type(&t, m);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_nrows(&nrows, m);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols, m);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_new(&a, t, nrows, ncols);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_wait(dp, GrB_MATERIALIZE);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_wait(dm, GrB_MATERIALIZE);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_nvals(&dp_nvals, dp);
	ASSERT(info == GrB_SUCCESS);
	info = GrB_Matrix_nvals(&dm_nvals, dm);
	ASSERT(info == GrB_SUCCESS);

	bool  additions  =  dp_nvals  >  0;
	bool  deletions  =  dm_nvals  >  0;

	//--------------------------------------------------------------------------
	// perform copy and deletions if needed
	//--------------------------------------------------------------------------
	
	// in case there are items to delete use mask otherwise just copy
	GrB_Matrix mask = deletions ? dm : NULL;
	GrB_Descriptor desc = deletions ? GrB_DESC_RSCT0 : GrB_DESC_RT0;
	info = GrB_transpose(a, mask, NULL, m, desc);
	ASSERT(info == GrB_SUCCESS);
	
	//--------------------------------------------------------------------------
	// perform additions
	//--------------------------------------------------------------------------

	if(additions) {
		GrB_Semiring s;
		s = (t == GrB_BOOL) ? GxB_ANY_PAIR_BOOL : GxB_ANY_PAIR_UINT64;
		info = GrB_Matrix_eWiseAdd_Semiring(a, NULL, NULL, s, a, dp,
				NULL);
		ASSERT(info == GrB_SUCCESS);
	}

	*A = a;

	return info;
}

