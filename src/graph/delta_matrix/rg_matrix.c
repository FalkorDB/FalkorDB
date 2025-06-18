/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"

void Delta_Matrix_setDirty
(
	Delta_Matrix C
) {
	ASSERT(C);
	C->dirty = true;
	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) C->transposed->dirty = true;
}

Delta_Matrix Delta_Matrix_getTranspose
(
	const Delta_Matrix C
) {
	ASSERT(C != NULL);
	return C->transposed;
}

bool Delta_Matrix_isDirty
(
	const Delta_Matrix C
) {
	ASSERT(C);

	if(C->dirty) {
		return true;
	}
	return false;
	int pending_M;
	int pending_DP;
	int pending_DM;

	GrB_Info info = GrB_Matrix_get_INT32(DELTA_MATRIX_M(C), &pending_M, 
		GxB_WILL_WAIT);
	ASSERT(info == GrB_SUCCESS)
	info = GrB_Matrix_get_INT32(DELTA_MATRIX_DELTA_PLUS(C), &pending_DP, 
		GxB_WILL_WAIT);
	ASSERT(info == GrB_SUCCESS)
	info = GrB_Matrix_get_INT32(DELTA_MATRIX_DELTA_MINUS(C), &pending_DM, 
		GxB_WILL_WAIT);
	ASSERT(info == GrB_SUCCESS)
	return (pending_M | pending_DM | pending_DP);
}

// checks if C is fully synced
// a synced delta matrix does not contains any entries in
// either its delta-plus and delta-minus internal matrices
bool Delta_Matrix_Synced
(
	const Delta_Matrix C  // matrix to inquery
) {
	ASSERT(C);

	// quick indication, if the matrix is marked as dirty that means
	// entires exists in either DP or DM
	if(C->dirty) {
		return false;
	}

	GrB_Index dp_nvals;
	GrB_Index dm_nvals;
	GrB_Matrix_nvals(&dp_nvals, DELTA_MATRIX_DELTA_PLUS(C));
	GrB_Matrix_nvals(&dm_nvals, DELTA_MATRIX_DELTA_MINUS(C));

	return ((dp_nvals + dm_nvals) == 0);
}

// locks the matrix
void Delta_Matrix_lock
(
	Delta_Matrix C
) {
	ASSERT(C);
	pthread_mutex_lock(&C->mutex);
}

// unlocks the matrix
void Delta_Matrix_unlock
(
	Delta_Matrix C
) {
	ASSERT(C);
	pthread_mutex_unlock(&C->mutex);
}

GrB_Info Delta_Matrix_nrows
(
	GrB_Index *nrows,
	const Delta_Matrix C
) {
	ASSERT(C);
	ASSERT(nrows);

	GrB_Matrix m = DELTA_MATRIX_M(C);
	return GrB_Matrix_nrows(nrows, m);
}

GrB_Info Delta_Matrix_ncols
(
	GrB_Index *ncols,
	const Delta_Matrix C
) {
	ASSERT(C);
	ASSERT(ncols);

	GrB_Matrix m = DELTA_MATRIX_M(C);
	return GrB_Matrix_ncols(ncols, m);
}

GrB_Info Delta_Matrix_nvals    // get the number of entries in a matrix
(
    GrB_Index *nvals,       // matrix has nvals entries
    const Delta_Matrix A       // matrix to query
) {
	ASSERT(A      !=  NULL);
	ASSERT(nvals  !=  NULL);

	GrB_Matrix  m;
	GrB_Matrix  dp;
	GrB_Matrix  dm;
	GrB_Info    info;

	GrB_Index  m_nvals   =  0;
	GrB_Index  dp_nvals  =  0;
	GrB_Index  dm_nvals  =  0;

	// nvals = nvals(M) + nvals(DP) - nvals(DM)

	m   =  DELTA_MATRIX_M(A);
	dp  =  DELTA_MATRIX_DELTA_PLUS(A);
	dm  =  DELTA_MATRIX_DELTA_MINUS(A);

	info = GrB_Matrix_nvals(&m_nvals, m);
	ASSERT(info == GrB_SUCCESS);
	info = GrB_Matrix_nvals(&dp_nvals, dp);
	ASSERT(info == GrB_SUCCESS);
	info = GrB_Matrix_nvals(&dm_nvals, dm);
	ASSERT(info == GrB_SUCCESS);

	*nvals = m_nvals + dp_nvals - dm_nvals;
	return info;
}

GrB_Info Delta_Matrix_clear
(
    Delta_Matrix A
) {
	ASSERT(A !=  NULL);

	GrB_Matrix  m            =  DELTA_MATRIX_M(A);
	GrB_Info    info         =  GrB_SUCCESS;
	GrB_Matrix  delta_plus   =  DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix  delta_minus  =  DELTA_MATRIX_DELTA_MINUS(A);

	info = GrB_Matrix_clear(m);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_clear(m);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_clear(m);
	ASSERT(info == GrB_SUCCESS);

	A->dirty = false;
	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(A)) A->transposed->dirty = false;

	return info;
}

GrB_Info Delta_Matrix_type
(
	GrB_Type *type,
	Delta_Matrix A
) {
	ASSERT(A     !=  NULL);
	ASSERT(type  !=  NULL);

	GrB_Matrix M = DELTA_MATRIX_M(A);
	GrB_Info info = GxB_Matrix_type(type, M);
	ASSERT(info == GrB_SUCCESS)
	return info;
}

// return # of bytes used for a matrix
GrB_Info Delta_Matrix_memoryUsage
(
    size_t *size,           // # of bytes used by the matrix C
    const Delta_Matrix C    // matrix to query
) {
	ASSERT(C     !=  NULL);
	ASSERT(size  !=  NULL);
	size_t temp_size  = 0;
	size_t _size      = 0;
	GrB_Info info = GxB_Matrix_memoryUsage(&temp_size, DELTA_MATRIX_M(C));
	ASSERT(info == GrB_SUCCESS)
	_size += temp_size;
	info = GxB_Matrix_memoryUsage(&temp_size, DELTA_MATRIX_DELTA_PLUS(C));
	ASSERT(info == GrB_SUCCESS)
	_size += temp_size;
	info = GxB_Matrix_memoryUsage(&temp_size, DELTA_MATRIX_DELTA_MINUS(C));
	ASSERT(info == GrB_SUCCESS)
	_size += temp_size;
	*size = _size;
	return info;
}

GrB_Info Delta_Matrix_setM
(
	Delta_Matrix C,  // delta matrix
	GrB_Matrix M     // new M
) {
	GrB_Index nvals = 0;
	GrB_Index tot   = 0;
	GrB_Info info   = GrB_Matrix_nvals(&nvals, DELTA_MATRIX_M(C));
	ASSERT(info == GrB_SUCCESS);
	tot |= nvals;
	info = GrB_Matrix_nvals(&nvals, DELTA_MATRIX_DELTA_PLUS(C));
	ASSERT(info == GrB_SUCCESS);
	tot |= nvals;
	info = GrB_Matrix_nvals(&nvals, DELTA_MATRIX_DELTA_MINUS(C));
	ASSERT(info == GrB_SUCCESS);
	tot |= nvals;
	if (tot != 0)
		return GrB_ALREADY_SET;
	info = GrB_free(&DELTA_MATRIX_M(C));
	ASSERT(info == GrB_SUCCESS);

	DELTA_MATRIX_M(C) = M;
	return GrB_SUCCESS;
}

const GrB_Matrix Delta_Matrix_M
(
	const Delta_Matrix C  // delta matrix
) {
	return DELTA_MATRIX_M(C);
}