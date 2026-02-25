/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "GraphBLAS.h"
#include "RG.h"
#include "tensor.h"
#include "../delta_matrix/delta_matrix.h"
#include "../delta_matrix/delta_utils.h"

static GrB_UnaryOp free_op  = NULL;
static GrB_Scalar  empty    = NULL;
static GrB_Scalar  bool_one = NULL;

// free vector entries of a tensor
static void _free_vectors
(
	uint64_t *z,       // [ignored] new value
	const uint64_t *x  // current entry
) {
	// see if entry is a vector
	uint64_t _x = *x;
	if(!SCALAR_ENTRY(_x)) {
		// free vector
		GrB_Vector V = AS_VECTOR(_x);
		GrB_free(&V);
	}
	*z = MSB_MASK;
}

// FUTURE: GraphBLAS may add specific sparse diagonal multiplication or a
// cartesian mask to speed up this function.
// GrB_Info _get_matrix_rows
// (
// 	GrB_Matrix rows,
// 	const GrB_Matrix A,
// 	const GrB_Vector i,   // row indices
// 	GrB_Descriptor desc,  // must have GxB_ROWINDEX_LIST == GxB_COLINDEX_LIST
// )
// {
// }

// remove all entries in the given rows
GrB_Info Tensor_RemoveRows
(
	Tensor T,                  // matrix to remove entry from
	Delta_Matrix *dels,        // [optional] A matrix the size of T containing
	                           // the values deleted from T, if NULL, entries
	                           // won't be returned, instead GrB_free will be
	                           // called on the multi edges
	const GrB_Vector i,        // row index
	const GrB_Descriptor desc  // use INP0 transpose to remove column
	                           // use GxB_ROWINDEX_LIST to manage
	                           // interpretation of i
) {
	// This function will remove the given rows or columns with calls to
	// GraphBLAS.
	//
	// If the dels matrix is present, this function will allocate it and return
	// a matrix of the same dimensions as T, but containing the values of the
	// entries that have been deleted.
	//
	// When desc does not specify transpose:
	// rows = T.DP[i,:]
	// rows [i,:] = rows        # make rows of the same dimension as T, only
	//                          # containing deleted rows.
	// dels = &rows
	// T.DP [i,:] = empty       # remove the entries in DP
	// t_rows = transpose(rows)
	// T.TDP <t_rows> = empty
	//
	// rows = T.M[i,:]
	// rows [i,:] = rows        # make rows of the same dimension as T, only 
	//                          # containing deleted rows.
	// dels = &rows
	// T.M [i,:] = MSB_MASK     # This prevents hanging pointers in T
	// T.DM <T.M> [i,:] = true  # mark all entries in this row as deleted
	// t_rows = transpose(rows)
	// T.TM <t_rows>  = MSB_MASK
	// T.TDM <t_rows> = true
	//
	// When transposed, the operations are almost the same, except
	// that we get the columns first from the transpose matrix's rows
	// and then retrive the values by
	// t_rows <t_rows> = T.DP
	// and
	// t_rows <t_rows> = T.M
	// after which we save t_rows into the output matrix.

	ASSERT (T != NULL) ;
	ASSERT (i != NULL) ;
	ASSERT (DELTA_MATRIX_MAINTAIN_TRANSPOSE(T)) ;

	GrB_Index  ncols;
	GrB_Index  nrows;
	GrB_Matrix     diag     = NULL;
	GrB_Matrix     rows     = NULL;
	GrB_Matrix     rows_t   = NULL;
	GrB_Descriptor _desc    = NULL;
	Delta_Matrix   _dels = NULL;

	GrB_Matrix     m     = DELTA_MATRIX_M(T);
	GrB_Matrix     dm    = DELTA_MATRIX_DELTA_MINUS(T);
	GrB_Matrix     dp    = DELTA_MATRIX_DELTA_PLUS (T);
	GrB_Matrix     tm    = DELTA_MATRIX_TM(T) ;
	GrB_Matrix     tdm   = DELTA_MATRIX_TDELTA_MINUS(T) ;
	GrB_Matrix     tdp   = DELTA_MATRIX_TDELTA_PLUS (T) ;

	// initialize frequently used static objects once
	if (free_op == NULL) {
		// WARNING: operator has side effects. Should only be called with an
		// unmasked, inplace apply.
		GrB_OK (GrB_UnaryOp_new (&free_op, (GxB_unary_function) _free_vectors,
			GrB_UINT64, GrB_UINT64)) ;
		GrB_OK (GrB_Scalar_new (&empty, GrB_UINT64)) ;
		GrB_OK (GrB_Scalar_new (&bool_one, GrB_BOOL)) ;
		GrB_OK (GrB_Scalar_setElement_BOOL(bool_one, true));
	}

	// get info from descriptor
	int32_t inp0 = GrB_DEFAULT;
	GrB_Descriptor_get_INT32 (desc, &inp0, GrB_INP0);
	bool transpose = inp0 == GrB_TRAN;

	int32_t index_list = GrB_DEFAULT;
	GrB_Descriptor_get_INT32 (desc, &index_list, GxB_ROWINDEX_LIST);
	ASSERT (index_list == GrB_DEFAULT || index_list == GxB_USE_INDICES);
	
	GrB_OK (GrB_Descriptor_new (&_desc)) ;
	GrB_OK (GrB_set (_desc, index_list, GxB_ROWINDEX_LIST)) ;
	GrB_OK (GrB_set (_desc, index_list, GxB_COLINDEX_LIST)) ;
	GrB_OK (GrB_set (_desc, GrB_STRUCTURE, GrB_MASK)) ;

	GrB_OK (GrB_Matrix_nrows(&nrows, m));
	GrB_OK (GrB_Matrix_ncols(&ncols, m));

	// build diagonal matrix to multiply by
	GrB_Index diag_n= transpose ? ncols : nrows;
	GrB_OK (GrB_Matrix_new(&diag, GrB_BOOL, diag_n, diag_n));
	GrB_OK (GxB_Matrix_build_Scalar_Vector(diag, i, i, bool_one, _desc));

	// Semiring for the diagonal mat mul
	GrB_Semiring semiring = transpose ?
		GxB_ANY_PAIR_BOOL : GxB_ANY_SECOND_UINT64;

	GrB_OK (GrB_Matrix_new (&rows_t, GrB_UINT64, ncols, nrows));

	// "rows" accumulates the values extracted from dp (or m) at the targeted
	// row indices. When dels != NULL, rows is a sub-matrix of _dels so that
	// deleted DP entries land in _dels.DP and deleted M entries land in _dels.M
	if (dels == NULL) {
		GrB_OK (GrB_Matrix_new ( &rows, GrB_UINT64, nrows, ncols));
	} else {
		// Allocate the output matrix; deleted DP entries will be written into
		// its DP sub-matrix first, then deleted M entries into its M sub-matrix.
		GrB_OK (Delta_Matrix_new(dels, GrB_UINT64, nrows, ncols, false));
		_dels = *dels;
		rows = DELTA_MATRIX_DELTA_PLUS(_dels);
	}

	// alias local pointers to reduce transpose branches in every function call.
	// swap (rows, rows_t); swap(m, tm); . . .
	if (transpose) {
		GrB_Matrix temp;
		temp = rows; rows = rows_t; rows_t = temp;
		temp = dp; dp = tdp; tdp = temp;
		temp = dm; dm = tdm; tdm = temp;
		temp = m; m = tm; tm = temp;
	}

	// -------------------------------------------------------------------------
	// Remove entries from DP and TDP
	// -------------------------------------------------------------------------
	GrB_OK (GrB_mxm(rows, NULL, NULL, semiring, diag, dp, NULL));

	// delete the given rows in dp and corresponding column in tdp
	// don't use the mask because GraphBLAS can delete the row easily
	GrB_OK (GxB_Matrix_assign_Scalar_Vector (
		dp, NULL,  NULL, empty, i, NULL, _desc));

	GrB_OK (GrB_transpose(rows_t, NULL, NULL, rows, NULL));

	if (dels == NULL || transpose) {
		GrB_OK (GrB_Matrix_clear (rows));
	}

	if (transpose) {
		// extract the entries that are being deleted
		GrB_OK (GrB_Matrix_assign (rows_t, rows_t, GrB_SECOND_UINT64, tdp,
				GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));
	}

	// use rows_t as a structural mask, avoiding a full-matrix scan
	GrB_OK (GrB_Matrix_assign_Scalar(
		tdp, rows_t, NULL, empty, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));

	if (dels == NULL) {
		// free any vector entries captured in rows_t before clearing it
		GrB_OK (GrB_Matrix_apply (rows_t, NULL, NULL, free_op, rows_t, NULL)) ;
		GrB_OK (GrB_Matrix_clear(rows_t)) ;
	} else if (transpose) {
		rows_t = DELTA_MATRIX_M(_dels);
	} else {
		rows = DELTA_MATRIX_M(_dels);
		GrB_OK (GrB_Matrix_clear(rows_t)) ;
	}

	// -------------------------------------------------------------------------
	// Add to DM and free entries in M
	// -------------------------------------------------------------------------
	GrB_OK (GrB_mxm(
		rows, NULL, NULL, semiring, diag, m, NULL));

	// set any hanging pointer in m to a NULL pointer to prevent use after free
	if (!transpose) {
		GrB_OK (GrB_Matrix_assign_UINT64 (
			m, rows, NULL, MSB_MASK, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));
	}

	// also add deletions into dm and tdm
	GrB_OK (GrB_Matrix_assign_BOOL (
		dm, rows, NULL, true, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));


	GrB_OK (GrB_transpose(rows_t, NULL, NULL, rows, NULL));

	if (dels == NULL || transpose) {
		GrB_OK (GrB_free (&rows));
	}

	// NOTE: these should not change the structure of m and so should not
	// make pending changes to m
	if (transpose) {
		// extract the entries that are being deleted
		GrB_OK (GrB_Matrix_assign (rows_t, rows_t, GrB_SECOND_UINT64, tm,
			GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));

		// assign deleted entries a value to remove hanging pointers in m
		GrB_OK (GrB_Matrix_assign_UINT64 (
			tm, rows_t, NULL, MSB_MASK, GrB_ALL, 0, GrB_ALL, 0, NULL));
	}

	// use rows_t as a structural mask, avoiding a full-matrix scan
	GrB_OK (GrB_Matrix_assign_BOOL (
		tdm, rows_t, NULL, true, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));

	if (dels == NULL) {
		// free any vector entries captured in rows_t before clearing it
		GrB_OK (GrB_Matrix_apply (rows_t, NULL, NULL, free_op, rows_t, NULL)) ;
	}

	if (dels == NULL || !transpose) {
		GrB_free (&rows_t);
	}

	GrB_free (&_desc);
	GrB_free (&diag);

	Delta_Matrix_validate(T, true) ;
	return GrB_SUCCESS;
}
