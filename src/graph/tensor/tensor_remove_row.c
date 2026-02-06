/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "GraphBLAS.h"
#include "RG.h"
#include "tensor.h"
#include "../delta_matrix/delta_matrix.h"
#include "../delta_matrix/delta_utils.h"

static GrB_UnaryOp free_op = NULL;

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

// remove all entries in the given rows
GrB_Info Tensor_RemoveRows
(
	Tensor T,                  // matrix to remove entry from
	Delta_Matrix *dels,        // A [nvals x ncols] matrix containing the values
	                           // deleted from T, if NULL entries won't be
	                           // returned, instead GrB_free will be called on
	                           // the multi edges
	const GrB_Vector i,        // row index
	const GrB_Descriptor desc  // use INP0 transpose to remove column
	                           // use GxB_ROWINDEX_LIST to manage
	                           // interpretation of i
) {
	ASSERT (T != NULL) ;
	ASSERT (i != NULL) ;
	ASSERT (DELTA_MATRIX_MAINTAIN_TRANSPOSE(T)) ;

	GrB_Index  ncols;
	GrB_Index  nvals;
	GrB_Scalar     empty    = NULL;
	GrB_Scalar     bool_one = NULL;
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

	// initialize unaryop only once
	// WARNING: operator has side effects. Should only be called with an
	// unmasked, inplace apply.
	if (free_op == NULL) {
		GrB_OK (GrB_UnaryOp_new (&free_op, (GxB_unary_function) _free_vectors,
			GrB_UINT64, GrB_UINT64)) ;
	}

	int32_t inp0 = GrB_DEFAULT;
	GrB_Descriptor_get_INT32 (desc, &inp0, GrB_INP0);
	bool transpose = inp0 == GrB_TRAN;

	if (transpose) {
		GrB_OK (GrB_Matrix_nrows(&ncols, m));
	} else {
		GrB_OK (GrB_Matrix_ncols(&ncols, m));
	}

	GrB_OK (GrB_Vector_nvals (&nvals, i));
	GrB_OK (GrB_Scalar_new (&empty, GrB_UINT64)) ;
	GrB_OK (GrB_Scalar_new (&bool_one, GrB_BOOL)) ;
	GrB_OK (GrB_Descriptor_new (&_desc)) ;
	GrB_OK (GrB_Scalar_setElement_BOOL(bool_one, true));

	if (dels == NULL){
		GrB_OK (GrB_Matrix_new (
			&rows, GrB_BOOL, nvals, ncols));
		GrB_OK (GrB_Matrix_new (
			&rows_t, GrB_UINT64, ncols, nvals));
	} else if (transpose) {
		GrB_OK (GrB_Matrix_new (&rows, GrB_UINT64, nvals, ncols));
		GrB_OK (Delta_Matrix_new(&_dels, GrB_UINT64, ncols, nvals, false));
		rows_t = DELTA_MATRIX_DELTA_PLUS(_dels);
	} else {
		GrB_OK (GrB_Matrix_new (&rows, GrB_UINT64, nvals, ncols));
		GrB_OK (Delta_Matrix_new(&_dels, GrB_UINT64, nvals, ncols, false));
		rows = DELTA_MATRIX_DELTA_PLUS(_dels);
	}

	int32_t index_list = GrB_DEFAULT;

	// Get info from descriptor
	GrB_Descriptor_get_INT32 (desc, &index_list, GxB_ROWINDEX_LIST);
	ASSERT (index_list == GrB_DEFAULT || index_list == GxB_USE_INDICES);
	
	GrB_set (_desc, index_list, GxB_ROWINDEX_LIST) ;
	GrB_set (_desc, index_list, GxB_COLINDEX_LIST) ;
	GrB_set (_desc, GrB_STRUCTURE, GrB_MASK) ;

	// get the rows to be deleted
	GrB_OK (GxB_Matrix_extract_Vector (
		rows, NULL, NULL, transpose ? tdp : dp, i, NULL, _desc));

	// delete the given rows in dp and corresponding column in tdp
	// don't use the mask because GraphBLAS can delete the row easily
	GrB_OK (GxB_Matrix_assign_Scalar_Vector (
		transpose ? tdp : dp, NULL,  NULL, empty, i, NULL, _desc));

	GrB_OK (GrB_transpose(rows_t, NULL, NULL, rows, NULL));

	if (dels == NULL || transpose) {
		GrB_OK (GrB_Matrix_clear (rows));
	}

	if (transpose) {
		// extract the entries that are being deleted
		GrB_OK (GxB_Matrix_extract_Vector (
			rows_t, rows_t,  NULL, dp, NULL, i, _desc)) ;
	}

	if (dels == NULL) {
		// call free on these value
		GrB_OK (GrB_Matrix_apply (rows_t, NULL, NULL, free_op, rows_t, NULL)) ;
	} else if (transpose) {
		rows_t = DELTA_MATRIX_M(_dels);
	} else {
		rows = DELTA_MATRIX_M(_dels);
	}


	// use the extracted rows (transposed) as a mask
	// to prevent a full scan of the matrix
	GrB_OK (GxB_Matrix_subassign_Scalar_Vector (
		transpose ? dp : tdp, rows_t,  NULL, empty, NULL, i, _desc)) ;
	GrB_OK (GrB_Matrix_clear(rows_t)) ;

	// -------------------------------------------------------------------------
	// Add to DM and free entries in M
	// -------------------------------------------------------------------------
	GrB_OK (GxB_Matrix_extract_Vector (
		rows, NULL, NULL, transpose ? tm : m, i, NULL, _desc));

	// empty will now be used to overwrite hanging pointers
	GrB_OK (GxB_Scalar_setElement_UINT64(empty, MSB_MASK));

	// set any hanging pointer in m to a NULL pointer to prevent use after free
	// also add deletions into dm and tdm
	GrB_OK (GxB_Matrix_subassign_Scalar_Vector (
		transpose ? tdm : dm, rows, NULL, bool_one, i, NULL, _desc));

	if (!transpose) {
		GrB_OK (GxB_Matrix_subassign_Scalar_Vector (
			m, rows, NULL, empty, i, NULL, _desc));
	}

	GrB_OK (GrB_transpose(rows_t, NULL, NULL, rows, NULL));
	if (dels == NULL || transpose) {
		GrB_OK (GrB_Matrix_clear (rows));
	}

	// NOTE: these should not change the structure of m and so should not
	// make pending changes to m
	if (transpose) {
		// extract the entries that are being deleted
		GrB_OK (GxB_Matrix_extract_Vector (
			rows_t, rows_t,  GrB_SECOND_UINT64, m, NULL, i, _desc)) ;

		// assign deleted entries a value to remove hanging pointers in m
		GrB_OK (GxB_Matrix_subassign_Scalar_Vector (
			m, rows_t, NULL, empty, NULL, i, _desc));
	// } else {
	// 	// assign deleted entries a value to remove hanging pointers in m
	// 	// burble shows m dups which will tank preformance
	// 	// can be fixed with a subassign using a non-transposed row matrix

	// 	GrB_OK (GxB_Matrix_assign_Scalar_Vector (
	// 		m, m, NULL, s, i, NULL, _desc));
	}

	if (dels == NULL) {
		// call free on these value
		GrB_OK (GrB_Matrix_apply (rows_t, NULL, NULL, free_op, rows_t, NULL)) ;
	} else if (transpose) {
		GrB_OK (GxB_Matrix_extract_Vector(
			DELTA_MATRIX_DELTA_MINUS(_dels), rows_t, NULL, dm, NULL, i, NULL));
	} else {
		GrB_OK (GxB_Matrix_extract_Vector(
			DELTA_MATRIX_DELTA_MINUS(_dels), NULL, NULL, dm, i, NULL, NULL));
	}

	GrB_OK (GxB_Matrix_subassign_Scalar_Vector (
		transpose ? dm : tdm, rows_t, NULL, bool_one, NULL, i, _desc));

	if (dels == NULL || transpose) {
		GrB_free (&rows);
	}
	if (dels == NULL || !transpose) {
		GrB_free (&rows_t);
	}
	GrB_free (&_desc);
	GrB_free (&empty);
	GrB_free (&bool_one);

	Delta_Matrix_validate(T, false) ;
	return GrB_SUCCESS;
}
