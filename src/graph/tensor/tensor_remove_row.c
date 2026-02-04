/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "GraphBLAS.h"
#include "RG.h"
#include "tensor.h"
#include "util/arr.h"
#include "../delta_matrix/delta_matrix.h"
#include "../delta_matrix/delta_utils.h"

// free vector entries of a tensor
static void _free_vectors
(
	void *z,       // [ignored] new value
	const void *x  // current entry
) {
	// see if entry is a vector
	uint64_t _x = *(uint64_t*)(x);
	if(!SCALAR_ENTRY(_x)) {
		// free vector
		GrB_Vector V = AS_VECTOR(_x);
		GrB_free(&V);
	}
}

// remove all entries in the given row for a tensor with multi edges
GrB_Info Tensor_RemoveRow
(
	Tensor T,                 // matrix to remove entry from
	const GrB_Index i,        // row index
	const GrB_Descriptor desc // use transpose to remove column
) {
	ASSERT (T);
	ASSERT (DELTA_MATRIX_MAINTAIN_TRANSPOSE(T));
	GrB_Index  nrows;
	GrB_Vector empty = NULL;
	GrB_Vector row   = NULL;

	GrB_Matrix m     = DELTA_MATRIX_M(T);
	GrB_Matrix dm    = DELTA_MATRIX_DELTA_MINUS(T);
	GrB_Matrix dp    = DELTA_MATRIX_DELTA_PLUS (T);
	GrB_Matrix tm    = DELTA_MATRIX_TM(T) ;
	GrB_Matrix tdm   = DELTA_MATRIX_TDELTA_MINUS(T) ;
	GrB_Matrix tdp   = DELTA_MATRIX_TDELTA_PLUS (T) ;

	// initialize unaryop only once
	// WARNING: operator has side effects. Should only be called with an unmasked apply
	GrB_UnaryOp free_op = NULL;
	GrB_OK (GrB_UnaryOp_new (&free_op, _free_vectors, GrB_UINT64,
				GrB_UINT64)) ;

	GrB_OK (GrB_Matrix_nrows(&nrows, m));
	GrB_OK (GrB_Vector_new (&empty, GrB_UINT64, nrows)) ;
	GrB_OK (GrB_Vector_new (&row, GrB_UINT64, nrows));

	// check if we remove a row or column
	
	int32_t inp0 = GrB_DEFAULT;
	GrB_Descriptor_get_INT32 (desc, &inp0, GrB_INP0);
	bool transpose = inp0 == GrB_TRAN;

	// swap matricies if transposed
	if (transpose) {
		GrB_Matrix temp = dp;
		dp = tdp;
		tdp = temp;
		temp = dm;
		dm = tdm;
		tdm = temp;
	}

	// get the row to be deleted
	GrB_OK (GrB_Col_extract (
		row, NULL, NULL, dp, GrB_ALL, 0, i, GrB_DESC_T0));

	//--------------------------------------------------------------------------
	// delete the given row in dp and corresponding column in tdp
	//--------------------------------------------------------------------------
	GrB_OK (GrB_Row_assign(dp, NULL, NULL, empty, i, GrB_ALL, 0, NULL));

	if (transpose) {
		// TODO: is row duped?
		GrB_OK (GrB_Col_extract(
			row, row, GrB_SECOND_UINT64, tdp, GrB_ALL, 0, i, GrB_DESC_S)) ;
	}

	// free the entries in row
	GrB_OK (GrB_Vector_apply (row, NULL, NULL, free_op, row, NULL)) ;

	// Scalar_Vector assign on the row instead of w/ GrB all to speed up
	GrB_OK (GrB_Col_assign(tdp, row, NULL, empty, GrB_ALL, 0, i, GrB_DESC_S)) ;

	//--------------------------------------------------------------------------
	// free the tensors in the given row in m and corresponding column in tm
	// also add the entries to dm and tdm
	//--------------------------------------------------------------------------

	// get the row to be deleted from m
	GrB_OK (GrB_Col_extract (
		row, NULL, NULL, transpose ? tm : m, GrB_ALL, 0, i, GrB_DESC_T0));

	if (transpose) {
		// extract the entries that are being deleted
		// TODO: Is row duped?
		GrB_OK (GrB_Col_extract(
			row, row, GrB_SECOND_UINT64, m, GrB_ALL, 0, i, GrB_DESC_S)) ;
	}

	// free the pointers
	GrB_OK (GrB_Vector_apply (row, NULL, NULL, free_op, row, NULL)) ;

	// ensure that row is iso (and has no hanging pointers)
	GrB_OK (GrB_Vector_assign_UINT64 (
		row, row, NULL, MSB_MASK, GrB_ALL, 0, GrB_DESC_S)) ;

	// Add the row to dm
	GrB_OK (GrB_Row_assign (dm, row, NULL, row, i, GrB_ALL, 0, GrB_DESC_S));

	// remove possible hanging pointers
	// NOTE: these should not change the structure of m and so should not
	// make pending changes to m
	if (transpose) {
		GrB_OK (GrB_Col_assign (m, NULL, NULL, row, GrB_ALL, 0, i, NULL));
	} else {
		GrB_OK (GrB_Row_assign (m, NULL, NULL, row, i, GrB_ALL, 0, NULL));
	}

	// Add the column to tdm
	GrB_OK (GrB_Col_assign (tdm, row, NULL, row, GrB_ALL, 0, i, GrB_DESC_S));
	
	GrB_free (&row);
	GrB_free (&empty);
	GrB_free (&free_op) ;

	Delta_Matrix_validate(T, true);
	return GrB_SUCCESS;
}

// remove all entries in the given rows
GrB_Info Tensor_RemoveRows
(
	Delta_Matrix T,            // matrix to remove entry from
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
	GrB_Scalar     s     = NULL;
	GrB_Matrix     rows  = NULL;
	GrB_Descriptor _desc = NULL;

	GrB_Matrix     m     = DELTA_MATRIX_M(T);
	GrB_Matrix     dm    = DELTA_MATRIX_DELTA_MINUS(T);
	GrB_Matrix     dp    = DELTA_MATRIX_DELTA_PLUS (T);
	GrB_Matrix     tm    = DELTA_MATRIX_TM(T) ;
	GrB_Matrix     tdm   = DELTA_MATRIX_TDELTA_MINUS(T) ;
	GrB_Matrix     tdp   = DELTA_MATRIX_TDELTA_PLUS (T) ;

	// initialize unaryop only once
	// WARNING: operator has side effects. Should only be called with an
	// unmasked, inplace apply.
	static GrB_UnaryOp free_op = NULL;
	GrB_OK (GrB_UnaryOp_new(
		&free_op, _free_vectors, GrB_UINT64, GrB_UINT64));

	int32_t inp0 = GrB_DEFAULT;
	GrB_Descriptor_get_INT32 (desc, &inp0, GrB_INP0);
	bool transpose = inp0 == GrB_TRAN;

	if (transpose) {
		GrB_OK (GrB_Matrix_nrows(&ncols, m));
	} else {
		GrB_OK (GrB_Matrix_ncols(&ncols, m));
	}

	GrB_OK (GrB_Vector_nvals (&nvals, i));
	GrB_OK (GrB_Scalar_new (&s, GrB_UINT64)) ;
	GrB_OK (GrB_Matrix_new (&rows, GrB_UINT64, nvals, ncols));
	GrB_OK (GrB_Descriptor_new (&_desc)) ;

	int32_t index_list = GrB_DEFAULT;

	// Get info from descriptor
	GrB_Descriptor_get_INT32 (desc, &index_list, GxB_ROWINDEX_LIST);
	ASSERT (index_list == GrB_DEFAULT || index_list == GxB_USE_INDICES);
	
	GrB_set (_desc, index_list, GxB_ROWINDEX_LIST) ;
	GrB_set (_desc, index_list, GxB_COLINDEX_LIST) ;
	GrB_set (_desc, GrB_STRUCTURE, GrB_MASK) ;

	// swap matricies if transposed
	if (transpose) {
		GrB_Matrix temp = dp;
		dp = tdp;
		tdp = temp;
		temp = dm;
		dm = tdm;
		tdm = temp;
	}

	// get the rows to be deleted
	GrB_OK (GxB_Matrix_extract_Vector (rows, NULL, NULL, dp, i, NULL, _desc));


	// delete the given rows in dp and corresponding column in tdp
	// don't use the mask because GraphBLAS can delete the row easily
	GrB_OK (GxB_Matrix_assign_Scalar_Vector (
		dp, NULL,  NULL, s, i, NULL, _desc));

	// rows = transpose(rows)
	// manually lazy transpose to ensure no extra work is done
	GxB_Container cont = NULL;
	GrB_OK ( GxB_Container_new (&cont)) ;
	GrB_OK ( GxB_unload_Matrix_into_Container (rows, cont, NULL));
	cont->orientation = cont->orientation == GrB_ROWMAJOR?
		GrB_COLMAJOR : GrB_ROWMAJOR;
	GrB_Index temp = cont->nrows;
	cont->nrows = cont-> ncols;
	cont->ncols = temp;
	GrB_OK (GxB_load_Matrix_from_Container (rows, cont, NULL)) ;

	if (transpose) {
		// extract the entries that are being deleted
		GrB_OK (GxB_Matrix_extract_Vector (
			rows, rows,  NULL, tdp, NULL, i, _desc)) ;
	}

	// call free on these value
	GrB_OK (GrB_Matrix_apply (rows, NULL, NULL, free_op, rows, NULL)) ;

	// use the extracted rows (transposed) as a mask
	// to prevent a full scan of the matrix
	GrB_OK (GxB_Matrix_subassign_Scalar_Vector (
		tdp, rows,  NULL, s, NULL, i, _desc)) ;
	GrB_OK (GrB_Matrix_clear(rows)) ;
	GrB_OK (GrB_Matrix_resize(rows, nvals, ncols)) ;

	GrB_OK (GxB_Matrix_extract_Vector (
		rows, NULL, NULL, transpose ? tm : m, i, NULL, _desc));

	GrB_OK (GxB_Scalar_setElement_UINT64(s, MSB_MASK));

	// set any hanging pointer in m to a NULL pointer to prevent use after free
	// also add deletions into dm and tdm
	GrB_OK (GxB_Matrix_subassign_Scalar_Vector (
		dm, rows, NULL, s, i, NULL, _desc));

	if (!transpose) {
		GrB_OK (GxB_Matrix_subassign_Scalar_Vector (
			m, rows, NULL, s, i, NULL, _desc));
	}

	// rows = transpose(rows)
	// manually lazy transpose to ensure no extra work is done
	GrB_OK ( GxB_unload_Matrix_into_Container (rows, cont, NULL));
	cont->orientation = cont->orientation == GrB_ROWMAJOR?
		GrB_COLMAJOR : GrB_ROWMAJOR;
	temp = cont->nrows;
	cont->nrows = cont-> ncols;
	cont->ncols = temp;
	GrB_OK (GxB_load_Matrix_from_Container (rows, cont, NULL)) ;

	// NOTE: these should not change the structure of m and so should not
	// make pending changes to m
	GrB_set(GrB_GLOBAL, true, GxB_BURBLE);
	if (transpose) {
		// extract the entries that are being deleted
		// TODO: is row duped?
		GrB_OK (GxB_Matrix_extract_Vector (
			rows, rows,  GrB_SECOND_UINT64, m, NULL, i, _desc)) ;

		// assign deleted entries a value to remove hanging pointers in m
		GrB_OK (GxB_Matrix_subassign_Scalar_Vector (
			m, rows, NULL, s, NULL, i, _desc));
	// } else {
	// 	// assign deleted entries a value to remove hanging pointers in m
	// 	// FIXME: burble appears to show m dups which will tank preformance
	// 	// can be fixed with a subassign using a non-transposed row matrix

	// 	GrB_OK (GxB_Matrix_assign_Scalar_Vector (
	// 		m, m, NULL, s, i, NULL, _desc));
	}

	GrB_set(GrB_GLOBAL, false, GxB_BURBLE);

	// free the entries extracted from the matrix
	GrB_OK (GrB_Matrix_apply (rows, NULL, NULL, free_op, rows, NULL)) ;

	GrB_OK (GxB_Matrix_subassign_Scalar_Vector (
		tdm, rows, NULL, s, NULL, i, _desc));

	GrB_free (&_desc);
	GrB_free (&rows);
	GrB_free (&s);
	GrB_free (&free_op);

	Delta_Matrix_validate(T, false) ;
	return GrB_SUCCESS;
}
