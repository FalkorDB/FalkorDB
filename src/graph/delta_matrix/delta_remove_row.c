
/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "GraphBLAS.h"
#include "RG.h"
#include "delta_matrix.h"
#include "delta_utils.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"
#include "../../globals.h"
#include <stdint.h>

// remove all entries in the given row
GrB_Info Delta_Matrix_removeRow
(
	Delta_Matrix C,     // matrix to remove entry from
	GrB_Index i,        // row index
	GrB_Descriptor desc // use transpose to remove column
) {
	bool       has_t = DELTA_MATRIX_MAINTAIN_TRANSPOSE(C);
	GrB_Index  nrows;
	GrB_Scalar empty = NULL;
	GrB_Vector row   = NULL;

	GrB_Matrix m     = DELTA_MATRIX_M(C);
	GrB_Matrix dm    = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix dp    = DELTA_MATRIX_DELTA_PLUS (C);
	GrB_Matrix tm    = has_t ? DELTA_MATRIX_TM(C) : NULL;
	GrB_Matrix tdm   = has_t ? DELTA_MATRIX_TDELTA_MINUS(C) : NULL;
	GrB_Matrix tdp   = has_t ? DELTA_MATRIX_TDELTA_PLUS (C) : NULL;

	GrB_OK (GrB_Matrix_nrows(&nrows, m));
	GrB_OK (GrB_Scalar_new (&empty, GrB_BOOL)) ;
	GrB_OK (GrB_Vector_new (&row, GrB_BOOL, nrows));

	// check if we remove a row or column
	
	int32_t inp0 = GrB_DEFAULT;
	GrB_Descriptor_get_INT32 (desc, &inp0, GrB_INP0);
	bool transpose = inp0 == GrB_TRAN;

	ASSERT (has_t || !transpose);

	// swap matricies if transposed
	if (transpose) {
		GrB_Matrix temp = dp;
		dp = tdp;
		tdp = temp;
		temp = dm;
		dm = tdm;
		tdm = temp;
	}
	
	GrB_set (GrB_GLOBAL, true, GxB_BURBLE) ;
	// delete the given row in dp and corresponding column in tdp
	GrB_OK (GrB_Matrix_assign_Scalar (
		dp, NULL,  NULL, empty, &i, 1, GrB_ALL, 0, NULL));

	// TODO: this does a full scan. make it faster? extract row from dp, then 
	// delete corresponding elements
	if (has_t) {
		// Scalar_Vector assign on the row instead of w/ GrB all to speed up
		GrB_OK (GrB_Matrix_assign_Scalar (
			tdp, NULL,  NULL, empty, GrB_ALL, 0, &i, 1, NULL));
	}
	GrB_set (GrB_GLOBAL, false, GxB_BURBLE) ;

	// get the row to be deleted
	GrB_OK (GrB_Col_extract (
		row, NULL, NULL, transpose ? tm : m, GrB_ALL, 0, i, GrB_DESC_T0));


	// ensure row is iso (falses could result from typecasting) this is a
	// special case in GraphBLAS so is very quick
	GrB_OK (GrB_Vector_assign_BOOL(row, row, NULL, (bool) true, GrB_ALL, 0,
		GrB_DESC_S)) ;

	// add to dm and tdm
	GrB_OK (GrB_Row_assign (
		dm, NULL, GrB_ONEB_BOOL, row, i, GrB_ALL, 0, NULL));

	if (tdm) {
		GrB_OK (GrB_Col_assign (
			tdm, NULL, GrB_ONEB_BOOL, row, GrB_ALL, 0, i, NULL));
	}
	
	GrB_free(&row);
	GrB_free(&empty);
	return GrB_SUCCESS;
}

// remove all entries in the given rows
GrB_Info Delta_Matrix_removeRows
(
	Delta_Matrix C,            // matrix to remove entry from
	const GrB_Vector i,        // row index
	const GrB_Descriptor desc  // use INP0 transpose to remove column
	                           // use GxB_ROWINDEX_LIST to manage 
	                           // interpretation of i
) {
	GrB_Index  ncols;
	GrB_Index  nvals;
	bool           has_t = DELTA_MATRIX_MAINTAIN_TRANSPOSE(C);
	GrB_Scalar     empty = NULL;
	GrB_Matrix     rows  = NULL;
	GrB_Descriptor d     = NULL;

	GrB_Matrix     m     = DELTA_MATRIX_M(C);
	GrB_Matrix     dm    = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix     dp    = DELTA_MATRIX_DELTA_PLUS (C);
	GrB_Matrix     tm    = has_t ? DELTA_MATRIX_TM(C) : NULL;
	GrB_Matrix     tdm   = has_t ? DELTA_MATRIX_TDELTA_MINUS(C) : NULL;
	GrB_Matrix     tdp   = has_t ? DELTA_MATRIX_TDELTA_PLUS (C) : NULL;

	int32_t inp0 = GrB_DEFAULT;
	GrB_Descriptor_get_INT32 (desc, &inp0, GrB_INP0);
	bool transpose = inp0 == GrB_TRAN;

	ASSERT (has_t || !transpose);

	if (transpose) {
		GrB_OK (GrB_Matrix_nrows(&ncols, m));
	} else {
		GrB_OK (GrB_Matrix_ncols(&ncols, m));
	}

	GrB_OK (GrB_Vector_nvals(&nvals, i));
	GrB_OK (GrB_Scalar_new (&empty, GrB_BOOL)) ;
	GrB_OK (GrB_Matrix_new (&rows, GrB_BOOL, nvals, ncols));
	GrB_OK (GrB_Descriptor_new (&d)) ;

	int32_t index_list = GrB_DEFAULT;

	// Get info from descriptor
	GrB_Descriptor_get_INT32 (desc, &index_list, GxB_ROWINDEX_LIST);
	ASSERT (index_list == GrB_DEFAULT || index_list == GxB_USE_INDICES);
	
	GrB_set (d, index_list, GxB_ROWINDEX_LIST) ;
	GrB_set (d, index_list, GxB_COLINDEX_LIST) ;
	GrB_set (d, GrB_STRUCTURE, GrB_MASK) ;

	// swap matricies if transposed
	if (transpose) {
		GrB_Matrix temp = dp;
		dp = tdp;
		tdp = temp;
		temp = dm;
		dm = tdm;
		tdm = temp;
	}

	// TODO: extract rows from dp before deleting to mask entries in tdp

	// get the rows to be deleted
	GrB_OK (GxB_Matrix_extract_Vector (rows, NULL, NULL, dp, i, NULL, d));

	// ensure row is iso (falses could result from typecasting) special case in
	// GraphBLAS so this is very quick (at most calls free on the x vector)
	GrB_OK (GrB_Matrix_assign_BOOL(rows, rows, NULL, (bool) 1, GrB_ALL, 0,
		GrB_ALL, 0, GrB_DESC_S));

	// delete the given rows in dp and corresponding column in tdp
	// don't use the mask because GraphBLAS can delete the row easily
	GrB_OK (GxB_Matrix_assign_Scalar_Vector (
		dp, NULL,  NULL, empty, i, NULL, d));

	if (tdp) {
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

		// use the extracted rows (transposed) as a mask
		// to prevent a full scan of the matrix
		GrB_OK (GxB_Matrix_subassign_Scalar_Vector (
			tdp, rows,  NULL, empty, NULL, i, d));
		GrB_Matrix_clear(rows);
		GrB_Matrix_resize(rows, nvals, ncols);
	}

	// get the rows to be deleted
	GrB_OK (GxB_Matrix_extract_Vector (
		rows, NULL, NULL, transpose ? tm : m, i, NULL, d));

	// ensure row is iso (falses could result from typecasting) this is a
	// special case in GraphBLAS so is very quick
	GrB_OK (GrB_Matrix_assign_BOOL(rows, rows, NULL, (bool) 1, GrB_ALL, 0,
		GrB_ALL, 0, GrB_DESC_S));

	// add to dm and tdm
	GrB_OK (GxB_Matrix_assign_Vector (
		dm, NULL, GrB_ONEB_BOOL, rows, i, NULL, d));

	if (tdm) {
		GrB_set (d, GrB_TRAN, GrB_INP0) ;
		GrB_OK (GxB_Matrix_assign_Vector (
			tdm, NULL, GrB_ONEB_BOOL, rows, NULL, i, d));
	}

	GrB_free(&d);
	GrB_free(&rows);
	GrB_free(&empty);
	
	return GrB_SUCCESS;
}
