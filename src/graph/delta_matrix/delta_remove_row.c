
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

	int32_t transpose;
	GrB_Descriptor_get_INT32 (desc, &transpose, GrB_INP0);
	transpose = transpose == GrB_TRAN;

	// swap matricies if transposed
	if (transpose) {
		GrB_Matrix temp = dp;
		dp = tdp;
		tdp = temp;
		temp = dm;
		dm = tdm;
		tdm = temp;
	}
	
	// delete the given row in dp and corresponding column in tdp
	if (dp) {
		GrB_OK (GrB_Matrix_assign_Scalar (
			dp, NULL,  NULL, empty, &i, 1, GrB_ALL, 0, NULL));
	}
	if (tdp) {
		GrB_OK (GrB_Matrix_assign_Scalar (
			tdp, NULL,  NULL, empty, GrB_ALL, 0, &i, 1, NULL));
	}

	// get the row to be deleted
	if (has_t) {
		GrB_OK (GrB_Col_extract (
			row, NULL, NULL, transpose ? tm : m, GrB_ALL, 0, i, GrB_DESC_T0));
	} else {
		// NOTE: this is slow when extracting columns.
		GrB_OK (GrB_Col_extract (
			row, NULL, NULL, m, GrB_ALL, 0, i, transpose ? NULL : GrB_DESC_T0));
	}


	// ensure row is iso (falses could result from typecasting) this is a
	// special case in GraphBLAS so is very quick
	GrB_OK (GrB_Vector_assign_BOOL(row, row, NULL, (bool) 1, GrB_ALL, 0,
		GrB_DESC_S));

	// add to dm and tdm
	if(dm) {
		GrB_OK (GrB_Row_assign (
			dm, NULL, GrB_ONEB_BOOL, row, i, GrB_ALL, 0, NULL));
	}
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
	Delta_Matrix C,      // matrix to remove entry from
	GrB_Vector i,        // row index
	GrB_Descriptor desc  // use INP0 transpose to remove column
	                     // use GxB_ROWINDEX_LIST to manage interpretation of i
) {
	GrB_Index  nrows;
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

	GrB_OK (GrB_Matrix_nrows(&nrows, m));
	GrB_OK (GrB_Vector_nvals(&nvals, i));
	GrB_OK (GrB_Scalar_new (&empty, GrB_BOOL)) ;
	GrB_OK (GrB_Matrix_new (&rows, GrB_BOOL, nvals, nrows));
	GrB_OK (GrB_Descriptor_new (&d)) ;

	int32_t transpose = GrB_DEFAULT;
	int32_t index_list = GrB_DEFAULT;

	// get info from descriptor
	GrB_Descriptor_get_INT32 (desc, &transpose, GrB_INP0);
	GrB_Descriptor_get_INT32 (desc, &index_list, GxB_ROWINDEX_LIST);
	transpose = transpose == GrB_TRAN;
	
	GrB_set (d, index_list, GxB_ROWINDEX_LIST) ;
	GrB_set (d, index_list, GxB_COLINDEX_LIST) ;

	// swap matricies if transposed
	if (transpose) {
		GrB_Matrix temp = dp;
		dp = tdp;
		tdp = temp;
		temp = dm;
		dm = tdm;
		tdm = temp;
	}

	// delete the given rows in dp and corresponding column in tdp
	if (dp) {
		GrB_OK (GxB_Matrix_assign_Scalar_Vector (
			dp, NULL,  NULL, empty, i, NULL, d));
	}

	if (tdp) {
		GrB_OK (GxB_Matrix_assign_Scalar_Vector (
			tdp, NULL,  NULL, empty, NULL, i, d));
	}

	// get the rows to be deleted
	if (has_t || !transpose) {
		GrB_OK (GxB_Matrix_extract_Vector (
			rows, NULL, NULL, transpose ? tm : m, i, NULL, d));
	} else {
		// NOTE: avoid this path. It is slow
		GrB_set (d, GrB_TRAN, GrB_INP0) ;
		GrB_OK (GxB_Matrix_extract_Vector (
			rows, NULL, NULL, m, i, NULL, d));
		GrB_set (d, GrB_DEFAULT, GrB_INP0) ;
	}

	// ensure row is iso (falses could result from typecasting) this is a
	// special case in GraphBLAS so is very quick
	GrB_OK (GrB_Matrix_assign_BOOL(rows, rows, NULL, (bool) 1, GrB_ALL, 0,
		GrB_ALL, 0, GrB_DESC_S));

	// add to dm and tdm
	if (dm) {
		GrB_OK (GxB_Matrix_assign_Vector (
			dm, NULL, GrB_ONEB_BOOL, rows, i, NULL, d));
	}

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

