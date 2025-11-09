//------------------------------------------------------------------------------
// GrB_Matrix_exportHint: determine sizes of arrays for GrB_Matrix_export
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "transpose/GB_transpose.h"
#define GB_FREE_ALL ;

GrB_Info GrB_Matrix_exportHint  // suggest the best export format
(
    int *format,            // export format
    GrB_Matrix A            // matrix to export
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (format) ;
    GB_RETURN_IF_NULL (A) ;
    GB_WHERE_1 (A, "GrB_Matrix_exportHint (&format, A)") ;

    // finish any pending work since this can change the sparsity of A
    GB_MATRIX_WAIT (A) ;

    int sparsity = GB_sparsity (A) ;
    bool is_csc = A->is_csc ;

    //--------------------------------------------------------------------------
    // determine format that requires the least amount of modification
    //--------------------------------------------------------------------------

    switch (sparsity)
    {
        default:
        case GxB_SPARSE : 
            // CSR and CSC formats are supported by GraphBLAS, so if the matrix
            // is sparse by-row or sparse by-column, then suggest CSR or CSC.
            // The matrix can be exported with no change at all.
        case GxB_BITMAP : 
            // Bitmap is not supported as a GrB_Format.  It cannot be exported
            // as full, in general, so select CSR or CSC.
            (*format) = is_csc ? GrB_CSC_FORMAT : GrB_CSR_FORMAT ;
            break ;

        case GxB_HYPERSPARSE : 
            // Hypersparse is not supported as a GrB_Format.  Expanding a huge
            // hypersparse matrix to sparse can be costly, so suggest COO.
            (*format) = GrB_COO_FORMAT ;
            break ;

        case GxB_FULL : 
            // Full is not supported by GraphBLAS
            (*format) = is_csc ? GrB_CSC_FORMAT : GrB_CSR_FORMAT ;
            // if full was supported by GraphBLAS;
//          (*format) = is_csc ?  GrB_DENSE_COL_FORMAT : GrB_DENSE_ROW_FORMAT ; 
            break ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

