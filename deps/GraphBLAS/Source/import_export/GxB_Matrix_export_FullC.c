//------------------------------------------------------------------------------
// GxB_Matrix_export_FullC: export a full matrix, held by column
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "import_export/GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_export_FullC  // export and free a full matrix, by column
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    uint64_t *nrows,    // number of rows of the matrix
    uint64_t *ncols,    // number of columns of the matrix

    void **Ax,          // values
    uint64_t *Ax_size,  // size of Ax in bytes
    bool *iso,          // if true, A is iso

    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL (*A) ;
    GB_WHERE_1 (*A, "GxB_Matrix_export_FullC (&A, &type, &nrows, &ncols, "
        "&Ax, &Ax_size, &iso, desc)") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (*A) ;
    if (!GB_is_dense (*A))
    { 
        // A must be dense or full
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // ensure the matrix is full by-col
    //--------------------------------------------------------------------------

    // ensure the matrix is in by-col format
    if (!((*A)->is_csc))
    { 
        // A = A', done in-place, to put A in by-col format
        GB_OK (GB_transpose_in_place (*A, true, Werk)) ;
        GB_MATRIX_WAIT (*A) ;
    }

    GB_convert_any_to_full (*A) ;

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_FULL (*A)) ;
    ASSERT ((*A)->is_csc) ;
    ASSERT (!GB_ZOMBIES (*A)) ;
    ASSERT (!GB_JUMBLED (*A)) ;
    ASSERT (!GB_PENDING (*A)) ;

    int sparsity ;
    bool is_csc ;

    info = GB_export (false, A, type, nrows, ncols, false,
        NULL, NULL,     // Ap
        NULL, NULL,     // Ah
        NULL, NULL,     // Ab
        NULL, NULL,     // Ai
        Ax,   Ax_size,  // Ax
        NULL, NULL, NULL,
        &sparsity, &is_csc,                 // full by col
        iso, Werk) ;

    if (info == GrB_SUCCESS)
    {
        ASSERT (sparsity == GxB_FULL) ;
        ASSERT (is_csc) ;
    }
    return (info) ;
}

