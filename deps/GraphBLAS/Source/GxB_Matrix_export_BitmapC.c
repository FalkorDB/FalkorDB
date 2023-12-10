//------------------------------------------------------------------------------
// GxB_Matrix_export_BitmapC: export a bitmap matrix, held by column
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_export_BitmapC  // export and free a bitmap matrix, by col
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix

    int8_t **Ab,        // bitmap
    void **Ax,          // values
    GrB_Index *Ab_size, // size of Ab in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso

    GrB_Index *nvals,   // # of entries in bitmap
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_export_BitmapC (&A, &type, &nrows, &ncols, "
        "&Ab, &Ax, &Ab_size, &Ax_size, &iso, &nvals, desc)") ;
    // GB_BURBLE_START ("GxB_Matrix_export_BitmapC") ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*A) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // ensure the matrix is bitmap by-col
    //--------------------------------------------------------------------------

    // ensure the matrix is in by-col format
    if (!((*A)->is_csc))
    { 
        // A = A', done in-place, to put A in by-col format
        GBURBLE ("(export transpose) ") ;
        GB_OK (GB_transpose_in_place (*A, true, Werk)) ;
    }

    GB_OK (GB_convert_any_to_bitmap (*A, Werk)) ;

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_BITMAP (*A)) ;
    ASSERT (((*A)->is_csc)) ;
    ASSERT (!GB_ZOMBIES (*A)) ;
    ASSERT (!GB_JUMBLED (*A)) ;
    ASSERT (!GB_PENDING (*A)) ;

    int sparsity ;
    bool is_csc ;

    info = GB_export (false, A, type, nrows, ncols, false,
        NULL, NULL,     // Ap
        NULL, NULL,     // Ah
        Ab,   Ab_size,  // Ab
        NULL, NULL,     // Ai
        Ax,   Ax_size,  // Ax
        nvals, NULL, NULL,                  // nvals for bitmap
        &sparsity, &is_csc,                 // bitmap by col
        iso, Werk) ;

    if (info == GrB_SUCCESS)
    {
        ASSERT (sparsity == GxB_BITMAP) ;
        ASSERT (is_csc) ;
    }
    // GB_BURBLE_END ;
    return (info) ;
}

