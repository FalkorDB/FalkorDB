//------------------------------------------------------------------------------
// GxB_Matrix_pack_BitmapC: pack a matrix in bitmap format, held by column
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "import_export/GB_export.h"

GrB_Info GxB_Matrix_pack_BitmapC  // pack a bitmap matrix, held by column
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    int8_t **Ab,        // bitmap, Ab_size >= nrows*ncols
    void **Ax,          // values, Ax_size >= nrows*ncols * (type size)
                        // or Ax_size >= (type size), if iso is true
    uint64_t Ab_size,   // size of Ab in bytes
    uint64_t Ax_size,   // size of Ax in bytes
    bool iso,           // if true, A is iso
    uint64_t nvals,     // # of entries in bitmap
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (A) ;
    GB_WHERE_1 (A, "GxB_Matrix_pack_BitmapC (A, &Ab, &Ax, Ab_size, Ax_size,"
        " iso, nvals, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_pack_BitmapC") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // pack the matrix
    //--------------------------------------------------------------------------

    info = GB_import (true, &A, A->type, GB_NROWS (A), GB_NCOLS (A), false,
        NULL, 0,        // Ap
        NULL, 0,        // Ah
        Ab,   Ab_size,  // Ab
        NULL, 0,        // Ai
        Ax,   Ax_size,  // Ax
        nvals, false, 0,                    // nvals for bitmap
        GxB_BITMAP, true,                   // bitmap by col
        iso, fast_import, true, Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

