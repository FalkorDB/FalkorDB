//------------------------------------------------------------------------------
// GxB_Matrix_pack_FullC: pack a matrix in full format, held by column
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Matrix_pack_FullC  // pack a full matrix, held by column
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    void **Ax,          // values, Ax_size >= nrows*ncols * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_pack_FullC (A, "
        "&Ax, Ax_size, iso, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_pack_FullC") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // pack the matrix
    //--------------------------------------------------------------------------

    info = GB_import (true, &A, A->type, GB_NROWS (A), GB_NCOLS (A), false,
        NULL, 0,        // Ap
        NULL, 0,        // Ah
        NULL, 0,        // Ab
        NULL, 0,        // Ai
        Ax,   Ax_size,  // Ax
        0, false, 0,
        GxB_FULL, true,                     // full by col
        iso, fast_import, true, Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

