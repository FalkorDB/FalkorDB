//------------------------------------------------------------------------------
// GxB_Matrix_pack_CSC: pack a matrix in CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "import_export/GB_export.h"

GrB_Info GxB_Matrix_pack_CSC      // pack a CSC matrix
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    uint64_t **Ap,      // col "pointers", Ap_size >= (ncols+1)*sizeof(int64_t)
    uint64_t **Ai,      // row indices, Ai_size >= nvals(A)*sizeof(int64_t)
    void **Ax,          // values, Ax_size >= nvals(A) * (type size)
                        // or Ax_size >= (type size), if iso is true
    uint64_t Ap_size,   // size of Ap in bytes
    uint64_t Ai_size,   // size of Ai in bytes
    uint64_t Ax_size,   // size of Ax in bytes
    bool iso,           // if true, A is iso
    bool jumbled,       // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (A) ;
    GB_WHERE_1 (A, "GxB_Matrix_pack_CSC (A, &Ap, &Ai, &Ax, Ap_size, Ai_size,"
        " Ax_size, iso, jumbled, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_pack_CSC") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // pack the matrix
    //--------------------------------------------------------------------------

    info = GB_import (true, &A, A->type, GB_NROWS (A), GB_NCOLS (A), false,
        Ap,   Ap_size,  // Ap
        NULL, 0,        // Ah
        NULL, 0,        // Ab
        Ai,   Ai_size,  // Ai
        Ax,   Ax_size,  // Ax
        0, jumbled, 0,                      // jumbled or not
        GxB_SPARSE, true,                   // sparse by col
        iso, fast_import, true, Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

