//------------------------------------------------------------------------------
// GxB_Matrix_pack_CSR: pack a matrix in CSR format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Matrix_pack_CSR      // pack a CSR matrix
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    GrB_Index **Ap,     // row "pointers", Ap_size >= (nrows+1)* sizeof(int64_t)
    GrB_Index **Aj,     // column indices, Aj_size >= nvals(A) * sizeof(int64_t)
    void **Ax,          // values, Ax_size >= nvals(A) * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ap_size,  // size of Ap in bytes
    GrB_Index Aj_size,  // size of Aj in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    bool jumbled,       // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_pack_CSR (A, "
        "&Ap, &Aj, &Ax, Ap_size, Aj_size, Ax_size, iso, "
        "jumbled, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_pack_CSR") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // pack the matrix
    //--------------------------------------------------------------------------

    info = GB_import (true, &A, A->type, GB_NCOLS (A), GB_NROWS (A), false,
        Ap,   Ap_size,  // Ap
        NULL, 0,        // Ah
        NULL, 0,        // Ab
        Aj,   Aj_size,  // Ai
        Ax,   Ax_size,  // Ax
        0, jumbled, 0,                      // jumbled or not
        GxB_SPARSE, false,                  // sparse by row
        iso, fast_import, true, Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

