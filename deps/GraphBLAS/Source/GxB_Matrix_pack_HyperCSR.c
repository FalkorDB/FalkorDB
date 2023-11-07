//------------------------------------------------------------------------------
// GxB_Matrix_pack_HyperCSR: pack a matrix in hypersparse CSR format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Matrix_pack_HyperCSR      // pack a hypersparse CSR matrix
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    GrB_Index **Ap,     // row "pointers", Ap_size >= (plen+1)*sizeof(int64_t)
    GrB_Index **Ah,     // row indices, Ah_size >= plen*sizeof(int64_t)
                        // where plen = 1 if nrows = 1, or nvec otherwise.
    GrB_Index **Aj,     // column indices, Aj_size >= nvals(A)*sizeof(int64_t)
    void **Ax,          // values, Ax_size >= nvals(A) * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ap_size,  // size of Ap in bytes
    GrB_Index Ah_size,  // size of Ah in bytes
    GrB_Index Aj_size,  // size of Aj in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    GrB_Index nvec,     // number of rows that appear in Ah
    bool jumbled,       // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_pack_HyperCSR (A, "
        "&Ap, &Ah, &Aj, &Ax, Ap_size, Ah_size, Aj_size, Ax_size, iso, "
        "nvec, jumbled, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_pack_HyperCSR") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // pack the matrix
    //--------------------------------------------------------------------------

    info = GB_import (true, &A, A->type, GB_NCOLS (A), GB_NROWS (A), false,
        Ap,   Ap_size,  // Ap
        Ah,   Ah_size,  // Ah
        NULL, 0,        // Ab
        Aj,   Aj_size,  // Aj
        Ax,   Ax_size,  // Ax
        0, jumbled, nvec,                   // jumbled or not
        GxB_HYPERSPARSE, false,             // hypersparse by row
        iso, fast_import, true, Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

