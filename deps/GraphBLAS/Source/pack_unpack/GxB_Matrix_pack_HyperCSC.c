//------------------------------------------------------------------------------
// GxB_Matrix_pack_HyperCSC: pack a matrix in hypersparse CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "import_export/GB_export.h"

GrB_Info GxB_Matrix_pack_HyperCSC      // pack a hypersparse CSC matrix
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    uint64_t **Ap,      // col "pointers",
                        // Ap_memsize >= (plen+1)*sizeof(int64_t)
    uint64_t **Ah,      // column indices,
                        // Ah_memsize >= plen*sizeof(int64_t)
                        // where plen = 1 if ncols = 1, or nvec otherwise.
    uint64_t **Ai,      // row indices, Ai_memsize >= nvals(A)*sizeof(int64_t)
    void **Ax,          // values, Ax_memsize >= nvals(A)*(type size)
                        // or Ax_memsize >= (type size), if iso is true
    uint64_t Ap_memsize,   // size of Ap in bytes
    uint64_t Ah_memsize,   // size of Ah in bytes
    uint64_t Ai_memsize,   // size of Ai in bytes
    uint64_t Ax_memsize,   // size of Ax in bytes
    bool iso,           // if true, A is iso
    uint64_t nvec,      // number of columns that appear in Ah
    bool jumbled,       // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (A) ;
    GB_WHERE_1 (A, "GxB_Matrix_pack_HyperCSC (A, &Ap, &Ah, &Ai, &Ax, "
        "Ap_memsize, Ah_memsize, Ai_memsize, Ax_memsize, iso, nvec, jumbled, "
        "desc)") ;
    GB_BURBLE_START ("GxB_Matrix_pack_HyperCSC") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // pack the matrix
    //--------------------------------------------------------------------------

    info = GB_import (true, &A, A->type, GB_NROWS (A), GB_NCOLS (A), false,
        Ap,   Ap_memsize,  // Ap
        Ah,   Ah_memsize,  // Ah
        NULL, 0,        // Ab
        Ai,   Ai_memsize,  // Ai
        Ax,   Ax_memsize,  // Ax
        0, jumbled, nvec,                   // jumbled or not
        GxB_HYPERSPARSE, true,              // hypersparse by col
        iso, fast_import, true, Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

