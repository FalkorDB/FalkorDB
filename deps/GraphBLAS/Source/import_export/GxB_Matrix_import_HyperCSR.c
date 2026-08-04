//------------------------------------------------------------------------------
// GxB_Matrix_import_HyperCSR: import a matrix in hypersparse CSR (HISTORICAL)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input arrays Ap, Ah, Aj, and Ax are assumed to be in the data arena
// defined by the current Context, or the global context if no Context is
// engaged.  Results are undefined if these arrays are in a different arena.

// The output matrix A is created in the same data arena.
// The new header for A is created in the header arena defined by
// the current Context, or the global context if no Context is enganged.

#include "import_export/GB_export.h"

GrB_Info GxB_Matrix_import_HyperCSR      // import a hypersparse CSR matrix
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    uint64_t nrows,     // number of rows of the matrix
    uint64_t ncols,     // number of columns of the matrix

    uint64_t **Ap,      // row "pointers"
    uint64_t **Ah,      // row indices
    uint64_t **Aj,      // column indices
    void **Ax,          // values
    uint64_t Ap_memsize,   // size of Ap in bytes
    uint64_t Ah_memsize,   // size of Ah in bytes
    uint64_t Aj_memsize,   // size of Aj in bytes
    uint64_t Ax_memsize,   // size of Ax in bytes
    bool iso,           // if true, A is iso

    uint64_t nvec,      // number of rows that appear in Ah
    bool jumbled,       // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE0 ("GxB_Matrix_import_HyperCSR (&A, type, nrows, ncols, "
        "&Ap, &Ah, &Aj, &Ax, Ap_memsize, Ah_memsize, Aj_memsize, Ax_memsize, "
        "iso, nvec, jumbled, desc)") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    info = GB_import (false, A, type, ncols, nrows, false,
        Ap,   Ap_memsize,  // Ap
        Ah,   Ah_memsize,  // Ah
        NULL, 0,        // Ab
        Aj,   Aj_memsize,  // Aj
        Ax,   Ax_memsize,  // Ax
        0, jumbled, nvec,                   // jumbled or not
        GxB_HYPERSPARSE, false,             // hypersparse by row
        iso, fast_import, true, Werk) ;

    return (info) ;
}

