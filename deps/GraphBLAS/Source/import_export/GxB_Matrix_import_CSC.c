//------------------------------------------------------------------------------
// GxB_Matrix_import_CSC: import a matrix in CSC format (HISTORICAL)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input arrays Ap, Ai, and Ax are assumed to be in the data arena
// defined by the current Context, or the global context if no Context is
// engaged.  Results are undefined if these arrays are in a different arena.

// The output matrix A is created in the same data arena.
// The new header for A is created in the header arena defined by
// the current Context, or the global context if no Context is enganged.

#include "import_export/GB_export.h"

GrB_Info GxB_Matrix_import_CSC      // import a CSC matrix
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    uint64_t nrows,     // number of rows of the matrix
    uint64_t ncols,     // number of columns of the matrix

    uint64_t **Ap,      // column "pointers"
    uint64_t **Ai,      // row indices
    void **Ax,          // values
    uint64_t Ap_memsize,   // size of Ap in bytes
    uint64_t Ai_memsize,   // size of Ai in bytes
    uint64_t Ax_memsize,   // size of Ax in bytes
    bool iso,           // if true, A is iso

    bool jumbled,       // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE0 ("GxB_Matrix_import_CSC (&A, type, nrows, ncols, "
        "&Ap, &Ai, &Ax, Ap_memsize, Ai_memsize, Ax_memsize, iso, "
        "jumbled, desc)") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    info = GB_import (false, A, type, nrows, ncols, false,
        Ap,   Ap_memsize,  // Ap
        NULL, 0,        // Ah
        NULL, 0,        // Ab
        Ai,   Ai_memsize,  // Ai
        Ax,   Ax_memsize,  // Ax
        0, jumbled, 0,                      // jumbled or not
        GxB_SPARSE, true,                   // sparse by col
        iso, fast_import, true, Werk) ;

    return (info) ;
}

