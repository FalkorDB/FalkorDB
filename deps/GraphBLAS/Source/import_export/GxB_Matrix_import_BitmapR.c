//------------------------------------------------------------------------------
// GxB_Matrix_import_BitmapR: import in bitmap format, held by row (HISTORICAL)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input arrays Ab and Ax are assumed to be in the data arena
// defined by the current Context, or the global context if no Context is
// engaged.  Results are undefined if these arrays are in a different arena.

// The output matrix A is created in the same data arena.
// The new header for A is created in the header arena defined by
// the current Context, or the global context if no Context is enganged.

#include "import_export/GB_export.h"

GrB_Info GxB_Matrix_import_BitmapR  // import a bitmap matrix, held by row
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    uint64_t nrows,     // number of rows of the matrix
    uint64_t ncols,     // number of columns of the matrix

    int8_t **Ab,        // bitmap
    void **Ax,          // values
    uint64_t Ab_memsize,   // size of Ab in bytes
    uint64_t Ax_memsize,   // size of Ax in bytes
    bool iso,           // if true, A is iso

    uint64_t nvals,     // # of entries in bitmap
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE0 ("GxB_Matrix_import_BitmapR (&A, type, nrows, ncols, "
        "&Ab, &Ax, Ab_memsize, Ax_memsize, iso, nvals, desc)") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    info = GB_import (false, A, type, ncols, nrows, false,
        NULL, 0,        // Ap
        NULL, 0,        // Ah
        Ab,   Ab_memsize,  // Ab
        NULL, 0,        // Ai
        Ax,   Ax_memsize,  // Ax
        nvals, false, 0,                    // nvals for bitmap
        GxB_BITMAP, false,                  // bitmap by row
        iso, fast_import, true, Werk) ;

    return (info) ;
}

