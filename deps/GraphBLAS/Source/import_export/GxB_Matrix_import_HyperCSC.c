//------------------------------------------------------------------------------
// GxB_Matrix_import_HyperCSC: import a matrix in hypersparse CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "import_export/GB_export.h"

GrB_Info GxB_Matrix_import_HyperCSC      // import a hypersparse CSC matrix
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    uint64_t nrows,     // number of rows of the matrix
    uint64_t ncols,     // number of columns of the matrix

    uint64_t **Ap,      // column "pointers"
    uint64_t **Ah,      // column indices
    uint64_t **Ai,      // row indices
    void **Ax,          // values
    uint64_t Ap_size,   // size of Ap in bytes
    uint64_t Ah_size,   // size of Ah in bytes
    uint64_t Ai_size,   // size of Ai in bytes
    uint64_t Ax_size,   // size of Ax in bytes
    bool iso,           // if true, A is iso

    uint64_t nvec,      // number of columns that appear in Ah
    bool jumbled,       // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE0 ("GxB_Matrix_import_HyperCSC (&A, type, nrows, ncols, "
        "&Ap, &Ah, &Ai, &Ax, Ap_size, Ah_size, Ai_size, Ax_size, iso, "
        "nvec, jumbled, desc)") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    info = GB_import (false, A, type, nrows, ncols, false,
        Ap,   Ap_size,  // Ap
        Ah,   Ah_size,  // Ah
        NULL, 0,        // Ab
        Ai,   Ai_size,  // Ai
        Ax,   Ax_size,  // Ax
        0, jumbled, nvec,                   // jumbled or not
        GxB_HYPERSPARSE, true,              // hypersparse by col
        iso, fast_import, true, Werk) ;

    return (info) ;
}

