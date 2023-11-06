//------------------------------------------------------------------------------
// GxB_Vector_import_CSC: import a vector in CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Vector_import_CSC  // import a vector in CSC format
(
    GrB_Vector *v,      // handle of vector to create
    GrB_Type type,      // type of vector to create
    GrB_Index n,        // vector length

    GrB_Index **vi,     // indices
    void **vx,          // values
    GrB_Index vi_size,  // size of Ai in bytes
    GrB_Index vx_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso

    GrB_Index nvals,    // # of entries in vector
    bool jumbled,       // if true, indices may be unsorted
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_import_CSC (&v, type, n, "
        "&vi, &vx, vi_size, vx_size, iso, nvals, jumbled, desc)") ;
    // GB_BURBLE_START ("GxB_Vector_import_CSC") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // import the vector
    //--------------------------------------------------------------------------

    info = GB_import (false, (GrB_Matrix *) v, type, n, 1, true,
        NULL, 0,        // Ap
        NULL, 0,        // Ah
        NULL, 0,        // Ab
        vi,   vi_size,  // Ai
        vx,   vx_size,  // Ax
        nvals, jumbled, 0,                  // jumbled or not
        GxB_SPARSE, true,                   // sparse by col
        iso, fast_import, true, Werk) ;

    // GB_BURBLE_END ;
    return (info) ;
}

