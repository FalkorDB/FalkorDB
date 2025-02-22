//------------------------------------------------------------------------------
// GxB_Vector_import_Bitmap: import a vector in bitmap format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "import_export/GB_export.h"

GrB_Info GxB_Vector_import_Bitmap // import a bitmap vector
(
    GrB_Vector *v,      // handle of vector to create
    GrB_Type type,      // type of vector to create
    uint64_t n,         // vector length

    int8_t **vb,        // bitmap
    void **vx,          // values
    uint64_t vb_size,   // size of vb in bytes
    uint64_t vx_size,   // size of vx in bytes
    bool iso,           // if true, A is iso

    uint64_t nvals,     // # of entries in bitmap
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE0 ("GxB_Vector_import_Bitmap (&v, type, n, "
        "&vb, &vx, vb_size, vx_size, iso, nvals, desc)") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // import the vector
    //--------------------------------------------------------------------------

    info = GB_import (false, (GrB_Matrix *) v, type, n, 1, false,
        NULL, 0,        // Ap
        NULL, 0,        // Ah
        vb,   vb_size,  // Ab
        NULL, 0,        // Ai
        vx,   vx_size,  // Ax
        nvals, false, 0,                    // nvals for bitmap
        GxB_BITMAP, true,                   // bitmap by col
        iso, fast_import, true, Werk) ;

    return (info) ;
}

