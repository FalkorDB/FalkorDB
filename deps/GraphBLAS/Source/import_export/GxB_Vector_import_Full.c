//------------------------------------------------------------------------------
// GxB_Vector_import_Full: import a vector in full format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "import_export/GB_export.h"

GrB_Info GxB_Vector_import_Full // import a full vector
(
    GrB_Vector *v,      // handle of vector to create
    GrB_Type type,      // type of vector to create
    uint64_t n,         // vector length

    void **vx,          // values
    uint64_t vx_size,   // size of vx in bytes
    bool iso,           // if true, v is iso

    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE0 ("GxB_Vector_import_Full (&v, type, n, "
        "&vx, vx_size, iso, desc)") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // import the vector
    //--------------------------------------------------------------------------

    info = GB_import (false, (GrB_Matrix *) v, type, n, 1, false,
        NULL, 0,        // Ap
        NULL, 0,        // Ah
        NULL, 0,        // Ab
        NULL, 0,        // Ai
        vx,   vx_size,  // Ax
        0, false, 0,
        GxB_FULL, true,                     // full by col
        iso, fast_import, true, Werk) ;

    return (info) ;
}

