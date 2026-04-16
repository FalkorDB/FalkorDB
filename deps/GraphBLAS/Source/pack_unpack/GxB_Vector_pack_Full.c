//------------------------------------------------------------------------------
// GxB_Vector_pack_Full: pack a vector in full format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "import_export/GB_export.h"

GrB_Info GxB_Vector_pack_Full // pack a full vector
(
    GrB_Vector v,       // vector to create (type and length unchanged)
    void **vx,          // values, vx_memsize >= nvals(v) * (type size)
                        // or vx_memsize >= (type size), if iso is true
    uint64_t vx_memsize,   // size of vx in bytes
    bool iso,           // if true, v is iso
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (v) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (v) ;
    GB_WHERE_1 (v, "GxB_Vector_pack_Full (v, &vx, vx_memsize, iso, desc)") ;
    GB_BURBLE_START ("GxB_Vector_pack_Full") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // pack the vector
    //--------------------------------------------------------------------------

    info = GB_import (true, (GrB_Matrix *) (&v), v->type, v->vlen, 1, false,
        NULL, 0,        // Ap
        NULL, 0,        // Ah
        NULL, 0,        // Ab
        NULL, 0,        // Ai
        vx,   vx_memsize,  // Ax
        0, false, 0,
        GxB_FULL, true,                     // full by col
        iso, fast_import, true, Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

