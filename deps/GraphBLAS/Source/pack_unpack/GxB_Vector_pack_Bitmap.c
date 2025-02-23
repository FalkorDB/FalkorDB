//------------------------------------------------------------------------------
// GxB_Vector_pack_Bitmap: pack a vector in bitmap format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "import_export/GB_export.h"

GrB_Info GxB_Vector_pack_Bitmap // pack a bitmap vector
(
    GrB_Vector v,       // vector to create (type and length unchanged)
    int8_t **vb,        // bitmap, vb_size >= n
    void **vx,          // values, vx_size >= n * (type size)
                        // or vx_size >= (type size), if iso is true
    uint64_t vb_size,   // size of vb in bytes
    uint64_t vx_size,   // size of vx in bytes
    bool iso,           // if true, v is iso
    uint64_t nvals,     // # of entries in bitmap
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (v) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (v) ;
    GB_WHERE_1 (v, "GxB_Vector_pack_Bitmap (v, &vb, &vx, vb_size, vx_size,"
        " iso, nvals, desc)") ;
    GB_BURBLE_START ("GxB_Vector_pack_Bitmap") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // pack the vector
    //--------------------------------------------------------------------------

    info = GB_import (true, (GrB_Matrix *) (&v), v->type, v->vlen, 1, false,
        NULL, 0,        // Ap
        NULL, 0,        // Ah
        vb,   vb_size,  // Ab
        NULL, 0,        // Ai
        vx,   vx_size,  // Ax
        nvals, false, 0,                    // nvals for bitmap
        GxB_BITMAP, true,                   // bitmap by col
        iso, fast_import, true, Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

