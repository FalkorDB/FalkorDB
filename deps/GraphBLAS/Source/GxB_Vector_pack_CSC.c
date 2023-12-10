//------------------------------------------------------------------------------
// GxB_Vector_pack_CSC: pack a vector in CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Vector_pack_CSC  // pack a vector in CSC format
(
    GrB_Vector v,       // vector to create (type and length unchanged)
    GrB_Index **vi,     // indices, vi_size >= nvals(v) * sizeof(int64_t)
    void **vx,          // values, vx_size >= nvals(v) * (type size)
                        // or vx_size >= (type size), if iso is true
    GrB_Index vi_size,  // size of vi in bytes
    GrB_Index vx_size,  // size of vx in bytes
    bool iso,           // if true, v is iso
    GrB_Index nvals,    // # of entries in vector
    bool jumbled,       // if true, indices may be unsorted
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_pack_CSC (v, "
        "&vi, &vx, vi_size, vx_size, iso, nvals, jumbled, desc)") ;
    GB_BURBLE_START ("GxB_Vector_pack_CSC") ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // pack the vector
    //--------------------------------------------------------------------------

    info = GB_import (true, (GrB_Matrix *) (&v), v->type, v->vlen, 1, true,
        NULL, 0,        // Ap
        NULL, 0,        // Ah
        NULL, 0,        // Ab
        vi,   vi_size,  // Ai
        vx,   vx_size,  // Ax
        nvals, jumbled, 0,                  // jumbled or not
        GxB_SPARSE, true,                   // sparse by col
        iso, fast_import, true, Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

