//------------------------------------------------------------------------------
// GxB_Vector_deserialize_arena: create a vector from serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// deserialize: create a GrB_Vector from a blob of bytes

#include "GB.h"
#include "serialize/GB_serialize.h"

GrB_Info GxB_Vector_deserialize_arena // deserialize blob into a GrB_Vector
(
    // output:
    GrB_Vector *w,      // output vector created from the blob, created in the
                        // header and data arena given by inputs below
    // input:
    GrB_Type type,      // type of the vector w.  Required if the blob holds a
                        // vector of user-defined type.  May be NULL if blob
                        // holds a built-in type; otherwise must match the
                        // type of w.
    const void *blob,   // the blob
    uint64_t blob_memsize, // size of the blob
    const int header_arena,
    const int data_arena,
    const GrB_Descriptor desc       // to control # of threads used
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (w) ;

    GB_BURBLE_START ("GxB_Vector_deserialize") ;

    GrB_Info info ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // deserialize the blob into a vector
    //--------------------------------------------------------------------------

    info = GB_deserialize ((GrB_Matrix *) w, type, (const GB_void *) blob,
        blob_memsize, header_arena, data_arena) ;
    GB_BURBLE_END ;
    return (info) ;
}

