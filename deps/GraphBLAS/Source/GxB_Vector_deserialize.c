//------------------------------------------------------------------------------
// GxB_Vector_deserialize: create a vector from a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// deserialize: create a GrB_Vector from a blob of bytes

// Identical to GrB_Vector_deserialize, except that this method has
// a descriptor as the last parameter, to control the # of threads used.

#include "GB.h"
#include "GB_serialize.h"

GrB_Info GxB_Vector_deserialize     // deserialize blob into a GrB_Vector
(
    // output:
    GrB_Vector *w,      // output vector created from the blob
    // input:
    GrB_Type type,      // type of the vector w.  Required if the blob holds a
                        // vector of user-defined type.  May be NULL if blob
                        // holds a built-in type; otherwise must match the
                        // type of w.
    const void *blob,       // the blob
    GrB_Index blob_size,    // size of the blob
    const GrB_Descriptor desc       // to control # of threads used
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_deserialize (&w, type, blob, blob_size, desc)") ;
    GB_BURBLE_START ("GxB_Vector_deserialize") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (w) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // deserialize the blob into a vector
    //--------------------------------------------------------------------------

    info = GB_deserialize ((GrB_Matrix *) w, type, (const GB_void *) blob,
        (size_t) blob_size) ;
    GB_BURBLE_END ;
    return (info) ;
}

