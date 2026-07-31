//------------------------------------------------------------------------------
// GxB_Vector_deserialize: create a vector from a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// deserialize: create a GrB_Vector from a blob of bytes

// The vector is allocated in arenas determined by the current Context.

#include "GB.h"
#include "serialize/GB_serialize.h"

GrB_Info GxB_Vector_deserialize     // deserialize blob into a GrB_Vector
(
    // output:
    GrB_Vector *w,      // output vector created from the blob, created in the
                        // header and data arena of the current Context
    // input:
    GrB_Type type,      // type of the vector w.  Required if the blob holds a
                        // vector of user-defined type.  May be NULL if blob
                        // holds a built-in type; otherwise must match the
                        // type of w.
    const void *blob,   // the blob
    uint64_t blob_memsize, // size of the blob
    const GrB_Descriptor desc       // to control # of threads used
)
{ 
    int header_arena = GB_Context_header_arena ( ) ;
    int data_arena = GB_Context_data_arena ( ) ;
    return (GxB_Vector_deserialize_arena (w, type, blob, blob_memsize,
        header_arena, data_arena, desc)) ;
}

