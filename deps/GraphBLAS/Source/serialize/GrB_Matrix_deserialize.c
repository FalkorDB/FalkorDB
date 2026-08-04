//------------------------------------------------------------------------------
// GrB_Matrix_deserialize: create a matrix from a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// deserialize: create a GrB_Matrix from a blob of bytes

// Identical to GxB_Matrix_deserialize, except that this method does not take
// a descriptor as the last parameter.

// The matrix is allocated in arenas determined by the current Context.

#include "GB.h"
#include "serialize/GB_serialize.h"

GrB_Info GrB_Matrix_deserialize     // deserialize blob into a GrB_Matrix
(
    // output:
    GrB_Matrix *C,      // output matrix created from the blob, created in the
                        // header and data arena of the current Context
    // input:
    GrB_Type type,      // type of the matrix C.  Required if the blob holds a
                        // matrix of user-defined type.  May be NULL if blob
                        // holds a built-in type; otherwise must match the
                        // type of C.
    const void *blob,   // the blob
    uint64_t blob_memsize  // size of the blob
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_BURBLE_START ("GrB_Matrix_deserialize") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (C) ;

    //--------------------------------------------------------------------------
    // deserialize the blob into a matrix
    //--------------------------------------------------------------------------

    int header_arena = GB_Context_header_arena ( ) ;
    int data_arena = GB_Context_data_arena ( ) ;

    GrB_Info info = GB_deserialize (C, type, (const GB_void *) blob,
        blob_memsize, header_arena, data_arena) ;
    GB_BURBLE_END ;
    return (info) ;
}

