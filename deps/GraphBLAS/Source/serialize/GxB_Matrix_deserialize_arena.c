//------------------------------------------------------------------------------
// GxB_Matrix_deserialize_arena: create a matrix from serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// deserialize: create a GrB_Matrix from a blob of bytes

#include "GB.h"
#include "serialize/GB_serialize.h"

GrB_Info GxB_Matrix_deserialize_arena // deserialize blob into a GrB_Matrix
(
    // output:
    GrB_Matrix *C,      // output matrix created from the blob, created in the
                        // header and data arena given by inputs below
    // input:
    GrB_Type type,      // type of the matrix C.  Required if the blob holds a
                        // matrix of user-defined type.  May be NULL if blob
                        // holds a built-in type; otherwise must match the
                        // type of C.
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
    GB_RETURN_IF_NULL (C) ;
    (*C) = NULL ;

    GB_BURBLE_START ("GxB_Matrix_deserialize") ;

    GrB_Info info ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // deserialize the blob into a matrix
    //--------------------------------------------------------------------------

    info = GB_deserialize (C, type, (const GB_void *) blob, blob_memsize,   
        header_arena, data_arena) ;
    GB_BURBLE_END ;
    return (info) ;
}

