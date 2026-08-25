//------------------------------------------------------------------------------
// GxB_Vector_serialize: copy a vector into a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// serialize a GrB_Vector into a blob of bytes

// This method is similar to GxB_Matrix_serialize.  Like that method, it
// allocates the blob itself, and hands over the allocated space to the user
// application.  The blob must be freed by the same free function passed in to
// GxB_init, or by the C11 free() if GrB_init was used.  On input, the
// blob_memsize need not be initialized; it is returned as the size of the blob
// as allocated.

// This method includes the descriptor as the last parameter, which allows
// for the compression method to be selected, and controls the # of threads
// used to create the blob.  Example usage:

/*
    void *blob = NULL ;
    uint64_t blob_memsize = 0 ;
    GrB_Vector u, B = NULL ;
    // construct a vector u, then serialized it:
    GxB_Vector_serialize (&blob, &blob_memsize, u, NULL) ; // GxB mallocs blob
    GxB_Vector_deserialize (&B, atype, blob, blob_memsize, NULL) ;
    free (blob) ;                                   // user frees the blob
*/

// The blob is created in the current data arena, as defined by the Context.

#include "GB.h"
#include "serialize/GB_serialize.h"

GrB_Info GxB_Vector_serialize       // serialize a GrB_Vector to a blob
(
    // output:
    void **blob_handle,             // the blob, allocated on output
    uint64_t *blob_memsize_handle,     // size of the blob on output
    // input:
    GrB_Vector u,                   // vector to serialize
    const GrB_Descriptor desc       // descriptor to select compression method
                                    // and to control # of threads used
)
{ 
    int data_arena = GB_Context_data_arena ( ) ;
    return (GxB_Vector_serialize_arena (blob_handle, blob_memsize_handle, u,
        data_arena, desc)) ;
}

