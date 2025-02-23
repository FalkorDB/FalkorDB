//------------------------------------------------------------------------------
// GrB_Matrix_serialize: copy a matrix into a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// serialize a GrB_Matrix into a blob of bytes

// This method is similar to GxB_Matrix_serialize.  In contrast with the GrB*
// method, this method requires the user application to allocate the blob
// first, which must be non-NULL on input.  The required size of the blob is
// computed by GrB_Matrix_serializeSize.  Example usage:

/*
    void *blob = NULL ;
    uint64_t blob_size = 0 ;
    GrB_Matrix A, B = NULL ;
    // construct a matrix A, then serialized it:
    GrB_Matrix_serializeSize (&blob_size, A) ;      // loose upper bound
    blob = malloc (blob_size) ;                     // user mallocs the blob
    GrB_Matrix_serialize (blob, &blob_size, A) ;    // returns actual size
    blob = realloc (blob, blob_size) ;              // user can shrink the blob
    GrB_Matrix_deserialize (&B, atype, blob, blob_size) ;
    free (blob) ;                                   // user frees the blob
*/

#include "GB.h"
#include "serialize/GB_serialize.h"

GrB_Info GrB_Matrix_serialize       // serialize a GrB_Matrix to a blob
(
    // output:
    void *blob,                     // the blob, already allocated in input
    // input/output:
    uint64_t *blob_size_handle,     // size of the blob on input.  On output,
                                    // the # of bytes used in the blob.
    // input:
    GrB_Matrix A                    // matrix to serialize
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (blob_size_handle) ;
    GB_RETURN_IF_NULL (A) ;
    GB_WHERE_1 (A, "GrB_Matrix_serialize (blob, &blob_size, A)") ;
    GB_BURBLE_START ("GrB_Matrix_serialize") ;

    // no descriptor, so assume the default method
    int method = GxB_DEFAULT ;

    // Werk will hold the default # of threads, which can be controlled
    // by GxB_Global_Option_set.

    //--------------------------------------------------------------------------
    // serialize the matrix into the preallocated blob
    //--------------------------------------------------------------------------

    size_t blob_size = (size_t) (*blob_size_handle) ;
    info = GB_serialize ((GB_void **) &blob, &blob_size, A, method, Werk) ;
    if (info == GrB_SUCCESS)
    { 
        (*blob_size_handle) = (uint64_t) blob_size ;
    }
    GB_BURBLE_END ;
    #pragma omp flush
    return (info) ;
}

