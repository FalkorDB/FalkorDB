//------------------------------------------------------------------------------
// GxB_Matrix_serialize: copy a matrix into a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// serialize a GrB_Matrix into a blob of bytes

// This method is similar to GrB_Matrix_serialize.  In contrast with the GrB*
// method, this method allocates the blob itself, and hands over the allocated
// space to the user application.  The blob must be freed by the same free
// function passed in to GxB_init, or by the C11 free() if GrB_init was
// used.  On input, the blob_size need not be initialized; it is returned as
// the size of the blob as allocated.

// This method also includes the descriptor as the last parameter, which allows
// for the compression method to be selected, and controls the # of threads
// used to create the blob.  Example usage:

/*
    void *blob = NULL ;
    GrB_Index blob_size = 0 ;
    GrB_Matrix A, B = NULL ;
    // construct a matrix A, then serialized it:
    GxB_Matrix_serialize (&blob, &blob_size, A, NULL) ; // GxB mallocs the blob
    GxB_Matrix_deserialize (&B, atype, blob, blob_size, NULL) ;
    free (blob) ;                                   // user frees the blob
*/

#include "GB.h"
#include "GB_serialize.h"

GrB_Info GxB_Matrix_serialize       // serialize a GrB_Matrix to a blob
(
    // output:
    void **blob_handle,             // the blob, allocated on output
    GrB_Index *blob_size_handle,    // size of the blob on output
    // input:
    GrB_Matrix A,                   // matrix to serialize
    const GrB_Descriptor desc       // descriptor to select compression method
                                    // and to control # of threads used
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_serialize (&blob, &blob_size, A, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_serialize") ;
    GB_RETURN_IF_NULL (blob_handle) ;
    GB_RETURN_IF_NULL (blob_size_handle) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    // get the compression method from the descriptor
    int method = (desc == NULL) ? GxB_DEFAULT : desc->compression ;

    //--------------------------------------------------------------------------
    // serialize the matrix
    //--------------------------------------------------------------------------

    (*blob_handle) = NULL ;
    size_t blob_size = 0 ;
    info = GB_serialize ((GB_void **) blob_handle, &blob_size, A, method,
        Werk) ;
    (*blob_size_handle) = (GrB_Index) blob_size ;
    GB_BURBLE_END ;
    #pragma omp flush
    return (info) ;
}

