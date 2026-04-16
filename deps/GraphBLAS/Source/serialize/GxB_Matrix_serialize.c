//------------------------------------------------------------------------------
// GxB_Matrix_serialize: copy a matrix into a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// serialize a GrB_Matrix into a blob of bytes

// This method is similar to GrB_Matrix_serialize.  In contrast with the GrB*
// method, this method allocates the blob itself, and hands over the allocated
// space to the user application.  The blob must be freed by the same free
// function passed in to GxB_init, or by the C11 free() if GrB_init was
// used.  On input, the blob_memsize need not be initialized; it is returned as
// the size of the blob as allocated.

// This method also includes the descriptor as the last parameter, which allows
// for the compression method to be selected, and controls the # of threads
// used to create the blob.  Example usage:

/*
    void *blob = NULL ;
    uint64_t blob_memsize = 0 ;
    GrB_Matrix A, B = NULL ;
    // construct a matrix A, then serialized it:
    GxB_Matrix_serialize (&blob, &blob_memsize, A, NULL) ; // GxB mallocs blob
    GxB_Matrix_deserialize (&B, atype, blob, blob_memsize, NULL) ;
    free (blob) ;                                   // user frees the blob
*/

#include "GB.h"
#include "serialize/GB_serialize.h"

GrB_Info GxB_Matrix_serialize       // serialize a GrB_Matrix to a blob
(
    // output:
    void **blob_handle,             // the blob, allocated on output
    uint64_t *blob_memsize_handle,     // size of the blob on output
    // input:
    GrB_Matrix A,                   // matrix to serialize
    const GrB_Descriptor desc       // descriptor to select compression method
                                    // and to control # of threads used
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (blob_handle) ;
    GB_RETURN_IF_NULL (blob_memsize_handle) ;
    GB_RETURN_IF_NULL (A) ;
    GB_WHERE_1 (A, "GxB_Matrix_serialize (&blob, &blob_memsize, A, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_serialize") ;

    int memlane = GB_memlane (A->header_mem) ;
    uint64_t mem = GB_mem (memlane, 0) ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    // get the compression method from the descriptor
    int method = (desc == NULL) ? GxB_DEFAULT : desc->compression ;

    //--------------------------------------------------------------------------
    // serialize the matrix
    //--------------------------------------------------------------------------

    (*blob_handle) = NULL ;
    uint64_t blob_memsize = mem ;
    info = GB_serialize ((GB_void **) blob_handle, &blob_memsize, A, method,
        Werk) ;
    (*blob_memsize_handle) = blob_memsize ;
    GB_BURBLE_END ;
    #pragma omp flush
    return (info) ;
}

