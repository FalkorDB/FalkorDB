//------------------------------------------------------------------------------
// GxB_Vector_serialize: copy a vector into a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// serialize a GrB_Vector into a blob of bytes

// This method is similar to GxB_Matrix_serialize.  Like that method, it
// allocates the blob itself, and hands over the allocated space to the user
// application.  The blob must be freed by the same free function passed in to
// GxB_init, or by the C11 free() if GrB_init was used.  On input, the
// blob_size need not be initialized; it is returned as the size of the blob as
// allocated.

// This method includes the descriptor as the last parameter, which allows
// for the compression method to be selected, and controls the # of threads
// used to create the blob.  Example usage:

/*
    void *blob = NULL ;
    GrB_Index blob_size = 0 ;
    GrB_Vector u, B = NULL ;
    // construct a vector u, then serialized it:
    GxB_Vector_serialize (&blob, &blob_size, u, NULL) ; // GxB mallocs the blob
    GxB_Vector_deserialize (&B, atype, blob, blob_size, NULL) ;
    free (blob) ;                                   // user frees the blob
*/

#include "GB.h"
#include "GB_serialize.h"

GrB_Info GxB_Vector_serialize       // serialize a GrB_Vector to a blob
(
    // output:
    void **blob_handle,             // the blob, allocated on output
    GrB_Index *blob_size_handle,    // size of the blob on output
    // input:
    GrB_Vector u,                   // vector to serialize
    const GrB_Descriptor desc       // descriptor to select compression method
                                    // and to control # of threads used
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_serialize (&blob, &blob_size, u, desc)") ;
    GB_BURBLE_START ("GxB_Vector_serialize") ;
    GB_RETURN_IF_NULL (blob_handle) ;
    GB_RETURN_IF_NULL (blob_size_handle) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    // get the compression method from the descriptor
    int method = (desc == NULL) ? GxB_DEFAULT : desc->compression ;

    //--------------------------------------------------------------------------
    // serialize the vector
    //--------------------------------------------------------------------------

    (*blob_handle) = NULL ;
    size_t blob_size = 0 ;
    info = GB_serialize ((GB_void **) blob_handle, &blob_size, (GrB_Matrix) u,
        method, Werk) ;
    (*blob_size_handle) = (GrB_Index) blob_size ;
    GB_BURBLE_END ;
    #pragma omp flush
    return (info) ;
}

