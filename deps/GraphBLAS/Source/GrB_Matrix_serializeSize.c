//------------------------------------------------------------------------------
// GrB_Matrix_serializeSize: return an upper bound on the blob size
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_Matrix_serialize and GxB_Matrix_serialize both serialize a GrB_Matrix
// into a blob of bytes.  This function provides an estimate of the # of bytes
// the blob would have, assuming the default method and default # of threads,
// using the dryrun option in GB_serialize.

#include "GB.h"
#include "GB_serialize.h"

GrB_Info GrB_Matrix_serializeSize   // estimate the size of a blob
(
    // output:
    GrB_Index *blob_size_handle,    // upper bound on the required size of the
                                    // blob on output.
    // input:
    GrB_Matrix A                    // matrix to serialize
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_serializeSize (&blob_size, A)") ;
    GB_BURBLE_START ("GrB_Matrix_serialize") ;
    GB_RETURN_IF_NULL (blob_size_handle) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    // no descriptor, so assume the default method
    int method = GxB_DEFAULT ;

    // Werk will hold the default # of threads, which can be controlled
    // by GxB_Global_Option_set.

    //--------------------------------------------------------------------------
    // serialize the matrix
    //--------------------------------------------------------------------------

    size_t blob_size ;
    GrB_Info info = GB_serialize (NULL, &blob_size, A, method, Werk) ;
    (*blob_size_handle) = (GrB_Index) blob_size ;
    GB_BURBLE_END ;
    #pragma omp flush
    return (info) ;
}

