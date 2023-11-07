//------------------------------------------------------------------------------
// GB_mex_test19: serialize/deserialize tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test19"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;

    //--------------------------------------------------------------------------
    // test serialize/deserialize for a dense matrix
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, C = NULL ;
    int n = 100 ;
    OK (GrB_Matrix_new (&A, GrB_FP32, n, n)) ;
    OK (GrB_assign (A, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GrB_Matrix_setElement (A, 0, 0, 0)) ;
    OK (GrB_wait (A, GrB_MATERIALIZE)) ;
    // OK (GxB_print (A, 2)) ;

    void *blob = NULL ;
    GrB_Index blob_size ;

    // default compression
    OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;
    OK (GxB_Matrix_deserialize (&C, NULL, blob, blob_size, NULL)) ;
    printf ("default compression, blob size %lu\n", blob_size) ;
    CHECK (GB_mx_isequal (A, C, 0)) ;
    GrB_free (&C) ;

    // mangle the blob (set Cx_Sblocks [0] = -1)
    GrB_Info expected = GrB_INVALID_OBJECT ;
    ERR (GxB_Matrix_deserialize (&C, NULL, blob, blob_size - 4, NULL)) ;
    size_t s = 
          sizeof (uint64_t) 
        + sizeof (int32_t)  * 14
        + sizeof (int64_t)  * 11
        + sizeof (float) * 2 ;
    int64_t Cx_Sblocks = -1 ;
    memcpy (((uint8_t *) blob) + s, &Cx_Sblocks, sizeof (int64_t)) ;
    ERR (GxB_Matrix_deserialize (&C, NULL, blob, blob_size, NULL)) ;

    // free the blob
    mxFree (blob) ;

    // no compression
    GrB_Descriptor desc ;
    OK (GrB_Descriptor_new (&desc)) ;
    OK (GxB_set (desc, GxB_COMPRESSION, GxB_COMPRESSION_NONE)) ;
    OK (GxB_Matrix_serialize (&blob, &blob_size, A, desc)) ;
    OK (GxB_Matrix_deserialize (&C, NULL, blob, blob_size, NULL)) ;
    printf ("no compression, blob size %lu\n", blob_size) ;
    CHECK (GB_mx_isequal (A, C, 0)) ;
    GrB_free (&C) ;

    // mangle the blob (set Cx_nblocks to 4)
    s = 
          sizeof (uint64_t) 
        + sizeof (int32_t)  * 12
        + sizeof (int64_t)  * 11
        + sizeof (float) * 2 ;
    int32_t Cx_nblocks = 4 ;
    memcpy (((uint8_t *) blob) + s, &Cx_nblocks, sizeof (int32_t)) ;
    ERR (GxB_Matrix_deserialize (&C, NULL, blob, blob_size, NULL)) ;

    // free the blob
    mxFree (blob) ;
    GrB_free (&desc) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&A) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test19:  all tests passed\n\n") ;
}

