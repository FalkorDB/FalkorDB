//------------------------------------------------------------------------------
// GB_mex_serialize: copy a matrix, using serialize/deserialize
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// copy a matrix using one of:
// GxB_Matrix_serialize and GxB_Matrix_deserialize
// GrB_Matrix_serialize and GrB_Matrix_deserialize
// GxB_Vector_serialize and GxB_Vector_deserialize
// GrB_Vector_serialize and GrB_Vector_deserialize

#include "GB_mex.h"
#include "GB_mex_errors.h"

// method:
// -2                          // GrB*serialize with default LZ4 compression
// GxB_COMPRESSION_NONE -1     // no compression
// GxB_COMPRESSION_DEFAULT 0   // ZSTD level 1
// GxB_COMPRESSION_LZ4   1000  // LZ4
// GxB_COMPRESSION_LZ4HC 2000  // LZ4HC, with default level 9
// GxB_COMPRESSION_LZ4HC 2001  // LZ4HC:1
// ...
// GxB_COMPRESSION_LZ4HC 2009  // LZ4HC:9

// GxB_COMPRESSION_ZSTD  3000  // ZSTD with default level 3

#define USAGE "C = GB_mex_serialize (A, method)"

#define FREE_ALL                        \
{                                       \
    mxFree (blob) ;                     \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&C) ;              \
    GrB_Descriptor_free_(&desc) ;       \
    GB_mx_put_global (true) ;           \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL, C = NULL ;
    GrB_Descriptor desc = NULL ;
    void *blob = NULL ;
    GrB_Index blob_size = 0 ;

    // check inputs
    if (nargout > 1 || nargin < 1 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY  ;
    #define FREE_DEEP_COPY ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get the type of A
    GrB_Type atype ;
    GxB_Matrix_type (&atype, A) ;

    // get method
    int GET_SCALAR (1, int, method, 0) ;
    bool use_GrB_serialize = (method < -1) ;
    if (use_GrB_serialize)
    {
        method = 0 ;
    }
    if (method != 0)
    {
        GrB_Descriptor_new (&desc) ;
        GxB_Desc_set (desc, GxB_COMPRESSION, method) ;
    }

    // serialize A into the blob and then deserialize into C
    if (use_GrB_serialize)
    {
//      if (GB_VECTOR_OK (A))
//      {
//          // test the vector methods
//          METHOD (GrB_Vector_serializeSize (&blob_size, (GrB_Vector) A)) ;
//          blob = mxMalloc (blob_size) ;
//          METHOD (GrB_Vector_serialize (blob, &blob_size, (GrB_Vector) A)) ;
//          METHOD (GrB_Vector_deserialize ((GrB_Vector *) &C, atype,
//              blob, blob_size)) ;
//      }
//      else
        {
            // test the matrix methods
            METHOD (GrB_Matrix_serializeSize (&blob_size, A)) ;
            blob = mxMalloc (blob_size) ;
            METHOD (GrB_Matrix_serialize (blob, &blob_size, A)) ;
            METHOD (GrB_Matrix_deserialize (&C, atype, blob, blob_size)) ;
        }
    }
    else
    {
        if (GB_VECTOR_OK (A))
        {
            // test the vector methods
            METHOD (GxB_Vector_serialize (&blob, &blob_size, (GrB_Vector) A,
                desc));
            METHOD (GxB_Vector_deserialize ((GrB_Vector *) &C, atype,
                blob, blob_size, desc)) ;
        }
        else
        {
            // test the matrix methods
            METHOD (GxB_Matrix_serialize (&blob, &blob_size, A, desc)) ;
            METHOD (GxB_Matrix_deserialize (&C, atype, blob, blob_size, desc)) ;
        }
    }

    // check the type
    char type_name1 [GxB_MAX_NAME_LEN], type_name2 [GxB_MAX_NAME_LEN],
         type_name3 [GxB_MAX_NAME_LEN] ;
    GxB_Matrix_type_name (type_name1, C) ;
    GxB_Matrix_type_name (type_name2, A) ;
    GxB_deserialize_type_name (type_name3, blob, blob_size) ;
    CHECK (MATCH (type_name1, type_name2)) ;
    CHECK (MATCH (type_name1, type_name3)) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;
    FREE_ALL ;
}

