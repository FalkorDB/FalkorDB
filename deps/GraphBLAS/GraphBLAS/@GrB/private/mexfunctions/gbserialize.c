//------------------------------------------------------------------------------
// gbserialize: serialize a matrix into a blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbserialize is an interface to GxB_Matrix_serialize.

// Usage:

// blob = gbserialize (A, method)

#include "gb_interface.h"

#define USAGE "usage: blob = GrB.serialize (A, method, level)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage ((nargin >= 1 && nargin <= 3) && nargout <= 1, USAGE) ;
    GrB_Matrix A = gb_get_shallow (pargin [0]) ;

    bool debug = false ;
    GrB_Descriptor desc = NULL ;
    if (nargin > 1)
    {
        // create the descriptor
        OK (GrB_Descriptor_new (&desc)) ;
        // get the method
        int method = GxB_COMPRESSION_DEFAULT ;
        int level = 0 ;     // use whatever is the default for the method
        #define LEN 64
        char method_name [LEN+2] ;
        gb_mxstring_to_string (method_name, LEN, pargin [1], "method") ;
        if (MATCH (method_name, "none"))
        {
            method = GxB_COMPRESSION_NONE ;
        }
        else if (MATCH (method_name, "lz4"))
        { 
            method = GxB_COMPRESSION_LZ4 ;
        }
        else if (MATCH (method_name, "lz4hc"))
        { 
            method = GxB_COMPRESSION_LZ4HC ;
        }
        else if (MATCH (method_name, "default") || MATCH (method_name, "zstd"))
        {
            // the default is ZSTD, with level 1
            method = GxB_COMPRESSION_ZSTD ;
        }
        else if (MATCH (method_name, "debug"))
        { 
            // use GrB_Matrix_serializeSize and GrB_Matrix_serialize, just
            // for testing
            debug = true ;
        }
        else
        { 
            ERROR ("unknown method") ;
        }
        // get the method level
        if (nargin > 2)
        { 
            level = (int) mxGetScalar (pargin [2]) ;
        }
        if (level < 0 || level > 999) level = 0 ;
        // set the descriptor
        OK (GxB_Desc_set (desc, GxB_COMPRESSION, method + level)) ;
    }

    //--------------------------------------------------------------------------
    // serialize the matrix into the blob
    //--------------------------------------------------------------------------

    void *blob = NULL ;
    GrB_Index blob_size ;

    if (debug)
    { 
        // debug GrB_Matrix_serializeSize and GrB_Matrix_serialize
        OK (GrB_Matrix_serializeSize (&blob_size, A)) ;
        blob = mxMalloc (blob_size) ;
        OK (GrB_Matrix_serialize (blob, &blob_size, A)) ;
        // shrink the blob to its actual size
        blob = mxRealloc (blob, blob_size) ;
    }
    else
    { 
        // use GxB_Matrix_serialize by default
        OK (GxB_Matrix_serialize (&blob, &blob_size, A, desc)) ;
    }

    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // free the shallow matrix A
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;

    //--------------------------------------------------------------------------
    // return the blob to MATLAB as a uint8 dense blobsize-by-1 array
    //--------------------------------------------------------------------------

    pargout [0] = mxCreateNumericMatrix (0, 1, mxUINT8_CLASS, mxREAL) ;
    mxFree (mxGetData (pargout [0])) ;
    mxSetData (pargout [0], blob) ;
    mxSetM (pargout [0], blob_size) ;
    GB_WRAPUP ;
}

