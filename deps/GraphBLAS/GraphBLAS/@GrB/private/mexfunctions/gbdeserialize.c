//------------------------------------------------------------------------------
// gbdeserialize: deserialize a blob into a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbdeserialize is an interface to GrB_Matrix_deserialize.

// Usage:

// A = gbdeserialize (blob)

#include "gb_interface.h"

#define USAGE "usage: A = GrB.deserialize (blob)"

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

    gb_usage ((nargin >= 1 || nargin <= 3) && nargout <= 1, USAGE) ;
    CHECK_ERROR (mxGetClassID (pargin [0]) != mxUINT8_CLASS
        || mxIsSparse (pargin [0]), "blob must be a uint8 dense matrix/vector");

    //--------------------------------------------------------------------------
    // get the blob, normally a row or column vector, but can be a dense matrix
    //--------------------------------------------------------------------------

    void *blob = mxGetData (pargin [0]) ;
    GrB_Index m = (GrB_Index) mxGetM (pargin [0]) ;
    GrB_Index n = (GrB_Index) mxGetN (pargin [0]) ;
    GrB_Index blob_size = m*n ;

    //--------------------------------------------------------------------------
    // deserialize the blob into a matrix
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL ;
    OK (GrB_Matrix_deserialize (&C, NULL, blob, blob_size)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, KIND_GRB) ;
    GB_WRAPUP ;
}

