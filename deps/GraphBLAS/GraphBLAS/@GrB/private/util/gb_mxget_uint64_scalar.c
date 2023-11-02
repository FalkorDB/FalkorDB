//------------------------------------------------------------------------------
// gb_mxget_int64_scalar: return an int64 scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

uint64_t gb_mxget_uint64_scalar // return uint64 value of a MATLAB scalar
(
    const mxArray *mxscalar,    // MATLAB scalar to extract
    char *name                  // name of the scalar
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!gb_mxarray_is_scalar (mxscalar))
    { 
        GB_COV_PUT ;
        mexErrMsgIdAndTxt ("GrB:error", "%s must be a scalar", name) ;
    }

    //--------------------------------------------------------------------------
    // extract the scalar
    //--------------------------------------------------------------------------

    uint64_t *p, scalar ;

    switch (mxGetClassID (mxscalar))
    {
        case mxINT64_CLASS    : 
        case mxUINT64_CLASS   : 
            p = (uint64_t *) mxGetData (mxscalar) ;
            scalar = p [0] ;
            break ;

        default               : 
            scalar = (uint64_t) mxGetScalar (mxscalar) ;
            break ;
    }

    return (scalar) ;
}

