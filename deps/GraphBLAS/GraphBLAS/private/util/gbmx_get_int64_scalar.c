//------------------------------------------------------------------------------
// gbmx_get_int64_scalar: return an int64 scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

int64_t gbmx_get_int64_scalar   // return int64 value of a MATLAB scalar
(
    const mxArray *mxscalar,    // MATLAB scalar to extract
    char *name                  // name of the scalar
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!gbmx_mxarray_is_scalar (mxscalar))
    {
        ERROR2 ("%s must be a scalar", name, GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // extract the scalar
    //--------------------------------------------------------------------------

    int64_t *p, scalar ;

    switch (mxGetClassID (mxscalar))
    {
        case mxINT64_CLASS    : 
        case mxUINT64_CLASS   : 
            p = (int64_t *) mxGetData (mxscalar) ;
            scalar = p [0] ;
            break ;

        default               : 
            scalar = (int64_t) mxGetScalar (mxscalar) ;
            break ;
    }

    return (scalar) ;
}

