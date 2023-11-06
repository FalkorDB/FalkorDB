//------------------------------------------------------------------------------
// gbsize: dimension and type of a GraphBLAS or built-in matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard built-in
// matrix.  Note that the output may be int64, to accomodate huge hypersparse
// matrices.  Also returns the type of the matrix.

// Usage:

// [m, n, type] = gbsize (X)

#include "gb_interface.h"

#define USAGE "usage: [m n type] = gbsize (X)"

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

    gb_usage (nargin == 1 && nargout <= 4, USAGE) ;

    //--------------------------------------------------------------------------
    // get the # of rows and columns of a GraphBLAS or built-in matrix
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols ;
    int typecode = -1 ;

    if (mxIsStruct (pargin [0]))
    { 

        //----------------------------------------------------------------------
        // get the size of a GraphBLAS matrix
        //----------------------------------------------------------------------

        // get the type
        mxArray *mx_type = mxGetField (pargin [0], 0, "GraphBLASv7_3") ;
        if (mx_type == NULL)
        {
            // check if it is a GraphBLASv5_1 struct
            mx_type = mxGetField (pargin [0], 0, "GraphBLASv5_1") ;
        }
        if (mx_type == NULL)
        {
            // check if it is a GraphBLASv5 struct
            mx_type = mxGetField (pargin [0], 0, "GraphBLASv5") ;
        }
        if (mx_type == NULL)
        {
            // check if it is a GraphBLASv4 struct
            mx_type = mxGetField (pargin [0], 0, "GraphBLASv4") ;
        }
        if (mx_type == NULL)
        {
            // check if it is a GraphBLASv3 struct
            mx_type = mxGetField (pargin [0], 0, "GraphBLAS") ;
        }
        CHECK_ERROR (mx_type == NULL, "invalid GraphBLAS struct") ;

        // get the scalar info
        mxArray *opaque = mxGetField (pargin [0], 0, "s") ;
        CHECK_ERROR (opaque == NULL, "invalid GraphBLAS struct") ;
        // use mxGetData (best for Octave, fine for MATLAB)
        int64_t *s = (int64_t *) mxGetData (opaque) ;
        int64_t vlen = s [1] ;
        int64_t vdim = s [2] ;
        bool is_csc = (bool) (s [6]) ;

        nrows = (is_csc) ? vlen : vdim ;
        ncols = (is_csc) ? vdim : vlen ;

        //----------------------------------------------------------------------
        // return type of a GraphBLAS matrix, if requested
        //----------------------------------------------------------------------

        if (nargout > 2)
        { 
            // return the type
            pargout [2] = mxDuplicateArray (mx_type) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // get the size of a built-in matrix
        //----------------------------------------------------------------------

        nrows = (GrB_Index) mxGetM (pargin [0]) ;
        ncols = (GrB_Index) mxGetN (pargin [0]) ;

        //----------------------------------------------------------------------
        // get the type of a built-in matrix, if requested
        //----------------------------------------------------------------------

        if (nargout > 2)
        { 
            mxClassID class = mxGetClassID (pargin [0]) ;
            bool is_complex = mxIsComplex (pargin [0]) ;
            pargout [2] = gb_mxclass_to_mxstring (class, is_complex) ;
        }
    }

    //--------------------------------------------------------------------------
    // return the size as int64 or double
    //--------------------------------------------------------------------------

    if (nrows > FLINTMAX || ncols > FLINTMAX)
    { 
        // output is int64 to avoid flint overflow
        int64_t *p ;
        pargout [0] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        // use mxGetData (best for Octave, fine for MATLAB)
        p = (int64_t *) mxGetData (pargout [0]) ;
        p [0] = (int64_t) nrows ;
        pargout [1] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        p = (int64_t *) mxGetData (pargout [1]) ;
        p [0] = (int64_t) ncols ;
    }
    else
    { 
        // output is double
        pargout [0] = mxCreateDoubleScalar ((double) nrows) ;
        pargout [1] = mxCreateDoubleScalar ((double) ncols) ;
    }

    GB_WRAPUP ;
}

