//------------------------------------------------------------------------------
// gbmex_size: dimension, type, and bytes of a GraphBLAS or built-in matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS GrB or GhB matrix or a standard built-in
// matrix.  Note that the [m n] output can be int64 to accomodate huge
// hypersparse matrices.  Optionally returns the type of the matrix.

// Usage:

// [m, n, type, bytes] = gbmex_size (A)

#include "gb_interface.h"
#include "gbmx_interface.h"
#include "GB_opaque.h"

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&A_to_free) ;

#define USAGE "usage: [m n type bytes] = gbmex_size (A)"

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

    GrB_Matrix A = NULL, A_to_free = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    GBMX_USAGE (nargin == 1 && nargout <= 4, USAGE) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix properties
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;

    uint64_t anrows, ancols ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;

    GrB_Type type ;
    OK (GxB_Matrix_type (&type, A)) ;

    size_t bytes ;
    OK (GxB_Matrix_memoryUsage (&bytes, A)) ;

    // The input matrix is freed, so that mx* methods can allocate memory
    // below.  This eliminates any potential memory leaks if A is a handle GrB
    // matrix using malloc/free.

    FREE_ALL ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // return the # of bytes used
    //--------------------------------------------------------------------------

    if (nargout > 3)
    { 
        pargout [3] = mxCreateDoubleScalar ((double) bytes) ;
    }

    //--------------------------------------------------------------------------
    // return the type
    //--------------------------------------------------------------------------

    if (nargout > 2)
    { 
        pargout [2] = gbmx_type_to_mxstring (type) ;
    }

    //--------------------------------------------------------------------------
    // return the size as int64 or double
    //--------------------------------------------------------------------------

    if (anrows > FLINTMAX || ancols > FLINTMAX)
    { 
        // output is int64 to avoid flint overflow
        int64_t *p ;
        pargout [0] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        // use mxGetData (best for Octave, fine for MATLAB)
        p = (int64_t *) mxGetData (pargout [0]) ;
        p [0] = (int64_t) anrows ;
        pargout [1] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        p = (int64_t *) mxGetData (pargout [1]) ;
        p [0] = (int64_t) ancols ;
    }
    else
    { 
        // output is double
        pargout [0] = mxCreateDoubleScalar ((double) anrows) ;
        pargout [1] = mxCreateDoubleScalar ((double) ancols) ;
    }
    gb_wrapup ( ) ;
}

