//------------------------------------------------------------------------------
// gbmex_extractvalues: extract all entries from a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// X = gbmex_extractvalues (A)

// X is returned as a MATLAB matrix of size nvals-by-1.
// The input matrix A must have no pending work.

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL                    \
    gb_free (&x, xarena) ;          \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Vector_free (&X_vector) ;

#define USAGE "usage: X = GrB.extractvalues (A)"

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
    GrB_Vector X_vector = NULL ;
    GrB_Type xtype = NULL ;
    void *x = NULL ;
    int xarena = GrB_DEFAULT ;  // revised below
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    GBMX_USAGE (nargin == 1 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    CHECK_ERROR (Matrix [0].will_wait, "matrix must have no pending work") ;

    //--------------------------------------------------------------------------
    // construct X output
    //--------------------------------------------------------------------------

    uint64_t nvals = Matrix [0].nvals ;
    GrB_Type X_type = Matrix [0].type ;
    size_t X_typesize = Matrix [0].typesize ;

    pargout [0] = gbmx_new_matlab_matrix (nvals, 1, X_type) ;
    void *X_out = mxGetData (pargout [0]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the # of threads to use
    //--------------------------------------------------------------------------

    int nthreads ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &nthreads, GxB_NTHREADS)) ;

    //--------------------------------------------------------------------------
    // get the matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
    uint64_t nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    int burble ;
    bool disable_burble = (nrows <= 1 && ncols <= 1) ;
    if (disable_burble)
    { 
        OK (GrB_Global_get_INT32 (GrB_GLOBAL, &burble, GxB_BURBLE)) ;
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, false, GxB_BURBLE)) ;
    }

    //--------------------------------------------------------------------------
    // extract the tuples
    //--------------------------------------------------------------------------

    int handling = 0 ;
    uint64_t X_memsize = 0, nvals2 = 0 ;

    OK (GxB_Vector_new_arena (&X_vector, X_type, 0, arena, arena)) ;
    OK (GxB_Matrix_extractTuples_Vector (NULL, NULL, X_vector, A, NULL)) ;
    OK (GxB_Vector_unload (X_vector, &x, &xtype, &nvals2, &X_memsize,
        &handling, NULL)) ;
    xarena = (handling >= GxB_IS_READONLY) ?
        (handling - GxB_IS_READONLY) : handling ;
    ASSERT (xtype == X_type) ;
    ASSERT (nvals == nvals2) ;
    GB_memcpy (X_out, x, nvals * X_typesize, nthreads) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    if (disable_burble)
    { 
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, burble, GxB_BURBLE)) ;
    }

    FREE_ALL ;
    gb_wrapup ( ) ;
}

