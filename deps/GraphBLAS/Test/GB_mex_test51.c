//------------------------------------------------------------------------------
// GB_mex_test51: test reshape with large matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// See issue #462, where the in-place reshape fails if the input matrix and
// output matrix require different integer sizes.  Fixed in GraphBLAS 10.5.2.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                    \
{                                   \
    GrB_Matrix_free (&A) ;          \
}

//------------------------------------------------------------------------------
// GB_mex_test51 mexFunction
//------------------------------------------------------------------------------

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
    GrB_Matrix A = NULL ;

    //--------------------------------------------------------------------------
    // test reshape
    //--------------------------------------------------------------------------

    // n = 2^16 + 1 = 65537
    GrB_Index n = 65537 ;

    for (int trial = 0 ; trial <= 1 ; trial++)
    {

        OK (GrB_Matrix_new (&A, GrB_BOOL, n, n)) ;

        // setElement with 0-based indices:
        OK (GrB_Matrix_setElement_BOOL (A, true, n-1, n-1)) ;

        printf ("printing A in 1-based indices:\n") ;
        OK (GxB_Matrix_fprint (A, "A", 5, NULL)) ;

        GrB_Info info ;
        if (trial == 0)
        {
            printf ("\n=== in-place reshape to 1-by-(n*n):\n") ;
            OK (GxB_Matrix_reshape (A, true, 1, n * n, NULL)) ;
        }
        else
        {
            printf ("\n=== in-place reshape to (n*n)-by-1:\n") ;
            OK (GxB_Matrix_reshape (A, true, n * n, 1, NULL)) ;
        }

        printf ("\nafter reshape (in 1-based indices):\n") ;
        OK (GxB_Matrix_fprint (A, "A", 5, NULL)) ;

        GrB_Index nvals = 0 ;
        OK (GrB_Matrix_nvals (&nvals, A)) ;
        CHECK (nvals == 1) ;

        GrB_Index I [2], J [2] ;
        bool X [2] ;

        OK (GrB_Matrix_extractTuples_BOOL (I, J, X, &nvals, A)) ;
        CHECK (nvals == 1) ;

        // extractTuples returns 0-based indices:
        printf("Extracted edge: (%" PRIu64 ", %" PRIu64 ")\n", I [0], J [0]) ;

        if (trial == 0)
        {
            CHECK (I [0] == 0) ;
            CHECK (J [0] == (n*n) - 1) ;
        }
        else
        {
            CHECK (I [0] == (n*n) - 1) ;
            CHECK (J [0] == 0) ;
        }

        FREE_ALL ;
    }

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("GB_mex_test51:  all tests passed\n") ;
}

