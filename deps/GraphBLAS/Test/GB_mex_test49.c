//------------------------------------------------------------------------------
// GB_mex_test49: test resize, vector to matrix with pending tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&A) ;

//------------------------------------------------------------------------------
// GB_mex_test49 mexFunction
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
    OK (GrB_set (GrB_GLOBAL, true, GxB_BURBLE)) ;
 
    //--------------------------------------------------------------------------
    // test resize: vector to matrix
    //--------------------------------------------------------------------------

    // Test for resizing an n-by-1 matrix to n-by-k with k>1, when pending
    // tuples exist.  This triggers a bug in GraphBLAS v10.5.0 (see
    // https://github.com/DrTimothyAldenDavis/GraphBLAS/issues/459 ).

    GrB_Matrix A = NULL ;
    OK (GrB_Matrix_new (&A, GrB_BOOL, 8, 1)) ;
    OK (GrB_Matrix_setElement_BOOL (A, true, 2, 0)) ;
    OK (GrB_Matrix_setElement_BOOL (A, true, 1, 0)) ;
    OK (GxB_Matrix_fprint (A, "before resize", 5, NULL)) ;
    OK (GrB_Matrix_resize (A, 8, 2)) ;
    OK (GxB_Matrix_fprint (A, "after resize", 5, NULL)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // test resize:  small to large
    //--------------------------------------------------------------------------

    uint64_t n = 2 * ((uint64_t) UINT32_MAX) ;
    OK (GrB_Matrix_setElement_BOOL (A, true, 2, 1)) ;
    OK (GrB_Matrix_resize (A, 4, n)) ;
    OK (GxB_Matrix_fprint (A, "after 2nd resize", 5, NULL)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;

    OK (GrB_Matrix_setElement_BOOL (A, true, 3, 1)) ;
    OK (GrB_Matrix_resize (A, n, 4)) ;
    OK (GxB_Matrix_fprint (A, "after 3rd resize", 5, NULL)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;

    OK (GrB_Matrix_setElement_BOOL (A, true, n-1, 1)) ;
    OK (GrB_Matrix_resize (A, n, n)) ;
    OK (GxB_Matrix_fprint (A, "after 4th resize", 5, NULL)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    FREE_ALL ;
    OK (GrB_set (GrB_GLOBAL, false, GxB_BURBLE)) ;
    GB_mx_put_global (true) ;
    printf ("GB_mex_test49:  all tests passed\n") ;
}

