//------------------------------------------------------------------------------
// GB_mex_test8: still more basic tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Test a particular case of GrB_mxm for 0-by-0 iso full matrices.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test8"
#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    GrB_Matrix A = NULL, C = NULL ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;
    int expected = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // GrB_mxm with 0-by-0 iso full matrices
    //--------------------------------------------------------------------------

    // Test for bug fix in GB_iso_reduce_worker (the correct test is "n <= 0").
    // Bug caught by Henry Amuasi in the python grblas interface, on Mar 8,
    // 2022, which causes a stack overflow because of an infinite recursion,
    // and segfaults in v6.2.3 and earlier.  The bug first occurs in v5.1.1,
    // released on June 29, 2021.

    GrB_init (GrB_NONBLOCKING) ;
    // GxB_set (GxB_BURBLE, true) ;
    GrB_Matrix M_0 ;
    GrB_Vector v_0, v_1 ;
    GrB_Matrix_new (&M_0, GrB_INT64, 0, 0) ;
    GrB_Matrix_assign_INT64 (M_0, NULL, NULL, 1, GrB_ALL, 0, GrB_ALL, 0, NULL) ;
    GxB_print (M_0, 3) ;
    GrB_Vector_new (&v_0, GrB_INT64, 0) ;
    GrB_Vector_assign_INT64 (v_0, NULL, NULL, 1, GrB_ALL, 0, NULL) ;
    GxB_print (v_0, 3) ;
    GrB_Vector_new (&v_1, GrB_INT64, 0) ;
    GrB_mxv (v_1, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_INT64, M_0, v_0, NULL) ;
    GxB_print (v_1, 3) ;
    GrB_free (&M_0) ;
    GrB_free (&v_0) ;
    GrB_free (&v_1) ;

    //--------------------------------------------------------------------------
    // reshape error handling
    //--------------------------------------------------------------------------

    GrB_Index n =  (1L << 40) ;
    OK (GrB_Matrix_new (&C, GrB_BOOL, n, n)) ;
    expected = GrB_OUT_OF_MEMORY ;
    ERR (GxB_Matrix_reshape (C, true, n/2, 2*n, NULL)) ;
    OK (GrB_Matrix_free (&C)) ;

    n = 12 ;
    OK (GrB_Matrix_new (&C, GrB_BOOL, n, n)) ;
    expected = GrB_DIMENSION_MISMATCH ;
    ERR (GxB_Matrix_reshape (C, true, n, 2*n, NULL)) ;
    OK (GrB_Matrix_free (&C)) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;   
    printf ("\nGB_mex_test8: all tests passed\n\n") ;
}

