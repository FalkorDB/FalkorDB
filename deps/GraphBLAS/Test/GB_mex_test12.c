//------------------------------------------------------------------------------
// GB_mex_test12: more simple tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test12"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

typedef double mytype ;
#define MYTYPE_DEFN "typedef double mytype ;"

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

    //--------------------------------------------------------------------------
    // assign_scalar
    //--------------------------------------------------------------------------

    GrB_Matrix A ;
    GrB_Vector v ;
    GrB_Scalar scalar ;
    GrB_Type MyType ; 
    OK (GrB_Type_new (&MyType, sizeof (mytype))) ;
    OK (GxB_print (MyType, 3)) ;
    OK (GrB_Matrix_new (&A, MyType, 3, 3)) ;
    OK (GrB_Vector_new (&v, MyType, 3)) ;
    OK (GrB_Scalar_new (&scalar, GrB_FP32)) ;
    OK (GrB_Scalar_setElement (scalar, 3.1416)) ;

    GrB_Info expected = GrB_DOMAIN_MISMATCH ;
    const char *error ;

    ERR (GrB_Matrix_assign_Scalar (A, NULL, NULL, scalar,
        GrB_ALL, 3, GrB_ALL, 3, NULL)) ;
    OK (GrB_Matrix_error (&error, A)) ;
    printf ("expected: %s\n", error) ;

    ERR (GxB_Matrix_subassign_Scalar (A, NULL, NULL, scalar,
        GrB_ALL, 3, GrB_ALL, 3, NULL)) ;
    OK (GrB_Matrix_error (&error, A)) ;
    printf ("expected: %s\n", error) ;

    ERR (GrB_Vector_assign_Scalar (v, NULL, NULL, scalar, GrB_ALL, 3, NULL)) ;
    OK (GrB_Vector_error (&error, v)) ;
    printf ("expected: %s\n", error) ;

    ERR (GxB_Vector_subassign_Scalar (v, NULL, NULL, scalar, GrB_ALL, 3, NULL));
    OK (GrB_Vector_error (&error, v)) ;
    printf ("expected: %s\n", error) ;

    GrB_free (&A) ;
    GrB_free (&v) ;
    GrB_free (&MyType) ;
    GrB_free (&scalar) ;

    //--------------------------------------------------------------------------
    // GB_as_if_full
    //--------------------------------------------------------------------------

    CHECK (!GB_as_if_full (NULL)) ;

#if 0
    //--------------------------------------------------------------------------
    // GRAPHBLAS_CACHE_PATH
    //--------------------------------------------------------------------------

    if (getenv ("GRAPHBLAS_CACHE_PATH") != NULL)
    {
        setenv ("GRAPHBLAS_CACHE_PATH", "/tmp/grbcache3", 0) ;
        unsetenv ("GRAPHBLAS_CACHE_PATH") ;
    }
#endif

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test12:  all tests passed\n\n") ;
}

