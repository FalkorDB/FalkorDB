//------------------------------------------------------------------------------
// GB_mex_test17: reduce to vector with user-data type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test17"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

 typedef struct { int32_t re ; int32_t im ; } mygauss ;
#define MYGAUSS_DEFN \
"typedef struct { int32_t re ; int32_t im ; } mygauss ;"

void myadd (mygauss *z, const mygauss *x, const mygauss *y) ;
void myadd (mygauss *z, const mygauss *x, const mygauss *y)
{
    z->re = x->re + y->re ;
    z->im = x->im + y->im ;
}

#define MYADD_DEFN \
"void myadd (mygauss *z, const mygauss *x, const mygauss *y)    \n" \
"{                                                              \n" \
"    z->re = x->re + y->re ;                                    \n" \
"    z->im = x->im + y->im ;                                    \n" \
"}"

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
    OK (GxB_set (GxB_BURBLE, true)) ;

    //--------------------------------------------------------------------------
    // create the Gauss type and Add operator and monoid
    //--------------------------------------------------------------------------

    GrB_Type Gauss = NULL ;
    OK (GxB_Type_new (&Gauss, sizeof (mygauss), "mygauss", MYGAUSS_DEFN)) ;
    OK (GxB_print (Gauss, 3)) ;
    GrB_BinaryOp MyAdd = NULL ;
    OK (GxB_BinaryOp_new (&MyAdd, (GxB_binary_function) myadd,
        Gauss, Gauss, Gauss, "myadd", MYADD_DEFN)) ;
    OK (GxB_print (MyAdd, 3)) ;

    mygauss zero ;
    zero.re = 0 ;
    zero.im = 0 ;
    GrB_Monoid Monoid = NULL ;
    OK (GrB_Monoid_new_UDT (&Monoid, MyAdd, (void *) &zero)) ;
    OK (GxB_print (Monoid, 3)) ;

    //--------------------------------------------------------------------------
    // create a Gauss matrix and reduce it to a vector
    //--------------------------------------------------------------------------

    int n = 4 ;
    GrB_Matrix A = NULL ;
    GrB_Vector v = NULL ;
    OK (GrB_Matrix_new (&A, Gauss, n, n)) ;
    OK (GrB_Vector_new (&v, Gauss, n)) ;
    mygauss z ;
    int k = 1 ;
    for (int i = 0 ; i < n ; i++)
    {
        for (int j = i ; j < n ; j++)
        {
            z.re = k++ ;
            z.im = (i-j) ;
            OK (GrB_Matrix_setElement_UDT (A, (void *) &z, i, j)) ;
            printf ("A (%d, %d) = (%d, %d)\n", i, j, z.re, z.im) ;
        }
    }
    OK (GrB_wait (A, GrB_MATERIALIZE)) ;
    OK (GxB_print (A, 3)) ;

    #undef  GET_DEEP_COPY
    #define GET_DEEP_COPY  GrB_Vector_new (&v, Gauss, n) ;
    #undef  FREE_DEEP_COPY
    #define FREE_DEEP_COPY GrB_free (&v) ;

    METHOD (GrB_reduce (v, NULL, NULL, Monoid, A, NULL)) ;
    OK (GxB_print (v, 3)) ;

    printf ("\nvector v:\n") ;
    for (int i = 0 ; i < n ; i++)
    {
        mygauss t ;
        info = GrB_Vector_extractElement_UDT ((void *) &t, v, i) ;
        printf ("v (%d) ", i) ;
        if (info == GrB_NO_VALUE)
        {
            printf (" not present\n") ;
        }
        else
        {
            printf (" = (%d,%d)\n", t.re, t.im) ;
        }
        switch (i)
        {
            case 0: CHECK (t.re == 10 && t.im == -6) ;  break ;
            case 1: CHECK (t.re == 18 && t.im == -3) ;  break ;
            case 2: CHECK (t.re == 17 && t.im == -1) ;  break ;
            case 3: CHECK (t.re == 10 && t.im ==  0) ;  break ;
            default:;
        }
    }

    z.re = 2 ;
    z.im = 0 ;
    GrB_free (&A) ;
    OK (GrB_Matrix_new (&A, Gauss, n, n)) ;
    OK (GrB_assign (A, NULL, NULL, (void *) &z, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GxB_print (A, 3)) ;
    METHOD (GrB_reduce (v, NULL, NULL, Monoid, A, NULL)) ;
    OK (GxB_print (v, 3)) ;

    for (int i = 0 ; i < n ; i++)
    {
        mygauss t ;
        t.re = -999 ;
        t.im = -999 ;
        info = GrB_Vector_extractElement_UDT ((void *) &t, v, i) ;
        printf ("v (%d) ", i) ;
        if (info == GrB_NO_VALUE)
        {
            printf (" not present\n") ;
        }
        else
        {
            printf (" = (%d,%d)\n", t.re, t.im) ;
        }
        CHECK (t.re == 8 && t.im == 0) ;
    }

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&Gauss) ;
    GrB_free (&MyAdd) ;
    GrB_free (&Monoid) ;
    GrB_free (&A) ;
    GrB_free (&v) ;

    OK (GxB_set (GxB_BURBLE, false)) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test17:  all tests passed\n\n") ;
}

