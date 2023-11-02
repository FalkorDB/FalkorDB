//------------------------------------------------------------------------------
// GB_mex_test22: reduce to scalar with user-data types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test22"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

 typedef struct { double a ; double b ; double c ; double d ; } my32 ;
#define MY32_DEFN \
"typedef struct { double a ; double b ; double c ; double d ; } my32 ;"

void myadd32 (my32 *z, const my32 *x, const my32 *y) ;
void myadd32 (my32 *z, const my32 *x, const my32 *y)
{
    z->a = x->a + y->a ;
    z->b = x->b + y->b ;
    z->c = x->c + y->c ;
    z->d = x->d + y->d ;
}

#define MYADD32_DEFN \
"void myadd32 (my32 *z, const my32 *x, const my32 *y)   \n" \
"{                                                      \n" \
"    z->a = x->a + y->a ;                               \n" \
"    z->b = x->b + y->b ;                               \n" \
"    z->c = x->c + y->c ;                               \n" \
"    z->d = x->d + y->d ;                               \n" \
"}"

 typedef struct {
    double a ; double b ; double c ; double d ;
    double e ; double f ; double g ; double h ; } my64 ;
#define MY64_DEFN \
"typedef struct { \n"   \
"   double a ; double b ; double c ; double d ; \n" \
"   double e ; double f ; double g ; double h ; } my64 ; "

void myadd64 (my64 *z, const my64 *x, const my64 *y) ;
void myadd64 (my64 *z, const my64 *x, const my64 *y)
{
    z->a = x->a + y->a ;
    z->b = x->b + y->b ;
    z->c = x->c + y->c ;
    z->d = x->d + y->d ;
    z->e = x->e + y->e ;
    z->f = x->f + y->f ;
    z->g = x->g + y->g ;
    z->h = x->h + y->h ;
}

#define MYADD64_DEFN \
"void myadd64 (my64 *z, const my64 *x, const my64 *y)   \n" \
"{                                                      \n" \
"    z->a = x->a + y->a ;                               \n" \
"    z->b = x->b + y->b ;                               \n" \
"    z->c = x->c + y->c ;                               \n" \
"    z->d = x->d + y->d ;                               \n" \
"    z->e = x->e + y->e ;                               \n" \
"    z->f = x->f + y->f ;                               \n" \
"    z->g = x->g + y->g ;                               \n" \
"    z->h = x->h + y->h ;                               \n" \
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

    //--------------------------------------------------------------------------
    // create the my32 type and Add operator and monoid
    //--------------------------------------------------------------------------

    GrB_Type My32 = NULL ;
    OK (GxB_Type_new (&My32, sizeof (my32), "my32", MY32_DEFN)) ;
    OK (GxB_print (My32, 3)) ;
    GrB_BinaryOp MyAdd32 = NULL ;
    OK (GxB_BinaryOp_new (&MyAdd32, (GxB_binary_function) myadd32,
        My32, My32, My32, "myadd32", MYADD32_DEFN)) ;
    OK (GxB_print (MyAdd32, 3)) ;

    my32 zero32 ;
    zero32.a = 0 ;
    zero32.b = 0 ;
    zero32.c = 0 ;
    zero32.d = 0 ;
    GrB_Monoid Monoid32 = NULL ;
    OK (GrB_Monoid_new_UDT (&Monoid32, MyAdd32, (void *) &zero32)) ;
    OK (GxB_print (Monoid32, 3)) ;

    //--------------------------------------------------------------------------
    // create a my32 matrix and reduce it to a scalar
    //--------------------------------------------------------------------------

    int n = 4 ;
    GrB_Matrix A = NULL ;
    OK (GrB_Matrix_new (&A, My32, n, n)) ;
    my32 z32 ;
    int k = 1 ;
    for (int i = 0 ; i < n ; i++)
    {
        for (int j = i ; j < n ; j++)
        {
            z32.a = k++ ;
            z32.b = (2*i+j) ;
            z32.c = 0 ;
            z32.d = 1 ;
            OK (GrB_Matrix_setElement_UDT (A, (void *) &z32, i, j)) ;
            printf ("A (%d, %d) = (%g, %g, %g, %g)\n", i, j,
                z32.a, z32.b, z32.c, z32.d) ;
        }
    }
    OK (GrB_wait (A, GrB_MATERIALIZE)) ;
    OK (GxB_print (A, 3)) ;

    OK (GrB_Matrix_reduce_UDT ((void *) &z32, NULL, Monoid32, A, NULL)) ;
    printf ("sum: (%g, %g, %g, %g)\n", z32.a, z32.b, z32.c, z32.d) ;
    CHECK (z32.a == 55) ;
    CHECK (z32.b == 40) ;
    CHECK (z32.c == 0) ;
    CHECK (z32.d == 10) ;
    GrB_free (&A) ;

    //--------------------------------------------------------------------------
    // create the my64 type and Add operator and monoid
    //--------------------------------------------------------------------------

    GrB_Type My64 = NULL ;
    OK (GxB_Type_new (&My64, sizeof (my64), "my64", MY64_DEFN)) ;
    OK (GxB_print (My64, 3)) ;
    GrB_BinaryOp MyAdd64 = NULL ;
    OK (GxB_BinaryOp_new (&MyAdd64, (GxB_binary_function) myadd64,
        My64, My64, My64, "myadd64", MYADD64_DEFN)) ;
    OK (GxB_print (MyAdd64, 3)) ;

    my64 zero64 ;
    zero64.a = 0 ;
    zero64.b = 0 ;
    zero64.c = 0 ;
    zero64.d = 0 ;
    zero64.e = 0 ;
    zero64.f = 0 ;
    zero64.g = 0 ;
    zero64.h = 0 ;
    GrB_Monoid Monoid64 = NULL ;
    OK (GrB_Monoid_new_UDT (&Monoid64, MyAdd64, (void *) &zero64)) ;
    OK (GxB_print (Monoid64, 3)) ;

    //--------------------------------------------------------------------------
    // create a my64 matrix and reduce it to a scalar
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, My64, n, n)) ;
    my64 z64 ;
    k = 1 ;
    for (int i = 0 ; i < n ; i++)
    {
        for (int j = i ; j < n ; j++)
        {
            z64.a = k++ ;
            z64.b = (2*i+j) ;
            z64.c = 0 ;
            z64.d = 1 ;
            z64.e = 1-k ;
            z64.f = 99 ;
            z64.g = i+j+k ;
            z64.h = k-j ;
            OK (GrB_Matrix_setElement_UDT (A, (void *) &z64, i, j)) ;
            printf ("A (%d, %d) = (%g, %g, %g, %g, %g, %g, %g, %g)\n", i, j,
                z64.a, z64.b, z64.c, z64.d,
                z64.e, z64.f, z64.g, z64.h) ;
        }
    }
    OK (GrB_wait (A, GrB_MATERIALIZE)) ;
    OK (GxB_print (A, 3)) ;

    OK (GrB_Matrix_reduce_UDT ((void *) &z64, NULL, Monoid64, A, NULL)) ;
    printf ("sum: (%g, %g, %g, %g, %g, %g, %g, %g)\n",
        z64.a, z64.b, z64.c, z64.d, z64.e, z64.f, z64.g, z64.h) ;
    CHECK (z64.a == 55) ;
    CHECK (z64.b == 40) ;
    CHECK (z64.c == 0) ;
    CHECK (z64.d == 10) ;
    CHECK (z64.e == -55) ;
    CHECK (z64.f == 990) ;
    CHECK (z64.g == 95) ;
    CHECK (z64.h == 45) ;
    GrB_free (&A) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&My32) ;
    GrB_free (&MyAdd32) ;
    GrB_free (&Monoid32) ;
    GrB_free (&My64) ;
    GrB_free (&MyAdd64) ;
    GrB_free (&Monoid64) ;

    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test22:  all tests passed\n\n") ;
}

