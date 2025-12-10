//------------------------------------------------------------------------------
// GB_mex_test33: test GrB_get and GrB_set (context)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

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

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info, expected ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    GrB_Vector v = NULL ;
    GrB_Scalar s = NULL, s_fp64 = NULL, s_int32 = NULL, s_fp32 = NULL ;
    GxB_Context context = NULL ;
    uint8_t stuff [256] ;
    void *nothing = stuff ;
    size_t size ;
    char name [256] ;
    char defn [2048] ;
    int32_t code, i ;
    float fvalue ;
    double dvalue ;

    OK (GrB_Scalar_new (&s_fp64, GrB_FP64)) ;
    OK (GrB_Scalar_new (&s_fp32, GrB_FP32)) ;
    OK (GrB_Scalar_new (&s_int32, GrB_INT32)) ;

    //--------------------------------------------------------------------------
    // GxB_Context get/set
    //--------------------------------------------------------------------------

    int32_t nthreads1 = 999, nthreads2 = 777 ;
    GxB_get (GxB_NTHREADS, &nthreads1) ;
    printf ("nthreads: %d\n", nthreads1) ;

    OK (GxB_Context_get_INT_ (GxB_CONTEXT_WORLD, &nthreads2, GxB_NTHREADS)) ;
    printf ("nthreads: %d\n", nthreads2) ;
    CHECK (nthreads1 == nthreads2) ;

    OK (GxB_Context_set_INT_ (GxB_CONTEXT_WORLD, 7, GxB_NTHREADS)) ;
    OK (GxB_Context_get_INT_ (GxB_CONTEXT_WORLD, &nthreads2, GxB_NTHREADS)) ;
    CHECK (nthreads2 == 7) ;

    OK (GxB_Global_Option_get (GxB_NTHREADS, &i)) ;
    CHECK (i == 7) ;

    OK (GrB_Scalar_setElement_FP64 (s_int32, 31)) ;
    OK (GxB_Context_set_Scalar_ (GxB_CONTEXT_WORLD, s_int32, GxB_NTHREADS)) ;
    OK (GxB_Context_get_Scalar_ (GxB_CONTEXT_WORLD, s_fp64, GxB_NTHREADS)) ;
    OK (GrB_Scalar_extractElement_FP64 (&dvalue, s_fp64)) ;
    CHECK (dvalue == 31) ;

    GxB_set (GxB_NTHREADS, nthreads1) ;

    double chunk ;
    OK (GxB_Context_get_Scalar_ (GxB_CONTEXT_WORLD, s_fp64, GxB_CHUNK)) ;
    OK (GrB_Scalar_extractElement_FP64 (&chunk, s_fp64)) ;
    printf ("chunk: %g\n", chunk) ;

    OK (GrB_Scalar_setElement_FP64 (s_fp64, 2048)) ;
    OK (GxB_Context_set_Scalar_ (GxB_CONTEXT_WORLD, s_fp64, GxB_CHUNK)) ;
    OK (GxB_Context_get_Scalar_ (GxB_CONTEXT_WORLD, s_fp32, GxB_CHUNK)) ;
    OK (GrB_Scalar_extractElement_FP32 (&fvalue, s_fp32)) ;
    CHECK (fvalue == 2048) ;
    printf ("new chunk: %g\n", fvalue) ;

    OK (GxB_Context_get_SIZE_ (GxB_CONTEXT_WORLD, &size, GrB_NAME)) ;
    CHECK (size == GxB_MAX_NAME_LEN) ;
    OK (GxB_Context_get_String_ (GxB_CONTEXT_WORLD, name, GrB_NAME)) ;
    printf ("name of world [%s]\n", name) ;
    CHECK (MATCH (name, "GxB_CONTEXT_WORLD")) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Context_set_String_ (GxB_CONTEXT_WORLD, "newname", GrB_NAME)) ;
    OK (GxB_Context_get_String_ (GxB_CONTEXT_WORLD, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GxB_CONTEXT_WORLD")) ;

    ERR (GxB_Context_get_SIZE_ (GxB_CONTEXT_WORLD, &size, GxB_FORMAT)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Context_get_VOID_ (GxB_CONTEXT_WORLD, nothing, 0)) ;
    ERR (GxB_Context_set_VOID_ (GxB_CONTEXT_WORLD, nothing, 0, 0)) ;
    ERR (GxB_Context_get_String_ (GxB_CONTEXT_WORLD, name, 999)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Context_get_INT_ (GxB_CONTEXT_WORLD, &i, GrB_NAME)) ;
    ERR (GxB_Context_get_Scalar_ (GxB_CONTEXT_WORLD, s_fp32, GrB_NAME)) ;
    ERR (GxB_Context_set_INT_ (GxB_CONTEXT_WORLD, 7, GrB_NAME)) ;
    ERR (GxB_Context_set_Scalar_ (GxB_CONTEXT_WORLD, s_fp64, GrB_NAME)) ;

    expected = GrB_EMPTY_OBJECT ;
    OK (GrB_Scalar_clear (s_int32)) ;
    ERR (GxB_Context_set_Scalar_ (GxB_CONTEXT_WORLD, s_int32, GxB_NTHREADS)) ;

    OK (GxB_Context_new (&context)) ;
    OK (GxB_Context_get_String_ (context, name, GrB_NAME)) ;
    CHECK (MATCH (name, "")) ;
    OK (GxB_Context_set_String_ (context, "another_name", GrB_NAME)) ;
    OK (GxB_Context_get_String_ (context, name, GrB_NAME)) ;
    CHECK (MATCH (name, "another_name")) ;
    OK (GxB_Context_get_SIZE_ (context, &size, GrB_NAME)) ;
    CHECK (size == strlen (name) + 1) ;

    OK (GxB_Context_disengage (NULL)) ;
    GrB_free (&context) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&A) ;
    GrB_free (&v) ;
    GrB_free (&s) ;
    GrB_free (&s_fp64) ;
    GrB_free (&s_fp32) ;
    GrB_free (&s_int32) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test33:  all tests passed\n\n") ;
}

