//------------------------------------------------------------------------------
// GB_mex_test10: still more basic tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test10"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info, expected ;
    GrB_Matrix A = NULL ;
    GrB_Vector v = NULL ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    double t = GB_omp_get_wtime ( ) ;
    GrB_Descriptor desc = NULL ;
    bool malloc_debug = GB_mx_get_global (true) ;

    //--------------------------------------------------------------------------
    // get/set tests
    //--------------------------------------------------------------------------

    double chunk = 0 ;
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Desc_set_INT32 (GrB_DESC_ST0, GrB_OUTP, GrB_REPLACE)) ;
    ERR (GxB_Global_Option_set_FP64 (-1, 0)) ;
    ERR (GxB_Global_Option_set_FP64_ARRAY (-1, NULL)) ;
    ERR (GxB_Global_Option_set_INT64_ARRAY (-1, NULL)) ;
    ERR (GxB_Global_Option_set_FUNCTION (-1, NULL)) ;
    ERR (GxB_Desc_get_FP64 (NULL, -1, &chunk)) ;

    #define FREE_ALL ;
    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;
    int value = -1 ;
    METHOD (GrB_Descriptor_new (&desc)) ;
    OK (GxB_Desc_get_INT32 (desc, GxB_IMPORT, &value)) ;
    CHECK (value == GxB_DEFAULT) ;
    OK (GxB_Desc_set_INT32 (desc, GxB_IMPORT, GxB_SECURE_IMPORT)) ;
    OK (GxB_Desc_get_INT32 (desc, GxB_IMPORT, &value)) ;
    CHECK (value == GxB_SECURE_IMPORT) ;

    OK (GxB_Global_Option_set_FP64 (GxB_GLOBAL_CHUNK, 2e6)) ;
    OK (GxB_Global_Option_get_FP64 (GxB_GLOBAL_CHUNK, &chunk)) ;
    CHECK (chunk = 2e6) ;

    int32_t ver [3] ;
    const char *compiler ;
    OK (GxB_Global_Option_get_INT32 (GxB_COMPILER_VERSION, ver)) ;
    OK (GxB_Global_Option_get_CHAR (GxB_COMPILER_NAME, &compiler)) ;
    printf ("compiler: %s %d.%d.%d\n", compiler, ver [0], ver [1], ver [2]) ;

    OK (GxB_Global_Option_set_INT32 (GxB_BURBLE, 1)) ;
    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 10)) ;
    OK (GrB_transpose (A, NULL, NULL, A, desc)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_BURBLE, 0)) ;
    OK (GrB_transpose (A, NULL, NULL, A, desc)) ;

    OK (GxB_Matrix_Option_get_INT32 (A, GxB_SPARSITY_STATUS, &value)) ;
    CHECK (value == GxB_HYPERSPARSE) ;

    double bswitch1 = 0.5, bswitch2 = 1.0 ;
    OK (GxB_Matrix_Option_set (A, GxB_BITMAP_SWITCH, bswitch1)) ;
    OK (GxB_Matrix_Option_get (A, GxB_BITMAP_SWITCH, &bswitch2)) ;
    CHECK (bswitch1 == bswitch2) ;

    ERR (GxB_Matrix_Option_set_FP64 (A, -1, 0)) ;

    OK (GrB_Vector_new (&v, GrB_FP64, 10)) ;
    ERR (GxB_Vector_Option_set_FP64 (v, -1, 0)) ;
    ERR (GxB_Vector_Option_set_FP64 (v, -1, 0)) ;
    ERR (GxB_Vector_Option_get_FP64 (v, -1, &chunk)) ;
    ERR (GxB_Vector_Option_set_INT32 (v, -1, 0)) ;

    OK (GrB_Descriptor_free (&desc)) ;
    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Vector_free (&v)) ;

    void *f1 = NULL ;
    OK (GxB_Global_Option_get (GxB_MALLOC_FUNCTION, &f1)) ;
    CHECK (f1 == (void *) mxMalloc) ;
    OK (GxB_Global_Option_get (GxB_CALLOC_FUNCTION, &f1)) ;
    CHECK (f1 == (void *) mxCalloc) ;
    OK (GxB_Global_Option_get (GxB_REALLOC_FUNCTION, &f1)) ;
    CHECK (f1 == (void *) mxRealloc) ;
    OK (GxB_Global_Option_get (GxB_FREE_FUNCTION, &f1)) ;
    CHECK (f1 == (void *) mxFree) ;

    OK (GxB_Global_Option_get_FUNCTION (GxB_MALLOC_FUNCTION, &f1)) ;
    CHECK (f1 == (void *) mxMalloc) ;
    OK (GxB_Global_Option_get_FUNCTION (GxB_CALLOC_FUNCTION, &f1)) ;
    CHECK (f1 == (void *) mxCalloc) ;
    OK (GxB_Global_Option_get_FUNCTION (GxB_REALLOC_FUNCTION, &f1)) ;
    CHECK (f1 == (void *) mxRealloc) ;
    OK (GxB_Global_Option_get_FUNCTION (GxB_FREE_FUNCTION, &f1)) ;
    CHECK (f1 == (void *) mxFree) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    int nthreads_max = GB_omp_get_max_threads ( ) ;
    t = GB_omp_get_wtime ( ) - t ;
    printf ("test time %g sec, max threads %d\n", t, nthreads_max) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test10: all tests passed\n\n") ;
}

