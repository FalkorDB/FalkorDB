//------------------------------------------------------------------------------
// GB_mex_test38: misc tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "../Source/werk/include/GB_werk.h"

#undef  FREE_ALL
#define FREE_ALL                        \
{                                       \
    mxFree (W) ; W = NULL ;             \
    GrB_Vector_free (&V) ;              \
    GrB_Matrix_free (&A) ;              \
}

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

    GrB_Info info ;
    GrB_Matrix A = NULL, B = NULL ;
    GrB_Vector V = NULL ;
    uint32_t *W = NULL ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GB_WERK ("GB_mex_test38") ;

    //--------------------------------------------------------------------------
    // create a test matrix
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 10)) ;

    double x = 1 ;
    for (int64_t i = 0 ; i < 9 ; i++)
    {
        OK (GrB_Matrix_setElement_FP64 (A, x, i, i)) ;
        x = x*1.2 ;
        OK (GrB_Matrix_setElement_FP64 (A, x, i, i+1)) ;
        x = x*1.2 ;
        OK (GrB_Matrix_setElement_FP64 (A, x, i+1, i)) ;
        x = x*1.2 ;
    }

    OK (GrB_Matrix_setElement_FP64 (A, x, 9, 9)) ;
    x = x - 1000 ;
    OK (GrB_Matrix_setElement_FP64 (A, x, 5, 2)) ;
    OK (GxB_print (A, 5)) ;

    //--------------------------------------------------------------------------
    // test convert_int with pending tuples
    //--------------------------------------------------------------------------

    int expected = GrB_SUCCESS ;
    OK (GB_convert_int (NULL, false, false, false, true)) ;
    OK (GB_convert_int (A, false, false, false, true)) ;
    OK (GxB_print (A, 5)) ;
    OK (GrB_wait (A, GrB_MATERIALIZE)) ;
    OK (GxB_print (A, 5)) ;

    //--------------------------------------------------------------------------
    // test 32-bit cumsum with integer overflow
    //--------------------------------------------------------------------------

    uint32_t n = 1000000 ;
    W = mxMalloc (n * sizeof (uint32_t)) ;
    for (int k = 0 ; k < n ; k++)
    {
        W [k] = UINT32_MAX ;
    }

    int64_t kresult ;
    bool ok ;

    ok = GB_cumsum (W, true, n, NULL, 1, Werk) ;
    CHECK (!ok) ;
    ok = GB_cumsum (W, true, n, NULL, 4, Werk) ;
    CHECK (!ok) ;

    ok = GB_cumsum (W, true, n, &kresult, 1, Werk) ;
    CHECK (!ok) ;
    ok = GB_cumsum (W, true, n, &kresult, 4, Werk) ;
    CHECK (!ok) ;

    //--------------------------------------------------------------------------
    // test GB_index methods
    //--------------------------------------------------------------------------

    ok = GB_determine_p_is_32 (true, INT64_MAX / 4) ;
    CHECK (!ok) ;

    OK (GB_convert_int (A, true, true, true, true)) ;
    OK (GrB_set (A, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_set (A, GrB_COLMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GxB_print (A, 5)) ;
    int64_t save = A->vlen ;
    A->vlen = GrB_INDEX_MAX ;

    if (A->i_is_32)
    {
        // matrix is too large for 32-bit integers:
        expected = GrB_INVALID_OBJECT ;
        ERR (GB_valid_matrix (A)) ;
        ERR (GxB_print (A, 5)) ;
    }
    else
    {
        // matrix is OK with 64-bit integers
        OK (GB_valid_matrix (A)) ;
        OK (GxB_print (A, 5)) ;
    }

    A->vlen = save ;
    OK (GB_valid_matrix (A)) ;
    OK (GxB_print (A, 5)) ;

    // index out of range:
    uint32_t *Ai = A->i ;
    save = Ai [0] ;
    Ai [0] = 1000 ;
    expected = GrB_INVALID_OBJECT ;
    ERR (GxB_print (A, 5)) ;
    Ai [0] = save ;
    OK (GxB_print (A, 5)) ;

    //--------------------------------------------------------------------------
    // test GB_new_bix
    //--------------------------------------------------------------------------

    // matrix is too large for 32-bit integers
    expected = GrB_INVALID_VALUE ;
    ERR (GB_new_bix (&B, GrB_FP64, INT64_MAX / 4, INT64_MAX / 4, GB_ph_null,
        true, GxB_HYPERSPARSE, false, 0.1, 2, 2, true, true,
        true, true, true)) ;
    CHECK (B == NULL) ;

    //--------------------------------------------------------------------------
    // test GrB_Matrix_sort with A == P (not supported)
    //--------------------------------------------------------------------------

    expected = GrB_NOT_IMPLEMENTED ;
    GrB_Matrix_free (&A) ;
    OK (GrB_Matrix_new (&A, GrB_INT64, 10, 10)) ;
    ERR (GxB_Matrix_sort (NULL, A, GrB_LT_FP64, A, NULL)) ;

    //--------------------------------------------------------------------------
    // test GxB_*_extractTuples_Vector with I == A (not supported)
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&V, GrB_INT64, 10)) ;
    ERR (GxB_Vector_extractTuples_Vector (V, V, V, NULL)) ;
    ERR (GxB_Matrix_extractTuples_Vector (V, V, V, A, NULL)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test38:  all tests passed\n\n") ;
}

