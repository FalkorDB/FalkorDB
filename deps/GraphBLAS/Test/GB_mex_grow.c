//------------------------------------------------------------------------------
// GB_mex_grow: MATLAB version of GraphBLAS/Demo/Program/grow_demo.c
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_grow (A)"

#define WALLCLOCK GB_omp_get_wtime ( )

// macro used by OK(...) to free workspace if an error occurs
#undef  FREE_ALL
#define FREE_ALL                            \
    GrB_Matrix_free (&A) ;                  \
    GrB_Matrix_free (&C) ;                  \
    GrB_Matrix_free (&T) ;                  \
    GrB_Matrix_free (&W) ;                  \
    GrB_Vector_free (&w) ;                  \
    GB_mx_put_global (true) ;               \

GrB_Info check_result (GrB_Matrix A1, GrB_Matrix C1, GrB_BinaryOp eq) ;
GrB_Info check_result (GrB_Matrix A1, GrB_Matrix C1, GrB_BinaryOp eq)
{
    double t = WALLCLOCK ;
    GrB_Info info ;
    GrB_Matrix A = NULL, C = NULL, T = NULL, W = NULL ;
    GrB_Vector w = NULL ;
    uint64_t anvals, cnvals, tnvals, anrows, ancols ;
    OK (GrB_Matrix_nrows (&anrows, A1)) ;
    OK (GrB_Matrix_ncols (&ancols, A1)) ;
    OK (GrB_Matrix_nvals (&anvals, A1)) ;
    OK (GrB_Matrix_nvals (&cnvals, C1)) ;
    CHECK (anvals == cnvals) ;
    OK (GrB_Matrix_new (&T, GrB_BOOL, anrows, ancols)) ;
    OK (GrB_set (T, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_eWiseMult (T, NULL, NULL, eq, A1, C1, NULL)) ;
    // OK (GxB_print (T, 2)) ;
    OK (GrB_Matrix_nvals (&tnvals, T)) ;
    CHECK (anvals == tnvals) ;
    bool ok = true ;
    OK (GrB_reduce (&ok, NULL, GrB_LAND_MONOID_BOOL, T, NULL)) ;
    CHECK (ok) ;
    t = (WALLCLOCK - t) ;
    GrB_Matrix_free (&T) ;
    printf ("A and C match, time %g\n", t) ;
    return (GrB_SUCCESS) ;
}


void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL, C = NULL, T = NULL, W = NULL ;
    GrB_Vector w = NULL ;
    GrB_Info info ;

    // check inputs
    if (nargout > 0 || nargin != 1)
    {
        FREE_ALL ;
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A (deep copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A", true, true) ;
    if (A == NULL)
    {
        mexErrMsgTxt ("A failed") ;
    }

    int nthreads ;
    OK (GrB_get (GrB_GLOBAL, &nthreads, GxB_NTHREADS)) ;
    fprintf (stderr, "grow demo: nthreads %d\n", nthreads) ;

    //--------------------------------------------------------------------------
    // get A matrix
    //--------------------------------------------------------------------------

    uint64_t anrows, ancols ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;

    int type_code ;
    OK (GrB_Matrix_get_INT32 (A, &type_code, GrB_EL_TYPE_CODE)) ;
    GrB_Type atype = NULL ;
    // OK (GxB_print (A, 5)) ;
    // printf ("type_code: %d\n", type_code) ;
    GrB_BinaryOp eq = NULL ;

    switch (type_code)
    {
        case GrB_BOOL_CODE   : atype = GrB_BOOL   ; eq = GrB_EQ_BOOL   ; break ;
        case GrB_INT8_CODE   : atype = GrB_INT8   ; eq = GrB_EQ_INT8   ; break ;
        case GrB_UINT8_CODE  : atype = GrB_UINT8  ; eq = GrB_EQ_UINT8  ; break ;
        case GrB_INT16_CODE  : atype = GrB_INT16  ; eq = GrB_EQ_INT16  ; break ;
        case GrB_UINT16_CODE : atype = GrB_UINT16 ; eq = GrB_EQ_UINT16 ; break ;
        case GrB_INT32_CODE  : atype = GrB_INT32  ; eq = GrB_EQ_INT32  ; break ;
        case GrB_UINT32_CODE : atype = GrB_UINT32 ; eq = GrB_EQ_UINT32 ; break ;
        case GrB_INT64_CODE  : atype = GrB_INT64  ; eq = GrB_EQ_INT64  ; break ;
        case GrB_UINT64_CODE : atype = GrB_UINT64 ; eq = GrB_EQ_UINT64 ; break ;
        case GrB_FP32_CODE   : atype = GrB_FP32   ; eq = GrB_EQ_FP32   ; break ;
        case GrB_FP64_CODE   : atype = GrB_FP64   ; eq = GrB_EQ_FP64   ; break ;
        case GxB_FC32_CODE   : atype = GxB_FC32   ; eq = GxB_EQ_FC32   ; break ;
        case GxB_FC64_CODE   : atype = GxB_FC64   ; eq = GxB_EQ_FC64   ; break ;
        default              : ;
    }

    // OK (GxB_print (atype, 5)) ;
    CHECK (atype != NULL) ;
    OK (GxB_print (A, 1)) ;

    //--------------------------------------------------------------------------
    // C = A, one row at a time
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, atype, anrows, ancols)) ;
    OK (GrB_Vector_new (&w, atype, ancols)) ;
    // OK (GrB_set (GrB_GLOBAL, true, GxB_BURBLE)) ;
    int32_t hyper_hash ;
    OK (GrB_get (C, &hyper_hash, GxB_HYPER_HASH)) ;
    CHECK (hyper_hash) ;
    OK (GrB_set (C, false, GxB_HYPER_HASH)) ;
    OK (GrB_get (C, &hyper_hash, GxB_HYPER_HASH)) ;
    CHECK (!hyper_hash) ;
    OK (GrB_set (C, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_set (w, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    // printf ("\n\nC empty:\n") ;
    // OK (GxB_print (C, 1)) ;

    // convert A and C to CSR format
    OK (GrB_set (A, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_set (C, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;

    double t, tt [4] = {0, 0, 0, 0}, t2 [4] = {0, 0, 0, 0} ;
    tt [0] = WALLCLOCK ;

    for (int64_t i = 0 ; i < anrows ; i++)
    {
        if (i % 100000 == 0)
        {
            printf (".") ;
            fflush (stdout) ;
        }

        // w = A (i,:), using A' via the descriptor
        t = WALLCLOCK ;
        OK (GrB_Col_extract (w, NULL, NULL, A, GrB_ALL, ancols, i,
            GrB_DESC_T0)) ;
        tt [1] += (WALLCLOCK - t) ;
        // OK (GxB_print (w, 3)) ;

        // C (i,:) = w
        t = WALLCLOCK ;
        OK (GrB_Row_assign_(C, NULL, NULL, w, i, GrB_ALL, ancols, NULL)) ;
        tt [2] += (WALLCLOCK - t) ;

        // ensure C is finished
        t = WALLCLOCK ;
        OK (GrB_wait (C, GrB_MATERIALIZE)) ;
        tt [3] += (WALLCLOCK - t) ;
        // OK (GxB_print (C, 1)) ;
    }

    OK (GrB_set (GrB_GLOBAL, false, GxB_BURBLE)) ;
    // OK (GxB_print (C, 1)) ;

    tt [0] = WALLCLOCK - tt [0] ;
    printf ("one row at a time:\n") ;
    printf ("total time: %g\n", tt [0]) ;
    printf ("extract:    %g\n", tt [1]) ;
    printf ("assign:     %g\n", tt [2]) ;
    printf ("wait:       %g\n", tt [3]) ;

    // check to see if A and C are equal
    OK (check_result (A, C, eq)) ;
    GrB_Matrix_free (&C) ;
    GrB_Vector_free (&w) ;

    //--------------------------------------------------------------------------
    // C = A, multiple rows at a time
    //--------------------------------------------------------------------------

    // OK (GrB_set (GrB_GLOBAL, true, GxB_BURBLE)) ;
    OK (GrB_Matrix_new (&C, atype, anrows, ancols)) ;
    OK (GrB_set (C, false, GxB_HYPER_HASH)) ;
    OK (GrB_set (C, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_set (C, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;

    t2 [0] = WALLCLOCK ;
    int64_t i1 = 0 ;
    int64_t ilast = anrows - 1 ;
    while (i1 <= ilast)
    {
        // determine the rows i1:i2 of A to append
        int64_t irows = (ilast - i1) / 2 ;
        if (irows == 0) irows = 1 ;
        int64_t i2 = i1 + irows - 1 ;
        if (i2 > ilast) i2 = ilast ;
        irows = i2 - i1 + 1 ;
        // printf ("i1: %ld i2: %ld irows %ld ilast: %ld\n",
        //     i1, i2, irows, ilast) ;

        // W = A (i1:i2,:)
        t = WALLCLOCK ;
        OK (GrB_Matrix_new (&W, atype, irows, ancols)) ;
        OK (GrB_set (W, false, GxB_HYPER_HASH)) ;
        OK (GrB_set (W, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
        OK (GrB_set (W, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
        uint64_t Icolon [3] ;       // OK
        Icolon [GxB_BEGIN] = i1 ;
        Icolon [GxB_INC  ] = 1 ;
        Icolon [GxB_END  ] = i2 ;
        // OK (GxB_print (W, 3)) ;
        info = GrB_extract (W, NULL, NULL, A, Icolon, GxB_RANGE,
            GrB_ALL, ancols, NULL) ;
        // printf ("got here: %d\n", info) ;
        OK (info) ;
        t2 [1] += (WALLCLOCK - t) ;
        // OK (GxB_print (W, 3)) ;

        // C (i1:i2,:) = W
        t = WALLCLOCK ;
        OK (GrB_Matrix_assign_(C, NULL, NULL, W, Icolon, GxB_RANGE,
            GrB_ALL, ancols, NULL)) ;
        t2 [2] += (WALLCLOCK - t) ;

        // ensure C is finished
        t = WALLCLOCK ;
        OK (GrB_wait (C, GrB_MATERIALIZE)) ;
        t2 [3] += (WALLCLOCK - t) ;
        // OK (GxB_print (C, 1)) ;

        GrB_Matrix_free (&W) ;

        i1 += irows ;
    }
    OK (GrB_set (GrB_GLOBAL, false, GxB_BURBLE)) ;

    t2 [0] = WALLCLOCK - t2 [0] ;
    printf ("\nmany rows at a time:\n") ;
    printf ("total time: %g\n", t2 [0]) ;
    printf ("extract:    %g\n", t2 [1]) ;
    printf ("assign:     %g\n", t2 [2]) ;
    printf ("wait:       %g\n", t2 [3]) ;

    // check to see if A and C are equal
    OK (check_result (A, C, eq)) ;
    GrB_Matrix_free (&C) ;

    //--------------------------------------------------------------------------
    // C = A, using dup
    //--------------------------------------------------------------------------

    printf ("\nsingle call to dup:\n") ;
    {
        GrB_free (&T) ;
        t = WALLCLOCK ;
        OK (GrB_Matrix_dup (&T, A)) ;
        t = (WALLCLOCK - t) ;
        GrB_free (&T) ;
        printf ("dup:        %g (%d threads)\n", t, nthreads) ;
    }

    //--------------------------------------------------------------------------
    // C = A
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, atype, anrows, ancols)) ;
    OK (GrB_set (C, false, GxB_HYPER_HASH)) ;
    OK (GrB_set (C, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_set (A, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_wait (A, GrB_MATERIALIZE)) ;
    OK (GrB_wait (C, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_apply_BinaryOp2nd_FP64 (C, NULL, NULL,
        GrB_PLUS_FP64, A, (double) 0, NULL)) ;
    OK (check_result (A, C, eq)) ;
    GrB_Matrix_free (&C) ;

    printf ("GB_mex_grow: all tests passed\n") ;
    FREE_ALL ;
}

