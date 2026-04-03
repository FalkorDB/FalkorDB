//------------------------------------------------------------------------------
// GB_mex_test42: test vector unload
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                        \
{                                       \
    GrB_Vector_free (&V) ;              \
    if (X != NULL) mxFree (X) ;         \
    X = NULL ;                          \
}

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
    GrB_Vector V = NULL ;
    uint64_t n = 32, n2 = 0 ;
    double *X = NULL ;
    GrB_Type type = NULL ;
    int handling = 99 ;
    bool malloc_debug = GB_mx_get_global (true) ;
    uint64_t xsize = 0 ;

    //--------------------------------------------------------------------------
    // create a dense vector with pending work
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&V, GrB_FP64, n)) ;
    OK (GxB_Vector_fprint (V, "V empty", 5, NULL)) ;
    OK (GrB_Vector_set_INT32 (V, GxB_SPARSE + GxB_FULL, GxB_SPARSITY_CONTROL)) ;

    for (int i = 0 ; i < n ; i++)
    {
        OK (GrB_Vector_setElement_FP64 (V, (double) 2*i+1, i)) ;
        if (i == 16)
        {
            OK (GrB_wait (V, GrB_MATERIALIZE)) ;
        }
    }

    int will_wait = false ;
    OK (GrB_Vector_get_INT32 (V, &will_wait, GxB_WILL_WAIT)) ;
    CHECK (will_wait) ;

    //--------------------------------------------------------------------------
    // unload the vector and check the result
    //--------------------------------------------------------------------------

    OK (GxB_Vector_fprint (V, "V dense with pending work", 5, NULL)) ;

    OK (GxB_Vector_unload (V, (void **) &X, &type, &n2, &xsize, &handling,
        NULL)) ;

    CHECK (type == GrB_FP64) ;
    CHECK (n == n2) ;
    CHECK (handling == GrB_DEFAULT) ;
    CHECK (xsize >= n * sizeof (double)) ;
    for (int i = 0 ; i < n ; i++)
    {
        CHECK (X [i] == (double) (2*i + 1)) ;
    }

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test42:  all tests passed\n\n") ;
}

