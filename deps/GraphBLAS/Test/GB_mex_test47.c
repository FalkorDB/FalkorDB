//------------------------------------------------------------------------------
// GB_mex_test47: test vector load in different arena
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL GrB_Vector_free (&V) ;

//------------------------------------------------------------------------------
// GB_mex_test47 mexFunction
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

    //--------------------------------------------------------------------------
    // test GxB_Vector_new_arena and GxB_Vector_load
    //--------------------------------------------------------------------------

    int n = 32 ;
    uint64_t siz = n * sizeof (uint8_t) ;
    uint8_t *p = mxMalloc (siz) ;
    for (int k = 0 ; k < n ; k++)
    {
        p [k] = k ;
    }

    GrB_Vector V = NULL ;
    OK (GxB_Vector_new_arena (&V, GrB_UINT8, n, GB_ARENA_TEST, GB_ARENA_TEST)) ;

    OK (GxB_Vector_load (V, &p, GrB_UINT8, n, siz, GxB_IS_READONLY + 
        GB_ARENA_TEST, NULL)) ;

    OK (GxB_Vector_fprint (V, "V with readonly components", 5, NULL)) ;

    int expected = GrB_INVALID_VALUE ;
    ERR (GxB_Vector_load (V, &p, GrB_UINT8, n, siz, GxB_IS_READONLY + 
        99, NULL)) ;    // arena out of range
    ERR (GxB_Vector_load (V, &p, GrB_UINT8, n, siz, GxB_IS_READONLY + 
        5, NULL)) ;     // arena not initialized

    OK (GrB_Vector_free (&V)) ;
    mxFree (p) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("GB_mex_test47:  all tests passed\n") ;
}

