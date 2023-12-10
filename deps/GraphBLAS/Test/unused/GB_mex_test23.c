//------------------------------------------------------------------------------
// GB_mex_test23: JIT error handling
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "GB_stringify.h"

#define USAGE "GB_mex_test23"

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

    GrB_Info info ;
    GB_mx_at_exit ( ) ;
    bool malloc_debug = GB_mx_get_global (true) ;

    //--------------------------------------------------------------------------
    // invalid cache path
    //--------------------------------------------------------------------------

    OK (GxB_set (GxB_BURBLE, true)) ;
    int save, c ;
    OK (GxB_get (GxB_JIT_C_CONTROL, &save)) ;
    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    OK (GxB_get (GxB_JIT_C_CONTROL, &c)) ;

    if (c == GxB_JIT_ON)
    {
        // JIT is enabled
        GrB_Info expected = GrB_INVALID_VALUE ;
        ERR (GxB_Global_Option_set_CHAR (GxB_JIT_CACHE_PATH, "/myroot")) ;
    }
    else
    {
        // JIT is disabled
        printf ("JIT disabled\n") ;
        OK (GxB_Global_Option_set_CHAR (GxB_JIT_CACHE_PATH, "/myroot")) ;
        const char *s ;
        OK (GxB_Global_Option_get_CHAR (GxB_JIT_CACHE_PATH, &s)) ;
        CHECK (MATCH (s, "/myroot")) ;
    }

    OK (GxB_set (GxB_JIT_C_CONTROL, save)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    GB_mx_at_exit ( ) ;
    printf ("\nGB_mex_test23:  all tests passed\n\n") ;
}

