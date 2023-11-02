//------------------------------------------------------------------------------
// GB_mex_test13: more JIT tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test13"

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
    // GRAPHBLAS_CACHE_PATH
    //--------------------------------------------------------------------------

    GrB_Info info ;
    char *cache_env = getenv ("GRAPHBLAS_CACHE_PATH") ;
    bool no_cache = (cache_env == NULL) ;
    printf ("cache env [%s]\n", cache_env) ;
    if (no_cache)
    {
        setenv ("GRAPHBLAS_CACHE_PATH", "/tmp/grbcache13", 1) ;
    }

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_at_exit ( ) ;
    bool malloc_debug = GB_mx_get_global (true) ;

    char *cache ;
    OK (GxB_get (GxB_JIT_CACHE_PATH, &cache)) ;
    printf ("cache: [%s]\n", cache) ;

    GrB_Info expected = GrB_NULL_POINTER ;
    ERR (GxB_Global_Option_set_CHAR (GxB_JIT_CACHE_PATH, NULL)) ;
    ERR (GxB_Global_Option_set_CHAR (GxB_JIT_C_COMPILER_NAME, NULL)) ;
    ERR (GxB_Global_Option_set_CHAR (GxB_JIT_C_LIBRARIES, NULL)) ;
    ERR (GxB_Global_Option_set_CHAR (GxB_JIT_C_CMAKE_LIBS, NULL)) ;
    ERR (GxB_Global_Option_set_CHAR (GxB_JIT_C_COMPILER_FLAGS, NULL)) ;
    ERR (GxB_Global_Option_set_CHAR (GxB_JIT_C_LINKER_FLAGS, NULL)) ;
    ERR (GxB_Global_Option_set_CHAR (GxB_JIT_C_PREFACE, NULL)) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_ERROR_LOG, NULL)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    if (no_cache)
    {
        unsetenv ("GRAPHBLAS_CACHE_PATH") ;
    }
    system ("rm -rf /tmp/grbcache13") ;
    cache_env = getenv ("GRAPHBLAS_CACHE_PATH") ;
    CHECK (cache_env == NULL) ;

    GB_mx_put_global (true) ;
    GB_mx_at_exit ( ) ;
    printf ("\nGB_mex_test13:  all tests passed\n\n") ;
}

