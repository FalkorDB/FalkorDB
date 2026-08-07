//------------------------------------------------------------------------------
// GB_mex_test46: test arenas for global malloc/free
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL ;

//------------------------------------------------------------------------------
// GB_mex_test46 mexFunction
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
    // test global malloc/free
    //--------------------------------------------------------------------------

    uint8_t *p = NULL ;
    p = GB_Global_malloc_function (32, 5) ;     // arena not initialized
    CHECK (p == NULL) ;

    p = GB_Global_malloc_function (32, GrB_DEFAULT) ;   // malloc/free arena
    CHECK (p != NULL) ;
    GB_Global_free_function (p, GrB_DEFAULT) ;
    p = NULL ;

    p = GB_Global_malloc_function (32, GB_ARENA_TEST) ; // mxMalloc/mxFree arena
    CHECK (p != NULL) ;
    GB_Global_free_function (p, GB_ARENA_TEST) ;
    p = NULL ;

    GB_calloc_function_t calloc_func = GB_Global_calloc_function_get (5) ;
    CHECK (calloc_func == NULL) ;               // arena not initialized

    calloc_func = GB_Global_calloc_function_get (99) ;
    CHECK (calloc_func == NULL) ;               // arena out of range

    calloc_func = GB_Global_calloc_function_get (GrB_DEFAULT) ;
    CHECK (calloc_func == calloc) ;             // malloc/free arena

    calloc_func = GB_Global_calloc_function_get (GB_ARENA_TEST) ;
    CHECK (calloc_func == mxCalloc) ;           // malloc/free arena

    bool have_realloc = GB_Global_realloc_function_have (GrB_DEFAULT) ;
    CHECK (have_realloc) ;

    have_realloc = GB_Global_realloc_function_have (GB_ARENA_TEST) ;
    CHECK (have_realloc) ;

    have_realloc = GB_Global_realloc_function_have (5) ;
    CHECK (!have_realloc) ;

    have_realloc = GB_Global_realloc_function_have (99) ;
    CHECK (!have_realloc) ;

    GB_free_function_t free_func = GB_Global_free_function_get (5) ;
    CHECK (free_func == NULL) ;                 // arena not initialized

    free_func = GB_Global_free_function_get (99) ;
    CHECK (free_func == NULL) ;                 // arena out of range

    free_func = GB_Global_free_function_get (GrB_DEFAULT) ;
    CHECK (free_func == free) ;                 // malloc/free arena

    free_func = GB_Global_free_function_get (GB_ARENA_TEST) ;
    CHECK (free_func == mxFree) ;               // malloc/free arena

    GB_Global_free_function (NULL, 99) ;        // arena out of range

    // create a new arena with just malloc/free
    int arena = 3 ;
    OK (GxB_arena_init (3, malloc, NULL, NULL, free)) ;
    int flag = false ;
    OK (GxB_arena_initialized (&flag, 3)) ;
    CHECK (flag == true) ;

    have_realloc = GB_Global_realloc_function_have (arena) ;
    CHECK (!have_realloc) ;

    uint64_t p_mem = GB_mem (arena, 0) ;
    p = GB_malloc_memory (32, sizeof (uint8_t), &p_mem) ;
    CHECK (p != NULL) ;

    for (int k = 0 ; k < 32 ; k++)
    {
        p [k] = k ;
    }

    bool ok = false ;
    uint8_t *pnew = GB_realloc_memory (64, sizeof (uint8_t), p, &p_mem, &ok) ;
    CHECK (ok) ;
    CHECK (pnew != NULL) ;

    for (int k = 0 ; k < 32 ; k++)
    {
        CHECK (pnew [k] == k) ;
    }

    GB_free_memory (&pnew, p_mem) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("GB_mex_test46:  all tests passed\n") ;
}

