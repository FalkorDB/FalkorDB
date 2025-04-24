//------------------------------------------------------------------------------
// LAGraph/src/test/test_Xinit.c:  test LAGr_Init and LAGraph_Global
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2023 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

#include "LAGraph_test.h"
#include "LAGraphX.h"

// functions defined in LAGr_Init.c:
LAGRAPH_PUBLIC void LG_set_LAGr_Init_has_been_called (bool setting) ;
LAGRAPH_PUBLIC bool LG_get_LAGr_Init_has_been_called (void) ;

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

char msg [LAGRAPH_MSG_LEN] ;

//------------------------------------------------------------------------------
// test_Xinit:  test LAGr_Init
//------------------------------------------------------------------------------

void test_Xinit (void)
{

    printf ("\nTesting LAGr_Init: with expected errors\n") ;

    TEST_CHECK (LAGr_Init (GrB_NONBLOCKING, NULL, NULL, NULL, NULL, msg)
        == GrB_NULL_POINTER) ;
    printf ("msg: [%s]\n", msg) ;

    TEST_CHECK (LAGr_Init (GrB_NONBLOCKING, malloc, NULL, NULL, NULL, msg)
        == GrB_NULL_POINTER) ;
    printf ("msg: [%s]\n", msg) ;

    TEST_CHECK (LAGr_Init (GrB_NONBLOCKING, NULL, NULL, NULL, free, msg)
        == GrB_NULL_POINTER) ;
    printf ("msg: [%s]\n", msg) ;

    OK (LAGr_Init (GrB_NONBLOCKING, malloc, calloc, realloc, free, msg)) ;
    printf ("msg: [%s]\n", msg) ;
    TEST_CHECK (LG_get_LAGr_Init_has_been_called ( ) == true) ;

    // LAGr_Init cannot be called twice
    int status = LAGr_Init (GrB_NONBLOCKING,
        malloc, calloc, realloc, free, msg) ;
    TEST_CHECK (status != GrB_SUCCESS) ;
    printf ("msg: [%s]\n", msg) ;

    OK (LAGraph_Finalize (msg)) ;

    // the flag is still set after LAGraph_Finalize has been called,
    // per LAGraph policy
    TEST_CHECK (LG_get_LAGr_Init_has_been_called ( ) == true) ;

    // reset and try again
    LG_set_LAGr_Init_has_been_called (false) ;
    TEST_CHECK (LG_get_LAGr_Init_has_been_called ( ) == false) ;
    OK (LAGr_Init (GrB_NONBLOCKING, malloc, calloc, realloc, free, msg)) ;
    TEST_CHECK (LG_get_LAGr_Init_has_been_called ( ) == true) ;
    OK (LAGraph_Finalize (msg)) ;
    TEST_CHECK (LG_get_LAGr_Init_has_been_called ( ) == true) ;
}

//------------------------------------------------------------------------------
// test_Xinit_brutal:  test LAGr_Init with brutal memory debug
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_Xinit_brutal (void)
{
    // no brutal memory failures, but test LG_brutal_malloc/calloc/realloc/free
    LG_brutal = -1 ;
    LG_nmalloc = 0 ;
    OK (LAGr_Init (GrB_NONBLOCKING,
        LG_brutal_malloc, LG_brutal_calloc, LG_brutal_realloc, LG_brutal_free,
        msg)) ;

    int32_t *p = LG_brutal_malloc (42 * sizeof (int32_t)) ;
    TEST_CHECK (p != NULL) ;
    LG_brutal_free (p) ;
    p = LG_brutal_calloc (42, sizeof (int32_t)) ;
    for (int k = 0 ; k < 42 ; k++)
    {
        TEST_CHECK (p [k] == 0) ;
    }
    p = LG_brutal_realloc (p, 99 * sizeof (int32_t)) ;
    for (int k = 0 ; k < 42 ; k++)
    {
        TEST_CHECK (p [k] == 0) ;
    }
    LG_brutal_free (p) ;
    p = LG_brutal_realloc (NULL, 4 * sizeof (int32_t)) ;
    for (int k = 0 ; k < 4 ; k++)
    {
        p [k] = k ;
    }
    LG_brutal_free (p) ;

    OK (LAGraph_Finalize (msg)) ;
    TEST_CHECK (LG_nmalloc == 0) ;

    // brutal tests: keep giving the method more malloc's until it succeeds

    for (int nbrutal = 0 ; nbrutal < 1000 ; nbrutal++)
    {
        LG_brutal = nbrutal ;
        GB_Global_GrB_init_called_set (false) ;
        GrB_Info info = GxB_init (GrB_NONBLOCKING, LG_brutal_malloc,
            LG_brutal_calloc, LG_brutal_realloc, LG_brutal_free) ;
        void *p = NULL, *pnew = NULL ;
        bool ok = false ;
        if (info == GrB_SUCCESS)
        {
            p = LG_brutal_realloc (NULL, 42) ;
            pnew = NULL ;
            ok = (p != NULL) ;
            if (ok)
            {
                pnew = LG_brutal_realloc (p, 107) ;
                ok = (pnew != NULL) ;
                LG_brutal_free (ok ? pnew : p) ;
            }
        }
        if (ok)
        {
            OK (GrB_finalize ( )) ;
            printf ("\nGxB_init, finally: %d %g\n", nbrutal,
                (double) LG_nmalloc) ;
            TEST_CHECK (LG_nmalloc == 0) ;
            break ;
        }
    }

    TEST_CHECK (LG_get_LAGr_Init_has_been_called ( ) == true) ;

    for (int nbrutal = 0 ; nbrutal < 1000 ; nbrutal++)
    {
        LG_brutal = nbrutal ;
        // reset both GraphBLAS and LAGraph
        GB_Global_GrB_init_called_set (false) ;
        LG_set_LAGr_Init_has_been_called (false) ;
        TEST_CHECK (LG_get_LAGr_Init_has_been_called ( ) == false) ;
        // try to initialize GraphBLAS and LAGraph
        int result = LAGr_Init (GrB_NONBLOCKING,
            LG_brutal_malloc, LG_brutal_calloc,
            LG_brutal_realloc, LG_brutal_free, msg) ;
        if (result == 0)
        {
            // success
            OK (LAGraph_Finalize (msg)) ;
            printf ("LAGr_Init: finally: %d %g\n", nbrutal,
                (double) LG_nmalloc) ;
            TEST_CHECK (LG_nmalloc == 0) ;
            break ;
        }
        // failure: free anything partially allocated
        OK (LAGraph_Finalize (msg)) ;
    }
}
#endif

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "Xinit", test_Xinit },
    #if LAGRAPH_SUITESPARSE
    { "Xinit_brutal", test_Xinit_brutal },
    #endif
    { NULL, NULL }
} ;
