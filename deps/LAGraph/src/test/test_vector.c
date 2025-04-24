//------------------------------------------------------------------------------
// LAGraph/src/test/test_vector.c:  test LG_check_vector
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
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

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

GrB_Vector X = NULL ;
int64_t *x = NULL ;
GrB_Index n = 10000 ;
int64_t missing = 42 ;
char msg [LAGRAPH_MSG_LEN] ;

//------------------------------------------------------------------------------
// test_vector
//------------------------------------------------------------------------------

void test_vector (void)
{
    OK (LAGraph_Init (msg)) ;
    OK (GrB_Vector_new (&X, GrB_INT64, n)) ;
    for (int i = 0 ; i < 10 ; i++)
    {
        OK (GrB_Vector_setElement (X, i, i)) ;
    }
    OK (LAGraph_Malloc ((void **) &x, n, sizeof (int64_t), msg)) ;
    OK (LG_check_vector (x, X, n, missing)) ;
    for (int i = 0 ; i < n ; i++)
    {
        TEST_CHECK (x [i] == ((i < 10) ? i : missing)) ;
    }
    OK (GrB_free (&X)) ;
    OK (LAGraph_Free ((void **) &x, NULL)) ;
    OK (LAGraph_Finalize (msg)) ;
}

//------------------------------------------------------------------------------
// test_vector_brutal
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_vector_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    printf ("\n") ;

    OK (LAGraph_Malloc ((void **) &x, n, sizeof (int64_t), msg)) ;

    for (int nbrutal = 0 ; ; nbrutal++)
    {
        /* allow for only nbrutal mallocs before 'failing' */
        LG_brutal = nbrutal ;
        /* try the method with brutal malloc */
        GrB_free (&X) ;
        int brutal_result = GrB_Vector_new (&X, GrB_INT64, n) ;
        if (brutal_result != GrB_SUCCESS) continue ;
        for (int i = 0 ; i < 10 ; i++)
        {
            brutal_result = GrB_Vector_setElement (X, i, i) ;
            if (brutal_result != GrB_SUCCESS) break ;
        }
        if (brutal_result != GrB_SUCCESS) continue ;
        brutal_result = LG_check_vector (x, X, n, missing) ;
        if (brutal_result >= 0)
        {
            /* the method finally succeeded */
            printf ("Finally: %d\n", nbrutal) ;
            break ;
        }
        if (nbrutal > 10000) { printf ("Infinite!\n") ; abort ( ) ; }
    }
    LG_brutal = -1 ;  /* turn off brutal mallocs */

    for (int i = 0 ; i < n ; i++)
    {
        TEST_CHECK (x [i] == ((i < 10) ? i : missing)) ;
    }
    OK (GrB_free (&X)) ;
    OK (LAGraph_Free ((void **) &x, NULL)) ;
    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "vector", test_vector },
    #if LAGRAPH_SUITESPARSE
    { "vector_brutal", test_vector_brutal },
    #endif
    { NULL, NULL }
} ;
