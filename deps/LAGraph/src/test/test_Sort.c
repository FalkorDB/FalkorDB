//-----------------------------------------------------------------------------
// LAGraph/src/test/test_Sort.c: test LG_msort* methods
//-----------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//-----------------------------------------------------------------------------

#include "LAGraph_test.h"
#include "LG_internal.h"

char msg [LAGRAPH_MSG_LEN] ;

//-----------------------------------------------------------------------------
// test_sort1
//-----------------------------------------------------------------------------

void test_sort1 (void)
{
    OK (LAGraph_Init (msg)) ;
    OK (LAGraph_SetNumThreads (1, 4, msg)) ;

    for (int trial = 0 ; trial <= 1 ; trial++)
    {
        int64_t n = (trial == 0) ? 1024 : (256 * 1024) ;

        int64_t *A0 ;
        OK (LAGraph_Malloc ((void **) &A0, n, sizeof (int64_t), msg)) ;

        uint64_t seed = 1 ;
        for (int k = 0 ; k < n ; k++)
        {
            A0 [k] = (int64_t) LG_Random15 (&seed) ;
        }

        OK (LG_msort1 (A0, n, msg)) ;

        for (int k = 1 ; k < n ; k++)
        {
            TEST_CHECK (A0 [k-1] <= A0 [k]) ;
        }

        for (int k = 0 ; k < n ; k++)
        {
            A0 [k] = (int64_t) (LG_Random15 (&seed) % 4) ;
        }

        OK (LG_msort1 (A0, n, msg)) ;

        for (int k = 1 ; k < n ; k++)
        {
            TEST_CHECK (A0 [k-1] <= A0 [k]) ;
        }

        LAGraph_Free ((void **) &A0, NULL) ;
    }

    OK (LAGraph_Finalize (msg)) ;
}

//-----------------------------------------------------------------------------
// test_sort2
//-----------------------------------------------------------------------------

void test_sort2 (void)
{
    OK (LAGraph_Init (msg)) ;
    OK (LAGraph_SetNumThreads (1, 4, msg)) ;

    int64_t n = 256 * 1024 ;

    int64_t *A0, *A1 ;
    OK (LAGraph_Malloc ((void **) &A0, n, sizeof (int64_t), msg)) ;
    OK (LAGraph_Malloc ((void **) &A1, n, sizeof (int64_t), msg)) ;

    uint64_t seed = 1 ;
    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = (int64_t) LG_Random15 (&seed) ;
        A1 [k] = (int64_t) LG_Random60 (&seed) ;
    }

    OK (LG_msort2 (A0, A1, n, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_2 (A0, A1, k-1, A0, A1, k)
            || (A0 [k-1] == A0 [k] && A1 [k-1] == A1 [k])) ;
    }

    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = 0 ;
        A1 [k] = (int64_t) (LG_Random15 (&seed) % 4) ;
    }

    OK (LG_msort2 (A0, A1, n, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_2 (A0, A1, k-1, A0, A1, k)
            || (A0 [k-1] == A0 [k] && A1 [k-1] == A1 [k])) ;
    }

    LAGraph_Free ((void **) &A0, NULL) ;
    LAGraph_Free ((void **) &A1, NULL) ;

    OK (LAGraph_Finalize (msg)) ;
}

//-----------------------------------------------------------------------------
// test_sort3
//-----------------------------------------------------------------------------

void test_sort3 (void)
{
    OK (LAGraph_Init (msg)) ;
    OK (LAGraph_SetNumThreads (1, 4, msg)) ;

    int64_t n = 256 * 1024 ;
    printf ("test sort3\n") ;

    int64_t *A0, *A1, *A2 ;
    OK (LAGraph_Malloc ((void **) &A0, n, sizeof (int64_t), msg)) ;
    OK (LAGraph_Malloc ((void **) &A1, n, sizeof (int64_t), msg)) ;
    OK (LAGraph_Malloc ((void **) &A2, n, sizeof (int64_t), msg)) ;

    uint64_t seed = 1 ;
    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = (int64_t) LG_Random15 (&seed) ;
        A1 [k] = (int64_t) LG_Random60 (&seed) ;
        A2 [k] = (int64_t) LG_Random60 (&seed) ;
    }

    OK (LG_msort3 (A0, A1, A2, n, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_3 (A0, A1, A2, k-1, A0, A1, A2, k)
            || (A0 [k-1] == A0 [k] &&
                A1 [k-1] == A1 [k] &&
                A2 [k-1] == A2 [k])) ;
    }

    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = 0 ;
        A1 [k] = (int64_t) (LG_Random15 (&seed) % 4) ;
        A2 [k] = (int64_t) (LG_Random15 (&seed) % 4) ;
    }

    OK (LG_msort3 (A0, A1, A2, n, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_3 (A0, A1, A2, k-1, A0, A1, A2, k)
            || (A0 [k-1] == A0 [k] &&
                A1 [k-1] == A1 [k] &&
                A2 [k-1] == A2 [k])) ;
    }

    LAGraph_Free ((void **) &A0, NULL) ;
    LAGraph_Free ((void **) &A1, NULL) ;
    LAGraph_Free ((void **) &A2, NULL) ;

    OK (LAGraph_Finalize (msg)) ;
}

//-----------------------------------------------------------------------------
// test_sort1_brutal
//-----------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_sort1_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    OK (LAGraph_SetNumThreads (1, 4, msg)) ;

    for (int trial = 0 ; trial <= 1 ; trial++)
    {
        int64_t n = (trial == 0) ? 1024 : (256 * 1024) ;

        int64_t *A0 ;
        OK (LAGraph_Malloc ((void **) &A0, n, sizeof (int64_t), msg)) ;

        uint64_t seed = 1 ;
        for (int k = 0 ; k < n ; k++)
        {
            A0 [k] = (int64_t) LG_Random15 (&seed) ;
        }

        LG_BRUTAL (LG_msort1 (A0, n, msg)) ;

        for (int k = 1 ; k < n ; k++)
        {
            TEST_CHECK (A0 [k-1] <= A0 [k]) ;
        }

        for (int k = 0 ; k < n ; k++)
        {
            A0 [k] = (int64_t) (LG_Random15 (&seed) % 4) ;
        }

        LG_BRUTAL (LG_msort1 (A0, n, msg)) ;

        for (int k = 1 ; k < n ; k++)
        {
            TEST_CHECK (A0 [k-1] <= A0 [k]) ;
        }

        LAGraph_Free ((void **) &A0, NULL) ;
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// test_sort2_brutal
//-----------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_sort2_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    OK (LAGraph_SetNumThreads (1, 4, msg)) ;

    int64_t n = 256 * 1024 ;

    int64_t *A0, *A1 ;
    OK (LAGraph_Malloc ((void **) &A0, n, sizeof (int64_t), msg)) ;
    OK (LAGraph_Malloc ((void **) &A1, n, sizeof (int64_t), msg)) ;

    uint64_t seed = 1 ;
    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = (int64_t) LG_Random15 (&seed) ;
        A1 [k] = (int64_t) LG_Random60 (&seed) ;
    }

    LG_BRUTAL (LG_msort2 (A0, A1, n, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_2 (A0, A1, k-1, A0, A1, k)
            || (A0 [k-1] == A0 [k] && A1 [k-1] == A1 [k])) ;
    }

    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = 0 ;
        A1 [k] = (int64_t) (LG_Random15 (&seed) % 4) ;
    }

    LG_BRUTAL (LG_msort2 (A0, A1, n, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_2 (A0, A1, k-1, A0, A1, k)
            || (A0 [k-1] == A0 [k] && A1 [k-1] == A1 [k])) ;
    }

    LAGraph_Free ((void **) &A0, NULL) ;
    LAGraph_Free ((void **) &A1, NULL) ;

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// test_sort3_brutal
//-----------------------------------------------------------------------------

void test_sort3_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    OK (LAGraph_SetNumThreads (1, 4, msg)) ;

    int64_t n = 256 * 1024 ;
    printf ("test sort3\n") ;

    int64_t *A0, *A1, *A2 ;
    OK (LAGraph_Malloc ((void **) &A0, n, sizeof (int64_t), msg)) ;
    OK (LAGraph_Malloc ((void **) &A1, n, sizeof (int64_t), msg)) ;
    OK (LAGraph_Malloc ((void **) &A2, n, sizeof (int64_t), msg)) ;

    uint64_t seed = 1 ;
    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = (int64_t) LG_Random15 (&seed) ;
        A1 [k] = (int64_t) LG_Random60 (&seed) ;
        A2 [k] = (int64_t) LG_Random60 (&seed) ;
    }

    LG_BRUTAL (LG_msort3 (A0, A1, A2, n, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_3 (A0, A1, A2, k-1, A0, A1, A2, k)
            || (A0 [k-1] == A0 [k] &&
                A1 [k-1] == A1 [k] &&
                A2 [k-1] == A2 [k])) ;
    }

    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = 0 ;
        A1 [k] = (int64_t) (LG_Random15 (&seed) % 4) ;
        A2 [k] = (int64_t) (LG_Random15 (&seed) % 4) ;
    }

    LG_BRUTAL (LG_msort3 (A0, A1, A2, n, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_3 (A0, A1, A2, k-1, A0, A1, A2, k)
            || (A0 [k-1] == A0 [k] &&
                A1 [k-1] == A1 [k] &&
                A2 [k-1] == A2 [k])) ;
    }

    LAGraph_Free ((void **) &A0, NULL) ;
    LAGraph_Free ((void **) &A1, NULL) ;
    LAGraph_Free ((void **) &A2, NULL) ;

    OK (LG_brutal_teardown (msg)) ;
}


//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST = {
    {"test_sort1", test_sort1},
    {"test_sort2", test_sort2},
    {"test_sort3", test_sort3},
    #if LAGRAPH_SUITESPARSE
    {"test_sort1_brutal", test_sort1_brutal},
    {"test_sort2_brutal", test_sort2_brutal},
    {"test_sort3_brutal", test_sort3_brutal},
    #endif
    {NULL, NULL}
};
