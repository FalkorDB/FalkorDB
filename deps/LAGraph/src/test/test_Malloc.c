//-----------------------------------------------------------------------------
// LAGraph/src/test/test_Malloc.c: test LAGraph_Malloc and related methods
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
char msg [LAGRAPH_MSG_LEN] ;

//-----------------------------------------------------------------------------
// test_malloc
//-----------------------------------------------------------------------------

void test_malloc (void)
{
    char msg [LAGRAPH_MSG_LEN] ;
    OK (LAGraph_Init (msg)) ;

    char *p ;
    OK (LAGraph_Malloc ((void **) &p, 42, sizeof (char), msg)) ;
    for (int k = 0 ; k < 42 ; k++)
    {
        p [k] = (char) k ;
    }
    OK (LAGraph_Free ((void **) &p, msg)) ;
    TEST_CHECK (p == NULL) ;

    size_t huge = 1 + SIZE_MAX/2 ;

    if (sizeof (size_t) >= sizeof (uint64_t))
    {
        LAGraph_Malloc ((void **) &p, huge, sizeof (char), msg) ;
        TEST_CHECK (p == NULL) ;    // was FAIL
        LAGraph_Calloc ((void **) &p, huge, sizeof (char), msg) ;
        TEST_CHECK (p == NULL) ;    // was FAIL
    }

    OK (LAGraph_Calloc ((void **) &p, 42, sizeof (char), msg)) ;
    for (int k = 0 ; k < 42 ; k++)
    {
        TEST_CHECK (*p == '\0') ;
    }
    OK (LAGraph_Free ((void **) &p, msg)) ;
    TEST_CHECK (p == NULL) ;

    OK (LAGraph_Free (NULL, NULL)) ;

    LAGraph_Calloc_function = NULL ;

    OK (LAGraph_Calloc ((void **) &p, 42, sizeof (char), msg)) ;
    TEST_CHECK (p != NULL) ;
    for (int k = 0 ; k < 42 ; k++)
    {
        TEST_CHECK (*p == '\0') ;
    }

    OK (LAGraph_Realloc ((void **) &p, 100, 42, sizeof (char), msg)) ;
    for (int k = 0 ; k < 42 ; k++)
    {
        TEST_CHECK (*p == '\0') ;
    }
    for (int k = 42 ; k < 100 ; k++)
    {
        p [k] = (char) k ;
    }
    OK (LAGraph_Free ((void **) &p, NULL)) ;
    TEST_CHECK (p == NULL) ;

    OK (LAGraph_Realloc ((void **) &p, 80, 0, sizeof (char), msg)) ;

    if (sizeof (size_t) >= sizeof (uint64_t))
    {
        int s = (LAGraph_Realloc ((void **) &p, huge, 80, sizeof (char), msg)) ;
        TEST_CHECK (s == GrB_OUT_OF_MEMORY) ;  // was FAIL
    }

    for (int k = 0 ; k < 80 ; k++)
    {
        p [k] = (char) k ;
    }

    OK (LAGraph_Realloc ((void **) &p, 80, 80, sizeof (char), msg)) ;
    for (int k = 0 ; k < 80 ; k++)
    {
        TEST_CHECK (p [k] == (char) k) ;    // was FAIL
    }

    LAGraph_Realloc_function = NULL ;

    OK (LAGraph_Realloc ((void **) &p, 100, 80, sizeof (char), msg)) ;
    for (int k = 0 ; k < 80 ; k++)
    {
        TEST_CHECK (p [k] == (char) k) ;    // was FAIL
    }

    OK (LAGraph_Free ((void **) &p, NULL)) ;
    TEST_CHECK (p == NULL) ;

    OK (LAGraph_Finalize (msg)) ;
}

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST = {
    {"test_malloc", test_malloc},
    // no brutal test needed
    {NULL, NULL}
};
