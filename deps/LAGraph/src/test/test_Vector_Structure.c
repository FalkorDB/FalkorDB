//------------------------------------------------------------------------------
// LAGraph/src/test/test_Vector_Structure.c:  test LAGraph_Vector_Structure
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
#include "LG_internal.h"

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

char msg [LAGRAPH_MSG_LEN] ;
GrB_Vector w = NULL, u = NULL ;
#define LEN 512
char wtype_name [LAGRAPH_MAX_NAME_LEN] ;

//------------------------------------------------------------------------------
// setup: start a test
//------------------------------------------------------------------------------

void setup (void)
{
    OK (LAGraph_Init (msg)) ;
}

//------------------------------------------------------------------------------
// teardown: finalize a test
//------------------------------------------------------------------------------

void teardown (void)
{
    OK (LAGraph_Finalize (msg)) ;
}

//------------------------------------------------------------------------------
// test_Vector_Structure:  test LAGraph_Vector_Structure
//------------------------------------------------------------------------------

void test_Vector_Structure (void)
{
    setup ( ) ;
    printf ("\n") ;

    // create a test vector
    GrB_Index n = 10 ;
    OK (GrB_Vector_new (&u, GrB_FP64, n)) ;
    OK (GrB_Vector_setElement_FP64 (u, 3.5, 0)) ;
    OK (GrB_Vector_setElement_FP64 (u, 4.7, 3)) ;
    OK (GrB_Vector_setElement_FP64 (u, 9.8, 7)) ;
    OK (LAGraph_Vector_Print (u, LAGraph_COMPLETE_VERBOSE, stdout, msg)) ;

    // get its structure
    int result = LAGraph_Vector_Structure (&w, u, msg) ;
    TEST_CHECK (result == GrB_SUCCESS) ;
    OK (LAGraph_Vector_Print (w, LAGraph_COMPLETE_VERBOSE, stdout, msg)) ;

    // check it
    GrB_Index nvals ;
    OK (GrB_Vector_size (&n, u)) ;
    TEST_CHECK (n == 10) ;
    OK (GrB_Vector_nvals (&nvals, u)) ;
    TEST_CHECK (nvals == 3) ;
    OK (LAGraph_Vector_TypeName (wtype_name, w, msg)) ;
    TEST_CHECK (MATCHNAME (wtype_name, "bool")) ;

    bool x = false ;
    result = GrB_Vector_extractElement_BOOL (&x, w, 0) ;
    TEST_CHECK (result == GrB_SUCCESS) ;
    TEST_CHECK (x) ;

    x = false ;
    result = GrB_Vector_extractElement_BOOL (&x, w, 3) ;
    TEST_CHECK (result == GrB_SUCCESS) ;
    TEST_CHECK (x) ;

    x = false ;
    result = GrB_Vector_extractElement_BOOL (&x, w, 7) ;
    TEST_CHECK (result == GrB_SUCCESS) ;
    TEST_CHECK (x) ;

    OK (GrB_free (&w)) ;
    OK (GrB_free (&u)) ;

    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_Vector_Structure_brutal
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_Vector_Structure_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    printf ("\n") ;

    // create a test vector
    GrB_Index n = 10 ;
    OK (GrB_Vector_new (&u, GrB_FP64, n)) ;
    OK (GrB_Vector_setElement_FP64 (u, 3.5, 0)) ;
    OK (GrB_Vector_setElement_FP64 (u, 4.7, 3)) ;
    OK (GrB_Vector_setElement_FP64 (u, 9.8, 7)) ;

    // get its structure
    int result = LAGraph_Vector_Structure (&w, u, msg) ;
    TEST_CHECK (result == GrB_SUCCESS) ;
    OK (LAGraph_Vector_Print (w, LAGraph_COMPLETE_VERBOSE, stdout, msg)) ;

    // check it
    GrB_Index nvals ;
    OK (GrB_Vector_size (&n, u)) ;
    TEST_CHECK (n == 10) ;
    OK (GrB_Vector_nvals (&nvals, u)) ;
    TEST_CHECK (nvals == 3) ;
    OK (LAGraph_Vector_TypeName (wtype_name, w, msg)) ;
    TEST_CHECK (MATCHNAME (wtype_name, "bool")) ;

    bool x = false ;
    result = GrB_Vector_extractElement_BOOL (&x, w, 0) ;
    TEST_CHECK (result == GrB_SUCCESS) ;
    TEST_CHECK (x) ;

    x = false ;
    result = GrB_Vector_extractElement_BOOL (&x, w, 3) ;
    TEST_CHECK (result == GrB_SUCCESS) ;
    TEST_CHECK (x) ;

    x = false ;
    result = GrB_Vector_extractElement_BOOL (&x, w, 7) ;
    TEST_CHECK (result == GrB_SUCCESS) ;
    TEST_CHECK (x) ;


    OK (GrB_free (&w)) ;
    OK (GrB_free (&u)) ;


    OK (LG_brutal_teardown (msg)) ;
}
#endif

//------------------------------------------------------------------------------
// test_Vector_Structure_failures: test LAGraph_Vector_Structure error handling
//------------------------------------------------------------------------------

void test_Vector_Structure_failures (void)
{
    setup ( ) ;

    w = NULL ;
    int result = LAGraph_Vector_Structure (NULL, NULL, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("\nmsg: [%s]\n", msg) ;
    result = LAGraph_Vector_Structure (&w, NULL, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("msg: [%s]\n", msg) ;
    TEST_CHECK (w == NULL) ;

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "Vector_Structure", test_Vector_Structure },
    { "Vector_Structure_failures", test_Vector_Structure_failures },
    #if LAGRAPH_SUITESPARSE
    { "Vector_Structure_brutal", test_Vector_Structure_brutal },
    #endif
    { NULL, NULL }
} ;
