//-----------------------------------------------------------------------------
// LAGraph/src/test/test_Vector_Print.c: test LAGraph_Vector_Print
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
GrB_Vector v = NULL ;
GrB_Index n = 40 ;

//-----------------------------------------------------------------------------
// test_print
//-----------------------------------------------------------------------------

void test_print (void)
{
    OK (LAGraph_Init (msg)) ;
    printf ("\n") ;
    LAGraph_PrintLevel pr = LAGraph_SHORT ;

    OK (GrB_Vector_new (&v, GrB_BOOL, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 1, 0)) ;
    OK (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_INT8, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, -8, 0)) ;
    OK (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_INT16, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, -16, 0)) ;
    OK (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_INT32, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, -32, 0)) ;
    OK (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_INT64, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, -64, 0)) ;
    OK (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_UINT8, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 8, 0)) ;
    OK (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_UINT16, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 16, 0)) ;
    OK (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_UINT32, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 32, 0)) ;
    OK (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_UINT64, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 64, 0)) ;
    OK (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_FP32, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 3.14159, 0)) ;
    OK (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_FP64, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 99.999, 0)) ;
    OK (LAGraph_Vector_Print (v, pr, stdout, msg)) ;

    // attempt to print to a NULL file, which should fail
    int result = LAGraph_Vector_Print (v, pr, NULL, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    OK (GrB_Vector_free (&v)) ;

    // attempt to print a vector with a user-defined type, which should fail
    GrB_Type type = NULL ;
    OK (GrB_Type_new (&type, sizeof (int))) ;
    OK (GrB_Vector_new (&v, type, n)) ;
    result = LAGraph_Vector_Print (v, pr, stdout, msg) ;
    TEST_CHECK (result == GrB_NOT_IMPLEMENTED) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Type_free (&type)) ;
    OK (LAGraph_Finalize (msg)) ;
}

//-----------------------------------------------------------------------------
// test_print_brutal
//-----------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_print_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    printf ("\n") ;
    LAGraph_PrintLevel pr = LAGraph_SHORT ;

    OK (GrB_Vector_new (&v, GrB_BOOL, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 1, 0)) ;
    OK (GrB_wait (v, GrB_MATERIALIZE)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_INT8, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, -8, 0)) ;
    OK (GrB_wait (v, GrB_MATERIALIZE)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_INT16, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, -16, 0)) ;
    OK (GrB_wait (v, GrB_MATERIALIZE)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_INT32, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, -32, 0)) ;
    OK (GrB_wait (v, GrB_MATERIALIZE)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_INT64, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, -64, 0)) ;
    OK (GrB_wait (v, GrB_MATERIALIZE)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_UINT8, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 8, 0)) ;
    OK (GrB_wait (v, GrB_MATERIALIZE)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_UINT16, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 16, 0)) ;
    OK (GrB_wait (v, GrB_MATERIALIZE)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_UINT32, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 32, 0)) ;
    OK (GrB_wait (v, GrB_MATERIALIZE)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_UINT64, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 64, 0)) ;
    OK (GrB_wait (v, GrB_MATERIALIZE)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_FP32, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 3.14159, 0)) ;
    OK (GrB_wait (v, GrB_MATERIALIZE)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (GrB_Vector_new (&v, GrB_FP64, n)) ;
    OK (GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Vector_setElement (v, 99.999, 0)) ;
    OK (GrB_wait (v, GrB_MATERIALIZE)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (v, pr, stdout, msg)) ;
    OK (GrB_Vector_free (&v)) ;

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST = {
    {"test_print", test_print},
    #if LAGRAPH_SUITESPARSE
    {"test_print_brutal", test_print_brutal},
    #endif
    {NULL, NULL}
};
