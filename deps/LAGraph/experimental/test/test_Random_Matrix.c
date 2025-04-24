//------------------------------------------------------------------------------
// LAGraph/src/test/test_Random_Matrix.c: test cases for random matrix generator
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

#include <stdio.h>
#include <acutest.h>
#include <LAGraphX.h>
#include <LAGraph_test.h>

char msg [LAGRAPH_MSG_LEN] ;
GrB_Vector Seed = NULL ;
GrB_Matrix A = NULL ;
GrB_Type MyInt = NULL ;
typedef int myint ;

//------------------------------------------------------------------------------
// test_Random_Matrix
//------------------------------------------------------------------------------

void test_Random_Matrix (void)
{
    LAGraph_Init (msg) ;
    OK (LAGraph_Random_Init (msg)) ;

    uint64_t seed = 42 ;
    LAGraph_PrintLevel pr = LAGraph_COMPLETE_VERBOSE ;

    for (int trial = 0 ; trial <= 4 ; trial++)
    {
        seed++ ;
        printf ("\n=============================== seed: %g\n", (double) seed) ;

        double d = (trial == 4) ? INFINITY : ((double) trial / 4) ;
        printf ("density: %g, expected values: %g\n", d, d*20) ;

        printf ("\n----------------bool:\n") ;
        OK (LAGraph_Random_Matrix (&A, GrB_BOOL, 4, 5, d, seed, msg)) ;
        OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
        OK (GrB_free (&A)) ;

        printf ("\n----------------int8:\n") ;
        OK (LAGraph_Random_Matrix (&A, GrB_INT8, 4, 5, d, seed, msg)) ;
        OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
        OK (GrB_free (&A)) ;

        printf ("\n----------------int16:\n") ;
        OK (LAGraph_Random_Matrix (&A, GrB_INT16, 4, 5, d, seed, msg)) ;
        OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
        OK (GrB_free (&A)) ;

        printf ("\n----------------int32:\n") ;
        OK (LAGraph_Random_Matrix (&A, GrB_INT32, 4, 5, d, seed, msg)) ;
        OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
        OK (GrB_free (&A)) ;

        printf ("\n----------------int64:\n") ;
        OK (LAGraph_Random_Matrix (&A, GrB_INT64, 4, 5, d, seed, msg)) ;
        OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
        OK (GrB_free (&A)) ;

        printf ("\n----------------uint8:\n") ;
        OK (LAGraph_Random_Matrix (&A, GrB_UINT8, 4, 5, d, seed, msg)) ;
        OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
        OK (GrB_free (&A)) ;

        printf ("\n----------------uint16:\n") ;
        OK (LAGraph_Random_Matrix (&A, GrB_UINT16, 4, 5, d, seed, msg)) ;
        OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
        OK (GrB_free (&A)) ;

        printf ("\n----------------uint32:\n") ;
        OK (LAGraph_Random_Matrix (&A, GrB_UINT32, 4, 5, d, seed, msg)) ;
        OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
        OK (GrB_free (&A)) ;

        printf ("\n----------------uint64:\n") ;
        OK (LAGraph_Random_Matrix (&A, GrB_UINT64, 4, 5, d, seed, msg)) ;
        OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
        OK (GrB_free (&A)) ;

        printf ("\n----------------fp32:\n") ;
        OK (LAGraph_Random_Matrix (&A, GrB_FP32, 4, 5, d, seed, msg)) ;
        OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
        OK (GrB_free (&A)) ;

        printf ("\n----------------fp64:\n") ;
        OK (LAGraph_Random_Matrix (&A, GrB_FP64, 4, 5, d, seed, msg)) ;
        OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
        OK (GrB_free (&A)) ;

    }

    printf ("\n----------------empty bool:\n") ;
    OK (LAGraph_Random_Matrix (&A, GrB_BOOL, 0, 5, 0.5, seed, msg)) ;
    OK (LAGraph_Matrix_Print (A, pr, stdout, NULL)) ;
    OK (GrB_free (&A)) ;

    printf ("\n----------------invalid type:\n") ;
    OK (GrB_Type_new (&MyInt, sizeof (myint))) ;
    int result = LAGraph_Random_Matrix (&A, MyInt, 0, 5, 0.5, seed, msg) ;
    printf ("result %d msg [%s]\n", result, msg) ;
    TEST_CHECK (result == GrB_NOT_IMPLEMENTED) ;
    OK (GrB_free (&MyInt)) ;

    OK (LAGraph_Random_Finalize (msg)) ;
    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// Test list
//------------------------------------------------------------------------------

TEST_LIST = {
    {"Random_Matrix", test_Random_Matrix},
    {NULL, NULL}
};
