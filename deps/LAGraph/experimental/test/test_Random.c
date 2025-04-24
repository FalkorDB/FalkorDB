//------------------------------------------------------------------------------
// LAGraph/src/test/test_Random.c: test cases for random vector generator
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

//------------------------------------------------------------------------------
// test_Random
//------------------------------------------------------------------------------

void test_Random (void)
{
    LAGraph_Init (msg) ;
    OK (LAGraph_Random_Init (msg)) ;

    uint64_t seed = 42 ;
    LAGraph_PrintLevel pr = LAGraph_COMPLETE_VERBOSE ;

    for (int trial = 0 ; trial <= 4 ; trial++)
    {
        seed++ ;
        printf ("\n=============================== seed: %g\n", (double) seed) ;

        // generate a seed vector (all entries present)
        printf ("\nDense random vector:\n") ;
        GrB_Index n = 8 ;
        OK (GrB_Vector_new (&Seed, GrB_UINT64, n)) ;
        OK (GrB_Vector_assign_UINT64 (Seed, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
        OK (LAGraph_Random_Seed (Seed, seed, msg)) ;
        OK (LAGraph_Vector_Print (Seed, pr, stdout, NULL)) ;
        printf ("\nnext dense random vector:\n") ;
        OK (LAGraph_Random_Next (Seed, msg)) ;
        OK (LAGraph_Vector_Print (Seed, pr, stdout, NULL)) ;

        // free all workspace
        OK (GrB_Vector_free (&Seed)) ;

        // generate a sparse seed vector
        printf ("\nSparse random vector (same seed):\n") ;
        OK (GrB_Vector_new (&Seed, GrB_UINT64, n)) ;
        for (int i = 0 ; i < n ; i += 2)
        {
            OK (GrB_Vector_setElement_UINT64 (Seed, 0, i)) ;
        }
        OK (LAGraph_Random_Seed (Seed, seed, msg)) ;
        OK (LAGraph_Vector_Print (Seed, pr, stdout, NULL)) ;
        printf ("\nnext sparse random vector:\n") ;
        OK (LAGraph_Random_Next (Seed, msg)) ;
        OK (LAGraph_Vector_Print (Seed, pr, stdout, NULL)) ;

        // free all workspace
        OK (GrB_Vector_free (&Seed)) ;
    }

    OK (LAGraph_Random_Finalize (msg)) ;
    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// Test list
//------------------------------------------------------------------------------

TEST_LIST = {
    {"Random", test_Random},
    {NULL, NULL}
};
