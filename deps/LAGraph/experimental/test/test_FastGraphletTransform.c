//----------------------------------------------------------------------------
// LAGraph/src/test/test_TriangleCount.cpp: test cases for triangle

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

//-----------------------------------------------------------------------------

#include <stdio.h>
#include <acutest.h>
#include <LAGraphX.h>
#include <LAGraph_test.h>
#include <LG_Xtest.h>
#include <LG_test.h>

#include <graph_zachary_karate.h>

char msg[LAGRAPH_MSG_LEN];
LAGraph_Graph G = NULL;

#define LEN 512
char filename [LEN+1] ;

int64_t karate_graphlet_counts [ ] =
{
    1,16,17,102,18,81,197,13,352,6,34,171,10,2,30,7,
    1,9,19,24,12,73,56,33,32,8,80,27,6,2,18,7,
    1,10,34,34,11,72,179,84,54,17,75,51,20,6,8,7,
    1,6,20,5,10,49,11,56,1,5,81,5,0,4,7,7,
    1,3,16,1,2,17,1,64,0,15,25,0,1,2,1,0,
    1,4,15,3,3,15,14,64,0,14,25,2,1,2,2,0,
    1,4,15,3,3,15,14,64,0,14,25,2,1,2,2,0,
    1,4,25,0,6,37,0,87,0,5,53,0,0,11,0,4,
    1,5,44,5,5,31,74,208,0,23,58,6,10,12,3,1,
    1,2,25,1,0,40,15,130,0,26,0,0,5,0,0,0,
    1,3,16,1,2,17,1,64,0,15,25,0,1,2,1,0,
    1,1,15,0,0,17,0,87,0,18,0,0,0,0,0,0,
    1,2,18,0,1,27,0,79,0,18,10,0,0,4,0,0,
    1,5,41,4,6,27,69,192,0,20,53,6,10,11,0,4,
    1,2,25,0,1,34,0,142,0,8,7,0,0,9,0,0,
    1,2,25,0,1,34,0,142,0,8,7,0,0,9,0,0,
    1,2,4,0,1,28,0,0,0,2,2,0,0,1,0,0,
    1,2,21,0,1,27,0,96,0,16,9,0,0,6,0,0,
    1,2,25,0,1,34,0,142,0,8,7,0,0,9,0,0,
    1,3,37,2,1,31,43,201,0,31,9,1,5,6,0,0,
    1,2,25,0,1,34,0,142,0,8,7,0,0,9,0,0,
    1,2,21,0,1,27,0,96,0,16,9,0,0,6,0,0,
    1,2,25,0,1,34,0,142,0,8,7,0,0,9,0,0,
    1,5,27,6,4,36,39,111,2,5,43,5,4,9,2,1,
    1,3,8,2,1,67,7,6,0,3,5,1,2,0,0,0,
    1,3,9,2,1,62,7,6,0,6,5,1,3,0,0,0,
    1,2,17,0,1,29,0,93,0,13,13,0,0,2,0,0,
    1,4,29,5,1,47,53,120,2,25,14,2,8,2,0,0,
    1,3,28,2,1,59,23,122,0,25,17,1,7,1,0,0,
    1,4,24,2,4,34,11,111,0,4,43,1,0,9,2,1,
    1,4,33,3,3,56,39,138,0,18,32,3,5,9,0,1,
    1,6,42,12,3,34,149,216,9,25,23,10,10,9,1,0,
    1,12,23,53,13,56,150,40,139,10,75,34,10,4,45,2,
    1,17,18,121,15,81,210,3,507,9,26,123,25,1,48,2
};

int64_t A_graphlet_counts [ ] =
{
    1,3,8,1,2,1,0,2,0,5,2,0,3,4,1,0,
    1,5,5,5,5,0,0,0,1,0,0,4,5,5,4,1,
    1,5,3,3,7,0,1,0,0,0,5,1,1,2,7,2,
    1,5,3,3,7,0,1,0,0,0,5,1,1,2,7,2,
    1,3,8,1,2,1,0,2,0,5,2,0,3,4,1,0,
    1,4,6,1,5,0,0,2,0,0,6,0,2,4,2,2,
    1,5,5,5,5,0,0,0,1,0,0,4,5,5,4,1
};

void test_FastGraphletTransform(void)
{
    LAGraph_Init (msg) ;
    #if LAGRAPH_SUITESPARSE

    GrB_Matrix A = NULL, F_net = NULL ;
    GrB_Index n ;
    bool ok = 1 ;

    //--------------------------------------------------------------------------
    // karate
    //--------------------------------------------------------------------------

    {
        // create the karate graph
        snprintf (filename, LEN, LG_DATA_DIR "%s", "karate.mtx") ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        OK (LAGraph_DeleteSelfEdges (G, msg)) ;

        // get the net frequency matrix
        OK (LAGraph_FastGraphletTransform (&F_net, G, true, msg)) ;
        printf ("# Matrix: %s\n", "karate.mtx") ;

        OK (GrB_Matrix_nrows (&n, G->A)) ;

        // check that each element matches fglt result
        for (int i = 0 ; i < n ; i++) {
            for (int j = 0 ; j < 16 ; j++) {
                int64_t x;
                if (GrB_Matrix_extractElement (&x, F_net, j, i) == GrB_NO_VALUE)
                    x = 0 ;
                ok &= (x == karate_graphlet_counts [16 * i + j]) ;
            }
        }

        TEST_CHECK (ok) ;

        OK (GrB_free (&F_net)) ;
        OK (LAGraph_Delete (&G, msg)) ;
    }

    //--------------------------------------------------------------------------
    // A
    //--------------------------------------------------------------------------

    {
        // create the A graph
        snprintf (filename, LEN, LG_DATA_DIR "%s", "A.mtx") ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        OK (LAGraph_DeleteSelfEdges (G, msg)) ;

        // get the net frequency matrix
        OK (LAGraph_FastGraphletTransform (&F_net, G, true, msg)) ;
        printf ("# Matrix: %s\n", "A.mtx") ;

        OK (GrB_Matrix_nrows (&n, G->A)) ;

        // check that each element matches fglt result
        for (int i = 0 ; i < n ; i++) {
            for (int j = 0 ; j < 16 ; j++) {
                int64_t x = 0 ;
                OK (GrB_Matrix_extractElement (&x, F_net, j, i)) ;
                ok &= (x == A_graphlet_counts [16 * i + j]) ;
            }
        }

        TEST_CHECK (ok) ;

        OK (GrB_free (&F_net)) ;
        OK (LAGraph_Delete (&G, msg)) ;
    }

    //--------------------------------------------------------------------------

    #endif
    LAGraph_Finalize (msg) ;
}


//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"FastGraphletTransform", test_FastGraphletTransform},
    {NULL, NULL}
};
