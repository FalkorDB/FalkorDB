//------------------------------------------------------------------------------
// LAGraph/experimental/test/test_MaximalIndependentSet
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
#include <LG_Xtest.h>

//------------------------------------------------------------------------------
// test cases
//------------------------------------------------------------------------------

const char *files [ ] =
{
    "A.mtx",
    "jagmesh7.mtx",
    "bcsstk13.mtx",
    "karate.mtx",
    "ldbc-cdlp-undirected-example.mtx",
    "ldbc-cdlp-directed-example.mtx",
    "ldbc-undirected-example-bool.mtx",
    "ldbc-undirected-example-unweighted.mtx",
    "ldbc-undirected-example.mtx",
    "ldbc-wcc-example.mtx",
    "LFAT5.mtx",
    "LFAT5_two.mtx",
    "cryg2500.mtx",
    "msf2.mtx",
    "olm1000.mtx",
    "west0067.mtx",
    ""
} ;

#define LEN 512
char filename [LEN+1] ;

char msg [LAGRAPH_MSG_LEN] ;
GrB_Vector mis = NULL, ignore = NULL ;
GrB_Matrix A = NULL, C = NULL, empty_row = NULL, empty_col = NULL ;
LAGraph_Graph G = NULL ;

//------------------------------------------------------------------------------
// setup: start a test
//------------------------------------------------------------------------------

void setup (void)
{
    OK (LAGraph_Init (msg)) ;
    OK (LAGraph_Random_Init (msg)) ;
}

//------------------------------------------------------------------------------
// teardown: finalize a test
//------------------------------------------------------------------------------

void teardown (void)
{
    OK (LAGraph_Random_Finalize (msg)) ;
    OK (LAGraph_Finalize (msg)) ;
}

//------------------------------------------------------------------------------
// test_MaximalIndependentSet:  test MIS
//------------------------------------------------------------------------------

void test_MIS (void)
{
    GrB_Info info ;
    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k] ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of valued matrix failed") ;
        printf ("\nMatrix: %s\n", aname) ;

        // C = structure of A
        OK (LAGraph_Matrix_Structure (&C, A, msg)) ;
        OK (GrB_free (&A)) ;

        // construct a directed graph G with adjacency matrix C
        OK (LAGraph_New (&G, &C, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (C == NULL) ;

        // error handling test
        int result = LAGraph_MaximalIndependentSet (&mis, G, 0, NULL, msg) ;
        TEST_CHECK (result == -105) ;
        TEST_CHECK (mis == NULL) ;

        // check if the pattern is symmetric
        OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;

        if (G->is_symmetric_structure == LAGraph_FALSE)
        {
            // make the adjacency matrix symmetric
            OK (LAGraph_Cached_AT (G, msg)) ;
            OK (GrB_eWiseAdd (G->A, NULL, NULL, GrB_LOR, G->A, G->AT, NULL)) ;
            G->is_symmetric_structure = LAGraph_TRUE ;
        }
        G->kind = LAGraph_ADJACENCY_UNDIRECTED ;

        // check for self-edges
        OK (LAGraph_Cached_NSelfEdges (G, msg)) ;
        if (G->nself_edges != 0)
        {
            // remove self-edges
            printf ("graph has %g self edges\n", (double) G->nself_edges) ;
            OK (LAGraph_DeleteSelfEdges (G, msg)) ;
            printf ("now has %g self edges\n", (double) G->nself_edges) ;
            TEST_CHECK (G->nself_edges == 0) ;
        }

        // compute the row degree
        OK (LAGraph_Cached_OutDegree (G, msg)) ;

        GrB_Index n ;
        GrB_Matrix_nrows (&n, G->A) ;

        // create ignore
        OK (GrB_Vector_new (&ignore, GrB_BOOL, n)) ;
        for (int i = 0 ; i < n ; i += 8)
        {
            OK (GrB_Vector_setElement (ignore, (bool) true, i)) ;
        }

        for (int64_t seed = 0 ; seed <= 4*n ; seed += n)
        {
            // compute the MIS with no ignored nodes
            OK (LAGraph_MaximalIndependentSet (&mis, G, seed, NULL, msg)) ;
            // check the result
            OK (LG_check_mis (G->A, mis, NULL, msg)) ;
            OK (GrB_free (&mis)) ;

            // compute the MIS with ignored nodes
            OK (LAGraph_MaximalIndependentSet (&mis, G, seed, ignore, msg)) ;
            // check the result
            OK (LG_check_mis (G->A, mis, ignore, msg)) ;

            OK (GrB_free (&mis)) ;
        }

        // create some singletons
        GrB_Index nsingletons = 0 ;
        GrB_Index I [1] ;
        OK (GrB_Matrix_new (&empty_col, GrB_BOOL, n, 1)) ;
        OK (GrB_Matrix_new (&empty_row, GrB_BOOL, 1, n)) ;
        for (int64_t i = 0 ; i < n ; i += 10)
        {
            // make node i a singleton
            I [0] = i ;
            OK (GrB_assign (G->A, NULL, NULL, empty_col, GrB_ALL, n, I, 1,
                NULL)) ;
            OK (GrB_assign (G->A, NULL, NULL, empty_row, I, 1, GrB_ALL, n,
                NULL)) ;
            nsingletons++ ;
        }
        printf ("creating at least %g singletons\n", (double) nsingletons) ;

        OK (LAGraph_DeleteCached (G, msg)) ;
        G->kind = LAGraph_ADJACENCY_UNDIRECTED ;
        G->is_symmetric_structure = LAGraph_TRUE ;
        G->nself_edges = 0 ;

        // recompute the out degree
        OK (LAGraph_Cached_OutDegree (G, msg)) ;

        GrB_Index nonsingletons, nsingletons_actual ;
        OK (GrB_Vector_nvals (&nonsingletons, G->out_degree)) ;
        nsingletons_actual = n - nonsingletons ;
        printf ("actual # of singletons: %g\n", (double) nsingletons_actual) ;
        TEST_CHECK (nsingletons <= nsingletons_actual) ;

        for (int64_t seed = 0 ; seed <= 4*n ; seed += n)
        {
            // compute the MIS with no ignored nodes
            OK (LAGraph_MaximalIndependentSet (&mis, G, seed, NULL, msg)) ;
            // check the result
            OK (LG_check_mis (G->A, mis, NULL, msg)) ;
            OK (GrB_free (&mis)) ;

            // compute the MIS with ignored nodes
            OK (LAGraph_MaximalIndependentSet (&mis, G, seed, ignore, msg)) ;
            // check the result
            OK (LG_check_mis (G->A, mis, ignore, msg)) ;

            OK (GrB_free (&mis)) ;
        }

        // convert to directed with symmetric structure and recompute the MIS
        G->kind = LAGraph_ADJACENCY_DIRECTED ;
        OK (LAGraph_MaximalIndependentSet (&mis, G, 0, NULL, msg)) ;
        // check the result
        OK (LG_check_mis (G->A, mis, NULL, msg)) ;
        OK (GrB_free (&mis)) ;

        // hack the random number generator to induce an error condition
        #if defined ( COVERAGE )
        printf ("Hack the random number generator to induce a stall:\n") ;
        random_hack = true ;
        result = LAGraph_MaximalIndependentSet (&mis, G, 0, NULL, msg) ;
        random_hack = false ;
        printf ("hack msg: %d %s\n", result, msg) ;
        TEST_CHECK (result == -111 || result == 0) ;
        if (result == 0)
        {
            OK (LG_check_mis (G->A, mis, NULL, msg)) ;
        }
        #endif

        OK (GrB_free (&mis)) ;
        OK (GrB_free (&ignore)) ;
        OK (GrB_free (&empty_col)) ;
        OK (GrB_free (&empty_row)) ;
        OK (LAGraph_Delete (&G, msg)) ;
    }
    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "MIS", test_MIS },
    { NULL, NULL }
} ;
