//------------------------------------------------------------------------------
// LAGraph/src/test/test_Cached_Degree.c:  test LAGraph_Cached_*Degree
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

LAGraph_Graph G = NULL ;
char msg [LAGRAPH_MSG_LEN] ;
GrB_Matrix A = NULL ;
#define LEN 512
char filename [LEN+1] ;

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
// check_degree: check a row or column degree vector
//------------------------------------------------------------------------------

void check_degree
(
    GrB_Vector Degree,
    GrB_Index n,
    const int *degree
)
{
    GrB_Index n2 ;
    OK (GrB_Vector_size (&n2, Degree)) ;
    TEST_CHECK (n == n2) ;
    for (int k = 0 ; k < n ; k++)
    {
        int64_t degk ;
        GrB_Info info = GrB_Vector_extractElement (&degk, Degree, k) ;
        TEST_CHECK (info == GrB_NO_VALUE || info == GrB_SUCCESS) ;
        if (info == GrB_NO_VALUE)
        {
            TEST_CHECK (degree [k] == 0) ;
        }
        else
        {
            TEST_CHECK (degree [k] == degk) ;
        }
    }
}

//------------------------------------------------------------------------------
// test_Cached_Degree:  test LAGraph_Cached_*Degree
//------------------------------------------------------------------------------

typedef struct
{
    const char *name ;
    const int out_deg [67] ;
    const int in_deg [67] ;
}
matrix_info ;

const matrix_info files [ ] =
{
    { "A.mtx",
        { 3, 5, 5, 5, 3, 4, 5,  },
        { 3, 5, 5, 5, 3, 4, 5,  }, },
     { "LFAT5.mtx",
        { 3, 2, 2, 4, 4, 3, 3, 5, 5, 2, 2, 4, 4, 3,  },
        { 3, 2, 2, 4, 4, 3, 3, 5, 5, 2, 2, 4, 4, 3,  }, },
     { "cover.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "cover_structure.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "full.mtx",
        { 3, 3, 3,  },
        { 3, 3, 3,  }, },
     { "full_symmetric.mtx",
        { 4, 4, 4, 4,  },
        { 4, 4, 4, 4,  }, },
     { "karate.mtx",
        { 16, 9, 10, 6, 3, 4, 4, 4, 5, 2, 3, 1, 2, 5, 2, 2, 2, 2, 2, 3,
          2, 2, 2, 5, 3, 3, 2, 4, 3, 4, 4, 6, 12, 17,  },
        { 16, 9, 10, 6, 3, 4, 4, 4, 5, 2, 3, 1, 2, 5, 2, 2, 2, 2, 2, 3,
          2, 2, 2, 5, 3, 3, 2, 4, 3, 4, 4, 6, 12, 17,  }, },
     { "ldbc-cdlp-directed-example.mtx",
        { 3, 2, 2, 2, 3, 2, 3, 1,  },
        { 2, 2, 2, 1, 3, 4, 3, 1,  }, },
     { "ldbc-cdlp-undirected-example.mtx",
        { 3, 2, 2, 3, 4, 3, 3, 4,  },
        { 3, 2, 2, 3, 4, 3, 3, 4,  }, },
     { "ldbc-directed-example-bool.mtx",
        { 2, 3, 4, 0, 3, 2, 1, 1, 1, 0,  },
        { 2, 0, 3, 5, 3, 0, 0, 2, 0, 2,  }, },
     { "ldbc-directed-example-unweighted.mtx",
        { 2, 3, 4, 0, 3, 2, 1, 1, 1, 0,  },
        { 2, 0, 3, 5, 3, 0, 0, 2, 0, 2,  }, },
     { "ldbc-directed-example.mtx",
        { 2, 3, 4, 0, 3, 2, 1, 1, 1, 0,  },
        { 2, 0, 3, 5, 3, 0, 0, 2, 0, 2,  }, },
     { "ldbc-undirected-example-bool.mtx",
        { 2, 4, 2, 3, 5, 2, 3, 2, 1,  },
        { 2, 4, 2, 3, 5, 2, 3, 2, 1,  }, },
     { "ldbc-undirected-example-unweighted.mtx",
        { 2, 4, 2, 3, 5, 2, 3, 2, 1,  },
        { 2, 4, 2, 3, 5, 2, 3, 2, 1,  }, },
     { "ldbc-undirected-example.mtx",
        { 2, 4, 2, 3, 5, 2, 3, 2, 1,  },
        { 2, 4, 2, 3, 5, 2, 3, 2, 1,  }, },
     { "ldbc-wcc-example.mtx",
        { 3, 3, 5, 5, 5, 2, 1, 3, 1, 2,  },
        { 3, 3, 5, 5, 5, 2, 1, 3, 1, 2,  }, },
     { "matrix_bool.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "matrix_fp32.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "matrix_fp32_structure.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "matrix_fp64.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "matrix_int16.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "matrix_int32.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "matrix_int64.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "matrix_int8.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "matrix_uint16.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "matrix_uint32.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "matrix_uint64.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "matrix_uint8.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "msf1.mtx",
        { 2, 2, 1, 1, 1, 1,  },
        { 1, 1, 2, 2, 0, 2,  }, },
     { "msf2.mtx",
        { 2, 3, 3, 2, 1, 1, 0, 0,  },
        { 0, 1, 1, 1, 2, 2, 2, 3,  }, },
     { "msf3.mtx",
        { 2, 2, 2, 1, 0,  },
        { 0, 1, 1, 2, 3,  }, },
     { "structure.mtx",
        { 2, 2, 1, 2, 1, 1, 3,  },
        { 1, 1, 3, 2, 2, 2, 1,  }, },
     { "sample.mtx",
        { 3, 2, 1, 2, 2, 1, 1, 0,  },
        { 0, 1, 3, 1, 3, 1, 1, 2,  }, },
     { "sample2.mtx",
        { 2, 3, 4, 3, 5, 5, 3, 3,  },
        { 2, 3, 4, 3, 5, 5, 3, 3,  }, },
     { "skew_fp32.mtx",
        { 3, 3, 3, 4, 3, 4,  },
        { 3, 3, 3, 4, 3, 4,  }, },
     { "skew_fp64.mtx",
        { 3, 3, 3, 4, 3, 4,  },
        { 3, 3, 3, 4, 3, 4,  }, },
     { "skew_int16.mtx",
        { 3, 3, 3, 4, 3, 4,  },
        { 3, 3, 3, 4, 3, 4,  }, },
     { "skew_int32.mtx",
        { 3, 3, 3, 4, 3, 4,  },
        { 3, 3, 3, 4, 3, 4,  }, },
     { "skew_int64.mtx",
        { 3, 3, 3, 4, 3, 4,  },
        { 3, 3, 3, 4, 3, 4,  }, },
     { "skew_int8.mtx",
        { 3, 3, 3, 4, 3, 4,  },
        { 3, 3, 3, 4, 3, 4,  }, },
     { "tree-example.mtx",
        { 1, 1, 2, 3, 2, 1,  },
        { 1, 1, 2, 3, 2, 1,  }, },
     { "west0067.mtx",
        { 3, 3, 3, 3, 5, 5, 5, 5, 5, 6, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5,
          3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 3, 3, 3, 3, 5,
          5, 5, 5, 5, 6, 3, 3, 3, 3, 4, 4, 4, 4, 4, 6, 1, 5, 5, 5, 5,
          5, 5, 5, 5, 5, 5, 5,  },
        { 10, 4, 4, 4, 4, 3, 5, 3, 3, 3, 3, 2, 5, 5, 5, 5, 4, 5, 2, 10,
          3, 3, 3, 3, 3, 4, 4, 4, 4, 3, 10, 3, 3, 3, 3, 3, 10, 5, 5, 5,
          5, 4, 5, 4, 4, 4, 4, 3, 10, 3, 3, 3, 3, 3, 10, 5, 5, 5, 5, 4,
          5, 4, 4, 4, 4, 3, 5,  }, },
     { "west0067_jumbled.mtx",
        { 3, 3, 3, 3, 5, 5, 5, 5, 5, 6, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5,
          3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 3, 3, 3, 3, 5,
          5, 5, 5, 5, 6, 3, 3, 3, 3, 4, 4, 4, 4, 4, 6, 1, 5, 5, 5, 5,
          5, 5, 5, 5, 5, 5, 5,  },
        { 10, 4, 4, 4, 4, 3, 5, 3, 3, 3, 3, 2, 5, 5, 5, 5, 4, 5, 2, 10,
          3, 3, 3, 3, 3, 4, 4, 4, 4, 3, 10, 3, 3, 3, 3, 3, 10, 5, 5, 5,
          5, 4, 5, 4, 4, 4, 4, 3, 10, 3, 3, 3, 3, 3, 10, 5, 5, 5, 5, 4,
          5, 4, 4, 4, 4, 3, 5,  }, },
    { "", { 0 }, { 0 }}
} ;

//-----------------------------------------------------------------------------
// test_Cached_Degree
//-----------------------------------------------------------------------------

void test_Cached_Degree (void)
{
    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        if (strlen (aname) == 0) break;
        const int *out_deg = files [k].out_deg ;
        const int *in_deg = files [k].in_deg ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct the graph G with adjacency matrix A
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;

        for (int trial = 0 ; trial <= 2 ; trial++)
        {
            // create the G->out_degree cached property and check it
            OK (LAGraph_Cached_OutDegree (G, msg)) ;
            GrB_Index n ;
            OK (GrB_Matrix_nrows (&n, G->A)) ;
            check_degree (G->out_degree, n, out_deg) ;

            if (trial == 2)
            {
                // use G->AT to compute G->in_degree
                OK (LAGraph_DeleteCached (G, msg)) ;
                OK (LAGraph_Cached_AT (G, msg)) ;
            }

            // create the G->in_degree cached property and check it
            OK (LAGraph_Cached_InDegree (G, msg)) ;
            OK (GrB_Matrix_ncols (&n, G->A)) ;
            check_degree (G->in_degree, n, in_deg) ;
        }

        OK (LAGraph_Delete (&G, msg)) ;
    }

    // check error handling
    int status = LAGraph_Cached_OutDegree (NULL, msg) ;
    printf ("\nstatus: %d, msg: %s\n", status, msg) ;
    TEST_CHECK (status == GrB_NULL_POINTER) ;
    status = LAGraph_Cached_InDegree (NULL, msg) ;
    printf ("status: %d, msg: %s\n", status, msg) ;
    TEST_CHECK (status == GrB_NULL_POINTER) ;

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// test_Cached_Degree_brutal
//-----------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_Cached_Degree_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        if (strlen (aname) == 0) break;
        const int *out_deg = files [k].out_deg ;
        const int *in_deg = files [k].in_deg ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct the graph G with adjacency matrix A
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;

        for (int trial = 0 ; trial <= 2 ; trial++)
        {
            // create the G->out_degree cached property and check it
            LG_BRUTAL (LAGraph_Cached_OutDegree (G, msg)) ;
            GrB_Index n ;
            OK (GrB_Matrix_nrows (&n, G->A)) ;
            check_degree (G->out_degree, n, out_deg) ;

            if (trial == 2)
            {
                // use G->AT to compute G->in_degree
                OK (LAGraph_DeleteCached (G, msg)) ;
                OK (LAGraph_Cached_AT (G, msg)) ;
            }

            // create the G->in_degree cached property and check it
            LG_BRUTAL (LAGraph_Cached_InDegree (G, msg)) ;
            OK (GrB_Matrix_ncols (&n, G->A)) ;
            check_degree (G->in_degree, n, in_deg) ;
        }

        OK (LAGraph_Delete (&G, msg)) ;
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "test_Degree", test_Cached_Degree },
    #if LAGRAPH_SUITESPARSE
    { "test_Degree_brutal", test_Cached_Degree_brutal },
    #endif
    { NULL, NULL }
} ;
