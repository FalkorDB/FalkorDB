//------------------------------------------------------------------------------
// LAGraph/src/test/test_Cached_Symmetric_Structure.c
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
// test_Cached_Symmetric_Structure:
//------------------------------------------------------------------------------

typedef struct
{
    bool symmetric_structure ;
    bool symmetric_values ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    1, 1, "A.mtx",
    1, 1, "LFAT5.mtx",
    1, 1, "bcsstk13.mtx",
    1, 0, "comments_full.mtx",
    0, 0, "comments_west0067.mtx",
//  1, 0, "complex.mtx",
    0, 0, "cover.mtx",
    0, 0, "cover_structure.mtx",
    0, 0, "cryg2500.mtx",
    0, 0, "empty.mtx",
    1, 0, "full.mtx",
    1, 1, "full_symmetric.mtx",
    1, 1, "jagmesh7.mtx",
    1, 1, "karate.mtx",
    0, 0, "ldbc-cdlp-directed-example.mtx",
    1, 1, "ldbc-cdlp-undirected-example.mtx",
    0, 0, "ldbc-directed-example-bool.mtx",
    0, 0, "ldbc-directed-example-unweighted.mtx",
    0, 0, "ldbc-directed-example.mtx",
    1, 1, "ldbc-undirected-example-bool.mtx",
    1, 1, "ldbc-undirected-example-unweighted.mtx",
    1, 1, "ldbc-undirected-example.mtx",
    1, 1, "ldbc-wcc-example.mtx",
    0, 0, "lp_afiro.mtx",
    0, 0, "lp_afiro_structure.mtx",
    0, 0, "matrix_bool.mtx",
    0, 0, "matrix_fp32.mtx",
    0, 0, "matrix_fp32_structure.mtx",
    0, 0, "matrix_fp64.mtx",
    0, 0, "matrix_int16.mtx",
    0, 0, "matrix_int32.mtx",
    0, 0, "matrix_int64.mtx",
    0, 0, "matrix_int8.mtx",
    0, 0, "matrix_uint16.mtx",
    0, 0, "matrix_uint32.mtx",
    0, 0, "matrix_uint64.mtx",
    0, 0, "matrix_uint8.mtx",
    0, 0, "msf1.mtx",
    0, 0, "msf2.mtx",
    0, 0, "msf3.mtx",
    0, 0, "olm1000.mtx",
    0, 0, "structure.mtx",
    0, 0, "sample.mtx",
    1, 1, "sample2.mtx",
    1, 0, "skew_fp32.mtx",
    1, 0, "skew_fp64.mtx",
    1, 0, "skew_int16.mtx",
    1, 0, "skew_int32.mtx",
    1, 0, "skew_int64.mtx",
    1, 0, "skew_int8.mtx",
    0, 0, "sources_7.mtx",
    1, 1, "tree-example.mtx",
    0, 0, "west0067.mtx",
    0, 0, "west0067_jumbled.mtx",
    0, 0, ""
} ;

void test_Cached_Symmetric_Structure (void)
{
    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        bool sym_structure = files [k].symmetric_structure ;
        bool sym_values  = files [k].symmetric_values ;
        if (strlen (aname) == 0) break;
        // printf ("%s:\n", aname) ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct a directed graph G with adjacency matrix A
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;

        // compute the is_symmetric_structure cached property
        OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;

        // check the result
        if (sym_structure)
        {
            TEST_CHECK (G->is_symmetric_structure == LAGraph_TRUE) ;
        }
        else
        {
            TEST_CHECK (G->is_symmetric_structure == LAGraph_FALSE) ;
        }

        // delete all cached properties
        OK (LAGraph_DeleteCached (G, msg)) ;

        // try again, but precompute G->AT
        OK (LAGraph_Cached_AT (G, msg)) ;
        OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;

        // check the result
        if (sym_structure)
        {
            TEST_CHECK (G->is_symmetric_structure == LAGraph_TRUE) ;
        }
        else
        {
            TEST_CHECK (G->is_symmetric_structure == LAGraph_FALSE) ;
        }

        // delete all cached properties
        OK (LAGraph_DeleteCached (G, msg)) ;

        // change the graph to directed, if matrix is symmetric
        if (sym_values)
        {
            G->kind = LAGraph_ADJACENCY_UNDIRECTED ;
            // recompute the symmetry cached property
            OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;
            TEST_CHECK (G->is_symmetric_structure == LAGraph_TRUE) ;
        }

        OK (LAGraph_Delete (&G, msg)) ;

    }

    // check error handling
    int status = LAGraph_Cached_IsSymmetricStructure (NULL, msg) ;
    printf ("\nstatus: %d, msg: %s\n", status, msg) ;
    TEST_CHECK (status == GrB_NULL_POINTER) ;

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// test_Cached_Symmetric_Structure_brutal
//-----------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_Cached_Symmetric_Structure_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        bool sym_structure = files [k].symmetric_structure ;
        bool sym_values  = files [k].symmetric_values ;
        if (strlen (aname) == 0) break;
        // printf ("%s:\n", aname) ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct a directed graph G with adjacency matrix A
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;

        // compute the is_symmetric_structure cached property
        LG_BRUTAL (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;

        // check the result
        if (sym_structure)
        {
            TEST_CHECK (G->is_symmetric_structure == LAGraph_TRUE) ;
        }
        else
        {
            TEST_CHECK (G->is_symmetric_structure == LAGraph_FALSE) ;
        }

        // delete all cached properties
        OK (LAGraph_DeleteCached (G, msg)) ;

        // try again, but precompute G->AT
        LG_BRUTAL (LAGraph_Cached_AT (G, msg)) ;
        LG_BRUTAL (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;

        // check the result
        if (sym_structure)
        {
            TEST_CHECK (G->is_symmetric_structure == LAGraph_TRUE) ;
        }
        else
        {
            TEST_CHECK (G->is_symmetric_structure == LAGraph_FALSE) ;
        }

        // delete all cached properties
        OK (LAGraph_DeleteCached (G, msg)) ;

        // change the graph to directed, if matrix is symmetric
        if (sym_values)
        {
            G->kind = LAGraph_ADJACENCY_UNDIRECTED ;
            // recompute the symmetry cached property
            LG_BRUTAL (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;
            TEST_CHECK (G->is_symmetric_structure == LAGraph_TRUE) ;
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
    { "test_Symmetric_Structure", test_Cached_Symmetric_Structure },
    #if LAGRAPH_SUITESPARSE
    { "test_Symmetric_Structure_brutal",
        test_Cached_Symmetric_Structure_brutal },
    #endif
    { NULL, NULL }
} ;
