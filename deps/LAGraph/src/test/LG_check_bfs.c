//------------------------------------------------------------------------------
// LAGraph/src/test/LG_check_bfs: stand-alone test for BFS
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

#define LG_FREE_WORK                                \
{                                                   \
    LAGraph_Free ((void **) &queue, NULL) ;         \
    LAGraph_Free ((void **) &level_check, NULL) ;   \
    LAGraph_Free ((void **) &level_in, NULL) ;      \
    LAGraph_Free ((void **) &parent_in, NULL) ;     \
    LAGraph_Free ((void **) &visited, NULL) ;       \
    LAGraph_Free ((void **) &neighbors, NULL) ;     \
    GrB_free (&Row) ;                               \
}

#define LG_FREE_ALL                                 \
{                                                   \
    LG_FREE_WORK ;                                  \
    LAGraph_Free ((void **) &Ap, NULL) ;            \
    LAGraph_Free ((void **) &Aj, NULL) ;            \
    LAGraph_Free ((void **) &Ax, NULL) ;            \
}

#include "LG_internal.h"
#include "LG_test.h"

//------------------------------------------------------------------------------
// test the results from a BFS
//------------------------------------------------------------------------------

// Because this method does on GxB_unpack on G->A, it should not be used in a
// brutal memory test, unless the caller is prepared to reconstruct G->A
// when the brutal test causes this method to return early.

int LG_check_bfs
(
    // input
    GrB_Vector Level,       // optional; may be NULL
    GrB_Vector Parent,      // optional; may be NULL
    LAGraph_Graph G,
    GrB_Index src,
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    double tt = LAGraph_WallClockTime ( ) ;

    GrB_Vector Row = NULL ;
    GrB_Index *Ap = NULL, *Aj = NULL, *neighbors = NULL ;
    void *Ax = NULL ;
    GrB_Index Ap_size, Aj_size, Ax_size, n, ncols ;
    int64_t *queue = NULL, *level_in = NULL, *parent_in = NULL,
        *level_check = NULL ;
    bool *visited = NULL ;
    LG_TRY (LAGraph_CheckGraph (G, msg)) ;
    GRB_TRY (GrB_Matrix_nrows (&n, G->A)) ;
    GRB_TRY (GrB_Matrix_ncols (&ncols, G->A)) ;
    bool print_timings = (n >= 2000) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    LG_TRY (LAGraph_Malloc ((void **) &queue, n, sizeof (int64_t), msg)) ;
    LG_TRY (LAGraph_Malloc ((void **) &level_check, n, sizeof (int64_t), msg)) ;

    //--------------------------------------------------------------------------
    // get the contents of the Level and Parent vectors
    //--------------------------------------------------------------------------

    if (Level != NULL)
    {
        LG_TRY (LAGraph_Malloc ((void **) &level_in, n, sizeof (int64_t), msg)) ;
        LG_TRY (LG_check_vector (level_in, Level, n, -1)) ;
    }

    if (Parent != NULL)
    {
        LG_TRY (LAGraph_Malloc ((void **) &parent_in, n, sizeof (int64_t), msg)) ;
        LG_TRY (LG_check_vector (parent_in, Parent, n, -1)) ;
    }

    //--------------------------------------------------------------------------
    // unpack the matrix in CSR form for SuiteSparse:GraphBLAS
    //--------------------------------------------------------------------------

    #if LAGRAPH_SUITESPARSE
    bool iso, jumbled ;
    GRB_TRY (GxB_Matrix_unpack_CSR (G->A,
        &Ap, &Aj, &Ax, &Ap_size, &Aj_size, &Ax_size, &iso, &jumbled, NULL)) ;
    #endif

    //--------------------------------------------------------------------------
    // compute the level of each node
    //--------------------------------------------------------------------------

    if (print_timings)
    {
        tt = LAGraph_WallClockTime ( ) - tt ;
        printf ("LG_check_bfs init  time: %g sec\n", tt) ;
        tt = LAGraph_WallClockTime ( ) ;
    }

    queue [0] = src ;
    int64_t head = 0 ;
    int64_t tail = 1 ;
    LG_TRY (LAGraph_Calloc ((void **) &visited, n, sizeof (bool), msg)) ;
    visited [src] = true ;      // src is visited, and is level 0

    for (int64_t i = 0 ; i < n ; i++)
    {
        level_check [i] = -1 ;
    }
    level_check [src] = 0 ;

    #if !LAGRAPH_SUITESPARSE
    GRB_TRY (GrB_Vector_new (&Row, GrB_BOOL, n)) ;
    LG_TRY (LAGraph_Malloc ((void **) &neighbors, n, sizeof (GrB_Index), msg)) ;
    #endif

    while (head < tail)
    {
        // dequeue the node at the head of the queue
        int64_t u = queue [head++] ;

        #if LAGRAPH_SUITESPARSE
        // directly access the indices of entries in A(u,:)
        GrB_Index degree = Ap [u+1] - Ap [u] ;
        GrB_Index *node_u_adjacency_list = Aj + Ap [u] ;
        #else
        // extract the indices of entries in A(u,:)
        GrB_Index degree = n ;
        GRB_TRY (GrB_Col_extract (Row, NULL, NULL, G->A, GrB_ALL, n, u,
            GrB_DESC_T0)) ;
        GRB_TRY (GrB_Vector_extractTuples_BOOL (neighbors, NULL, &degree, Row));
        GrB_Index *node_u_adjacency_list = neighbors ;
        #endif

        // traverse all entries in A(u,:)
        for (int64_t k = 0 ; k < degree ; k++)
        {
            // consider edge (u,v)
            int64_t v = node_u_adjacency_list [k] ;
            if (!visited [v])
            {
                // node v is not yet visited; set its level and add to the
                // end of the queue
                visited [v] = true ;
                level_check [v] = level_check [u] + 1 ;
                queue [tail++] = v ;
            }
        }
    }

    if (print_timings)
    {
        tt = LAGraph_WallClockTime ( ) - tt ;
        printf ("LG_check_bfs bfs   time: %g sec\n", tt) ;
        tt = LAGraph_WallClockTime ( ) ;
    }

    //--------------------------------------------------------------------------
    // repack the matrix in CSR form for SuiteSparse:GraphBLAS
    //--------------------------------------------------------------------------

    #if LAGRAPH_SUITESPARSE
    GRB_TRY (GxB_Matrix_pack_CSR (G->A,
        &Ap, &Aj, &Ax, Ap_size, Aj_size, Ax_size, iso, jumbled, NULL)) ;
    #endif

    //--------------------------------------------------------------------------
    // check the level of each node
    //--------------------------------------------------------------------------

    if (level_in != NULL)
    {
        for (int64_t i = 0 ; i < n ; i++)
        {
            bool ok = (level_in [i] == level_check [i]) ;
            LG_ASSERT_MSG (ok, -2000, "invalid level") ;
        }
    }

    //--------------------------------------------------------------------------
    // check the parent of each node
    //--------------------------------------------------------------------------

    if (parent_in != NULL)
    {
        for (int64_t i = 0 ; i < n ; i++)
        {
            if (i == src)
            {
                // src node is its own parent
                bool ok = (parent_in [src] == src) && (visited [src]) ;
                LG_ASSERT_MSG (ok, -2001, "invalid parent") ;
            }
            else if (visited [i])
            {
                int64_t pi = parent_in [i] ;
                // ensure the parent pi is valid and has been visited
                bool ok = (pi >= 0 && pi < n) && visited [pi] ;
                LG_ASSERT_MSG (ok, -2001, "invalid parent") ;
                // ensure the edge (pi,i) exists
                bool x ;
                int info = GrB_Matrix_extractElement_BOOL (&x, G->A, pi, i) ;
                ok = (info == GrB_SUCCESS) ;
                LG_ASSERT_MSG (ok, -2001, "invalid parent") ;
                // ensure the parent's level is ok
                ok = (level_check [i] == level_check [pi] + 1) ;
                LG_ASSERT_MSG (ok, -2001, "invalid parent") ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    LG_FREE_WORK ;

    if (print_timings)
    {
        tt = LAGraph_WallClockTime ( ) - tt ;
        printf ("LG_check_bfs check time: %g sec\n", tt) ;
    }
    return (GrB_SUCCESS) ;
}
