//------------------------------------------------------------------------------
// LAGraph_BF_basic: Bellman-Ford method for single source shortest paths
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Jinhao Chen and Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// LAGraph_BF_basic: Bellman-Ford single source shortest paths, returning just
// the shortest path lengths.

// LAGraph_BF_basic performs a Bellman-Ford to find out shortest path length
// from given source vertex s in the range of [0, n) on graph given as matrix A
// with size n by n. The sparse matrix A has entry A(i, j) if there is edge from
// vertex i to vertex j with weight w, then A(i, j) = w. Furthermore,
// LAGraph_BF_basic requires A(i, i) = 0 for all 0 <= i < n.

// LAGraph_BF_basic returns GrB_SUCCESS regardless of existence of
// negative-weight cycle. However, the GrB_Vector d(k) (i.e., *pd_output) will
// be NULL when negative-weight cycle detected. Otherwise, the vector d has
// d(k) as the shortest distance from s to k.

//------------------------------------------------------------------------------

#define LG_FREE_ALL        \
{                          \
    GrB_free(&d) ;         \
    GrB_free(&dtmp) ;      \
}

#include "LG_internal.h"
#include <LAGraphX.h>

// Given a n-by-n adjacency matrix A and a source vertex s.
// If there is no negative-weight cycle reachable from s, return the distances
// of shortest paths from s as vector d. Otherwise, return d=NULL if there is
// negative-weight cycle.
// pd_output = &d, where d is a GrB_Vector with d(k) as the shortest distance
// from s to k when no negative-weight cycle detected, otherwise, d = NULL.
// A has zeros on diagonal and weights on corresponding entries of edges
// s is given index for source vertex
GrB_Info LAGraph_BF_basic
(
    GrB_Vector *pd_output,      //the pointer to the vector of distance
    const GrB_Matrix A,         //matrix for the graph
    const GrB_Index s           //given index of the source
)
{
    GrB_Info info;
    char *msg = NULL ;
    GrB_Index nrows, ncols;
    // tmp vector to store distance vector after n (i.e., V) loops
    GrB_Vector d = NULL, dtmp = NULL;

    LG_ASSERT (A != NULL && pd_output != NULL, GrB_NULL_POINTER) ;

    *pd_output = NULL;
    GRB_TRY (GrB_Matrix_nrows (&nrows, A)) ;
    GRB_TRY (GrB_Matrix_ncols (&ncols, A)) ;
    LG_ASSERT_MSG (nrows == ncols, -1002, "A must be square") ;
    GrB_Index n = nrows;           // n = # of vertices in graph
    LG_ASSERT_MSG (s < n, GrB_INVALID_INDEX, "invalid source node") ;

    // Initialize distance vector, change the d[s] to 0
    GRB_TRY (GrB_Vector_new(&d, GrB_FP64, n));
    GRB_TRY (GrB_Vector_setElement_FP64(d, 0, s));

    // copy d to dtmp in order to create a same size of vector
    GRB_TRY (GrB_Vector_dup(&dtmp, d));

    int64_t iter = 0;      //number of iterations
    bool same = false;     //variable indicating if d=dtmp

    // terminate when no new path is found or more than n-1 loops
    while (!same && iter < n - 1)
    {

        double t = LAGraph_WallClockTime ( ) ;

        // execute semiring on d and A, and save the result to d
        GRB_TRY (GrB_vxm(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64, d, A,
            GrB_NULL));
        LG_TRY (LAGraph_Vector_IsEqual (&same, dtmp, d, NULL));
        if (!same)
        {
            GrB_Vector ttmp = dtmp;
            dtmp = d;
            d = ttmp;
        }
        iter++;
        t = LAGraph_WallClockTime ( ) - t ;
        GrB_Index dnz ;
        GRB_TRY (GrB_Vector_nvals (&dnz, d)) ;
//      printf ("step %3d time %16.4f sec, nvals %.16g\n", iter, t, (double) dnz);
        fflush (stdout) ;
    }

    // check for negative-weight cycle only when there was a new path in the
    // last loop, otherwise, there can't be a negative-weight cycle.
    if (!same)
    {
        // execute semiring again to check for negative-weight cycle
        GRB_TRY (GrB_vxm(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64, d, A,
            GrB_NULL));
        LG_TRY (LAGraph_Vector_IsEqual (&same, dtmp, d, NULL));

        // if d != dtmp, then there is a negative-weight cycle in the graph
        if (!same)
        {
            // printf("A negative-weight cycle found. \n");
            LG_FREE_ALL;
            return (GrB_NO_VALUE) ;
        }
    }

    (*pd_output) = d;
    d = NULL;
    LG_FREE_ALL;
    return (GrB_SUCCESS) ;
}
