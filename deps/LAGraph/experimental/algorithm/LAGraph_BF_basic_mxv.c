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

// LAGraph_BF_basic_mxv: Bellman-Ford single source shortest paths, returning
// just the shortest path lengths.

// LAGraph_BF_basic_mxv performs a Bellman-Ford to find out shortest path length
// from given source vertex s in the range of [0, n) on graph with n nodes.
// It works almost the same as LAGraph_BF_basic except that it performs update
// using GrB_mxv instead of GrB_vxm, therefore, it require the input matrix as
// the transpose of adjacency matrix A with size n by n. That is, the input
// sparse matrix has entry AT(i, j) if there is edge from vertex j to vertex i
// with weight w, then AT(i, j) = w. While same as LAGraph_BF_basic, it requires
// AT(i, i) = 0 for all 0 <= i < n.

// LAGraph_BF_basic_mxv returns GrB_SUCCESS if it succeeds. In this case, there
// are no negative-weight cycles in the graph, and the vector d is returned.
// d(k) is the shortest distance from s to k.

// If the graph has a negative-weight cycle, GrB_NO_VALUE is returned, and the
// GrB_Vector d (i.e., *pd_output) will be NULL.

// Otherwise, other errors such as GrB_OUT_OF_MEMORY, GrB_INVALID_OBJECT, and
// so on, can be returned, if these errors are found by the underlying
// GrB_* functions.
//------------------------------------------------------------------------------

#define LG_FREE_ALL        \
{                          \
    GrB_free(&d) ;         \
    GrB_free(&dtmp) ;      \
}

#include <LAGraph.h>
#include <LAGraphX.h>
#include <LG_internal.h>  // from src/utility


// Given the transposed of a n-by-n adjacency matrix A and a source vertex s.
// If there is no negative-weight cycle reachable from s, return the distances
// of shortest paths from s as vector d. Otherwise, return d=NULL if there is
// negative-weight cycle.
// pd_output = &d, where d is a GrB_Vector with d(k) as the shortest distance
// from s to k when no negative-weight cycle detected, otherwise, d = NULL.
// AT has zeros on diagonal and weights on corresponding entries of edges
// s is given index for source vertex
GrB_Info LAGraph_BF_basic_mxv
(
    GrB_Vector *pd_output,      //the pointer to the vector of distance
    const GrB_Matrix AT,        //transposed adjacency matrix for the graph
    const GrB_Index s           //given index of the source
)
{
    GrB_Info info;
    char *msg = NULL ;
    GrB_Index nrows, ncols;
    // tmp vector to store distance vector after n loops
    GrB_Vector d = NULL, dtmp = NULL;

    LG_ASSERT (AT != NULL && pd_output != NULL, GrB_NULL_POINTER) ;

    *pd_output = NULL;
    GRB_TRY (GrB_Matrix_nrows (&nrows, AT)) ;
    GRB_TRY (GrB_Matrix_ncols (&ncols, AT)) ;
    LG_ASSERT_MSG (nrows == ncols, -1002, "AT must be square") ;
    GrB_Index n = nrows;           // n = # of vertices in graph
    LG_ASSERT_MSG (s < n, GrB_INVALID_INDEX, "invalid source node") ;

    // Initialize distance vector, change the d[s] to 0
    GRB_TRY (GrB_Vector_new(&d, GrB_FP64, n));
    GRB_TRY (GrB_Vector_setElement_FP64(d, 0, s));

    // copy d to dtmp in order to create a same size of vector
    GRB_TRY (GrB_Vector_dup(&dtmp, d));

    int64_t iter = 0;      //number of iterations
    bool same = false;     //variable indicating if d == dtmp

    // terminate when no new path is found or more than n-1 loops
    while (!same && iter < n - 1)
    {
        // excute semiring on d and AT, and save the result to d
        GRB_TRY (GrB_mxv(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64,
            AT, d, GrB_NULL));

        LG_TRY (LAGraph_Vector_IsEqual (&same, dtmp, d, NULL));
        if (!same)
        {
            GrB_Vector ttmp = dtmp;
            dtmp = d;
            d = ttmp;
        }
        iter++;
    }

    // check for negative-weight cycle only when there was a new path in the
    // last loop, otherwise, there can't be a negative-weight cycle.
    if (!same)
    {
        // excute semiring again to check for negative-weight cycle
        GRB_TRY (GrB_mxv(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64,
            AT, d, GrB_NULL));

        // if d != dtmp, then there is a negative-weight cycle in the graph
        LG_TRY (LAGraph_Vector_IsEqual (&same, dtmp, d, NULL));
        if (!same)
        {
            // printf("AT negative-weight cycle found. \n");
            LG_FREE_ALL;
            return (GrB_NO_VALUE) ;
        }
    }

    (*pd_output) = d;
    d = NULL;
    LG_FREE_ALL;
    return (GrB_SUCCESS) ;
}
