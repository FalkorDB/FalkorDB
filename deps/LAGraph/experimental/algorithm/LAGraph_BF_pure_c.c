//------------------------------------------------------------------------------
// BF_pure_c.c: Bellman-Ford method, not using GraphBLAS
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

// LAGraph_BF_pure_c: Bellman-Ford single source shortest paths, returning
// both the path lengths and the shortest-path tree.

// LAGraph_BF_pure_c performs the Bellman-Ford algorithm to find out shortest
// path length, parent nodes along the path from given source vertex s in the
// range of [0, n) on graph with n nodes. It is implemented purely using
// conventional method, and is single-threaded. It is used here for checking
// the correctness of the result and comparison with the Bellman Ford
// implemented based on LAGraph.  Therefore, it require the graph represented
// as triplet format (I, J, W), which is an edge from vertex I(k) to vertex
// J(k) with weight W(k), and also the number of vertices and number of edges.

// LAGraph_BF_pure_c returns GrB_SUCCESS, or GrB_NO_VALUE if it detects of
// negative- weight cycle. The vector d(k) and pi(k) (i.e., *pd, and *ppi
// respectively) will be NULL when negative-weight cycle detected. Otherwise,
// the vector d has d(k) as the shortest distance from s to k. pi(k) = p, where
// p is the parent node of k-th node in the shortest path. In particular, pi(s)
// = -1.

//------------------------------------------------------------------------------

#define LG_FREE_ALL                     \
{                                       \
    LAGraph_Free ((void**) &d, NULL) ;  \
    LAGraph_Free ((void**) &pi, NULL) ; \
}

#include "LG_internal.h"
#include <LAGraphX.h>

// Given the edges and corresponding weights of a graph in tuple
// form {I, J, W} and a source vertex s. If there is no negative-weight
// cycle reachable from s, returns GrB_SUCCESS and the shortest distance
// d and the shortest path tree pi. Otherwise return NULL pointer for d
// and pi.

GrB_Info LAGraph_BF_pure_c
(
    int32_t **pd,     // pointer to distance vector d, d(k) = shorstest distance
                     // between s and k if k is reachable from s
    int64_t **ppi,   // pointer to parent index vector pi, pi(k) = parent of
                     // node k in the shortest path tree
    const int64_t s, // given source node index
    const int64_t n, // number of nodes
    const int64_t nz,// number of edges
    const int64_t *I,// row index vector
    const int64_t *J,// column index vector
    const int32_t  *W // weight vector, W(i) = weight of edge (I(i),J(i))
)
{
    char *msg = NULL ;
    int64_t i, j, k;
    int32_t *d = NULL;
    int64_t *pi = NULL;
    LG_ASSERT (I != NULL && J != NULL && W != NULL && pd != NULL &&
        ppi != NULL, GrB_NULL_POINTER) ;

    LAGraph_Free ((void **) pd, NULL) ;
    LAGraph_Free ((void **) ppi, NULL) ;

    LG_ASSERT_MSG (s < n, GrB_INVALID_INDEX, "invalid source node") ;

    // allocate d and pi
    LAGRAPH_TRY (LAGraph_Malloc((void **) &d,  n, sizeof(int32_t), msg));
    LAGRAPH_TRY (LAGraph_Malloc((void **) &pi, n, sizeof(int64_t), msg));

    // initialize d to a vector of INF while set d(s) = 0
    // and pi to a vector of -1
    for (i = 0; i < n; i++)
    {
        d[i] = INT32_MAX;
        pi[i] = -1;
    }
    d[s] = 0;

    // start the RELAX process and print results after each loop
    bool new_path = true;     //variable indicating if new path is found
    int64_t count = 0;        //number of loops
    // terminate when no new path is found or more than n-1 loops
    while(new_path && count < n-1)
    {
        new_path = false;
        for (k = 0; k < nz; k++)
        {
            i = I[k];
            j = J[k];
            if (d[i] != INT32_MAX && (d[j] == INT32_MAX || d[j] > d[i] + W[k]))
            {
                d[j] = d[i] + W[k];
                pi[j] = i;
                new_path = true;
            }
        }
        count++;
    }

    // check for negative-weight cycle only when there was a new path in the
    // last loop, otherwise, there can't be a negative-weight cycle.
    if (new_path)
    {
        // Do another loop of RELAX to check for negative loop,
        // return true if there is negative-weight cycle;
        // otherwise, print the distance vector and return false.
        for (k = 0; k < nz; k++)
        {
            i = I[k];
            j = J[k];
            if (d[i] != INT32_MAX && (d[j] == INT32_MAX || d[j] > d[i] + W[k]))
            {
                LG_FREE_ALL ;
                return (GrB_NO_VALUE) ;
            }
        }
    }

    *pd = d;
    *ppi = pi;
    return (GrB_SUCCESS) ;
}
