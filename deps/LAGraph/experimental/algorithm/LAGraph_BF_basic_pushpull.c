//------------------------------------------------------------------------------
// LAGraph_BF_basic_pushpull: Bellman-Ford method for single source shortest
// paths
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

// This is the fastest variant that computes just the path lengths,
// and not the parent vector.

// LAGraph_BF_basic_pushpull: Bellman-Ford single source shortest paths,
// returning just the shortest path lengths.

// LAGraph_BF_basic_pushpull performs a Bellman-Ford to find out shortest path
// length from given source vertex s in the range of [0, n) on graph given as
// matrix A with size n by n. The sparse matrix A has entry A(i, j) if there is
// edge from vertex i to vertex j with weight w, then A(i, j) = w. Furthermore,
// LAGraph_BF_basic requires A(i, i) = 0 for all 0 <= i < n.

// LAGraph_BF_basic returns GrB_SUCCESS if successful, or GrB_NO_VALUE if it
// detects a negative-weight cycle.  The GrB_Vector d(k) (i.e., *pd_output)
// will be NULL when negative-weight cycle detected. Otherwise, the vector d
// has d(k) as the shortest distance from s to k.

// todo: this is a candidate for inclusion as a src/algorithm

//------------------------------------------------------------------------------

#define LG_FREE_ALL        \
{                          \
    GrB_free(&d) ;         \
    GrB_free(&dtmp) ;      \
}

#include <LAGraph.h>
#include <LAGraphX.h>
#include <LG_internal.h>  // from src/utility


// Given a n-by-n adjacency matrix A and a source vertex s.
// If there is no negative-weight cycle reachable from s, return the distances
// of shortest paths from s as vector d. Otherwise, return d=NULL if there is
// negative-weight cycle.
// pd_output = &d, where d is a GrB_Vector with d(k) as the shortest distance
// from s to k when no negative-weight cycle detected, otherwise, d = NULL.

// A must have explicit zeros on the diagonal and weights on corresponding
// entries of edges.  s is given index for source vertex.

GrB_Info LAGraph_BF_basic_pushpull
(
    GrB_Vector *pd_output,      //the pointer to the vector of distance
    const GrB_Matrix A,         //matrix for the graph
    const GrB_Matrix AT,        //transpose of A (optional)
    const GrB_Index s           //given index of the source
)
{
    GrB_Info info;
    char *msg = NULL ;
    GrB_Index nrows, ncols, nvalA;
    // tmp vector to store distance vector after n (i.e., V) loops
    GrB_Vector d = NULL, dtmp = NULL;

    LG_ASSERT ((A != NULL || AT != NULL) && pd_output != NULL,
        GrB_NULL_POINTER);

    (*pd_output) = NULL;
    bool use_vxm_with_A;
    if (A == NULL)
    {
        GRB_TRY (GrB_Matrix_nrows (&nrows, AT)) ;
        GRB_TRY (GrB_Matrix_ncols (&ncols, AT)) ;
        GRB_TRY (GrB_Matrix_nvals (&nvalA, AT)) ;
        use_vxm_with_A = false;
    }
    else
    {
        GRB_TRY (GrB_Matrix_nrows (&nrows, A)) ;
        GRB_TRY (GrB_Matrix_ncols (&ncols, A)) ;
        GRB_TRY (GrB_Matrix_nvals (&nvalA, A)) ;
        use_vxm_with_A = true;
    }

    // push/pull requires both A and AT
    bool push_pull = (A != NULL && AT != NULL) ;

    LG_ASSERT_MSG (nrows == ncols, -1002, "A must be square") ;

    GrB_Index n = nrows;           // n = # of vertices in graph
    // average node degree
    double dA = (n == 0) ? 0 : (((double) nvalA) / (double) n) ;

    LG_ASSERT_MSG (s < n, GrB_INVALID_INDEX, "invalid source node") ;

    // values used to determine if d should be converted to dense
    // dthreshold is used when only A or AT is available
    int64_t dthreshold ;
    if (A == NULL)
    {
        dthreshold =  LAGRAPH_MAX (256, sqrt ((double) n)) ;
    }
    else
    {
        dthreshold = n/2;
    }
    // refer to GraphBLAS/Source/GB_AxB_select.c
    // convert d to dense when GB_AxB_select intend to use Gustavson
    size_t csize = sizeof(double) + sizeof(int64_t) ;
    double gs_memory = ((double) n * (double) csize)/1e9 ;

    bool dsparse = true;

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

        // double t2 = LAGraph_WallClockTime ( ) ;

        // excute semiring on d and A, and save the result to d
        if (!use_vxm_with_A)
        {
            GRB_TRY (GrB_mxv (dtmp, NULL, NULL, GrB_MIN_PLUS_SEMIRING_FP64,
                AT, d, NULL));
        }
        else
        {
            GRB_TRY (GrB_vxm (dtmp, NULL, NULL, GrB_MIN_PLUS_SEMIRING_FP64,
                d, A, NULL));
        }
        LG_TRY (LAGraph_Vector_IsEqual (&same, dtmp, d, NULL));
        if (!same)
        {
            GrB_Vector ttmp = dtmp;
            dtmp = d;
            d = ttmp;
        }
        iter++;

        // t2 = LAGraph_WallClockTime ( ) - t2 ;

        GrB_Index dnz ;
        GRB_TRY (GrB_Vector_nvals (&dnz, d)) ;

        if (dsparse)
        {
            if (!push_pull)
            {
                if (dnz > dthreshold)
                {
                    dsparse = false;
                }
            }
            else
            {
                double heap_memory = (((double) dnz+1) *
                                     5 * (double) (sizeof(int64_t))) / 1e9;
                int log2dnz = 0 ;
                while (dnz > 0)
                {
                    dnz = dnz / 2 ;
                    log2dnz++ ;
                }
                dsparse = (4 * log2dnz * heap_memory < gs_memory);
                use_vxm_with_A = dsparse;
            }

            if (!dsparse)
            {
                GRB_TRY (GrB_Vector_setElement_FP64(d, 1e-16, s));
                GRB_TRY (GrB_assign (d, d, NULL, INFINITY, GrB_ALL, n,
                    GrB_DESC_C)) ;
                GRB_TRY (GrB_Vector_setElement_FP64(d, 0, s));
            }
        }
    }

    // check for negative-weight cycle only when there was a new path in the
    // last loop, otherwise, there can't be a negative-weight cycle.
    if (!same)
    {
        // excute semiring again to check for negative-weight cycle
        if (!use_vxm_with_A)
        {
            GRB_TRY (GrB_mxv(dtmp, NULL, NULL,
                GrB_MIN_PLUS_SEMIRING_FP64, AT, d, NULL));
        }
        else
        {
            GRB_TRY (GrB_vxm(dtmp, NULL, NULL,
                GrB_MIN_PLUS_SEMIRING_FP64, d, A, NULL));
        }
        LG_TRY (LAGraph_Vector_IsEqual (&same, dtmp, d, NULL));

        // if d != dtmp, then there is a negative-weight cycle in the graph
        if (!same)
        {
            // printf("A negative-weight cycle found. \n");
            LG_FREE_ALL;
            return (GrB_NO_VALUE) ;
        }
    }

    //--------------------------------------------------------------------------
    // todo: make d sparse
    //--------------------------------------------------------------------------
    /*if (!dsparse)
    {
        GRB_TRY (GrB_assign (d, d, NULL, d, GrB_ALL, n, GrB_DESC_R)) ;
        GRB_TRY (GrB_Vector_setElement_FP64(d, 0, s));
        GrB_Index dnz ;
        GRB_TRY (GrB_Vector_nvals (&dnz, d)) ;
        printf ("final nvals %.16g\n", (double) dnz) ;
    }*/

    (*pd_output) = d;
    d = NULL;
    LG_FREE_ALL;
    return (GrB_SUCCESS) ;
}
