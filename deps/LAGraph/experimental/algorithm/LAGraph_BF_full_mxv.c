//------------------------------------------------------------------------------
// LAGraph_BF_full.c: Bellman-Ford single-source shortest paths, returns tree
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

// LAGraph_BF_full_mxv: Bellman-Ford single source shortest paths, returning
// both the path lengths and the shortest-path tree.

// LAGraph_BF_full_mxv performs a Bellman-Ford to find out shortest path
// length, parent nodes along the path and the hops (number of edges) in the
// path from given source vertex s in the range of [0, n) on graph with n nodes.
// It works almost the same as LAGraph_BF_full except that it performs update
// using GrB_mxv instead of GrB_vxm, therefore, it require the input matrix as
// the transpose of adjacency matrix A with size n by n. That is, the input
// sparse matrix has entry AT(i, j) if there is edge from vertex j to vertex i
// with weight w, then AT(i, j) = w. While same as LAGraph_BF_full, it requires
// AT(i, i) = 0 for all 0 <= i < n.

// LAGraph_BF_full_mxv returns GrB_SUCCESS if it succeeds.  In this case, there
// are no negative-weight cycles in the graph, and d, pi, and h are returned.
// The vector d has d(k) as the shortest distance from s to k. pi(k) = p+1,
// where p is the parent node of k-th node in the shortest path. In particular,
// pi(s) = 0. h(k) = hop(s, k), the number of edges from s to k in the shortest
// path.

// If the graph has a negative-weight cycle, GrB_NO_VALUE is returned, and the
// GrB_Vectors d(k), pi(k) and h(k)  (i.e., *pd_output, *ppi_output and
// *ph_output respectively) will be NULL when negative-weight cycle detected.

// Otherwise, other errors such as GrB_OUT_OF_MEMORY, GrB_INVALID_OBJECT, and
// so on, can be returned, if these errors are found by the underlying
// GrB_* functions.


//------------------------------------------------------------------------------

#define LG_FREE_WORK                   \
{                                      \
    GrB_free(&d);                      \
    GrB_free(&dtmp);                   \
    GrB_free(&Atmp);                   \
    GrB_free(&BF_Tuple3);              \
    GrB_free(&BF_lMIN_Tuple3);         \
    GrB_free(&BF_PLUSrhs_Tuple3);      \
    GrB_free(&BF_EQ_Tuple3);           \
    GrB_free(&BF_lMIN_Tuple3_Monoid);  \
    GrB_free(&BF_lMIN_PLUSrhs_Tuple3); \
    LAGraph_Free ((void**)&I, NULL);   \
    LAGraph_Free ((void**)&J, NULL);   \
    LAGraph_Free ((void**)&w, NULL);   \
    LAGraph_Free ((void**)&W, NULL);   \
    LAGraph_Free ((void**)&h, NULL);   \
    LAGraph_Free ((void**)&pi, NULL);  \
}

#define LG_FREE_ALL                    \
{                                      \
    LG_FREE_WORK ;                     \
    GrB_free (pd_output);              \
    GrB_free (ppi_output);             \
    GrB_free (ph_output);              \
}

#include <LAGraph.h>
#include <LAGraphX.h>
#include <LG_internal.h>  // from src/utility

typedef void (*LAGraph_binary_function) (void *, const void *, const void *) ;

//------------------------------------------------------------------------------
// data type for each entry of the adjacent matrix A and "distance" vector d;
// <INFINITY,INFINITY,INFINITY> corresponds to nonexistence of a path, and
// the value  <0, 0, NULL> corresponds to a path from a vertex to itself
//------------------------------------------------------------------------------

typedef struct
{
    double w;    // w  corresponds to a path weight.
    GrB_Index h; // h  corresponds to a path size or number of hops.
    GrB_Index pi;// pi corresponds to the penultimate vertex along a path.
                 // vertex indexed as 1, 2, 3, ... , V, and pi = 0 (as nil)
                 // for u=v, and pi = UINT64_MAX (as inf) for (u,v) not in E
}
BF_Tuple3_struct;

//------------------------------------------------------------------------------
// binary functions, z=f(x,y), where Tuple3xTuple3 -> Tuple3
//------------------------------------------------------------------------------

void BF_lMIN_mxv
(
    BF_Tuple3_struct *z,
    const BF_Tuple3_struct *y,
    const BF_Tuple3_struct *x
)
{
    if (x->w < y->w
        || (x->w == y->w && x->h < y->h)
        || (x->w == y->w && x->h == y->h && x->pi < y->pi))
    {
        if (z != x) { *z = *x; }
    }
    else
    {
        *z = *y;
    }
}

void BF_PLUSrhs_mxv
(
    BF_Tuple3_struct *z,
    const BF_Tuple3_struct *y,
    const BF_Tuple3_struct *x
)
{
    z->w = x->w + y->w;
    z->h = x->h + y->h;
    z->pi = (x->pi != UINT64_MAX && y->pi != 0) ?  y->pi : x->pi ;
}

void BF_EQ_mxv
(
    bool *z,
    const BF_Tuple3_struct *y,
    const BF_Tuple3_struct *x
)
{
    (*z) = (x->w == y->w && x->h == y->h && x->pi == y->pi) ;
}

// Given the transpose of a n-by-n adjacency matrix A and a source vertex s.
// If there is no negative-weight cycle reachable from s, return the distances
// of shortest paths from s and parents along the paths as vector d. Otherwise,
// returns d=NULL if there is a negtive-weight cycle.
// pd_output is pointer to a GrB_Vector, where the i-th entry is d(s,i), the
//   sum of edges length in the shortest path
// ppi_output is pointer to a GrB_Vector, where the i-th entry is pi(i), the
//   parent of i-th vertex in the shortest path
// ph_output is pointer to a GrB_Vector, where the i-th entry is h(s,i), the
//   number of edges from s to i in the shortest path
// AT has zeros on diagonal and weights on corresponding entries of edges
// s is given index for source vertex

GrB_Info LAGraph_BF_full_mxv
(
    GrB_Vector *pd_output,      //the pointer to the vector of distance
    GrB_Vector *ppi_output,     //the pointer to the vector of parent
    GrB_Vector *ph_output,      //the pointer to the vector of hops
    const GrB_Matrix AT,         //matrix for the graph
    const GrB_Index s           //given index of the source
)
{
    GrB_Info info;
    char *msg = NULL ;
    // tmp vector to store distance vector after n (i.e., V) loops
    GrB_Vector d = NULL, dtmp = NULL;
    GrB_Matrix Atmp = NULL;
    GrB_Type BF_Tuple3;

    GrB_BinaryOp BF_lMIN_Tuple3;
    GrB_BinaryOp BF_PLUSrhs_Tuple3;
    GrB_BinaryOp BF_EQ_Tuple3;

    GrB_Monoid BF_lMIN_Tuple3_Monoid;
    GrB_Semiring BF_lMIN_PLUSrhs_Tuple3;

    GrB_Index nrows, ncols, n, nz;  // n = # of row/col, nz = # of nnz in graph
    GrB_Index *I = NULL, *J = NULL; // for col/row indices of entries from AT
    GrB_Index *h = NULL, *pi = NULL;
    double *w = NULL;
    BF_Tuple3_struct *W = NULL;

    LG_ASSERT (AT != NULL && pd_output != NULL &&
        ppi_output != NULL && ph_output != NULL, GrB_NULL_POINTER) ;

    *pd_output  = NULL;
    *ppi_output = NULL;
    *ph_output  = NULL;
    GRB_TRY (GrB_Matrix_nrows (&nrows, AT)) ;
    GRB_TRY (GrB_Matrix_ncols (&ncols, AT)) ;
    GRB_TRY (GrB_Matrix_nvals (&nz, AT));
    LG_ASSERT_MSG (nrows == ncols, -1002, "A must be square") ;
    n = nrows;
    LG_ASSERT_MSG (s < n, GrB_INVALID_INDEX, "invalid source node") ;

    //--------------------------------------------------------------------------
    // create all GrB_Type GrB_BinaryOp GrB_Monoid and GrB_Semiring
    //--------------------------------------------------------------------------
    // GrB_Type
    GRB_TRY (GrB_Type_new(&BF_Tuple3, sizeof(BF_Tuple3_struct)));

    // GrB_BinaryOp
    GRB_TRY (GrB_BinaryOp_new(&BF_EQ_Tuple3,
        (LAGraph_binary_function) (&BF_EQ_mxv),
        GrB_BOOL, BF_Tuple3, BF_Tuple3));
    GRB_TRY (GrB_BinaryOp_new(&BF_lMIN_Tuple3,
        (LAGraph_binary_function) (&BF_lMIN_mxv),
        BF_Tuple3, BF_Tuple3, BF_Tuple3));
    GRB_TRY (GrB_BinaryOp_new(&BF_PLUSrhs_Tuple3,
        (LAGraph_binary_function)(&BF_PLUSrhs_mxv),
        BF_Tuple3, BF_Tuple3, BF_Tuple3));

    // GrB_Monoid
    BF_Tuple3_struct BF_identity = (BF_Tuple3_struct) { .w = INFINITY,
        .h = UINT64_MAX, .pi = UINT64_MAX };
    GRB_TRY (GrB_Monoid_new_UDT(&BF_lMIN_Tuple3_Monoid, BF_lMIN_Tuple3,
        &BF_identity));

    //GrB_Semiring
    GRB_TRY (GrB_Semiring_new(&BF_lMIN_PLUSrhs_Tuple3,
        BF_lMIN_Tuple3_Monoid, BF_PLUSrhs_Tuple3));

    //--------------------------------------------------------------------------
    // allocate arrays used for tuplets
    //--------------------------------------------------------------------------

    LAGRAPH_TRY (LAGraph_Malloc ((void **) &I, nz, sizeof(GrB_Index), msg)) ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &J, nz, sizeof(GrB_Index), msg)) ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &w, nz, sizeof(double), msg)) ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &W, nz, sizeof(BF_Tuple3_struct),
        msg)) ;

    //--------------------------------------------------------------------------
    // create matrix Atmp based on AT, while its entries become BF_Tuple3 type
    //--------------------------------------------------------------------------
    GRB_TRY (GrB_Matrix_extractTuples_FP64(I, J, w, &nz, AT));
    for (GrB_Index k = 0; k < nz; k++)
    {
        if (w[k] == 0)             //diagonal entries
        {
            W[k] = (BF_Tuple3_struct) { .w = 0, .h = 0, .pi = 0 };
        }
        else
        {
            // w(i, j) is the weight of edge from j to i, so the parent here
            // should be j
            W[k] = (BF_Tuple3_struct) { .w = w[k], .h = 1, .pi = J[k] + 1 };
        }
    }
    GRB_TRY (GrB_Matrix_new(&Atmp, BF_Tuple3, n, n));
    GRB_TRY (GrB_Matrix_build_UDT(Atmp, I, J, W, nz, BF_lMIN_Tuple3));
    LAGraph_Free ((void**)&I, NULL);
    LAGraph_Free ((void**)&J, NULL);
    LAGraph_Free ((void**)&W, NULL);
    LAGraph_Free ((void**)&w, NULL);

    //--------------------------------------------------------------------------
    // create and initialize "distance" vector d
    //--------------------------------------------------------------------------
    GRB_TRY (GrB_Vector_new(&d, BF_Tuple3, n));
    // initial distance from s to itself
    BF_Tuple3_struct d0 = (BF_Tuple3_struct) { .w = 0, .h = 0, .pi = 0 };
    GRB_TRY (GrB_Vector_setElement_UDT(d, &d0, s));

    //--------------------------------------------------------------------------
    // start the Bellman Ford process
    //--------------------------------------------------------------------------
    // copy d to dtmp in order to create a same size of vector
    GRB_TRY (GrB_Vector_dup(&dtmp, d));
    bool same= false;          // variable indicating if d == dtmp
    int64_t iter = 0;          // number of iterations

    // terminate when no new path is found or more than V-1 loops
    while (!same && iter < n - 1)
    {
        // execute semiring on d and AT, and save the result to dtmp
        GRB_TRY (GrB_mxv(dtmp, GrB_NULL, GrB_NULL, BF_lMIN_PLUSrhs_Tuple3,
            Atmp, d, GrB_NULL));
        LG_TRY (LAGraph_Vector_IsEqualOp (&same, dtmp, d, BF_EQ_Tuple3, NULL));
        if (!same)
        {
            GrB_Vector ttmp = dtmp;
            dtmp = d;
            d = ttmp;
        }
        iter ++;
    }

    // check for negative-weight cycle only when there was a new path in the
    // last loop, otherwise, there can't be a negative-weight cycle.
    if (!same)
    {
        // execute semiring again to check for negative-weight cycle
        GRB_TRY (GrB_mxv(dtmp, GrB_NULL, GrB_NULL, BF_lMIN_PLUSrhs_Tuple3,
            Atmp, d, GrB_NULL));
        LG_TRY (LAGraph_Vector_IsEqualOp (&same, dtmp, d, BF_EQ_Tuple3, NULL));

        // if d != dtmp, then there is a negative-weight cycle in the graph
        if (!same)
        {
            // printf("A negative-weight cycle found. \n");
            LG_FREE_ALL;
            return (GrB_NO_VALUE) ;
        }
    }

    //--------------------------------------------------------------------------
    // extract tuple from "distance" vector d and create GrB_Vectors for output
    //--------------------------------------------------------------------------

    LAGRAPH_TRY (LAGraph_Malloc ((void **) &I, n, sizeof(GrB_Index), msg)) ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &W, n, sizeof(BF_Tuple3_struct),
        msg)) ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &w, n, sizeof(double), msg)) ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &h, n, sizeof(GrB_Index), msg)) ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &pi, n, sizeof(GrB_Index), msg)) ;

    nz = n ;
    GRB_TRY (GrB_Vector_extractTuples_UDT (I, (void *) W, &nz, d));

    for (GrB_Index k = 0; k < n; k++)
    {
        w [k] = W[k].w ;
        h [k] = W[k].h ;
        pi[k] = W[k].pi;
    }

    GRB_TRY (GrB_Vector_new(pd_output,  GrB_FP64,   n));
    GRB_TRY (GrB_Vector_new(ppi_output, GrB_UINT64, n));
    GRB_TRY (GrB_Vector_new(ph_output,  GrB_UINT64, n));
    GRB_TRY (GrB_Vector_build (*pd_output , I, w , nz, GrB_MIN_FP64  ));
    GRB_TRY (GrB_Vector_build (*ppi_output, I, pi, nz, GrB_MIN_UINT64));
    GRB_TRY (GrB_Vector_build (*ph_output , I, h , nz, GrB_MIN_UINT64));
    LG_FREE_WORK;
    return (GrB_SUCCESS) ;
}
