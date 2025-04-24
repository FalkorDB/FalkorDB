//------------------------------------------------------------------------------
// LAGraph_VertexCentrality_triangle: vertex triangle-centrality
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Tim Davis, Texas A&M University.

//------------------------------------------------------------------------------

// LAGraph_VertexCentrality_Triangle: computes the TriangleCentrality of
// an undirected graph.  No self edges are allowed on the input graph.
// Methods 2 and 3 can tolerate any edge weights (they are ignored; only the
// structure of G->A is used).  Methods 1 and 1.5 require unit edge weights
// (this could be modified); results are undefined if this condition doesn't
// hold.

// P. Burkhardt, "Triangle centrality," https://arxiv.org/pdf/2105.00110.pdf,
// April 2021.

// Method 3 is by far the fastest.

// This method uses pure GrB* methods from the v2.0 C API only.
// It does not rely on any SuiteSparse:GraphBLAS extensions.

// TC0: in python (called TC1 in the first draft of the paper)
//
// def triangle_centrality1(A):
//          T = A.mxm(A, mask=A)
//          y = T.reduce_vector()
//          k = y.reduce_float()
//          return(1/k)*(3*(A @ y) - 2*(T @ y) + y)
//          note: T@y is wrong. should be plus_second semiring

//  def TC1(A):
//      # this was "Method 1.5" in a draft, note the T.one@y is now correct:
//      T = A.mxm(A, mask=A, desc=ST1)
//      y = T.reduce_vector()
//      k = y.reduce_float()
//      return (3 * (A @ y) - 2 * (T.one() @ y) + y) / k

//  def TC2(A):
//      # this was TC2 in the first submission
//      T = A.plus_pair(A, mask=A, desc=ST1)
//      y = Vector.dense(FP64, A.nrows)
//      T.reduce_vector(out=y, accum=FP64.plus)
//      k = y.reduce_float()
//      return (3 * A.plus_second(y) - 2 * T.plus_second(y) + y) / k

//  def TC3(A):
//      M = A.tril(-1)
//      T = A.plus_pair(A, mask=M, desc=ST1)
//      y = T.reduce() + T.reduce(desc=ST0)
//      k = y.reduce_float()
//      return (
//          3 * A.plus_second(y) -
//          (2 * (T.plus_second(y) + T.plus_second(y, desc=ST0))) + y
//      ) / k

//------------------------------------------------------------------------------

#define LG_FREE_WORK                \
{                                   \
    GrB_free (&T) ;                 \
    GrB_free (&u) ;                 \
    GrB_free (&w) ;                 \
    GrB_free (&y) ;                 \
    GrB_free (&L) ;                 \
}

#define LG_FREE_ALL                 \
{                                   \
    LG_FREE_WORK ;                  \
    GrB_free (centrality) ;         \
}

#include "LG_internal.h"

//------------------------------------------------------------------------------
// LAGraph_VertexCentrality_Triangle: vertex triangle-centrality
//------------------------------------------------------------------------------

int LAGraph_VertexCentrality_Triangle       // vertex triangle-centrality
(
    // outputs:
    GrB_Vector *centrality,     // centrality(i): triangle centrality of i
    uint64_t *ntriangles,       // # of triangles in the graph
    // inputs:
    int method,                 // 0, 1, 2, or 3
    LAGraph_Graph G,            // input graph
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Matrix T = NULL, L = NULL, A = NULL ;
    GrB_Vector y = NULL, u = NULL, w = NULL ;

    LG_ASSERT (centrality != NULL && ntriangles != NULL, GrB_NULL_POINTER) ;
    (*centrality) = NULL ;
    LG_TRY (LAGraph_CheckGraph (G, msg)) ;

    if (G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGraph_ADJACENCY_DIRECTED &&
        G->is_symmetric_structure == LAGraph_TRUE))
    {
        // the structure of A is known to be symmetric
        A = G->A ;
    }
    else
    {
        // A is not known to be symmetric
        LG_ASSERT_MSG (false, -1005, "G->A must be symmetric") ;
    }

    // no self edges can be present
    LG_ASSERT_MSG (G->nself_edges == 0, -1004, "G->nself_edges must be zero") ;

    //--------------------------------------------------------------------------
    // create the T matrix
    //--------------------------------------------------------------------------

    GrB_Index n ;
    GRB_TRY (GrB_Matrix_nrows (&n, A)) ;
    GRB_TRY (GrB_Matrix_new (&T, GrB_FP64, n, n)) ;
    double k = 0 ;

    //--------------------------------------------------------------------------
    // compute the Triangle Centrality
    //--------------------------------------------------------------------------

    if (method == 0 || method == 1)
    {

        //----------------------------------------------------------------------
        // TC0, TC1: simplest method, requires that A has all entries equal to 1
        //----------------------------------------------------------------------

        // todo: remove this method when moving this code from experimental/
        // to src/

        if (method == 0)
        {
            // T<A> = A*A : method 0 (was TC1 in the first paper submission)
            GRB_TRY (GrB_mxm (T, A, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, A,
                NULL)) ;
        }
        else
        {
            // this is faster than method 0
            // T<A> = A*A' : method TC1 (was method TC1.5)
            GRB_TRY (GrB_mxm (T, A, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, A,
                GrB_DESC_T1)) ;
        }

        // y = sum (T), where y(i) = sum (T (i,:)) and y(i)=0 of T(i,:) is empty
        GRB_TRY (GrB_Vector_new (&y, GrB_FP64, n)) ;
        GRB_TRY (GrB_reduce (y, NULL, NULL, GrB_PLUS_MONOID_FP64, T, NULL)) ;

        // k = sum (y)
        GRB_TRY (GrB_reduce (&k, NULL, GrB_PLUS_MONOID_FP64, y, NULL)) ;

        // T = spones (T)
        GRB_TRY (GrB_assign (T, T, NULL, (double) 1, GrB_ALL, n, GrB_ALL, n,
            GrB_DESC_S)) ;

        // centrality = (3*A*y - 2*T*y + y) / k

        // w = T*y
        GRB_TRY (GrB_Vector_new (&w, GrB_FP64, n)) ;
        GRB_TRY (GrB_mxv (w, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, T, y,
            NULL)) ;

        // w = (-2)*w
        double minus_two = -2 ;
        GRB_TRY (GrB_apply (w, NULL, NULL, GrB_TIMES_FP64, minus_two, w,
            NULL)) ;

        // u = A*y
        GRB_TRY (GrB_Vector_new (&u, GrB_FP64, n)) ;
        GRB_TRY (GrB_mxv (u, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, y,
            NULL)) ;

    }
    else if (method == 2)
    {

        //----------------------------------------------------------------------
        // TC2: using LAGraph_plus_one_fp64 semiring
        //----------------------------------------------------------------------

        // todo: remove this method when moving this code from experimental/
        // to src/

        // T{A} = A*A' (each triangle is seen 6 times)
        GRB_TRY (GrB_mxm (T, A, NULL, LAGraph_plus_one_fp64, A, A,
            GrB_DESC_ST1)) ;

        // y = sum (T), where y(i) = sum (T (i,:)) and y(i)=0 of T(i,:) is empty
        GRB_TRY (GrB_Vector_new (&y, GrB_FP64, n)) ;
        GRB_TRY (GrB_assign (y, NULL, NULL, ((double) 0), GrB_ALL, n, NULL)) ;
        GRB_TRY (GrB_reduce (y, NULL, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64, T,
            NULL)) ;

        // k = sum (y)
        GRB_TRY (GrB_reduce (&k, NULL, GrB_PLUS_MONOID_FP64, y, NULL)) ;

        // centrality = (3*A*y - 2*T*y + y) / k

        // w = T*y
        GRB_TRY (GrB_Vector_new (&w, GrB_FP64, n)) ;
        GRB_TRY (GrB_mxv (w, NULL, NULL, LAGraph_plus_second_fp64, T, y,
            NULL)) ;

        // w = (-2)*w
        double minus_two = -2 ;
        GRB_TRY (GrB_apply (w, NULL, NULL, GrB_TIMES_FP64, minus_two, w,
            NULL)) ;

        // u = A*y
        GRB_TRY (GrB_Vector_new (&u, GrB_FP64, n)) ;
        GRB_TRY (GrB_mxv (u, NULL, NULL, LAGraph_plus_second_fp64, A, y,
            NULL)) ;

    }
    else if (method == 3)
    {

        //----------------------------------------------------------------------
        // TC3: using tril.  This is the fastest method.
        //----------------------------------------------------------------------

        // todo: When this method is moved to src/, keep this method only.

        // L = tril (A,-1)
        GRB_TRY (GrB_Matrix_new (&L, GrB_FP64, n, n)) ;
        GRB_TRY (GrB_select (L, NULL, NULL, GrB_TRIL, A, (int64_t) (-1),
            NULL)) ;

        // T{L}= A*A' (each triangle is seen 3 times; T is lower triangular)
        GRB_TRY (GrB_mxm (T, L, NULL, LAGraph_plus_one_fp64, A, A,
            GrB_DESC_ST1)) ;
        GRB_TRY (GrB_free (&L)) ;

        // y = sum (T'), where y(j) = sum (T (:,j)) and y(j)=0 if T(:,j) empty
        GRB_TRY (GrB_Vector_new (&y, GrB_FP64, n)) ;
        GRB_TRY (GrB_assign (y, NULL, NULL, ((double) 0), GrB_ALL, n, NULL)) ;
        GRB_TRY (GrB_reduce (y, NULL, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64, T,
            GrB_DESC_T0)) ;
        // y += sum (T)
        GRB_TRY (GrB_reduce (y, NULL, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64, T,
            NULL)) ;

        // k = sum (y).  y is the same as the other methods, above, just
        // computed using the lower triangular matrix T.  So k/6 is the total
        // number of triangles in the graph.
        GRB_TRY (GrB_reduce (&k, NULL, GrB_PLUS_MONOID_FP64, y, NULL)) ;

        // centrality = (3*A*y - 2* (T*y + T'*y) + y) / k

        // w = T*y
        GRB_TRY (GrB_Vector_new (&w, GrB_FP64, n)) ;
        GRB_TRY (GrB_mxv (w, NULL, NULL, LAGraph_plus_second_fp64, T, y,
            NULL)) ;
        // w += T'*y
        GRB_TRY (GrB_mxv (w, NULL, GrB_PLUS_FP64, LAGraph_plus_second_fp64,
            T, y, GrB_DESC_T0)) ;

        // w = (-2)*w
        double minus_two = -2 ;
        GRB_TRY (GrB_apply (w, NULL, NULL, GrB_TIMES_FP64, minus_two, w,
            NULL)) ;

        // u = A*y
        GRB_TRY (GrB_Vector_new (&u, GrB_FP64, n)) ;
        GRB_TRY (GrB_mxv (u, NULL, NULL, LAGraph_plus_second_fp64, A, y,
            NULL)) ;

    }

    //--------------------------------------------------------------------------
    // centrality = (3*u + w + y) / k for all 4 methods
    //--------------------------------------------------------------------------

    // centrality = 3*u
    GRB_TRY (GrB_Vector_new (centrality, GrB_FP64, n)) ;
    const double three = 3 ;
    GRB_TRY (GrB_apply (*centrality, NULL, NULL, GrB_TIMES_FP64, three, u,
        NULL)) ;

    // centrality += (w + y)
    GRB_TRY (GrB_eWiseAdd (*centrality, NULL, GrB_PLUS_FP64, GrB_PLUS_FP64,
        w, y, NULL)) ;

    // centrality = centrality / k
    GRB_TRY (GrB_apply (*centrality, NULL, NULL, GrB_TIMES_FP64,
        ((k == 0) ? 1.0 : (1.0/k)), *centrality, NULL)) ;

    (*ntriangles) = (uint64_t) (k/6) ;     // # triangles is k/6 for all methods

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
