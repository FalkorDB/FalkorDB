//------------------------------------------------------------------------------
// LAGraph_AllKTruss.c: find all k-trusses of a graph
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

// LAGraph_AllKTruss: find all k-trusses of a graph via GraphBLAS.

// Given a symmetric graph A with no-self edges, LAGraph_AllKTruss finds all
// k-trusses of A.

// The output matrices Cset [3..kmax-1] are the k-trusses of A.  Their edges
// are a subset of A.  Each edge in C = Cset [k] is part of at least k-2
// triangles in C.  The structure of C is the adjacency matrix of the k-truss
// subgraph of A.  The edge weights of C are the support of each edge.  That
// is, C(i,j)=nt if the edge (i,j) is part of nt triangles in C.  All edges in
// C have support of at least k-2.  The total number of triangles in C is
// sum(C)/6.  The number of edges in C is nnz(C)/2.  C = Cset [k] is returned
// as symmetric with a zero-free diagonal.  Cset [kmax] is an empty matrix
// since the kmax-truss is empty.

// The arrays ntris, nedges, and nstepss hold the output statistics.
// ntris   [k] = # of triangles in the k-truss
// nedges  [k] = # of edges in the k-truss
// nstepss [k] = # of steps required to compute the k-truss

// Usage: constructs all k-trusses of A, for k = 3:kmax

//      int64_t kmax ;
//      GrB_Matrix_nrows (&n, A) ;
//      int64_t n4 = (n > 4) ? n : 4 ;
//      GrB_Matrix *Cset = LAGraph_malloc (n4, sizeof (GrB_Matrix)) ;
//      int64_t *ntris   = LAGraph_malloc (n4, sizeof (int64_t)) ;
//      int64_t *nedges  = LAGraph_malloc (n4, sizeof (int64_t)) ;
//      int64_t *nstepss = LAGraph_malloc (n4, sizeof (int64_t)) ;
//      int result = LAGraph_AllKTruss (&Cset, &kmax, ntris, nedges,
//          nstepss, G, msg) ;

// todo: add experimental/benchmark/ktruss_demo.c to benchmark k-truss
// and all-k-truss

// todo: consider LAGraph_KTrussNext to compute the (k+1)-truss from the
// k-truss

#define LG_FREE_ALL                         \
{                                           \
    for (int64_t kk = 3 ; kk <= k ; kk++)   \
    {                                       \
        GrB_free (&(Cset [kk])) ;           \
    }                                       \
}

#include "LG_internal.h"
#include "LAGraphX.h"

//------------------------------------------------------------------------------
// C = LAGraph_AllKTruss: find all k-trusses a graph
//------------------------------------------------------------------------------

int LAGraph_AllKTruss   // compute all k-trusses of a graph
(
    // outputs
    GrB_Matrix *Cset,   // size n, output k-truss subgraphs
    int64_t *kmax,      // smallest k where k-truss is empty
    int64_t *ntris,     // size max(n,4), ntris [k] is #triangles in k-truss
    int64_t *nedges,    // size max(n,4), nedges [k] is #edges in k-truss
    int64_t *nstepss,   // size max(n,4), nstepss [k] is #steps for k-truss
    // input
    LAGraph_Graph G,    // input graph
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    int64_t k = 0 ;
    LG_ASSERT (Cset != NULL && nstepss != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (kmax != NULL && ntris != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (nedges != NULL, GrB_NULL_POINTER) ;
    LG_TRY (LAGraph_CheckGraph (G, msg)) ;

    if (G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGraph_ADJACENCY_DIRECTED &&
        G->is_symmetric_structure == LAGraph_TRUE))
    {
        // the structure of A is known to be symmetric
        ;
    }
    else
    {
        // A is not known to be symmetric
        LG_ASSERT_MSG (false, -1005, "G->A must be symmetric") ;
    }

    // no self edges can be present
    LG_ASSERT_MSG (G->nself_edges == 0, -1004, "G->nself_edges must be zero") ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    for (k = 0 ; k <= 3 ; k++)
    {
        Cset [k] = NULL ;
        ntris   [k] = 0 ;
        nedges  [k] = 0 ;
        nstepss [k] = 0 ;
    }
    k = 3 ;
    (*kmax) = 0 ;

    //--------------------------------------------------------------------------
    // initialzations
    //--------------------------------------------------------------------------

    GrB_Index n ;
    GrB_Matrix S = G->A ;
    GRB_TRY (GrB_Matrix_nrows (&n, S)) ;
    GRB_TRY (GrB_Matrix_new (&(Cset [k]), GrB_UINT32, n, n)) ;
    GrB_Matrix C = Cset [k] ;
    GrB_Index nvals, nvals_last ;
    GRB_TRY (GrB_Matrix_nvals (&nvals_last, S)) ;
    int64_t nsteps = 0 ;

    //--------------------------------------------------------------------------
    // find all k-trusses
    //--------------------------------------------------------------------------

    while (true)
    {
        // C{S} = S*S'
        GRB_TRY (GrB_mxm (C, S, NULL, LAGraph_plus_one_uint32, S, S,
            GrB_DESC_RST1)) ;
        // keep entries in C that are >= k-2
        GRB_TRY (GrB_select (C, NULL, NULL, GrB_VALUEGE_UINT32, C, k-2, NULL)) ;
        nsteps++ ;
        // check if k-truss has been found
        GRB_TRY (GrB_Matrix_nvals (&nvals, C)) ;
        if (nvals == nvals_last)
        {
            // k-truss has been found
            int64_t nt = 0 ;
            GRB_TRY (GrB_reduce (&nt, NULL, GrB_PLUS_MONOID_INT64, C, NULL)) ;
            ntris   [k] = nt / 6 ;
            nedges  [k] = nvals / 2 ;
            nstepss [k] = nsteps ;
            nsteps = 0 ;
            if (nvals == 0)
            {
                // this is the last k-truss
                (*kmax) = k ;
                return (GrB_SUCCESS) ;
            }
            S = C ;             // S = current k-truss for k+1 iteration
            k++ ;               // advance to the next k-tryss
            GRB_TRY (GrB_Matrix_new (&(Cset [k]), GrB_UINT32, n, n)) ;
            C = Cset [k] ;      // C = new matrix for next k-truss
        }
        else
        {
            // advance to the next step, still computing the current k-truss
            nvals_last = nvals ;
            S = C ;
        }
    }
}
