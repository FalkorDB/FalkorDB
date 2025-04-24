//------------------------------------------------------------------------------
// LAGr_SingleSourceShortestPath: single-source shortest path
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Jinhao Chen, Scott Kolodziej and Tim Davis, Texas A&M
// University.  Adapted from GraphBLAS Template Library (GBTL) by Scott
// McMillan and Tze Meng Low.

//------------------------------------------------------------------------------

// This is an Advanced algorithm (G->emin is required).

// Single source shortest path with delta stepping.

// U. Sridhar, M. Blanco, R. Mayuranath, D. G. Spampinato, T. M. Low, and
// S. McMillan, "Delta-Stepping SSSP: From Vertices and Edges to GraphBLAS
// Implementations," in 2019 IEEE International Parallel and Distributed
// Processing Symposium Workshops (IPDPSW), 2019, pp. 241–250.
// https://ieeexplore.ieee.org/document/8778222/references
// https://arxiv.org/abs/1911.06895

// LAGr_SingleSourceShortestPath computes the shortest path lengths from the
// specified source vertex to all other vertices in the graph.

// The parent vector is not computed; see LAGraph_BF_* instead.

// NOTE: this method gets stuck in an infinite loop when there are negative-
// weight cycles in the graph.

// FUTURE: a Basic algorithm that picks Delta automatically

#define LG_FREE_WORK        \
{                           \
    GrB_free (&AL) ;        \
    GrB_free (&AH) ;        \
    GrB_free (&lBound) ;    \
    GrB_free (&uBound) ;    \
    GrB_free (&tmasked) ;   \
    GrB_free (&tReq) ;      \
    GrB_free (&tless) ;     \
    GrB_free (&s) ;         \
    GrB_free (&reach) ;     \
    GrB_free (&Empty) ;     \
}

#define LG_FREE_ALL         \
{                           \
    LG_FREE_WORK ;          \
    GrB_free (&t) ;         \
}

#include "LG_internal.h"

#define setelement(s, k)                                                      \
{                                                                             \
    switch (tcode)                                                            \
    {                                                                         \
        default:                                                              \
        case 0 : GrB_Scalar_setElement_INT32  (s, k * delta_int32 ) ; break ; \
        case 1 : GrB_Scalar_setElement_INT64  (s, k * delta_int64 ) ; break ; \
        case 2 : GrB_Scalar_setElement_UINT32 (s, k * delta_uint32) ; break ; \
        case 3 : GrB_Scalar_setElement_UINT64 (s, k * delta_uint64) ; break ; \
        case 4 : GrB_Scalar_setElement_FP32   (s, k * delta_fp32  ) ; break ; \
        case 5 : GrB_Scalar_setElement_FP64   (s, k * delta_fp64  ) ; break ; \
    }                                                                         \
}

int LAGr_SingleSourceShortestPath
(
    // output:
    GrB_Vector *path_length,    // path_length (i) is the length of the shortest
                                // path from the source vertex to vertex i
    // input:
    const LAGraph_Graph G,      // input graph, not modified
    GrB_Index source,           // source vertex
    GrB_Scalar Delta,           // delta value for delta stepping
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Scalar lBound = NULL ;  // the threshold for GrB_select
    GrB_Scalar uBound = NULL ;  // the threshold for GrB_select
    GrB_Matrix AL = NULL ;      // graph containing the light weight edges
    GrB_Matrix AH = NULL ;      // graph containing the heavy weight edges
    GrB_Vector t = NULL ;       // tentative shortest path length
    GrB_Vector tmasked = NULL ;
    GrB_Vector tReq = NULL ;
    GrB_Vector tless = NULL ;
    GrB_Vector s = NULL ;
    GrB_Vector reach = NULL ;
    GrB_Vector Empty = NULL ;

    LG_TRY (LAGraph_CheckGraph (G, msg)) ;
    LG_ASSERT (path_length != NULL && Delta != NULL, GrB_NULL_POINTER) ;
    (*path_length) = NULL ;

    GrB_Index nvals ;
    LG_TRY (GrB_Scalar_nvals (&nvals, Delta)) ;
    LG_ASSERT_MSG (nvals == 1, GrB_EMPTY_OBJECT, "Delta is missing") ;

    GrB_Matrix A = G->A ;
    GrB_Index n ;
    GRB_TRY (GrB_Matrix_nrows (&n, A)) ;
    LG_ASSERT_MSG (source < n, GrB_INVALID_INDEX, "invalid source node") ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    // get the type of the A matrix
    GrB_Type etype ;
    char typename [LAGRAPH_MAX_NAME_LEN] ;
    LG_TRY (LAGraph_Matrix_TypeName (typename, A, msg)) ;
    LG_TRY (LAGraph_TypeFromName (&etype, typename, msg)) ;

    GRB_TRY (GrB_Scalar_new (&lBound, etype)) ;
    GRB_TRY (GrB_Scalar_new (&uBound, etype)) ;
    GRB_TRY (GrB_Vector_new (&t, etype, n)) ;
    GRB_TRY (GrB_Vector_new (&tmasked, etype, n)) ;
    GRB_TRY (GrB_Vector_new (&tReq, etype, n)) ;
    GRB_TRY (GrB_Vector_new (&Empty, GrB_BOOL, n)) ;
    GRB_TRY (GrB_Vector_new (&tless, GrB_BOOL, n)) ;
    GRB_TRY (GrB_Vector_new (&s, GrB_BOOL, n)) ;
    GRB_TRY (GrB_Vector_new (&reach, GrB_BOOL, n)) ;

#if LAGRAPH_SUITESPARSE
    // optional hints for SuiteSparse:GraphBLAS
    GRB_TRY (GxB_set (t, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
    GRB_TRY (GxB_set (tmasked, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    GRB_TRY (GxB_set (tReq, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    GRB_TRY (GxB_set (tless, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    GRB_TRY (GxB_set (s, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    GRB_TRY (GxB_set (reach, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
#endif

    // select the operators, and set t (:) = infinity
    GrB_IndexUnaryOp ne, le, ge, lt, gt ;
    GrB_BinaryOp less_than ;
    GrB_Semiring min_plus ;
    int tcode ;
    int32_t  delta_int32  ;
    int64_t  delta_int64  ;
    uint32_t delta_uint32 ;
    uint64_t delta_uint64 ;
    float    delta_fp32   ;
    double   delta_fp64   ;

    bool negative_edge_weights = true ;

    if (etype == GrB_INT32)
    {
        GRB_TRY (GrB_Scalar_extractElement (&delta_int32, Delta)) ;
        GRB_TRY (GrB_assign (t, NULL, NULL, (int32_t) INT32_MAX,
            GrB_ALL, n, NULL)) ;
        ne = GrB_VALUENE_INT32 ;
        le = GrB_VALUELE_INT32 ;
        ge = GrB_VALUEGE_INT32 ;
        lt = GrB_VALUELT_INT32 ;
        gt = GrB_VALUEGT_INT32 ;
        less_than = GrB_LT_INT32 ;
        min_plus = GrB_MIN_PLUS_SEMIRING_INT32 ;
        tcode = 0 ;
    }
    else if (etype == GrB_INT64)
    {
        GRB_TRY (GrB_Scalar_extractElement (&delta_int64, Delta)) ;
        GRB_TRY (GrB_assign (t, NULL, NULL, (int64_t) INT64_MAX,
            GrB_ALL, n, NULL)) ;
        ne = GrB_VALUENE_INT64 ;
        le = GrB_VALUELE_INT64 ;
        ge = GrB_VALUEGE_INT64 ;
        lt = GrB_VALUELT_INT64 ;
        gt = GrB_VALUEGT_INT64 ;
        less_than = GrB_LT_INT64 ;
        min_plus = GrB_MIN_PLUS_SEMIRING_INT64 ;
        tcode = 1 ;
    }
    else if (etype == GrB_UINT32)
    {
        GRB_TRY (GrB_Scalar_extractElement (&delta_uint32, Delta)) ;
        GRB_TRY (GrB_assign (t, NULL, NULL, (uint32_t) UINT32_MAX,
            GrB_ALL, n, NULL)) ;
        ne = GrB_VALUENE_UINT32 ;
        le = GrB_VALUELE_UINT32 ;
        ge = GrB_VALUEGE_UINT32 ;
        lt = GrB_VALUELT_UINT32 ;
        gt = GrB_VALUEGT_UINT32 ;
        less_than = GrB_LT_UINT32 ;
        min_plus = GrB_MIN_PLUS_SEMIRING_UINT32 ;
        tcode = 2 ;
        negative_edge_weights = false ;
    }
    else if (etype == GrB_UINT64)
    {
        GRB_TRY (GrB_Scalar_extractElement (&delta_uint64, Delta)) ;
        GRB_TRY (GrB_assign (t, NULL, NULL, (uint64_t) UINT64_MAX,
            GrB_ALL, n, NULL)) ;
        ne = GrB_VALUENE_UINT64 ;
        le = GrB_VALUELE_UINT64 ;
        ge = GrB_VALUEGE_UINT64 ;
        lt = GrB_VALUELT_UINT64 ;
        gt = GrB_VALUEGT_UINT64 ;
        less_than = GrB_LT_UINT64 ;
        min_plus = GrB_MIN_PLUS_SEMIRING_UINT64 ;
        tcode = 3 ;
        negative_edge_weights = false ;
    }
    else if (etype == GrB_FP32)
    {
        GRB_TRY (GrB_Scalar_extractElement (&delta_fp32, Delta)) ;
        GRB_TRY (GrB_assign (t, NULL, NULL, (float) INFINITY,
            GrB_ALL, n, NULL)) ;
        ne = GrB_VALUENE_FP32 ;
        le = GrB_VALUELE_FP32 ;
        ge = GrB_VALUEGE_FP32 ;
        lt = GrB_VALUELT_FP32 ;
        gt = GrB_VALUEGT_FP32 ;
        less_than = GrB_LT_FP32 ;
        min_plus = GrB_MIN_PLUS_SEMIRING_FP32 ;
        tcode = 4 ;
    }
    else if (etype == GrB_FP64)
    {
        GRB_TRY (GrB_Scalar_extractElement (&delta_fp64, Delta)) ;
        GRB_TRY (GrB_assign (t, NULL, NULL, (double) INFINITY,
            GrB_ALL, n, NULL)) ;
        ne = GrB_VALUENE_FP64 ;
        le = GrB_VALUELE_FP64 ;
        ge = GrB_VALUEGE_FP64 ;
        lt = GrB_VALUELT_FP64 ;
        gt = GrB_VALUEGT_FP64 ;
        less_than = GrB_LT_FP64 ;
        min_plus = GrB_MIN_PLUS_SEMIRING_FP64 ;
        tcode = 5 ;
    }
    else
    {
        LG_ASSERT_MSG (false, GrB_NOT_IMPLEMENTED, "type not supported") ;
    }

    // check if the graph might have negative edge weights
    if (negative_edge_weights)
    {
        double emin = -1 ;
        if (G->emin != NULL &&
            (G->emin_state == LAGraph_VALUE ||
             G->emin_state == LAGraph_BOUND))
        {
            GRB_TRY (GrB_Scalar_extractElement_FP64 (&emin, G->emin)) ;
        }
//      else
//      {
//          // a future Basic algorithm should compute G->emin and perhaps
//          // G->emax, then compute Delta automatically.
//      }
        negative_edge_weights = (emin < 0) ;
    }

    // t (src) = 0
    GRB_TRY (GrB_Vector_setElement (t, 0, source)) ;

    // reach (src) = true
    GRB_TRY (GrB_Vector_setElement (reach, true, source)) ;

    // s (src) = true
    GRB_TRY (GrB_Vector_setElement (s, true, source)) ;

    // AL = A .* (A <= Delta)
    GRB_TRY (GrB_Matrix_new (&AL, etype, n, n)) ;
    GRB_TRY (GrB_select (AL, NULL, NULL, le, A, Delta, NULL)) ;
    GRB_TRY (GrB_wait (AL, GrB_MATERIALIZE)) ;

    // FUTURE: costly for some problems, taking up to 50% of the total time:
    // AH = A .* (A > Delta)
    GRB_TRY (GrB_Matrix_new (&AH, etype, n, n)) ;
    GRB_TRY (GrB_select (AH, NULL, NULL, gt, A, Delta, NULL)) ;
    GRB_TRY (GrB_wait (AH, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // while (t >= step*Delta) not empty
    //--------------------------------------------------------------------------

    for (int64_t step = 0 ; ; step++)
    {

        //----------------------------------------------------------------------
        // tmasked = all entries in t<reach> that are less than (step+1)*Delta
        //----------------------------------------------------------------------

        setelement (uBound, (step+1)) ;        // uBound = (step+1) * Delta
        GRB_TRY (GrB_Vector_clear (tmasked)) ;

        // tmasked<reach> = t
        // FUTURE: this is costly, typically using Method 06s in SuiteSparse,
        // which is a very general-purpose one.  Write a specialized kernel to
        // exploit the fact that reach and t are bitmap and tmasked starts
        // empty, or fuse this assignment with the GrB_select below.
        GRB_TRY (GrB_assign (tmasked, reach, NULL, t, GrB_ALL, n, NULL)) ;
        // tmasked = select (tmasked < (step+1)*Delta)
        GRB_TRY (GrB_select (tmasked, NULL, NULL, lt, tmasked, uBound, NULL)) ;
        // --- alternative:
        // FUTURE this is slower than the above but should be much faster.
        // GrB_select is computing a bitmap result then converting it to
        // sparse.  t and reach are both bitmap and tmasked finally sparse.
        // tmasked<reach> = select (t < (step+1)*Delta)
        // GRB_TRY (GrB_select (tmasked, reach, NULL, lt, t, uBound, NULL)) ;

        GrB_Index tmasked_nvals ;
        GRB_TRY (GrB_Vector_nvals (&tmasked_nvals, tmasked)) ;

        //----------------------------------------------------------------------
        // continue while the current bucket (tmasked) is not empty
        //----------------------------------------------------------------------

        while (tmasked_nvals > 0)
        {
            // tReq = AL'*tmasked using the min_plus semiring
            GRB_TRY (GrB_vxm (tReq, NULL, NULL, min_plus, tmasked, AL, NULL)) ;

            // s<struct(tmasked)> = true
            GRB_TRY (GrB_assign (s, tmasked, NULL, (bool) true, GrB_ALL, n,
                GrB_DESC_S)) ;

            // if nvals (tReq) is 0, no need to continue the rest of this loop
            GrB_Index tReq_nvals ;
            GRB_TRY (GrB_Vector_nvals (&tReq_nvals, tReq)) ;
            if (tReq_nvals == 0) break ;

            // tless = (tReq .< t) using set intersection
            GRB_TRY (GrB_eWiseMult (tless, NULL, NULL, less_than, tReq, t,
                NULL)) ;

            // remove explicit zeros from tless so it can be used as a
            // structural mask
            GrB_Index tless_nvals ;
            GRB_TRY (GrB_select (tless, NULL, NULL, ne, tless, 0, NULL)) ;
            GRB_TRY (GrB_Vector_nvals (&tless_nvals, tless)) ;
            if (tless_nvals == 0) break ;

            // update reachable node list/mask
            // reach<struct(tless)> = true
            GRB_TRY (GrB_assign (reach, tless, NULL, (bool) true, GrB_ALL, n,
                GrB_DESC_S)) ;

            // tmasked<struct(tless)> = select (tReq < (step+1)*Delta)
            GRB_TRY (GrB_Vector_clear (tmasked)) ;
            GRB_TRY (GrB_select (tmasked, tless, NULL, lt, tReq, uBound,
                GrB_DESC_S)) ;

            // For general graph with some negative weights:
            if (negative_edge_weights)
            {
                // If all entries of the graph are known to be positive, and
                // the entries of tmasked are at least step*Delta, tReq =
                // tmasked min.+ AL must be >= step*Delta.  Therefore, there is
                // no need to perform this GrB_select with ge to find tmasked
                // >= step*Delta from tReq.
                setelement (lBound, (step)) ;  // lBound = step*Delta
                // tmasked = select entries in tmasked that are >= step*Delta
                GRB_TRY (GrB_select (tmasked, NULL, NULL, ge, tmasked, lBound,
                    NULL)) ;
            }

            // t<struct(tless)> = tReq
            GRB_TRY (GrB_assign (t, tless, NULL, tReq, GrB_ALL, n, GrB_DESC_S));
            GRB_TRY (GrB_Vector_nvals (&tmasked_nvals, tmasked)) ;
        }

        // tmasked<s> = t
        GRB_TRY (GrB_Vector_clear (tmasked)) ;
        GRB_TRY (GrB_assign (tmasked, s, NULL, t, GrB_ALL, n, GrB_DESC_S)) ;

        // tReq = AH'*tmasked using the min_plus semiring
        GRB_TRY (GrB_vxm (tReq, NULL, NULL, min_plus, tmasked, AH, NULL)) ;

        // tless = (tReq .< t) using set intersection
        GRB_TRY (GrB_eWiseMult (tless, NULL, NULL, less_than, tReq, t, NULL)) ;

        // t<tless> = tReq, which computes t = min (t, tReq)
        GRB_TRY (GrB_assign (t, tless, NULL, tReq, GrB_ALL, n, NULL)) ;

        //----------------------------------------------------------------------
        // find out how many left to be computed
        //----------------------------------------------------------------------

        // update reachable node list
        // reach<tless> = true
        GRB_TRY (GrB_assign (reach, tless, NULL, (bool) true, GrB_ALL, n,
            NULL)) ;

        // remove previous buckets
        // reach<struct(s)> = Empty
        GRB_TRY (GrB_assign (reach, s, NULL, Empty, GrB_ALL, n, GrB_DESC_S)) ;
        GrB_Index nreach ;
        GRB_TRY (GrB_Vector_nvals (&nreach, reach)) ;
        if (nreach == 0) break ;

        GRB_TRY (GrB_Vector_clear (s)) ; // clear s for the next iteration
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*path_length) = t ;
    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
