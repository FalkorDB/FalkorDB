//------------------------------------------------------------------------------
// LAGraph_FastGraphletTransform: fast graphlet transform
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Tanner Hoke, Texas A&M University, ...

//------------------------------------------------------------------------------

// LAGraph_FastGraphletTransform: computes the Fast Graphlet Transform of
// an undirected graph.  No self edges are allowed on the input graph.

// fixme: rename this

// https://arxiv.org/pdf/2007.11111.pdf

//------------------------------------------------------------------------------
#define F_UNARY(f)  ((void (*)(void *, const void *)) f)

#define LG_FREE_WORK                \
{                                   \
    GrB_free (&C_3) ;               \
    GrB_free (&d_0) ;               \
    GrB_free (&d_1) ;               \
    GrB_free (&d_2) ;               \
    GrB_free (&d_3) ;               \
    GrB_free (&d_4) ;               \
    GrB_free (&d_5) ;               \
    GrB_free (&d_6) ;               \
    GrB_free (&d_7) ;               \
    GrB_free (&d_8) ;               \
    GrB_free (&d_9) ;               \
    GrB_free (&d_10) ;              \
    GrB_free (&d_11) ;              \
    GrB_free (&d_12) ;              \
    GrB_free (&d_13) ;              \
    GrB_free (&d_14) ;              \
    GrB_free (&d_15) ;              \
    GrB_free (&d_2) ;               \
    GrB_free (&v) ;                 \
    GrB_free (&p_1_minus_one) ;     \
    GrB_free (&p_1_minus_two) ;     \
    GrB_free (&two_c_3) ;           \
    GrB_free (&p_1_p_1_had) ;       \
    GrB_free (&C_42) ;              \
    GrB_free (&P_2) ;               \
    GrB_free (&D_1) ;               \
    GrB_free (&D_4c) ;              \
    GrB_free (&D_43) ;              \
    GrB_free (&U_inv) ;             \
    GrB_free (&F_raw) ;             \
    GrB_free (&C_4) ;               \
    GrB_free (&Sub_one_mult) ;      \
    GrB_free (&T) ;                 \
    if (A_Tiles != NULL)                                                \
    {                                                                   \
        for (int i = 0; i < tile_cnt; ++i) GrB_free (&A_Tiles [i]) ;    \
    }                                                                   \
    if (D_Tiles != NULL)                                                \
    {                                                                   \
        for (int i = 0; i < tile_cnt; ++i) GrB_free (&D_Tiles [i]) ;    \
    }                                                                   \
    if (C_Tiles != NULL)                                                \
    {                                                                   \
        for (int i = 0; i < tile_cnt; ++i) GrB_free (&C_Tiles [i]) ;    \
    }                                                                   \
}

#define LG_FREE_ALL                 \
{                                   \
    LG_FREE_WORK ;                  \
}

#define F_UNARY(f)  ((void (*)(void *, const void *)) f)

#include "LG_internal.h"
#include "LAGraphX.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void sub_one_mult (int64_t *z, const int64_t *x) { (*z) = (*x) * ((*x)-1) ; }

int LAGraph_FastGraphletTransform
(
    // outputs:
    GrB_Matrix *F_net,  // 16-by-n matrix of graphlet counts
    // inputs:
    LAGraph_Graph G,
    bool compute_d_15,  // probably this makes most sense
    char *msg
)
{
    LG_CLEAR_MSG ;
    GrB_Index const U_inv_I[] = {0, 1, 2, 2, 3, 3, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 14, 14, 15} ;
    GrB_Index const U_inv_J[] = {0, 1, 2, 4, 3, 4, 4, 5, 9, 10, 12, 13, 14, 15, 6, 10, 11, 12, 13, 14, 15, 7, 9, 10, 13, 14, 15, 8, 11, 14, 15, 9, 13, 15, 10, 13, 14, 15, 11, 14, 15, 12, 13, 14, 15, 13, 15, 14, 15, 15} ;
    int64_t const U_inv_X[] = {1, 1, 1, -2, 1, -1, 1, 1, -2, -1, -2, 4, 2, -6, 1, -1, -2, -2, 2, 4, -6, 1, -1, -1, 2, 1, -3, 1, -1, 1, -1, 1, -2, 3, 1, -2, -2, 6, 1, -2, 3, 1, -1, -1, 3, 1, -3, 1, -3, 1} ;
    GrB_Index const U_inv_nvals = 50;
    GrB_UnaryOp Sub_one_mult = NULL ;
    int tile_cnt = 0 ;
    GrB_Matrix *A_Tiles = NULL ;
    GrB_Matrix *D_Tiles = NULL ;
    GrB_Matrix *C_Tiles = NULL ;
    GrB_Index *Tile_nrows = NULL ;
    GrB_Matrix T = NULL ;

    GrB_Matrix C_3 = NULL,
	       A = NULL,
	       C_42 = NULL,
	       P_2 = NULL,
	       D_1 = NULL,
	       D_4c = NULL,
	       D_43 = NULL,
	       U_inv = NULL,
	       F_raw = NULL,
               C_4 = NULL ;

    GrB_Vector d_0 = NULL,
	       d_1 = NULL,
	       d_2 = NULL,
	       d_3 = NULL,
	       d_4 = NULL,
	       d_5 = NULL,
	       d_6 = NULL,
	       d_7 = NULL,
	       d_8 = NULL,
	       d_9 = NULL,
	       d_10 = NULL,
	       d_11 = NULL,
	       d_12 = NULL,
               d_13 = NULL,
               d_14 = NULL,
               d_15 = NULL;

    GrB_Vector v = NULL,
               two_c_3 = NULL,
	       p_1_minus_one = NULL,
	       p_1_minus_two = NULL,
	       p_1_p_1_had = NULL ;

    GrB_Index nvals ;
    int64_t ntri ;

#if !LAGRAPH_SUITESPARSE
    LG_ASSERT (false, GrB_NOT_IMPLEMENTED) ;
#else

    A = G->A ;

    GrB_Index n ;
    GRB_TRY (GrB_Matrix_nrows (&n, A)) ;

    //--------------------------------------------------------------------------
    // compute d_0 = e
    //--------------------------------------------------------------------------

    // d_0 = e
    GRB_TRY (GrB_Vector_new (&d_0, GrB_INT64, n)) ;
    GRB_TRY (GrB_assign (d_0, NULL, NULL, 1, GrB_ALL, n, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_1 = Ae (in_degree)
    //--------------------------------------------------------------------------

//  GRB_TRY (GrB_Vector_new (&d_1, GrB_INT64, n)) ;

    // d_1 = Ae (in_degree)
    LG_TRY (LAGraph_Cached_OutDegree (G, msg)) ;

    GRB_TRY (GrB_Vector_dup (&d_1, G->out_degree)) ;

    //--------------------------------------------------------------------------
    // compute d_2 = p_2
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (&d_2, GrB_INT64, n)) ;

    // d_2 = p_2 = A*p_1 - c_2 = A*d_1 - d_1
    GRB_TRY (GrB_mxv (d_2, NULL, NULL, GxB_PLUS_SECOND_INT64, A, d_1, NULL)) ;
    GRB_TRY (GrB_eWiseMult (d_2, NULL, NULL, GrB_MINUS_INT64, d_2, d_1, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_3 = hadamard(p_1, p_1 - 1) / 2
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (&d_3, GrB_INT64, n)) ;

    GRB_TRY (GrB_UnaryOp_new (&Sub_one_mult, F_UNARY (sub_one_mult), GrB_INT64, GrB_INT64)) ;

    GRB_TRY (GrB_apply (d_3, NULL, NULL, Sub_one_mult, d_1, NULL)) ;
    GRB_TRY (GrB_apply (d_3, NULL, NULL, GrB_DIV_INT64, d_3, (int64_t) 2, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_4 = C_3e/2
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_new (&C_3, GrB_INT64, n, n)) ;
    GRB_TRY (GrB_Vector_new (&d_4, GrB_INT64, n)) ;

    // C_3 = hadamard(A, A^2)
    GRB_TRY (GrB_mxm (C_3, A, NULL, GxB_PLUS_FIRST_INT64, A, A, GrB_DESC_ST1)) ;

    // d_4 = c_3 = C_3e/2
    GRB_TRY (GrB_reduce (d_4, NULL, NULL, GrB_PLUS_MONOID_INT64, C_3, NULL)) ;
    GRB_TRY (GrB_apply (d_4, NULL, NULL, GrB_DIV_INT64, d_4, (int64_t) 2, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_5 = p_3 = A*d_2 - hadamard(p_1, p_1 - 1) - 2c_3
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (&v, GrB_INT64, n)) ;
    GRB_TRY (GrB_Vector_new (&two_c_3, GrB_INT64, n)) ;
    GRB_TRY (GrB_Vector_new (&d_5, GrB_INT64, n)) ;

    // v = hadamard(p_1, p_1 - 1)
    GRB_TRY (GrB_apply (v, NULL, NULL, Sub_one_mult, d_1, NULL)) ;

    // two_c_3 = 2 * c_3 = 2 * d_4
    GRB_TRY (GrB_apply (two_c_3, NULL, NULL, GrB_TIMES_INT64, 2, d_4, NULL)) ;

    // d_5 = A * d_2
    GRB_TRY (GrB_mxv (d_5, NULL, NULL, GxB_PLUS_SECOND_INT64, A, d_2, NULL)) ;

    // d_5 -= hadamard(p_1, p_1 - 1)
    GRB_TRY (GrB_eWiseAdd (d_5, NULL, NULL, GrB_MINUS_INT64, d_5, v, NULL)) ;

    // d_5 -= two_c_3
    GRB_TRY (GrB_eWiseAdd (d_5, NULL, NULL, GrB_MINUS_INT64, d_5, two_c_3, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_6 = hadamard(d_2, p_1-1) - 2c_3
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (&p_1_minus_one, GrB_INT64, n)) ;
    GRB_TRY (GrB_Vector_new (&d_6, GrB_INT64, n)) ;

    // p_1_minus_one = p_1 - 1
    GRB_TRY (GrB_apply (p_1_minus_one, NULL, NULL, GrB_MINUS_INT64, d_1, (int64_t) 1, NULL)) ;

    // d_6 = hadamard(d_2, p_1-1)
    GRB_TRY (GrB_eWiseMult (d_6, NULL, NULL, GrB_TIMES_INT64, d_2, p_1_minus_one, NULL)) ;

    // d_6 -= 2c_3
    GRB_TRY (GrB_eWiseAdd (d_6, NULL, NULL, GrB_MINUS_INT64, d_6, two_c_3, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_7 = A*hadamard(p_1-1, p_1-2) / 2
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (&p_1_minus_two, GrB_INT64, n)) ;
    GRB_TRY (GrB_Vector_new (&p_1_p_1_had, GrB_INT64, n)) ;
    GRB_TRY (GrB_Vector_new (&d_7, GrB_INT64, n)) ;

    GRB_TRY (GrB_apply (p_1_minus_two, NULL, NULL, GrB_MINUS_INT64, d_1, (int64_t) 2, NULL)) ;
    GRB_TRY (GrB_eWiseMult (p_1_p_1_had, NULL, NULL, GrB_TIMES_INT64, p_1_minus_one, p_1_minus_two, NULL)) ;

    GRB_TRY (GrB_mxv (d_7, NULL, NULL, GxB_PLUS_SECOND_INT64, A, p_1_p_1_had, NULL)) ;
    GRB_TRY (GrB_apply (d_7, NULL, NULL, GrB_DIV_INT64, d_7, (int64_t) 2, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_8 = hadamard(p_1, p_1_p_1_had) / 6
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (&d_8, GrB_INT64, n)) ;

    GRB_TRY (GrB_eWiseMult (d_8, NULL, NULL, GrB_TIMES_INT64, d_1, p_1_p_1_had, NULL)) ;
    GRB_TRY (GrB_apply (d_8, NULL, NULL, GrB_DIV_INT64, d_8, (int64_t) 6, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_9 = A*c_3 - 2*c_3
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (&d_9, GrB_INT64, n)) ;

    GRB_TRY (GrB_mxv (d_9, NULL, NULL, GxB_PLUS_SECOND_INT64, A, d_4, NULL)) ;
    GRB_TRY (GrB_eWiseAdd (d_9, NULL, NULL, GrB_MINUS_INT64, d_9, two_c_3, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_10 = C_3 * (p_1 - 2)
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (&d_10, GrB_INT64, n)) ;

    GRB_TRY (GrB_mxv (d_10, NULL, NULL, GxB_PLUS_TIMES_INT64, C_3, p_1_minus_two, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_11 = hadamard(p_1 - 2, c_3)
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (&d_11, GrB_INT64, n)) ;

    GRB_TRY (GrB_eWiseMult (d_11, NULL, NULL, GrB_TIMES_INT64, p_1_minus_two, d_4, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_12 = c_4 = C_{4,2}e/2
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_new (&C_4, GrB_INT64, n, 1)) ;
    GRB_TRY (GrB_Matrix_new (&D_1, GrB_INT64, n, n)) ;

    GRB_TRY (GrB_Vector_new (&d_12, GrB_INT64, n)) ;

    // D_1 = diag(d_1)
    GRB_TRY (GxB_Matrix_diag (D_1, d_1, (int64_t) 0, NULL)) ;

    GRB_TRY (GrB_Matrix_nvals (&nvals, A));

    const GrB_Index entries_per_tile = 1000;
    GrB_Index ntiles = (nvals + entries_per_tile - 1) / entries_per_tile ;
    // FIXME: use LAGraph_Calloc here, and check if out of memory:
    A_Tiles = calloc (ntiles , sizeof (GrB_Matrix)) ;
    D_Tiles = calloc (ntiles , sizeof (GrB_Matrix)) ;
    C_Tiles = calloc (ntiles , sizeof (GrB_Matrix)) ;
    Tile_nrows = calloc (ntiles , sizeof (GrB_Index)) ;
    GrB_Index Tile_ncols [1] = {n} ;

    int64_t tot_deg = 0 ;
    GrB_Index last_row = -1 ;
    for (GrB_Index i = 0; i < n; ++i) {
        int64_t deg = 0 ;
        GRB_TRY (GrB_Vector_extractElement (&deg, d_1, i)) ;

        if (i == n - 1 || (tot_deg / entries_per_tile != (tot_deg + deg) / entries_per_tile)) {
            Tile_nrows [tile_cnt++] = i - last_row ;
            last_row = i ;
        }

        tot_deg += deg ;
    }

    GRB_TRY (GxB_Matrix_split (A_Tiles, tile_cnt, 1, Tile_nrows, Tile_ncols, A, NULL)) ;
    GRB_TRY (GxB_Matrix_split (D_Tiles, tile_cnt, 1, Tile_nrows, Tile_ncols, D_1, NULL)) ;

#define TRY(method)                         \
    {                                       \
        GrB_Info info2 = method ;           \
        if (info2 != GrB_SUCCESS)           \
        {                                   \
            GrB_free (&A_i) ;               \
            GrB_free (&C_Tiles [i_tile]) ;  \
            GrB_free (&e) ;                 \
            info1 = info2 ;                 \
            continue ;                      \
        }                                   \
    }

//  GxB_set (GxB_NTHREADS, 1) ;
    int save_nthreads_outer, save_nthreads_inner ;
    LG_TRY (LAGraph_GetNumThreads (&save_nthreads_outer, &save_nthreads_inner, msg)) ;
    LG_TRY (LAGraph_SetNumThreads (1, 1, msg)) ;

    int i_tile;
    GrB_Info info1 = GrB_SUCCESS ;
    #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic,1)
    for (i_tile = 0; i_tile < tile_cnt; ++i_tile) {
        GrB_Matrix A_i = NULL, e = NULL ;

        TRY (GrB_Matrix_new (&e, GrB_INT64, n, 1)) ;
        TRY (GrB_assign (e, NULL, NULL, (int64_t) 1, GrB_ALL, n, GrB_ALL, 1, NULL)) ;

        TRY (GrB_Matrix_new (&A_i, GrB_INT64, Tile_nrows [i_tile], n)) ;
        TRY (GrB_Matrix_new (&C_Tiles [i_tile], GrB_INT64, Tile_nrows [i_tile], 1)) ;

        TRY (GrB_mxm (A_i, NULL, NULL, GxB_PLUS_PAIR_INT64, A_Tiles [i_tile], A, NULL)) ;
        TRY (GrB_eWiseAdd (A_i, NULL, NULL, GrB_MINUS_INT64, A_i, D_Tiles [i_tile], NULL)) ;
        TRY (GrB_apply (A_i, NULL, NULL, Sub_one_mult, A_i, NULL)) ;

        // multiply A_i by it on the right
        TRY (GrB_mxm (C_Tiles [i_tile], NULL, NULL, GxB_PLUS_FIRST_INT64, A_i, e, NULL)) ;

        GrB_free (&A_i) ;
        GrB_free (&e) ;
    }

    GRB_TRY (info1) ;

//  GxB_set (GxB_NTHREADS, omp_get_max_threads()) ;
    LG_TRY (LAGraph_SetNumThreads (save_nthreads_outer, save_nthreads_inner, msg)) ;

    GRB_TRY (GxB_Matrix_concat (C_4, C_Tiles, tile_cnt, 1, NULL)) ;

    // d_12 = C_4
    GRB_TRY (GrB_reduce (d_12, NULL, NULL, GrB_PLUS_MONOID_INT64, C_4, NULL)) ;
    GRB_TRY (GrB_apply (d_12, NULL, NULL, GrB_DIV_INT64, d_12, 2, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_13 = D_{4,c}e/2
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_new (&D_4c, GrB_INT64, n, n)) ;
    GRB_TRY (GrB_Vector_new (&d_13, GrB_INT64, n)) ;

    GRB_TRY (GrB_eWiseMult (D_4c, NULL, NULL, GrB_MINUS_INT64, C_3, A, NULL)) ; // can be mult because we mask with A next
    GRB_TRY (GrB_mxm (D_4c, A, NULL, GxB_PLUS_SECOND_INT64, A, D_4c, GrB_DESC_S)) ;

    // d_13  = D_{4,c}*e/2
    GRB_TRY (GrB_reduce (d_13, NULL, NULL, GrB_PLUS_INT64, D_4c, NULL)) ;
    GRB_TRY (GrB_apply (d_13, NULL, NULL, GrB_DIV_INT64, d_13, (int64_t) 2, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_14 = D_{4,3}e/2 = hadamard(A, C_42)e/2
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_new (&D_43, GrB_INT64, n, n)) ;
    GRB_TRY (GrB_Vector_new (&d_14, GrB_INT64, n)) ;
    GRB_TRY (GrB_Matrix_new (&C_42, GrB_INT64, n, n)) ;
    GRB_TRY (GrB_Matrix_new (&P_2, GrB_INT64, n, n)) ;

    // P_2 = A*A - diag(d_1)
    GRB_TRY (GrB_eWiseAdd (P_2, A, NULL, GrB_MINUS_INT64, C_3, D_1, NULL)) ;

    // C_42 = hadamard(P_2, P_2 - 1)
    GRB_TRY (GrB_apply (C_42, A, NULL, Sub_one_mult, P_2, NULL)) ;

    GRB_TRY (GrB_eWiseMult (D_43, NULL, NULL, GrB_TIMES_INT64, A, C_42, NULL)) ;

    // d_14  = D_{4,3}*e/2
    GRB_TRY (GrB_reduce (d_14, NULL, NULL, GrB_PLUS_INT64, D_43, NULL)) ;
    GRB_TRY (GrB_apply (d_14, NULL, NULL, GrB_DIV_INT64, d_14, (int64_t) 2, NULL)) ;

    //--------------------------------------------------------------------------
    // compute d_15 = Te/6
    //--------------------------------------------------------------------------

    if (compute_d_15) {
        LG_TRY (LAGraph_KTruss (&T, G, 4, msg)) ;
        GRB_TRY (GrB_Vector_new (&d_15, GrB_INT64, n)) ;

        int nthreads = 1 ;
        // todo: parallelize this...
//#pragma omp parallel for num_threads(nthreads)
        //for (int tid = 0 ; tid < nthreads ; tid++)
        {
            GrB_Index *neighbors = (GrB_Index*) malloc(n * sizeof(GrB_Index));
            GrB_Index *k4cmn = (GrB_Index*) malloc(n * sizeof(GrB_Index));
            int64_t *f15 = (int64_t*) malloc(n * sizeof(int64_t));
            GrB_Index *I = (GrB_Index *) malloc(n * sizeof(GrB_Index));
            int *isNeighbor = (int*) malloc(n * sizeof(int));
            for (int i = 0; i < n; ++i) {
                neighbors [i] = k4cmn [i] = f15 [i] = isNeighbor [i] = 0 ;
                I [i] = i ;
            }


            // thread tid operates on T(row1:row2-1,:)
            GrB_Index row1 = 0;//tid * (n / nthreads) ;
            GrB_Index row2 = n;//(tid == nthreads - 1) ? n : ((tid+1) * (n / nthreads)) ;

            GxB_Iterator riterator ;
            GxB_Iterator_new (&riterator) ;
            GRB_TRY (GxB_rowIterator_attach (riterator, T, NULL)) ;

            GxB_Iterator iterator ;
            GxB_Iterator_new (&iterator) ;
            GRB_TRY (GxB_rowIterator_attach (iterator, T, NULL)) ;

            // seek to T(row1,:)
            GrB_Info info = GxB_rowIterator_seekRow (iterator, row1) ;
            while (info != GxB_EXHAUSTED)
            {
                // iterate over entries in T(i,:)
                GrB_Index idx2 = GxB_rowIterator_getRowIndex (iterator) ;
                if (idx2 >= row2) break ;
                int neighbor_cnt = 0 ;
                while (info == GrB_SUCCESS)
                {
                    // working with edge (idx2, j)
                    GrB_Index j = GxB_rowIterator_getColIndex (iterator) ;

                    if (j > idx2) {
                        neighbors [neighbor_cnt++] = j ;
                        isNeighbor [j] = 1 ;
                    }

                    info = GxB_rowIterator_nextCol (iterator) ;
                }

                for (int neighbor_id = 0 ; neighbor_id < neighbor_cnt ; ++neighbor_id) {
                    GrB_Index j = neighbors [neighbor_id] ;
                    int cmn_cnt = 0 ;
                    info = GxB_rowIterator_seekRow(riterator, j) ;

                    while (info == GrB_SUCCESS) { // iterate over neighbors of j
                        GrB_Index k = GxB_rowIterator_getColIndex (riterator) ;
                        if (k > j && isNeighbor [k]) {
                            k4cmn [cmn_cnt++] = k ;
                            isNeighbor [k] = -1 ;
                        }
                        info = GxB_rowIterator_nextCol (riterator) ;
                    }
                    // check every combination
                    for (int k_1 = 0 ; k_1 < cmn_cnt ; k_1++) {
                        GrB_Index k = k4cmn [k_1] ;
                        info = GxB_rowIterator_seekRow(riterator, k) ;

                        while (info == GrB_SUCCESS) { // iterate over neighbors of k
                            GrB_Index l = GxB_rowIterator_getColIndex (riterator) ;
                            if (l > k && isNeighbor [l] == -1) {
                                f15[idx2]++ ;
                                f15[j]++ ;
                                f15[k]++ ;
                                f15[l]++ ;
                            }
                            info = GxB_rowIterator_nextCol (riterator) ;
                        }
                    }
                    for (int k_1 = 0 ; k_1 < cmn_cnt ; k_1++) {
                        isNeighbor[k4cmn[k_1]] = 1 ;
                    }
                }

                for (int neighbor_id = 0 ; neighbor_id < neighbor_cnt ; ++neighbor_id) {
                    GrB_Index j = neighbors [neighbor_id] ;
                    isNeighbor [j] = 0 ;
                }

                // move to the next row, T(i+1,:)
                info = GxB_rowIterator_nextRow (iterator) ;
            }
            GrB_free (&iterator) ;
            GrB_free (&riterator) ;
            GrB_free (&T) ;
            GRB_TRY (GrB_Vector_build (d_15, I, f15, n, NULL)) ;

            free (neighbors) ;
            free (k4cmn) ;
            free (f15) ;
            free (I) ;
            free (isNeighbor) ;
        }

    }

    //--------------------------------------------------------------------------
    // construct raw frequencies matrix F_raw
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_new (&F_raw, GrB_INT64, 16, n)) ;

    GrB_Vector d[16] = {d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11, d_12, d_13, d_14, d_15} ;

    for (int i = 0; i < 15 + (compute_d_15 ? 1 : 0); ++i) {
        GRB_TRY (GrB_Vector_nvals (&nvals, d[i]));

        GrB_Index *J = (GrB_Index*) malloc (nvals*sizeof(GrB_Index)) ;
        int64_t *vals = (int64_t*) malloc (nvals*sizeof(int64_t)) ;

        GRB_TRY (GrB_Vector_extractTuples (J, vals, &nvals, d[i])) ;
        for (int j = 0; j < nvals; ++j) {
            GRB_TRY (GrB_Matrix_setElement (F_raw, vals[j], i, J[j])) ;
        }

        free (J) ;
        free (vals) ;
    }

    //--------------------------------------------------------------------------
    // construct U_inv
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_new (&U_inv, GrB_INT64, 16, 16)) ;

    GRB_TRY (GrB_Matrix_build (U_inv, U_inv_I, U_inv_J, U_inv_X, U_inv_nvals, GrB_PLUS_INT64)) ;
    //GRB_TRY (GxB_print (U_inv, 3)) ;

    //--------------------------------------------------------------------------
    // construct net frequencies matrix F_net
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_new (F_net, GrB_INT64, 16, n)) ;

    GRB_TRY (GrB_mxm (*F_net, NULL, NULL, GxB_PLUS_TIMES_INT64, U_inv, F_raw, NULL)) ;

    GrB_Vector f_net = NULL ;
    GRB_TRY (GrB_Vector_new (&f_net, GrB_INT64, 16)) ;
    GRB_TRY (GrB_reduce (f_net, NULL, NULL, GrB_PLUS_INT64, *F_net, NULL)) ;
    GRB_TRY (GxB_print (f_net, 3)) ;
    GRB_TRY (GrB_free (&f_net)) ;
    //GRB_TRY (GxB_print (*F_net, 3)) ;

    //--------------------------------------------------------------------------
    // free work
    //--------------------------------------------------------------------------

    LG_FREE_WORK ;
    free ((void *) A_Tiles) ;       A_Tiles = NULL ;
    free ((void *) D_Tiles) ;       D_Tiles = NULL ;
    free ((void *) C_Tiles) ;       C_Tiles = NULL ;
    free ((void *) Tile_nrows) ;    Tile_nrows = NULL ;

    return (0) ;
#endif
}
