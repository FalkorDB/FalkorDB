//------------------------------------------------------------------------------
// LAGraph_AllKCore: Full K-core Decomposition Using the GraphBLAS API
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Pranav Konduri, Texas A&M University

//------------------------------------------------------------------------------

#define LG_FREE_WORK                \
{                                   \
    GrB_free (&deg) ;               \
    GrB_free (&q) ;                 \
    GrB_free (&delta) ;             \
    GrB_free (&done) ;              \
}

#define LG_FREE_ALL                 \
{                                   \
    LG_FREE_WORK                    \
    GrB_free (decomp) ;             \
}

#include "LG_internal.h"


int LAGraph_KCore_All
(
    // outputs:
    GrB_Vector *decomp,     // kcore decomposition
    uint64_t *kmax,
    // inputs:
    LAGraph_Graph G,            // input graph
    char *msg
)
{
    LG_CLEAR_MSG ;

    // declare items
    GrB_Matrix A = NULL;
    GrB_Vector deg = NULL, q = NULL, done = NULL, delta = NULL;

    LG_ASSERT (decomp != NULL, GrB_NULL_POINTER) ;
    (*decomp) = NULL ;

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

    //create work scalars
    uint64_t level = 0; //don't set at 1 in case of empty graph getting returned as kmax = 1
    GrB_Index n, todo, nvals, maxDeg ;
    GRB_TRY (GrB_Matrix_nrows(&n, A)) ;

    //create deg vector using out_degree property
    LG_TRY (LAGraph_Cached_OutDegree(G, msg)) ;

    GRB_TRY (GrB_Vector_dup(&deg, G->out_degree)) ; //original deg vector is technically 1-core since 0 is omitted
    GRB_TRY (GrB_Vector_nvals(&todo, deg)) ; //use todo instead of n since some values are omitted (1-core)

    //retrieve the max degree level of the graph
    GRB_TRY (GrB_reduce(&maxDeg, GrB_NULL, GrB_MAX_MONOID_INT64, G->out_degree, GrB_NULL)) ;

    //select int type for work vectors and semirings
    GrB_Type int_type  = (maxDeg > INT32_MAX) ? GrB_INT64 : GrB_INT32 ;

    GRB_TRY (GrB_Vector_new(&q, int_type, n));
    GRB_TRY (GrB_Vector_new(&done, GrB_BOOL, n)) ;
    GRB_TRY (GrB_Vector_new(&delta, int_type, n)) ;
    //set output to int64
    GRB_TRY (GrB_Vector_new(decomp, int_type, n)) ;

    //change deg vector to int32 if needed
    if(int_type == GrB_INT32){
        GrB_free (&deg) ;
        GRB_TRY (GrB_Vector_new(&deg, int_type, n)) ;
        GRB_TRY (GrB_assign (deg, G->out_degree, NULL, G->out_degree, GrB_ALL, n, NULL)) ;
    }

    // determine semiring types
    GrB_IndexUnaryOp valueEQ = (maxDeg > INT32_MAX) ? GrB_VALUEEQ_INT64 : GrB_VALUEEQ_INT32 ;
    GrB_IndexUnaryOp valueLE = (maxDeg > INT32_MAX) ? GrB_VALUELE_INT64 : GrB_VALUELE_INT32 ;
    GrB_BinaryOp minus_op = (maxDeg > INT32_MAX) ? GrB_MINUS_INT64 : GrB_MINUS_INT32 ;
    GrB_Semiring semiring = (maxDeg > INT32_MAX) ? LAGraph_plus_one_int64 : LAGraph_plus_one_int32 ;

#if LG_SUITESPARSE
    GRB_TRY (GxB_set (done, GxB_SPARSITY_CONTROL, GxB_BITMAP + GxB_FULL)) ;    // try this
    //GRB_TRY (GxB_set (*decomp, GxB_SPARSITY_CONTROL, GxB_BITMAP + GxB_FULL)) ; // try this ... likely not needed...
#endif

    //printf ("\n================================== COMPUTING GrB_KCORE: ==================================\n") ;
    while(todo > 0){
        //printf("Level: %ld, todo: %ld\n", level, todo) ;
        level++;
        // Creating q
        GRB_TRY (GrB_select (q, GrB_NULL, GrB_NULL, valueEQ, deg, level, GrB_NULL)) ; // get all nodes with degree = level
        GRB_TRY (GrB_Vector_nvals(&nvals, q));

        //Assign values of deg into decomp (output)
        GRB_TRY (GrB_assign (*decomp, deg, NULL, level, GrB_ALL, n, GrB_NULL)) ;

        int round = 0;
        // while q not empty
        while(nvals > 0){
            // Decrease todo by number of nvals
            todo = todo - nvals ;
            //add anything in q as true into the done list
            GRB_TRY (GrB_assign (done, q, NULL, (bool) true, GrB_ALL, n, GrB_DESC_S)) ; //structure to take care of 0-node cases

            // Create delta (the nodes who lost friends, and how many they lost)
            GRB_TRY (GrB_vxm (delta, GrB_NULL, GrB_NULL, semiring, q, A, GrB_NULL));

            // Create new deg vector (keep anything not in done vector w/ replace command)
            GRB_TRY (GrB_eWiseAdd(deg, done, GrB_NULL, minus_op, deg, delta, GrB_DESC_RSC /* try GrB_DESC_RSC */)) ;

            // Update q, set new nvals
            GRB_TRY (GrB_select (q, GrB_NULL, GrB_NULL, valueLE, deg, level, GrB_NULL)) ;

            GRB_TRY (GrB_Vector_nvals(&nvals, q)) ;
            round++;
        }
    }
    //set kmax
    (*kmax) = level;

    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
