//------------------------------------------------------------------------------
// LAGr_HarmonicCentrality: Estimate the harmonic centrality for all nodes
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Gabriel A. Gomez, FalkorDB
//------------------------------------------------------------------------------

// The LAGr_HarmonicCentrality algorithm calculates an estimate of the "harmonic
// centrality" of all nodes in a graph. Harmonic Centrality is defined on a non-
// weighted graph as follows:
// HC(u) = \sum_{v \ne u} 1/d(u,v)
// where d(u,v) is the shortest path distance from u to v.
//
// HyperLogLog allows us to estimate the cardinality of the subsets which each
// node can reach at a given level, while using much less memory than a proper
// all pairs shortest paths approach.

// HLL (HyperLogLog) License:
// Copyright (c) 2015 Artem Zaytsev <arepo@nologin.ru>
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

#include "GraphBLAS.h"
#include "LAGraph.h"

#define LG_XSTR(x) LG_STR(x)
#define LG_STR(x) #x
#define LG_JIT_STRING(f, name) static const char* name = LG_XSTR(f); f

#ifndef GRB_CATCH
#define GRB_CATCH(info)                                                 \
{                                                                       \
    LG_FREE_ALL ;                                                       \
    return (info) ;                                                     \
}
#endif

//------------------------------------------------------------------------------
// LAGRAPH_CATCH: catch an error from LAGraph
//------------------------------------------------------------------------------

// A simple LAGRAPH_CATCH macro to be used by LAGRAPH_TRY.  If an LAGraph
// function wants something else, then #define a LAGRAPH_CATCH macro before the
// #include "LG_internal.h" statement.

#ifndef LAGRAPH_CATCH
#define LAGRAPH_CATCH(status)                                           \
{                                                                       \
    LG_FREE_ALL ;                                                       \
    return (status) ;                                                   \
}
#endif

//------------------------------------------------------------------------------
// LG_ASSERT_MSGF: assert an expression is true, and return if it is false
//------------------------------------------------------------------------------

// Identical to LG_ASSERT_MSG, except this allows a printf-style formatted
// message.

#define LG_ASSERT_MSGF(expression,error_status,expression_format,...)   \
{                                                                       \
    if (!(expression))                                                  \
    {                                                                   \
        LG_FREE_ALL ;                                                   \
        return (error_status) ;                                         \
    }                                                                   \
}

//------------------------------------------------------------------------------
// LG_ASSERT_MSG: assert an expression is true, and return if it is false
//------------------------------------------------------------------------------

// Identical to LG_ASSERT, except this allows a different string to be
// included in the message.

#define LG_ASSERT_MSG(expression,error_status,expression_message)       \
    LG_ASSERT_MSGF (expression,error_status,"%s",expression_message)

//------------------------------------------------------------------------------
// LG_ASSERT: assert an expression is true, and return if it is false
//------------------------------------------------------------------------------

// LAGraph methods can use this assertion macro for simple errors.

#define LG_ASSERT(expression, error_status)                                 \
{                                                                           \
    if (!(expression))                                                      \
    {                                                                       \
        LG_FREE_ALL ;                                                       \
        return (error_status) ;                                             \
    }                                                                       \
}

//------------------------------------------------------------------------------
// LG_TRY: check a condition and return on error
//------------------------------------------------------------------------------

// The msg is not modified.  This should be used when an LAGraph method calls
// another one.

#define LG_TRY(LAGraph_method)                  \
{                                               \
    int LAGraph_status = LAGraph_method ;       \
    if (LAGraph_status < 0)                     \
    {                                           \
        LG_FREE_ALL ;                           \
        return (LAGraph_status) ;               \
    }                                           \
}


#define CENTRALITY_MAX_ITER 100
#define HLL_P 10                   // Precision
#define HLL_REGISTERS (1 << HLL_P) // number of registers

LG_JIT_STRING(
typedef struct {
    uint8_t registers[(1 << 10)];
} HLL;
, HLL_jit)

static __inline uint8_t _hll_rank(uint32_t hash, uint8_t bits) {
    uint8_t i;

    for (i = 1; i <= 32 - bits; i++) {
        if (hash & 1)
            break;

        hash >>= 1;
    }

    return i;
}

static __inline void _hll_add_hash(HLL *hll, uint32_t hash) {
    uint32_t index = hash >> (32 - HLL_P);
    uint8_t rank = _hll_rank(hash, HLL_P);

    if (rank > hll->registers[index]) {
        hll->registers[index] = rank;
    }
}

LG_JIT_STRING (
void lg_hll_count(double *z, const HLL *x) {
    const HLL *hll = x;

    double alpha_mm = 0.7213 / (1.0 + 1.079 / (double)HLL_REGISTERS);

    alpha_mm *= ((double)HLL_REGISTERS * (double)HLL_REGISTERS);

    double sum = 0;
    for (uint32_t i = 0; i < HLL_REGISTERS; i++) {
        sum += 1.0 / (1 << hll->registers[i]);
    }

    double estimate = alpha_mm / sum;

    if (estimate <= 5.0 / 2.0 * (double)HLL_REGISTERS) {
        int zeros = 0;

        for (uint32_t i = 0; i < HLL_REGISTERS; i++)
            zeros += (hll->registers[i] == 0);

        if (zeros)
            estimate = (double)HLL_REGISTERS * log((double)HLL_REGISTERS / zeros);

    } else if (estimate > (1.0 / 30.0) * 4294967296.0) {
        estimate = -4294967296.0 * log(1.0 - (estimate / 4294967296.0)) ;
    }

    *z = estimate;
}, LG_HLL_COUNT)

//------------------------------------------------------------------------------
// GraphBLAS Ops
//------------------------------------------------------------------------------
#define GOLDEN_GAMMA 0x9E3779B97F4A7C15LL

// TODO: does this hash have good properties? Good spread is much more important
// than speed here.
uint64_t LG_HC_hash(GrB_Index i) {
    uint64_t result = (i + GOLDEN_GAMMA);
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9LL;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EBLL;
    result = (result ^ (result >> 31)) ;
    return result;
}

// no need to register in JIT, not proformance critical
// load a new HLL set with *x hashes in the set.
void lg_hll_init(HLL *z, const uint64_t *x, GrB_Index i, GrB_Index j,
                  bool theta) {
    int64_t weight = *x;
    memset(z->registers, 0, HLL_REGISTERS);
    // build a hash chain seeded from the node index i:
    // each iteration hashes the previous result, yielding 'weight'
    // distinct pseudo-random values for the HLL sketch
    uint32_t hash = LG_HC_hash(i);
    for (int64_t h = 0; h < weight; h++) {
        _hll_add_hash(z, hash);
        hash = LG_HC_hash(hash);
    }
}

LG_JIT_STRING(
void lg_hll_merge(HLL *z, const HLL *x, const HLL *y) {
    for (uint32_t i = 0; i < HLL_REGISTERS; i++) {
        z->registers[i] = y->registers[i] > x->registers[i] ?
                          y->registers[i] : x->registers[i];
    }
},
LG_HLL_MERGE_STR)

LG_JIT_STRING(
    void lg_hll_delta(double *z, const HLL *x, const HLL *y) {
    *z = 0;
    bool diff = 0 != memcmp(x->registers, y->registers, HLL_REGISTERS) ;
    if (diff) {
        uint32_t i;

        double alpha_mm = 0.7213 / (1.0 + 1.079 / (double)(1 << 10)) ;

        alpha_mm *= ((double)(HLL_REGISTERS) * (double)(HLL_REGISTERS)) ;

        double sum = 0;
        for (uint32_t i = 0; i < (HLL_REGISTERS); i++) {
            sum += 1.0 / (1 << x->registers[i]);
        }

        double estimate = alpha_mm / sum;

        if (estimate <= 5.0 / 2.0 * (double)(HLL_REGISTERS)) {
            int zeros = 0;

            for (i = 0; i < (HLL_REGISTERS); i++)
                zeros += (x->registers[i] == 0);

            if (zeros)
                estimate = (double)(HLL_REGISTERS) * log((double)(HLL_REGISTERS) / zeros);

        } else if (estimate > (1.0 / 30.0) * 4294967296.0) {
            estimate = -4294967296.0 * log(1.0 - (estimate / 4294967296.0)) ;
        }

        *z = estimate;

        sum = 0;
        for (uint32_t i = 0; i < (HLL_REGISTERS); i++) {
            sum += 1.0 / (1 << y->registers[i]);
        }

        estimate = alpha_mm / sum;

        if (estimate <= 5.0 / 2.0 * (double)(HLL_REGISTERS)) {
            int zeros = 0;

            for (i = 0; i < (HLL_REGISTERS); i++)
                zeros += (y->registers[i] == 0);

            if (zeros)
                estimate = (double)(HLL_REGISTERS) * log((double)(HLL_REGISTERS) / zeros);

        } else if (estimate > (1.0 / 30.0) * 4294967296.0) {
            estimate = -4294967296.0 * log(1.0 - (estimate / 4294967296.0)) ;
        }
        *z -= estimate;
    }
},
LG_HLL_DELTA_STR)

LG_JIT_STRING(
    void lg_hll_second(HLL *z, bool *x, const HLL *y) {
    memcpy(z->registers, y->registers, sizeof(z->registers));
},
LG_HLL_SECOND_STR)

// int64_t print_hll (
//     char *string,       // value is printed to the string
//     size_t string_size, // size of the string array
//     const void *value,  // HLL value to print
//     int verbose         // if >0, print verbosely; else tersely
// ) {
//     const HLL *hll = (const HLL *)value;
//
//     if (verbose > 0) {
//         return snprintf(string, string_size, "HLL{bits=%u, size=%u, count=%.2f}",
//                         HLL_P, HLL_REGISTERS, _hll_count(hll)) ;
//     }
//
//     return snprintf(string, string_size, "HLL{count=%.2f}", _hll_count(hll)) ;
// }

#undef LG_FREE_WORK
#define LG_FREE_WORK         \
GrB_free(&_A);               \
GrB_free(&score_cont);       \
GrB_free(&hll_t);            \
GrB_free(&new_sets);         \
GrB_free(&old_sets);         \
GrB_free(&old_cont);         \
GrB_free(&flat_scores);      \
GrB_free(&flat_weight);      \
GrB_free(&delta_vec);        \
GrB_free(&desc);             \
GrB_free(&init_hlls);        \
GrB_free(&shallow_second);   \
GrB_free(&merge_hll_biop);   \
GrB_free(&merge_hll);        \
GrB_free(&merge_second);     \
GrB_free(&delta_hll);        \
GrB_free(&count_hll);

#undef LG_FREE_ALL
#define LG_FREE_ALL          \
LG_FREE_WORK ;               \
GrB_free(scores) ;           \
GrB_free(reachable_nodes) ;

// compute harmonic closeness centrality estimates using HLL BFS propagation
//
// each node maintains an HLL sketch of "reachable nodes seen so far"
// at BFS level d, sketches are propagated along edges
// the harmonic contribution delta/d is accumulated into each node's score,
// where delta is the number of new nodes discovered at that level
int LAGr_HarmonicCentrality(
    // outputs:
    GrB_Vector *scores,          // FP64 scores by original node ID
    GrB_Vector *reachable_nodes, // [optional] estimate the number of reach-
                                 // able nodes from the given node.
    // inputs:
    const LAGraph_Graph G,         // input graph
    const GrB_Vector node_weights, // participating nodes and their weights
    char *msg
) {
    GrB_Matrix _A = NULL;
    GxB_Container score_cont = NULL;
    GrB_Index nrows = 0;
    GrB_Index nvals = 0;

    GrB_Type hll_t = NULL;
    GrB_Vector new_sets = NULL;
    GrB_Vector old_sets = NULL;
    GxB_Container old_cont = NULL;
    GrB_Vector flat_scores = NULL;
    GrB_Vector flat_weight = NULL;
    GrB_Vector delta_vec = NULL;

    GrB_Descriptor desc = NULL;
    GrB_IndexUnaryOp init_hlls = NULL;
    GrB_UnaryOp count_hll = NULL;
    GrB_BinaryOp shallow_second = NULL;
    GrB_BinaryOp merge_hll_biop = NULL;
    GrB_Monoid merge_hll = NULL;
    GrB_Semiring merge_second = NULL;
    GrB_BinaryOp delta_hll = NULL;

    LG_ASSERT(G != NULL, GrB_NULL_POINTER);
    LG_ASSERT(G->A != NULL, GrB_NULL_POINTER);
    LG_ASSERT(scores != NULL, GrB_NULL_POINTER);
    LG_ASSERT(node_weights != NULL, GrB_NULL_POINTER);

    GRB_TRY(GrB_Vector_size(&nrows, node_weights)) ;
    GRB_TRY(GrB_Vector_nvals(&nvals, node_weights)) ;
    GRB_TRY(GrB_Vector_new(scores, GrB_FP64, nrows)) ;

    if (nvals == 0) {
        return GrB_SUCCESS;
    }

    // double check weight type and maximum weight requirements
    GrB_Type weight_t = NULL;
    GRB_TRY(GxB_Vector_type(&weight_t, node_weights)) ;
    LG_ASSERT_MSG(weight_t == GrB_BOOL || weight_t == GrB_INT64,
                  GrB_DOMAIN_MISMATCH, "weight must be integer or uint64");

    if (weight_t == GrB_INT64) {
        int64_t max_w = 0, min_w = 0;
        GRB_TRY(GrB_Vector_reduce_INT64(&max_w, NULL, GrB_MAX_MONOID_INT64,
                                        node_weights, NULL)) ;
        GRB_TRY(GrB_Vector_reduce_INT64(&min_w, NULL, GrB_MIN_MONOID_INT64,
                                        node_weights, NULL)) ;
        LG_ASSERT_MSG(min_w >= 0, GrB_INVALID_VALUE,
                      "Negative node weights not supported");

        // TODO: is this cap reasonable?
        LG_ASSERT_MSG(max_w < 1000000, GrB_NOT_IMPLEMENTED,
                      "Node weights over 1000000 not supported");
    }

    //--------------------------------------------------------------------------
    // build compact submatrix _A
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Descriptor_new (&desc)) ;
    GRB_TRY (GrB_set (desc, GxB_USE_INDICES, GxB_COLINDEX_LIST)) ;
    GRB_TRY (GrB_set (desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST)) ;
    GRB_TRY (GrB_Matrix_new (&_A, GrB_BOOL, nvals, nvals)) ;
    GRB_TRY (GxB_Matrix_extract_Vector (
        _A, NULL, NULL, G->A, node_weights, node_weights, desc)) ;
    GRB_TRY(GrB_free(&desc)) ;

    //--------------------------------------------------------------------------
    // create scores vector (0.0 at each participating node)
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_assign_FP64 (
        *scores, node_weights, NULL, 0.0, GrB_ALL, 0, GrB_DESC_S)) ;
    GRB_TRY (GrB_set (*scores, GxB_SPARSE | GxB_FULL, GxB_SPARSITY_CONTROL)) ;
    GRB_TRY (GxB_Container_new (&score_cont)) ;
    GRB_TRY (GxB_unload_Vector_into_Container (*scores, score_cont, NULL)) ;

    //--------------------------------------------------------------------------
    // Initialize HLL
    //--------------------------------------------------------------------------

    GRB_TRY (GxB_Type_new (&hll_t, sizeof(HLL), "HLL", HLL_jit)) ;
    GRB_TRY (GrB_Vector_new (&new_sets, hll_t, nvals)) ;
    GRB_TRY (GrB_Vector_new (&old_sets, hll_t, nvals)) ;
    GRB_TRY (GxB_Container_new (&old_cont)) ;
    GRB_TRY (GrB_Vector_new (&flat_scores, GrB_FP64, nvals)) ;
    GRB_TRY (GrB_Vector_new (&flat_weight, GrB_INT64, nvals)) ;
    GRB_TRY (GxB_Vector_extractTuples_Vector (
        NULL, flat_weight, node_weights, NULL)) ;

    // count op
    GRB_TRY (GxB_UnaryOp_new (&count_hll, (GxB_unary_function) lg_hll_count,
        GrB_FP64, hll_t, "lg_hll_count", LG_HLL_COUNT)) ;

    // init op: weight (INT64) at row index i → HLL seeded with 'weight' hashes
    GRB_TRY(GrB_IndexUnaryOp_new(
        &init_hlls, (GxB_index_unary_function)lg_hll_init, hll_t,
        GrB_INT64, GrB_BOOL)) ;

    // merge binary op (HLL, HLL) -> HLL  in-place: z == x required
    GRB_TRY(GxB_BinaryOp_new(
        &merge_hll_biop, (GxB_binary_function)lg_hll_merge,
        hll_t, hll_t, hll_t, "lg_hll_merge", LG_HLL_MERGE_STR)) ;

    // second op
    GRB_TRY(GxB_BinaryOp_new(
        &shallow_second, (GxB_binary_function)lg_hll_second,
        hll_t, GrB_BOOL, hll_t, "lg_hll_second", LG_HLL_SECOND_STR)) ;

    // merge monoid - identity is an empty (all-zero) HLL sketch
    HLL hll_zero = {0};
    GRB_TRY(GrB_Monoid_new_UDT(&merge_hll, merge_hll_biop, &hll_zero)) ;

    // semiring: add = merge monoid, multiply = copy (pass-through second operand)
    GRB_TRY(GrB_Semiring_new(&merge_second, merge_hll, shallow_second)) ;

    // delta op: (HLL_old, HLL_new) → FP64 cardinality change
    GRB_TRY(GxB_BinaryOp_new(
        &delta_hll, (GxB_binary_function)lg_hll_delta,
        GrB_FP64, hll_t, hll_t, "lg_hll_delta", LG_HLL_DELTA_STR)) ;

    // delta output: one FP64 entry per participating node
    GRB_TRY(GrB_Vector_new(&delta_vec, GrB_FP64, nvals)) ;

    //--------------------------------------------------------------------------
    // Load vector values
    //--------------------------------------------------------------------------
    // initialize HLLs
    GRB_TRY(GrB_Vector_apply_IndexOp_BOOL(new_sets, NULL, NULL, init_hlls,
                                          flat_weight, false, NULL)) ;
    GRB_TRY(GrB_Vector_apply_IndexOp_BOOL(old_sets, NULL, NULL, init_hlls,
                                          flat_weight, false, NULL)) ;
    GRB_TRY(GrB_set(old_sets, GxB_BITMAP, GxB_SPARSITY_CONTROL)) ;

    GRB_TRY(GrB_free(&flat_weight)) ;
    GRB_TRY (GrB_Vector_assign_FP64 (
        flat_scores, NULL, NULL, 0.0, GrB_ALL, 0, NULL)) ;

    //--------------------------------------------------------------------------
    // HLL BFS propagation
    //--------------------------------------------------------------------------

    int64_t changes = 0;
    for (int d = 1; d <= CENTRALITY_MAX_ITER; d++) {
        changes = 0;

        // foward bfs
        // merge each neighbor's pre-round set into this node's set
        // target kernel for inplace adjustments
        GRB_TRY(GrB_mxv(
            new_sets, NULL, merge_hll_biop, merge_second, _A, old_sets, NULL)) ;

        GRB_TRY(GxB_unload_Vector_into_Container(old_sets, old_cont, NULL)) ;
        // find the delta between last round and this one

        GRB_TRY(GrB_eWiseMult(
            delta_vec, NULL, NULL, delta_hll, new_sets, old_cont->x, NULL)) ;
        GRB_TRY(GrB_Vector_assign(
            old_cont->x, NULL, NULL, new_sets, GrB_ALL, 0, NULL)) ;

        // old_set bitmap is the set of nodes with non-zero deltas
        GRB_TRY(GrB_apply(
            old_cont->b, NULL, NULL, GrB_IDENTITY_BOOL, delta_vec, NULL)) ;
        GRB_TRY(GrB_Vector_reduce_INT64(
            &changes, NULL, GrB_PLUS_MONOID_INT64, old_cont->b, NULL)) ;
        old_cont->nvals = changes;
        GRB_TRY(GxB_load_Vector_from_Container(old_sets, old_cont, NULL)) ;

        // use the deltas to update the score
        GRB_TRY(GrB_apply(flat_scores, NULL, GrB_PLUS_FP64,
            GrB_DIV_FP64, delta_vec, (double)d, NULL)) ;

        // stop when no HLL cardinality changed this round
        if (changes == 0) {
            break;
        }
    }

    //--------------------------------------------------------------------------
    // write flat_scores into score_cont->x, clear iso, reload scores vector
    //--------------------------------------------------------------------------
    GRB_TRY (GrB_free (&score_cont->x)) ;
    score_cont->iso = false;
    score_cont->x = flat_scores;
    flat_scores = NULL ;

    if (reachable_nodes) {
        // make reachable_nodes vector with same sparsity pattern as scores
        GRB_TRY (GrB_Vector_new (reachable_nodes, GrB_INT64, nrows)) ;
        GRB_TRY (GrB_apply (delta_vec, NULL, NULL, count_hll, new_sets, NULL)) ;
        GrB_Vector I_vec = (score_cont->format == GxB_FULL) ?
            NULL : score_cont->i;
        GRB_TRY (GxB_Vector_assign_Vector (
            *reachable_nodes, NULL, NULL, delta_vec, I_vec, NULL)) ;
    }

    GRB_TRY (GxB_load_Vector_from_Container (*scores, score_cont, NULL)) ;

    //--------------------------------------------------------------------------
    // cleanup
    //--------------------------------------------------------------------------

    // free operators (semiring before monoid)

    LG_FREE_WORK;
    return GrB_SUCCESS;
}
