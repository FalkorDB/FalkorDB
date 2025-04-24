//------------------------------------------------------------------------------
// LG_Xtest.h: include file for LAGraphX test library
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

#ifndef LG_XTEST_H
#define LG_XTEST_H

#include "LAGraphX.h"

int LG_check_mis        // check if iset is a valid MIS of A
(
    GrB_Matrix A,
    GrB_Vector iset,
    GrB_Vector ignore_node,     // if NULL, no nodes are ignored.  otherwise,
                        // ignore_node(i)=true if node i is to be ignored, and
                        // not added to the independent set.
    char *msg
) ;

int LG_check_ktruss
(
    // output
    GrB_Matrix *C_handle,   // the ktruss of G->A, of type GrB_UINT32
    // input
    LAGraph_Graph G,        // the structure of G->A must be symmetric
    uint32_t k,
    char *msg
) ;

int LG_check_kcore
(
    // outputs:
    GrB_Vector *decomp,     // kcore decomposition
    uint64_t *kmax,         // max kcore- if kfinal == -1, kmax = -1
    // inputs
    LAGraph_Graph G,        // input graph
    int64_t kfinal,         // max k to check for graph.
    char *msg
) ;

int LG_check_kcore_decompose
(
    // outputs:
    GrB_Matrix *D,              // kcore decomposition
    // inputs:
    LAGraph_Graph G,            // input graph
    GrB_Vector decomp,
    uint64_t k,
    char *msg
) ;

#endif
