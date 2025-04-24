//------------------------------------------------------------------------------
// LG_check_kcore: construct the kcore of a graph (simple method)
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

// An implementation of the BZ algorithm (2003) for k-core decomposition.
// This method is for testing only, to check the result of other, faster methods.
// Do not benchmark this method; it is simple by design.

#define LG_FREE_ALL                             \
{                                               \
    LAGraph_Free ((void **) &vert, msg) ;       \
    LAGraph_Free ((void **) &deg, msg) ;        \
    LAGraph_Free ((void **) &bin, msg) ;        \
    LAGraph_Free ((void **) &pos, msg) ;        \
    LAGraph_Free ((void **) &Ap, msg) ;         \
    LAGraph_Free ((void **) &Aj, msg) ;         \
    LAGraph_Free ((void **) &Ax, msg) ;         \
}

#include "LG_internal.h"
#include "LG_test.h"
#include "LG_test.h"
#include "LG_Xtest.h"

int LG_check_kcore
(
    // outputs:
    GrB_Vector *decomp,     // kcore decomposition
    uint64_t *kmax,         // max kcore- if kfinal == -1, kmax = -1
    // inputs
    LAGraph_Graph G,        // input graph
    int64_t kfinal,         // max k to check for graph.
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    uint64_t *vert = NULL, *pos = NULL, *bin = NULL, *deg = NULL ;
    uint64_t maxDeg = 0;
    GrB_Index *Ap = NULL, *Aj = NULL, *Ai = NULL ;
    void *Ax = NULL ;
    GrB_Index Ap_size, Aj_size, Ax_size, n, ncols, Ap_len, Aj_len, Ax_len ;
    LG_ASSERT (kmax != NULL, GrB_NULL_POINTER) ;
    LG_TRY (LAGraph_CheckGraph (G, msg)) ;
    LG_ASSERT (G->nself_edges == 0, LAGRAPH_NO_SELF_EDGES_ALLOWED) ;
    LG_ASSERT_MSG ((G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGraph_ADJACENCY_DIRECTED &&
        G->is_symmetric_structure == LAGraph_TRUE)),
        LAGRAPH_SYMMETRIC_STRUCTURE_REQUIRED,
        "G->A must be known to be symmetric") ;
    GRB_TRY (GrB_Matrix_nrows (&n, G->A)) ; //set n to number of rows
    GRB_TRY (GrB_Matrix_ncols (&ncols, G->A)) ;

    //--------------------------------------------------------------------------
    // export the matrix in CSR form
    //--------------------------------------------------------------------------

    size_t typesize ;
    LG_TRY (LG_check_export (G, &Ap, &Aj, &Ax, &Ap_len, &Aj_len, &Ax_len,
        &typesize, msg)) ;

    //--------------------------------------------------------------------------
    // compute the k-core
    //--------------------------------------------------------------------------
    //printf ("\n================================== COMPUTING BZ_KCORE: ==================================\n") ;
    //printf("ap_len = %ld, aj_len = %ld, ax_len = %ld\n", Ap_len, Aj_len, Ax_len) ;

    //create the arrays

    LAGraph_Malloc((void **) &deg, n, sizeof(uint64_t), msg) ;
    LAGraph_Malloc((void **) &vert, n, sizeof(uint64_t), msg) ;
    LAGraph_Malloc((void **) &pos, n, sizeof(uint64_t), msg) ;
    //core = AGraph_Malloc(n, sizeof(uint64_t)) ;

    for(uint64_t i = 0; i < n; i++){
        deg[i] = Ap[i+1] - Ap[i];
        if (deg[i] > maxDeg)
            maxDeg = deg[i];
    }

    //setup output vector
    GrB_Type int_type  = (maxDeg > INT32_MAX) ? GrB_INT64 : GrB_INT32 ;
    GRB_TRY (GrB_Vector_new(decomp, int_type, n)) ;
    GrB_IndexUnaryOp valueGE = (maxDeg > INT32_MAX) ? GrB_VALUEGE_INT64 : GrB_VALUEGE_INT32 ;

    //setup bin array
    LAGraph_Calloc((void **) &bin, maxDeg + 1, sizeof(uint64_t), msg) ;

    for(uint64_t i = 0; i < n; i++){
        bin[deg[i]]++;
    }

    uint64_t start = 0;
    for(uint64_t d = 0; d < maxDeg + 1; d++){
        uint64_t num = bin[d];
        bin[d] = start;
        start = start + num;
    }

    //Do bin-sort
    //vert -- contains the vertices in sorted order of degree
    //pos -- contains the positon of a vertex in vert array
    for(uint64_t i = 0; i < n; i++){
        pos[i] = bin[ deg[i] ];
        vert[pos[i]] = i;
        bin[deg[i]] ++;
    }

    for(uint64_t d = maxDeg; d >= 1; d --)
        bin[d] = bin[d-1];
    bin[0] = 0;

    uint64_t level = 0;

    //Compute k-core
    for(uint64_t i = 0; i < n; i++){
        //get the vertex to check
        uint64_t v = vert[i];

        //set the element in the output vector: if doing KCALL, then add deg.
        //If not, just set to kfinal's value.
        if((int64_t) deg[v] >= kfinal){
            if(kfinal == -1){
                GRB_TRY(GrB_Vector_setElement(*decomp, deg[v], v)) ;
            }
            else{
                GRB_TRY(GrB_Vector_setElement(*decomp, kfinal, v)) ;
            }
        }

        if(bin[deg[v]] == i){
            level = deg[v];
        }

        uint64_t start = Ap[v];
        int64_t original_deg = Ap[v+1] - Ap[v]; //original deg before decremented
        for(uint64_t j = 0; j < original_deg; j++){
            uint64_t u = Aj[start + j]; //a neighbor node of v

            //if we need to lower the neighbor's deg value, and relocate in bin
            if(deg[u] > deg[v]){
                uint64_t du = deg[u];
                uint64_t pu = pos[u];
                uint64_t pw = bin[du];
                uint64_t w = vert[pw]; //the vertex situated at the beginning of the bin

                //swap around the vertices- w goes to the end, u goes to the beginning
                if(u != w){
                    pos[u] = pw; vert[pu] = w;
                    pos[w] = pu; vert[pw] = u;
                }

                //increase starting index of bin @ du
                bin[du]++;
                //decrease degree of u
                deg[u]--;
            }
        }
    }

    LG_FREE_ALL;
    (*kmax) = level ;
    GRB_TRY (GrB_Vector_wait(*decomp, GrB_MATERIALIZE));
    return (GrB_SUCCESS);
}
