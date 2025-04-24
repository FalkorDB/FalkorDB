//------------------------------------------------------------------------------
// LG_check_kcoredecompose: deconstruct the graph into a k-core, given a
// decompostion vector (simple method)
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

// This method is for testing only, to check the result of other, faster methods.
// Do not benchmark this method; it is slow and simple by design.

#define LG_FREE_WORK                \
{                                   \
    GrB_free (&C) ;                 \
    GrB_free (&deg) ;               \
}

#define LG_FREE_ALL                 \
{                                   \
    LG_FREE_WORK                    \
    GrB_free (D) ;                  \
}


#include "LG_internal.h"
#include "LG_test.h"
#include "LG_Xtest.h"

int LG_check_kcore_decompose
(
    // outputs:
    GrB_Matrix *D,              // kcore decomposition
    // inputs:
    LAGraph_Graph G,            // input graph
    GrB_Vector decomp,
    uint64_t k,
    char *msg
)
{
    LG_CLEAR_MSG ;

    // declare items
    GrB_Matrix A = NULL, C = NULL;
    GrB_Vector deg = NULL;

    LG_ASSERT (D != NULL, GrB_NULL_POINTER) ;
    (*D) = NULL ;

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
    GrB_Index size_matrix, nvals_matrix, size_vector, nvals_vector;
    GRB_TRY (GrB_Matrix_nrows(&size_matrix, A)) ;
    GRB_TRY (GrB_Vector_size(&size_vector, decomp)) ;

    LG_ASSERT_MSG (size_matrix == size_vector, -1003, "Size of vector and of matrix must be same") ;

    //create D and nvals scalars
    GRB_TRY (GrB_Matrix_new(D, GrB_INT64, size_matrix, size_matrix)) ;
    GRB_TRY (GrB_Matrix_nvals(&nvals_matrix, A)) ;
    GRB_TRY (GrB_Vector_nvals(&nvals_vector, decomp));

    //extract out the values of the input graph
    GrB_Index *row = NULL, *col = NULL , *matrix_values = NULL, *vector = NULL, *vector_values = NULL;
    LG_TRY (LAGraph_Malloc ((void **) &row, nvals_matrix, sizeof (GrB_Index), msg));
    LG_TRY (LAGraph_Malloc ((void **) &col, nvals_matrix, sizeof (GrB_Index), msg));
    LG_TRY (LAGraph_Malloc ((void **) &matrix_values, nvals_matrix, sizeof (GrB_Index), msg));

    LG_TRY (LAGraph_Malloc ((void **) &vector, nvals_vector, sizeof (GrB_Index), msg));
    LG_TRY (LAGraph_Malloc ((void **) &vector_values, nvals_vector, sizeof (GrB_Index), msg));

    GRB_TRY(GrB_Matrix_extractTuples(row, col, (int64_t *) matrix_values, &nvals_matrix, A));
    GRB_TRY(GrB_Vector_extractTuples(vector, (int64_t *) vector_values, &size_vector, decomp));
    //take all values that have row and col indices
    for(uint64_t i = 0; i < nvals_matrix; i++){
        bool ok_row = false, ok_col = false;
        for(uint64_t j = 0; (j < nvals_vector) && (!ok_row || !ok_col); j++){
            if(row[i] == vector[j] && vector_values[j] >= k)
                ok_row = true;
            if(col[i] == vector[j] && vector_values[j] >= k)
                ok_col = true;
        }
        if(ok_row && ok_col){
            GRB_TRY(GrB_Matrix_setElement(*D, matrix_values[i], row[i], col[i]));
        }
    }
    LG_FREE_WORK;
    GRB_TRY (GrB_Matrix_wait(*D, GrB_MATERIALIZE));
    return (GrB_SUCCESS);
}
