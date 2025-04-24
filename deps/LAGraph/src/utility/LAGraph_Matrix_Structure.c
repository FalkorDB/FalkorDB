//------------------------------------------------------------------------------
// LAGraph_Matrix_Structure: return the structure of a matrix
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis and Scott Kolodziej, Texas A&M University

//------------------------------------------------------------------------------

// LAGraph_Matrix_Structure: return the structure of a matrix as a boolean
// matrix, where C(i,j)=true if the entry A(i,j) is present in the matrix A.

#define LG_FREE_ALL GrB_free (C) ;

#include "LG_internal.h"

int LAGraph_Matrix_Structure
(
    // output:
    GrB_Matrix *C,  // a boolean matrix with same structure of A, with C(i,j)
                    // set to true if A(i,j) appears in the sparsity structure
                    // of A.
    // input:
    GrB_Matrix A,
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Index nrows, ncols ;
    LG_ASSERT_MSG (C != NULL, GrB_NULL_POINTER, "&C != NULL") ;
    LG_ASSERT (A != NULL, GrB_NULL_POINTER) ;
    (*C) = NULL ;

    //--------------------------------------------------------------------------
    // get the size of A
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_nrows (&nrows, A)) ;
    GRB_TRY (GrB_Matrix_ncols (&ncols, A)) ;

    //--------------------------------------------------------------------------
    // C<s(A)> = true
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_new (C, GrB_BOOL, nrows, ncols)) ;
    GRB_TRY (GrB_assign (*C, A, NULL, (bool) true,
        GrB_ALL, nrows, GrB_ALL, nrows, GrB_DESC_S)) ;

    return (GrB_SUCCESS) ;
}
