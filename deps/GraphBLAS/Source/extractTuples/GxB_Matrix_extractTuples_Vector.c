//------------------------------------------------------------------------------
// GxB_Matrix_extractTuples_Vector: extract all tuples from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Extracts all tuples from a matrix, like [I,J,X] = find (A) in MATLAB.  If
// any parameter I, J and/or X is NULL, then that component is not extracted.
// I, J, and X are GrB_Vectors, and on output they are dense vectors of
// length (nvals (A)), with types revised to match the content of A.
//
// X is returned with the same type A.  I is returned as GrB_UINT32 if the # of
// rows of A is <= INT32_MAX, or GrB_UINT64 otherwise.  J is returned as
// GrB_UINT32 if the # of columns of A is <= INT32_MAX, or GrB_UINT64
// otherwise.

// If any parameter I, J, and/or X is NULL, that component is not extracted.
// For example, to extract just the row and col indices, pass I and J as
// non-NULL, and X as NULL.  This is like [I,J,~] = find (A) in MATLAB.

// If A is iso and X is not NULL, the iso scalar Ax [0] is expanded into X.

#include "GB.h"
#include "extractTuples/GB_extractTuples.h"
#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_extractTuples_Vector    // [I,J,X] = find (A)
(
    GrB_Vector I_vector,    // row indices
    GrB_Vector J_vector,    // col indices
    GrB_Vector X_vector,    // values
    const GrB_Matrix A,     // matrix to extract tuples from
    const GrB_Descriptor desc   // currently unused; for future expansion
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (I_vector) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (J_vector) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (X_vector) ;
    GB_WHERE_4 (I_vector, J_vector, X_vector, A,
        "GxB_Matrix_extractTuples_Vector (I, J, X, A, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_extractTuples_Vector") ;
    ASSERT_MATRIX_OK (A, "A for GxB_Matrix_extractTuples_Vector", GB0) ;

    if ((I_vector != NULL && (I_vector == X_vector || I_vector == J_vector)) ||
        (J_vector != NULL && (J_vector == X_vector)))
    { 
        return (GrB_NOT_IMPLEMENTED) ;  // input vectors cannot be aliased
    }

    //--------------------------------------------------------------------------
    // finish any pending work in A
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;

    //--------------------------------------------------------------------------
    // prepare the I, J, X vectors
    //--------------------------------------------------------------------------

    uint64_t nvals = GB_nnz (A) ;
    int64_t nrows = GB_NROWS (A) ;
    int64_t ncols = GB_NCOLS (A) ;
    bool I_is_32 = (nrows <= INT32_MAX) ;
    bool J_is_32 = (ncols <= INT32_MAX) ;
    GrB_Type I_type = (I_is_32) ? GrB_UINT32 : GrB_UINT64 ;
    GrB_Type J_type = (J_is_32) ? GrB_UINT32 : GrB_UINT64 ;
    GrB_Type X_type = A->type ;

    GB_OK (GB_extractTuples_prep (I_vector, nvals, I_type)) ;
    GB_OK (GB_extractTuples_prep (J_vector, nvals, J_type)) ;
    GB_OK (GB_extractTuples_prep (X_vector, nvals, X_type)) ;

    void *I = (I_vector == NULL) ? NULL : I_vector->x ;
    void *J = (J_vector == NULL) ? NULL : J_vector->x ;
    void *X = (X_vector == NULL) ? NULL : X_vector->x ;

    //--------------------------------------------------------------------------
    // extract tuples into the I, J, X vectors
    //--------------------------------------------------------------------------

    GB_OK (GB_extractTuples (I, I_is_32, J, J_is_32, X, &nvals, X_type, A,
        Werk)) ;

    //--------------------------------------------------------------------------
    // return results
    //--------------------------------------------------------------------------

    ASSERT_VECTOR_OK_OR_NULL (I_vector, "I: Matrix_extractTuples_Vector", GB0) ;
    ASSERT_VECTOR_OK_OR_NULL (J_vector, "J: Matrix_extractTuples_Vector", GB0) ;
    ASSERT_VECTOR_OK_OR_NULL (X_vector, "X: Matrix_extractTuples_Vector", GB0) ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

