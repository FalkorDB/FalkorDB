//------------------------------------------------------------------------------
// GxB_Matrix_build_Vector: build a sparse GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If dup is NULL: any duplicates result in an error.
// If dup is GxB_IGNORE_DUP: duplicates are ignored, which is not an error.
// If dup is a valid binary operator, it is used to reduce any duplicates to
// a single value.

#include "builder/GB_build.h"
#include "ij/GB_ij.h"
#define GB_FREE_ALL                             \
    if (I_size > 0) GB_FREE_MEMORY (&I, I_size) ;      \
    if (J_size > 0) GB_FREE_MEMORY (&J, J_size) ;      \
    if (X_size > 0) GB_FREE_MEMORY (&X, X_size) ;

GrB_Info GxB_Matrix_build_Vector // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,               // matrix to build
    const GrB_Vector I_vector,  // row indices
    const GrB_Vector J_vector,  // col indices
    const GrB_Vector X_vector,  // values
    const GrB_BinaryOp dup,     // binary function to assemble duplicates
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (I_vector) ;
    GB_RETURN_IF_NULL (J_vector) ;
    GB_RETURN_IF_NULL (X_vector) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_WHERE4 (C, I_vector, J_vector, X_vector,
        "GxB_Matrix_build_Vector (C, I, J, X, dup, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_build_Vector") ;
    ASSERT_VECTOR_OK (I_vector, "I_vector for build", GB0) ;
    ASSERT_VECTOR_OK (J_vector, "J_vector for build", GB0) ;
    ASSERT_VECTOR_OK (X_vector, "X_vector for build", GB0) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    void *I = NULL, *J = NULL, *X = NULL ;
    size_t I_size = 0, J_size = 0, X_size = 0 ;

    GB_MATRIX_WAIT (I_vector) ;
    GB_MATRIX_WAIT (J_vector) ;
    GB_MATRIX_WAIT (X_vector) ;
    int64_t nvals = GB_nnz ((GrB_Matrix) X_vector) ;
    int64_t ivals = GB_nnz ((GrB_Matrix) I_vector) ;
    int64_t jvals = GB_nnz ((GrB_Matrix) J_vector) ;
    if (nvals != ivals || nvals != jvals)
    { 
        GB_ERROR (GrB_INVALID_VALUE, "Input vectors I,J must have the "
            "same number of entries; nvals(I) = " GBd ", nvals(J) = " GBd
            " nvals(X) = " GBd, ivals, jvals, nvals) ;
    }

    //--------------------------------------------------------------------------
    // get the index vectors
    //--------------------------------------------------------------------------

    int64_t ni = 0, nj = 0, nx = 0 ;
    GrB_Type I_type = NULL, J_type = NULL, X_type = NULL ;
    GB_OK (GB_ijxvector (I_vector, false, 0, desc, true,
        &I, &ni, &I_size, &I_type, Werk)) ;
    GB_OK (GB_ijxvector (J_vector, false, 1, desc, true,
        &J, &nj, &J_size, &J_type, Werk)) ;
    GB_OK (GB_ijxvector (X_vector, false, 2, desc, true,
        &X, &nx, &X_size, &X_type, Werk)) ;
    bool I_is_32 = (I_type == GrB_UINT32) ;
    bool J_is_32 = (J_type == GrB_UINT32) ;

    // FUTURE: if they come from List->i, then I,J,X are known to be sorted
    // with no duplicates.  Exploit this in GB_build.

    // FUTURE: if I,J,X have been allocated, they can be ingested into C;
    // exploit this in GB_build.

    //--------------------------------------------------------------------------
    // build the matrix
    //--------------------------------------------------------------------------

    GB_OK (GB_build (C, I, J, X, nvals, dup, X_type, /* is_matrix: */ true,
        /* X iso: */ false, I_is_32, J_is_32, Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

