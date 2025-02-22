//------------------------------------------------------------------------------
// GxB_Matrix_build_Scalar_Vector: build a sparse GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GxB_Matrix_build_Scalar_Vector builds a matrix C whose values in its
// sparsity pattern are all equal to a value given by a GrB_Scalar.  Unlike the
// GrB_Matrix_build_* methods, there is no binary dup operator.  Instead, any
// duplicate indices are ignored, which is not an error condition.  The I and J
// GrB_Vectors are of size nvals, just like GrB_Matrix_build_*.

#include "builder/GB_build.h"
#include "ij/GB_ij.h"
#define GB_FREE_ALL                             \
    if (I_size > 0) GB_FREE_MEMORY (&I, I_size) ;      \
    if (J_size > 0) GB_FREE_MEMORY (&J, J_size) ;

GrB_Info GxB_Matrix_build_Scalar_Vector // build a matrix from (I,J,s) tuples
(
    GrB_Matrix C,               // matrix to build
    const GrB_Vector I_vector,  // row indices
    const GrB_Vector J_vector,  // col indices
    const GrB_Scalar scalar,    // value for all tuples
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (scalar) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_WHERE4 (C, I_vector, J_vector, scalar,
        "GxB_Matrix_build_Scalar_Vector (C, I, J, scalar, nvals, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_build_Scalar") ;
    ASSERT (GB_VECTOR_OK (I_vector)) ;
    ASSERT (GB_VECTOR_OK (J_vector)) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    void *I = NULL, *J = NULL ;
    size_t I_size = 0, J_size = 0 ;

    GB_MATRIX_WAIT (scalar) ;
    if (GB_nnz ((GrB_Matrix) scalar) != 1)
    { 
        GB_ERROR (GrB_EMPTY_OBJECT, "Scalar value is %s", "missing") ;
    }

    GB_MATRIX_WAIT (I_vector) ;
    GB_MATRIX_WAIT (J_vector) ;
    int64_t nvals = GB_nnz ((GrB_Matrix) I_vector) ;
    int64_t jvals = GB_nnz ((GrB_Matrix) J_vector) ;
    if (nvals != jvals)
    { 
        GB_ERROR (GrB_INVALID_VALUE, "Input vectors I,J must have the "
            "same number of entries; nvals(I) = " GBd ", nvals(J) = " GBd,
            nvals, jvals) ;
    }

    //--------------------------------------------------------------------------
    // get the index vectors
    //--------------------------------------------------------------------------

    int64_t ni = 0, nj = 0 ;
    GrB_Type I_type = NULL, J_type = NULL ;
    GB_OK (GB_ijxvector (I_vector, false, 0, desc, true,
        &I, &ni, &I_size, &I_type, Werk)) ;
    GB_OK (GB_ijxvector (J_vector, false, 1, desc, true,
        &J, &nj, &J_size, &J_type, Werk)) ;
    bool I_is_32 = (I_type == GrB_UINT32) ;
    bool J_is_32 = (J_type == GrB_UINT32) ;

    //--------------------------------------------------------------------------
    // build the matrix, ignoring duplicates
    //--------------------------------------------------------------------------

    GB_OK (GB_build (C, I, J, scalar->x, nvals, GxB_IGNORE_DUP, scalar->type,
        /* is_matrix: */ true, /* X_iso: */ true, I_is_32, J_is_32, Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

