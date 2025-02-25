//------------------------------------------------------------------------------
// GxB_Vector_build_Vector: build a sparse GraphBLAS vector
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
    if (X_size > 0) GB_FREE_MEMORY (&X, X_size) ;

GrB_Info GxB_Vector_build_Vector // build a vector from (I,X) tuples
(
    GrB_Vector w,               // vector to build
    const GrB_Vector I_vector,  // row indices
    const GrB_Vector X_vector,  // values
    const GrB_BinaryOp dup,     // binary function to assemble duplicates
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (I_vector) ;
    GB_RETURN_IF_NULL (X_vector) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;
    GB_WHERE3 (w, I_vector, X_vector,
        "GxB_Vector_build_Vector (w, I, X, dup, desc)") ;
    GB_BURBLE_START ("GxB_Vector_build_Vector") ;
    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (GB_VECTOR_OK (I_vector)) ;
    ASSERT (GB_VECTOR_OK (X_vector)) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    void *I = NULL, *X = NULL ;
    size_t I_size = 0, X_size = 0 ;

    GB_MATRIX_WAIT (I_vector) ;
    GB_MATRIX_WAIT (X_vector) ;
    int64_t nvals = GB_nnz ((GrB_Matrix) X_vector) ;
    int64_t ivals = GB_nnz ((GrB_Matrix) I_vector) ;
    if (nvals != ivals)
    { 
        GB_ERROR (GrB_INVALID_VALUE, "Input vectors I,X must have the "
            "same number of entries; nvals(I) = " GBd ", nvals(X) = " GBd,
            ivals, nvals) ;
    }

    //--------------------------------------------------------------------------
    // get the index vectors
    //--------------------------------------------------------------------------

    int64_t ni = 0, nx = 0 ;
    GrB_Type I_type = NULL, X_type = NULL ;
    bool need_copy = (w == I_vector || w == X_vector) ;
    GB_OK (GB_ijxvector (I_vector, need_copy, 0, desc, true,
        &I, &ni, &I_size, &I_type, Werk)) ;
    GB_OK (GB_ijxvector (X_vector, need_copy, 2, desc, true,
        &X, &nx, &X_size, &X_type, Werk)) ;
    bool I_is_32 = (I_type == GrB_UINT32) ;

    // FUTURE: if they come from List->i, then I,X are known to be sorted
    // with no duplicates.  Exploit this in GB_build.

    // FUTURE: if I,X have been allocated, they can be ingested into w.
    // Exploit this in GB_build.

    //--------------------------------------------------------------------------
    // build the vector
    //--------------------------------------------------------------------------

    GB_OK (GB_build ((GrB_Matrix) w, I, NULL, X, nvals, dup, X_type,
        /* is_matrix: */ false, /* X iso: */ false, I_is_32, false, Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

