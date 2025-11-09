//------------------------------------------------------------------------------
// GxB_Vector_build_Scalar_Vector: build a sparse GraphBLAS vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GxB_Vector_build_Scalar_Vector builds a vector w whose values in its
// sparsity pattern are all equal to a value given by a GrB_Scalar.  Unlike the
// GrB_Vector_build_* methods, there is no binary dup operator.  Instead, any
// duplicate indices are ignored, which is not an error condition.  The I
// GrB_Vector is of size nvals, just like GrB_Vector_build_*.

#include "builder/GB_build.h"
#include "ij/GB_ij.h"
#define GB_FREE_ALL                             \
    if (I_size > 0) GB_FREE_MEMORY (&I, I_size) ;

GrB_Info GxB_Vector_build_Scalar_Vector // build a vector from (I,s) tuples
(
    GrB_Vector w,               // vector to build
    const GrB_Vector I_vector,  // row indices
    const GrB_Scalar scalar,    // value for all tuples
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (scalar) ;
    GB_RETURN_IF_NULL (I_vector) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;
    GB_WHERE3 (w, scalar, I_vector,
        "GxB_Vector_build_Scalar_Vector (w, I, scalar, desc)") ;
    GB_BURBLE_START ("GxB_Vector_build_Scalar_Vector") ;
    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (GB_VECTOR_OK (I_vector)) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    void *I = NULL ;
    size_t I_size = 0 ;

    GB_MATRIX_WAIT (scalar) ;
    if (GB_nnz ((GrB_Matrix) scalar) != 1)
    { 
        GB_ERROR (GrB_EMPTY_OBJECT, "Scalar value is %s", "missing") ;
    }

    //--------------------------------------------------------------------------
    // get the index vector
    //--------------------------------------------------------------------------

    int64_t ni = 0 ;
    GrB_Type I_type = NULL ;
    bool need_copy = (w == I_vector) ;
    GB_OK (GB_ijxvector (I_vector, need_copy, 0, desc, true,
        &I, &ni, &I_size, &I_type, Werk)) ;
    bool I_is_32 = (I_type == GrB_UINT32) ;

    //--------------------------------------------------------------------------
    // build the vector
    //--------------------------------------------------------------------------

    GB_OK (GB_build ((GrB_Matrix) w, I, NULL, scalar->x, ni,
        GxB_IGNORE_DUP, scalar->type,
        /* is_matrix: */ false, /* X_iso: */ true,
        /* I,J is 32: */ I_is_32, false, Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

