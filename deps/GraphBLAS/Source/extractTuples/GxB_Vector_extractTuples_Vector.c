//------------------------------------------------------------------------------
// GxB_Vector_extractTuples_Vector: extract all tuples from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Extracts all tuples from a column vector, like [I,~,X] = find (v) in MATLAB.
// If any parameter I and/or X is NULL, then that component is not extracted.
// I and X are GrB_Vectors, and on output they are dense vectors of
// length (nvals (V)), with types revised to match the content of V.
//
// X is returned with the same type V.  I is returned as GrB_UINT32 if the
// length of V is <= INT32_MAX, or GrB_UINT64 otherwise.

// If any parameter I and/or X is NULL, that component is not extracted.  For
// example, to extract just the row indices, pass I as non-NULL, and X as NULL.
// This is like [I,~,~] = find (v) in MATLAB.

// If v is iso and X is not NULL, the iso scalar vx [0] is expanded into X.

#include "GB.h"
#include "extractTuples/GB_extractTuples.h"
#define GB_FREE_ALL ;

GrB_Info GxB_Vector_extractTuples_Vector    // [I,~,X] = find (V)
(
    GrB_Vector I_vector,    // row indices
    GrB_Vector X_vector,    // values
    const GrB_Vector V,     // vectors to extract tuples from
    const GrB_Descriptor desc   // currently unused; for future expansion
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (V) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (I_vector) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (X_vector) ;
    GB_WHERE_3 (I_vector, X_vector, V,
        "GxB_Vector_extractTuples_Vector (I, J, X, A, desc)") ;
    GB_BURBLE_START ("GxB_Vector_extractTuples_Vector") ;
    ASSERT_VECTOR_OK (V, "V for GxB_Vector_extractTuples_Vector", GB0) ;

    if (V == I_vector || V == X_vector ||
        (I_vector != NULL && I_vector == X_vector))
    { 
        return (GrB_NOT_IMPLEMENTED) ;  // input vectors cannot be aliased
    }

    //--------------------------------------------------------------------------
    // finish any pending work in V
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (V) ;

    //--------------------------------------------------------------------------
    // prepare the I, X vectors
    //--------------------------------------------------------------------------

    uint64_t nvals = GB_nnz ((GrB_Matrix) V) ;
    int64_t nrows = V->vlen ;
    bool I_is_32 = (nrows <= INT32_MAX) ;
    GrB_Type I_type = (I_is_32) ? GrB_UINT32 : GrB_UINT64 ;
    GrB_Type X_type = V->type ;

    GB_OK (GB_extractTuples_prep (I_vector, nvals, I_type)) ;
    GB_OK (GB_extractTuples_prep (X_vector, nvals, X_type)) ;

    void *I = (I_vector == NULL) ? NULL : I_vector->x ;
    void *X = (X_vector == NULL) ? NULL : X_vector->x ;

    //--------------------------------------------------------------------------
    // extract tuples into the I, X vectors
    //--------------------------------------------------------------------------

    GB_OK (GB_extractTuples (I, I_is_32, NULL, false, X, &nvals, X_type,
        (GrB_Matrix) V, Werk)) ;

    //--------------------------------------------------------------------------
    // return results
    //--------------------------------------------------------------------------

    ASSERT_VECTOR_OK_OR_NULL (I_vector, "I: Vector_extractTuples_Vector", GB0) ;
    ASSERT_VECTOR_OK_OR_NULL (X_vector, "X: Vector_extractTuples_Vector", GB0) ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

