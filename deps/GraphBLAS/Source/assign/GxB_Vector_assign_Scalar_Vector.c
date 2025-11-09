//------------------------------------------------------------------------------
// GxB_Vector_assign_Scalar_Vector: assign scalar to vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Assigns a single scalar to a vector, w<M>(Rows) = accum(w(Rows),x)
// The scalar x is implicitly expanded into a vector u of size ni-by-1,
// with each entry in u equal to x.

#include "assign/GB_assign.h"
#include "ij/GB_ij.h"
#define GB_FREE_ALL                             \
    if (I_size > 0) GB_FREE_MEMORY (&I, I_size) ;

GrB_Info GxB_Vector_assign_Scalar_Vector   // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    const GrB_Scalar scalar,        // scalar to assign to w(I)
    const GrB_Vector I_vector,      // row indices
    const GrB_Descriptor desc       // descriptor for w and mask
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE4 (w, mask, scalar, I_vector,
        "GxB_Vector_assign_Scalar_Vector (w, M, accum, s, I, desc)") ;
    GB_BURBLE_START ("GxB_Vector_assign_Scalar_Vector") ;

    //--------------------------------------------------------------------------
    // get the index vectors
    //--------------------------------------------------------------------------

    void *I = NULL ;
    size_t I_size = 0 ;
    int64_t ni = 0 ;
    GrB_Type I_type = NULL ;
    GB_OK (GB_ijxvector (I_vector, (w == I_vector), 0, desc, false,
        &I, &ni, &I_size, &I_type, Werk)) ;
    bool I_is_32 = (I_type == GrB_UINT32) ;

    //--------------------------------------------------------------------------
    // w<M>(I) = accum (w(I), scalar)
    //--------------------------------------------------------------------------

    GB_OK (GB_Vector_assign_scalar (w, mask, accum, scalar,
        I, I_is_32, ni, desc, Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

