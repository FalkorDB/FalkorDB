//------------------------------------------------------------------------------
// GxB_Vector_subassign_Vector: w(I)<M> = accum (w(I),u)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Compare with GrB_Vector_assign, which uses M and C_replace differently

#include "assign/GB_subassign.h"
#include "mask/GB_get_mask.h"
#include "ij/GB_ij.h"
#define GB_FREE_ALL                             \
    if (I_size > 0) GB_FREE_MEMORY (&I, I_size) ;

GrB_Info GxB_Vector_subassign_Vector // w(I)<mask> = accum (w(I),u)
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w(I),t)
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector I_vector,      // row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (u) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;
    GB_WHERE4 (w, mask, u, I_vector,
        "GxB_Vector_subassign_Vector (w, M, accum, u, I, desc)") ;
    GB_BURBLE_START ("GxB_Vector_subassign_Vector") ;

    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (mask == NULL || GB_VECTOR_OK (mask)) ;
    ASSERT (GB_VECTOR_OK (u)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) mask, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // get the index vector
    //--------------------------------------------------------------------------

    void *I = NULL ;
    size_t I_size = 0 ;
    int64_t ni = 0 ;
    GrB_Type I_type = NULL ;
    GB_OK (GB_ijxvector (I_vector, (w == I_vector), 0, desc, false,
        &I, &ni, &I_size, &I_type, Werk)) ;
    bool I_is_32 = (I_type == GrB_UINT32) ;

    //--------------------------------------------------------------------------
    // w(I)<M> = accum (w(I), u)
    //--------------------------------------------------------------------------

    GB_OK (GB_subassign (
        (GrB_Matrix) w, C_replace,      // w vector and its descriptor
        M, Mask_comp, Mask_struct,      // mask and its descriptor
        false,                          // do not transpose the mask
        accum,                          // for accum (w(I),u)
        (GrB_Matrix) u, false,          // u as a matrix; never transposed
        I, I_is_32, ni,                 // row indices
        GrB_ALL, false, 1,              // all column indices
        false, NULL, GB_ignore_code,    // no scalar expansion
        Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

