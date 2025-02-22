//------------------------------------------------------------------------------
// GxB_Row_subassign_Vector: C(i,J)<M'> = accum (C(i,J),u')
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "assign/GB_subassign.h"
#include "mask/GB_get_mask.h"
#include "ij/GB_ij.h"
#define GB_FREE_ALL                             \
    if (J_size > 0) GB_FREE_MEMORY (&J, J_size) ;

GrB_Info GxB_Row_subassign_Vector   // C(i,J)<mask'> = accum (C(i,J),u')
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for C(i,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(i,J),t)
    const GrB_Vector u,             // input vector
    uint64_t i,                     // row index
    const GrB_Vector J_vector,      // column indices
    const GrB_Descriptor desc       // descriptor for C(i,J) and mask
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (u) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_WHERE4 (C, mask, u, J_vector,
        "GxB_Row_subassign_Vector (C, M, accum, u, i, J, desc)") ;
    GB_BURBLE_START ("GxB_Row_subassign_Vector") ;

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

    void *J = NULL ;
    size_t J_size = 0 ;
    int64_t nj = 0 ;
    GrB_Type J_type = NULL ;
    GB_OK (GB_ijxvector (J_vector, false, 1, desc, false,
        &J, &nj, &J_size, &J_type, Werk)) ;
    bool J_is_32 = (J_type == GrB_UINT32) ;

    //--------------------------------------------------------------------------
    // C(i,J)<M'> = accum (C(i,J), u')
    //--------------------------------------------------------------------------

    // construct the index list I = [ i ] of length ni = 1
    uint64_t I [1] ;
    I [0] = i ;

    GB_OK (GB_subassign (
        C, C_replace,                   // C matrix and its descriptor
        M, Mask_comp, Mask_struct,      // mask and its descriptor
        true,                           // transpose the mask
        accum,                          // for accum (C(i,J),u)
        (GrB_Matrix) u, true,           // u as a matrix; always transposed
        I, false, 1,                    // a single row index
        J, J_is_32, nj,                 // column indices
        false, NULL, GB_ignore_code,    // no scalar expansion
        Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

