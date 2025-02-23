//------------------------------------------------------------------------------
// GxB_Col_subassign_Vector: C(I,j)<M> = accum (C(I,j),u)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "assign/GB_subassign.h"
#include "mask/GB_get_mask.h"
#include "ij/GB_ij.h"
#define GB_FREE_ALL                             \
    if (I_size > 0) GB_FREE_MEMORY (&I, I_size) ;

GrB_Info GxB_Col_subassign_Vector   // C(I,j)<M> = accum (C(I,j),u)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for C(I,j), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(I,j),t)
    const GrB_Vector u,             // input vector
    const GrB_Vector I_vector,      // row indices
    uint64_t j,                     // column index
    const GrB_Descriptor desc       // descriptor for C(I,j) and mask
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (u) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_WHERE4 (C, mask, u, I_vector,
        "GxB_Col_subassign_Vector (C, M, accum, u, I, ni, j, desc)") ;
    GB_BURBLE_START ("GxB_subassign") ;

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
    GB_OK (GB_ijxvector (I_vector, false, 0, desc, false,
        &I, &ni, &I_size, &I_type, Werk)) ;
    bool I_is_32 = (I_type == GrB_UINT32) ;

    //--------------------------------------------------------------------------
    // C(I,j)<M> = accum (C(I,j), u)
    //--------------------------------------------------------------------------

    // construct the index list J = [ j ] of length nj = 1
    uint64_t J [1] ;
    J [0] = j ;

    GB_OK (GB_subassign (
        C, C_replace,                   // C matrix and its descriptor
        M, Mask_comp, Mask_struct,      // mask and its descriptor
        false,                          // do not transpose the mask
        accum,                          // for accum (C(I,j),u)
        (GrB_Matrix) u, false,          // u as a matrix; never transposed
        I, I_is_32, ni,                 // row indices
        J, false, 1,                    // a single column index
        false, NULL, GB_ignore_code,    // no scalar expansion
        Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

