//------------------------------------------------------------------------------
// GxB_Matrix_assign_Vector: C<M>(I,J) = accum (C(I,J),A) or A'
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "assign/GB_assign.h"
#include "mask/GB_get_mask.h"
#include "ij/GB_ij.h"
#define GB_FREE_ALL                             \
    if (I_size > 0) GB_FREE_MEMORY (&I, I_size) ;      \
    if (J_size > 0) GB_FREE_MEMORY (&J, J_size) ;

GrB_Info GxB_Matrix_assign_Vector   // C<Mask>(I,J) = accum (C(I,J),A)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),T)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Vector I_vector,      // row indices
    const GrB_Vector J_vector,      // column indices
    const GrB_Descriptor desc       // descriptor for C, Mask, and A
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_WHERE5 (C, Mask, A, I_vector, J_vector,
        "GxB_Matrix_assign_Vector (C, M, accum, A, I, J, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_assign_Vector") ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (Mask, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // get the index vectors
    //--------------------------------------------------------------------------

    void *I = NULL, *J = NULL ;
    size_t I_size = 0, J_size = 0 ;
    int64_t ni = 0, nj = 0 ;
    GrB_Type I_type = NULL, J_type = NULL ;
    GB_OK (GB_ijxvector (I_vector, false, 0, desc, false,
        &I, &ni, &I_size, &I_type, Werk)) ;
    GB_OK (GB_ijxvector (J_vector, false, 1, desc, false,
        &J, &nj, &J_size, &J_type, Werk)) ;
    bool I_is_32 = (I_type == GrB_UINT32) ;
    bool J_is_32 = (J_type == GrB_UINT32) ;

    //--------------------------------------------------------------------------
    // C<M>(I,J) = accum (C(I,J), A)
    //--------------------------------------------------------------------------

    GB_OK (GB_assign (
        C, C_replace,                   // C matrix and its descriptor
        M, Mask_comp, Mask_struct,      // mask matrix and its descriptor
        false,                          // do not transpose the mask
        accum,                          // for accum (C(I,J),A)
        A, A_transpose,                 // A and its descriptor (T=A or A')
        I, I_is_32, ni,                 // row indices
        J, J_is_32, nj,                 // column indices
        false, NULL, GB_ignore_code,    // no scalar expansion
        GB_ASSIGN,
        Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

