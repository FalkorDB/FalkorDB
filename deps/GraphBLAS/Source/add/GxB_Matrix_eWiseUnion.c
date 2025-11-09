//------------------------------------------------------------------------------
// GxB_Matrix_eWiseUnion: matrix element-wise operations, set union
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C,A+B) and variations.

// if A(i,j) and B(i,j) both appear:
//      C(i,j) = add (A(i,j), B(i,j))
// else if A(i,j) appears but B(i,j) does not:
//      C(i,j) = add (A(i,j), beta)
// else if A(i,j) does not appear but B(i,j) does:
//      C(i,j) = add (alpha, B(i,j))

// by contrast, GrB_eWiseAdd does the following:

// if A(i,j) and B(i,j) both appear:
//      C(i,j) = add (A(i,j), B(i,j))
// else if A(i,j) appears but B(i,j) does not:
//      C(i,j) = A(i,j)
// else if A(i,j) does not appear but B(i,j) does:
//      C(i,j) = B(i,j)

#include "ewise/GB_ewise.h"
#include "mask/GB_get_mask.h"

//------------------------------------------------------------------------------
// GxB_Matrix_eWiseUnion: matrix addition
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_eWiseUnion      // C<M> = accum (C, A+B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // defines '+' for T=A+B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar alpha,
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Scalar beta,
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL (B) ;
    GB_RETURN_IF_NULL (alpha) ;
    GB_RETURN_IF_NULL (beta) ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_WHERE6 (C, M_in, A, alpha, B, beta,
        "GxB_Matrix_eWiseUnion (C, M, accum, op, A, alpha,  B, beta, desc)") ;
    GB_BURBLE_START ("GxB_eWiseUnion") ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct, A_tran,
        B_tran, xx, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // apply the eWise kernel (using set union)
    //--------------------------------------------------------------------------

    // C<M> = accum (C,T) where T = A+B, A'+B, A+B', or A'+B'
    info = GB_ewise (
        C,              C_replace,  // C and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // accumulate operator
        op,                         // operator that defines '+'
        A,              A_tran,     // A matrix and its descriptor
        B,              B_tran,     // B matrix and its descriptor
        true,                       // eWiseAdd
        true, alpha, beta,          // eWiseUnion
        Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

