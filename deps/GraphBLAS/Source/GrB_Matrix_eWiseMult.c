//------------------------------------------------------------------------------
// GrB_Matrix_eWiseMult: matrix element-wise operations, using set intersection
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C,A.*B) and variations.

#include "GB_ewise.h"
#include "GB_get_mask.h"

#define GB_EWISE(op)                                                        \
    /* check inputs */                                                      \
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;                                       \
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;                                       \
    GB_RETURN_IF_NULL_OR_FAULTY (B) ;                                       \
    GB_RETURN_IF_FAULTY (M_in) ;                                            \
    /* get the descriptor */                                                \
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,       \
        A_tran, B_tran, xx, xx7) ;                                          \
    /* get the mask */                                                      \
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;           \
    /* C<M> = accum (C,T) where T = A.*B, A'.*B, A.*B', or A'.*B' */        \
    info = GB_ewise (                                                       \
        C,              C_replace,  /* C and its descriptor        */       \
        M, Mask_comp, Mask_struct,  /* mask and its descriptor     */       \
        accum,                      /* accumulate operator         */       \
        op,                         /* operator that defines '.*'  */       \
        A,              A_tran,     /* A matrix and its descriptor */       \
        B,              B_tran,     /* B matrix and its descriptor */       \
        false,                      /* eWiseMult                   */       \
        false, NULL, NULL,          /* not eWiseUnion              */       \
        Werk) ;

//------------------------------------------------------------------------------
// GrB_Matrix_eWiseMult_BinaryOp: matrix element-wise multiplication
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_eWiseMult_BinaryOp       // C<M> = accum (C, A.*B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp mult,        // defines '.*' for T=A.*B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GrB_Matrix_eWiseMult_BinaryOp "
        "(C, M, accum, mult, A, B, desc)") ;
    GB_BURBLE_START ("GrB_eWiseMult") ;
    GB_RETURN_IF_NULL_OR_FAULTY (mult) ;

    //--------------------------------------------------------------------------
    // apply the eWise kernel (using set intersection)
    //--------------------------------------------------------------------------

    GB_EWISE (mult) ;
    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_eWiseMult_Monoid: matrix element-wise multiplication
//------------------------------------------------------------------------------

// C<M> = accum (C,A.*B) and variations.

GrB_Info GrB_Matrix_eWiseMult_Monoid         // C<M> = accum (C, A.*B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Monoid monoid,        // defines '.*' for T=A.*B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GrB_Matrix_eWiseMult_Monoid "
        "(C, M, accum, monoid, A, B, desc)") ;
    GB_BURBLE_START ("GrB_eWiseMult") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;

    //--------------------------------------------------------------------------
    // eWise multiply using the monoid operator
    //--------------------------------------------------------------------------

    GB_EWISE (monoid->op) ;
    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_eWiseMult_Semiring: matrix element-wise multiplication
//------------------------------------------------------------------------------

// C<M> = accum (C,A.*B) and variations.

GrB_Info GrB_Matrix_eWiseMult_Semiring       // C<M> = accum (C, A.*B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '.*' for T=A.*B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GrB_Matrix_eWiseMult_Semiring "
        "(C, M, accum, semiring, A, B, desc)") ;
    GB_BURBLE_START ("GrB_eWiseMult") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;

    //--------------------------------------------------------------------------
    // eWise multiply using the semiring multiply operator
    //--------------------------------------------------------------------------

    GB_EWISE (semiring->multiply) ;
    GB_BURBLE_END ;
    return (info) ;
}

