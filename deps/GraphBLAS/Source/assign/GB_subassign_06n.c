//------------------------------------------------------------------------------
// GB_subassign_06n: C(I,J)<M> = A ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 06n: C(I,J)<M> = A ; no S

// M:           present
// Mask_comp:   false
// C_replace:   false
// accum:       NULL
// A:           matrix
// S:           none (see also GB_subassign_06s)

// FULL: if A and C are dense, then C remains dense.

// If A is sparse and C dense, C will likely become sparse, except if M(i,j)=0
// wherever A(i,j) is not present.  So if M==A is aliased and A is sparse, then
// C remains dense.  Need C(I,J)<A,struct>=A kernel.  Then in that case, if C
// is dense it remains dense, even if A is sparse.   If that change is made,
// this kernel can start with converting C to sparse if A is sparse.

// C is not bitmap: GB_bitmap_assign is used if C is bitmap.
// M and A are not bitmap: 06s is used instead, if M or A are bitmap.

#include "assign/GB_subassign_methods.h"
#include "jitifyer/GB_stringify.h"
#define GB_FREE_ALL ;

GrB_Info GB_subassign_06n
(
    GrB_Matrix C,
    // input:
    #define C_replace false
    const GrB_Index *I,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    #define Mask_comp false
    const bool Mask_struct,
    #define accum NULL
    const GrB_Matrix A,
    #define scalar NULL
    #define scalar_type NULL
    #define assign_kind GB_SUBASSIGN
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix S = NULL ;           // not constructed
    ASSERT (!GB_IS_BITMAP (C)) ; ASSERT (!GB_IS_FULL (C)) ;
    ASSERT (!GB_IS_BITMAP (M)) ;    // Method 06n is not used for M bitmap
    ASSERT (!GB_IS_BITMAP (A)) ;    // Method 06n is not used for A bitmap
    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M
    ASSERT (!GB_any_aliased (C, A)) ;   // NO ALIAS of C==A

    ASSERT_MATRIX_OK (C, "C input for 06n", GB0) ;
    ASSERT_MATRIX_OK (M, "M input for 06n", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for 06n", GB0) ;

    GB_UNJUMBLE (C) ;
    GB_UNJUMBLE (M) ;
    GB_UNJUMBLE (A) ;

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    info = GB_subassign_jit (C,
        /* C_replace: */ false,
        I, ni, nI, Ikind, Icolon,
        J, nj, nJ, Jkind, Jcolon,
        M,
        /* Mask_comp: */ false,
        Mask_struct,
        /* accum: */ NULL,
        A,
        /* scalar, scalar_type: */ NULL, NULL,
        /* S: */ NULL,
        GB_SUBASSIGN, GB_JIT_KERNEL_SUBASSIGN_06n, "subassign_06n",
        Werk) ;
    if (info != GrB_NO_VALUE)
    { 
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    GBURBLE ("(generic assign) ") ;
    #define GB_GENERIC
    #define GB_SCALAR_ASSIGN 0
    #include "assign/include/GB_assign_shared_definitions.h"
    #include "assign/template/GB_subassign_06n_template.c"
}

