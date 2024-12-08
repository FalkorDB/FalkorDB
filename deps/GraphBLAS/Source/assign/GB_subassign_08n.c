//------------------------------------------------------------------------------
// GB_subassign_08n: C(I,J)<M> += A ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 08n: C(I,J)<M> += A ; no S

// M:           present
// Mask_struct: true or false
// Mask_comp:   false
// C_replace:   false
// accum:       present
// A:           matrix
// S:           none

// C not bitmap; C can be full since no zombies are inserted in that case.
// If C is bitmap, then GB_bitmap_assign_M_accum is used instead.
// M, A: not bitmap; Method 08s is used instead if M or A are bitmap.

#include "assign/GB_subassign_methods.h"
#include "jitifyer/GB_stringify.h"
#define GB_FREE_ALL ;

GrB_Info GB_subassign_08n
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
    const GrB_BinaryOp accum,
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
    ASSERT (!GB_IS_BITMAP (C)) ;
    ASSERT (!GB_IS_BITMAP (M)) ;    // Method 08s is used if M is bitmap
    ASSERT (!GB_IS_BITMAP (A)) ;    // Method 08s is used if A is bitmap
    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M
    ASSERT (!GB_any_aliased (C, A)) ;   // NO ALIAS of C==A
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
        accum,
        A,
        /* scalar, scalar_type: */ NULL, NULL,
        /* S: */ NULL,
        GB_SUBASSIGN, GB_JIT_KERNEL_SUBASSIGN_08n, "subassign_08n",
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
    #include "assign/template/GB_subassign_08n_template.c"
}

