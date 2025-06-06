//------------------------------------------------------------------------------
// GB_subassign_12_and_20: C(I,J)<M or !M,repl> += A ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 12: C(I,J)<M,repl> += A ; using S
// Method 20: C(I,J)<!M,repl> += A ; using S

// M:           present
// Mask_stuct:  true or false
// Mask_comp:   true or false
// C_replace:   true
// accum:       present
// A:           matrix
// S:           constructed

// C: not bitmap: use GB_bitmap_assign instead
// M, A: any sparsity structure.

#include "assign/GB_subassign_methods.h"
#include "jitifyer/GB_stringify.h"
#define GB_FREE_ALL GB_Matrix_free (&S) ;

GrB_Info GB_subassign_12_and_20
(
    GrB_Matrix C,
    // input:
    #define C_replace true
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_comp,           // if true, !M, else use M
    const bool Mask_struct,         // if true, use the only structure of M
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
    GrB_Matrix S = NULL ;
    ASSERT (!GB_IS_BITMAP (C)) ; ASSERT (!GB_IS_FULL (C)) ;
    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M
    ASSERT (!GB_any_aliased (C, A)) ;   // NO ALIAS of C==A
    GB_UNJUMBLE (M) ;
    GB_UNJUMBLE (A) ;

    //--------------------------------------------------------------------------
    // S = C(I,J)
    //--------------------------------------------------------------------------

    struct GB_Matrix_opaque S_header ;
    GB_CLEAR_MATRIX_HEADER (S, &S_header) ;
    GB_OK (GB_subassign_symbolic (S, C, I, I_is_32, ni, J, J_is_32, nj, true,
        Werk)) ;

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    info = GB_subassign_jit (C,
        /* C_replace: */ true,
        I, I_is_32, ni, nI, Ikind, Icolon,
        J, J_is_32, nj, nJ, Jkind, Jcolon,
        M,
        Mask_comp,
        Mask_struct,
        accum,
        A,
        /* scalar, scalar_type: */ NULL, NULL,
        S,
        GB_SUBASSIGN, GB_JIT_KERNEL_SUBASSIGN_12, "subassign_12",
        Werk) ;
    if (info != GrB_NO_VALUE)
    { 
        GB_FREE_ALL ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    GB_IDECL (I, const, u) ; GB_IPTR (I, I_is_32) ;
    GB_IDECL (J, const, u) ; GB_IPTR (J, J_is_32) ;

    GBURBLE ("(generic assign) ") ;
    #define GB_GENERIC
    #define GB_SCALAR_ASSIGN 0
    #include "assign/include/GB_assign_shared_definitions.h"
    #include "assign/template/GB_subassign_12_template.c"
}

