//------------------------------------------------------------------------------
// GB_subassign_27: C<C,struct> += A, no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 27: C<C,struct> += A ; no S

// M:           present, and aliased exactly with C
// Mask_struct: true
// Mask_comp:   false
// C_replace:   false
// accum:       present
// A:           matrix
// S:           none

// C not bitmap; C can be full since no zombies are inserted in that case.
// If C is bitmap, then GB_bitmap_assign_M_accum is used instead.

#include "assign/GB_subassign_methods.h"
#include "jitifyer/GB_stringify.h"
#define GB_FREE_ALL ;

GrB_Info GB_subassign_27
(
    GrB_Matrix C,
    // input:
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix S = NULL ;           // not constructed
    ASSERT (!GB_IS_BITMAP (C)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (!GB_any_aliased (C, A)) ;   // NO ALIAS of C==A
    GB_UNJUMBLE (C) ;
    GB_UNJUMBLE (A) ;

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    info = GB_subassign_jit (C,
        /* C_replace: */ false,
        /* I, ni, nI, Ikind, Icolon: */ NULL, false, 0, 0, GB_ALL, NULL,
        /* J, nj, nJ, Jkind, Jcolon: */ NULL, false, 0, 0, GB_ALL, NULL,
        /* M is C: */ C,
        /* Mask_comp: */ false,
        /* Mask_struct: */ true,
        accum,
        A,
        /* scalar, scalar_type: */ NULL, NULL,
        /* S: */ NULL,
        GB_SUBASSIGN, GB_JIT_KERNEL_SUBASSIGN_27, "subassign_27",
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
    #include "assign/template/GB_subassign_27_template.c"
}

