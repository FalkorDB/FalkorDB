//------------------------------------------------------------------------------
// GB_subassign_03: C(I,J) += scalar ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 03: C(I,J) += scalar ; using S

// M:           NULL
// Mask_comp:   false
// C_replace:   false
// accum:       present
// A:           scalar
// S:           constructed

// C: not bitmap

#include "assign/GB_subassign_methods.h"
#include "jitifyer/GB_stringify.h"
#define GB_FREE_ALL GB_Matrix_free (&S) ;

GrB_Info GB_subassign_03
(
    GrB_Matrix C,
    // input:
    #define C_replace false
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
    #define M NULL
    #define Mask_comp false
    #define Mask_struct true
    const GrB_BinaryOp accum,
    #define A NULL
    const void *scalar,
    const GrB_Type scalar_type,
    #define assign_kind GB_SUBASSIGN
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix S = NULL ;
    ASSERT (!GB_IS_BITMAP (C)) ;

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
        /* C_replace: */ false,
        I, I_is_32, ni, nI, Ikind, Icolon,
        J, J_is_32, nj, nJ, Jkind, Jcolon,
        /* M: */ NULL,
        /* Mask_comp: */ false,
        /* Mask_struct: */ true,
        accum,
        /* A: */ NULL,
        scalar, scalar_type,
        S,
        GB_SUBASSIGN, GB_JIT_KERNEL_SUBASSIGN_03, "subassign_03",
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
    #define GB_SCALAR_ASSIGN 1
    #include "assign/include/GB_assign_shared_definitions.h"
    #include "assign/template/GB_subassign_03_template.c"
}

